"""Clean RAPTOR run folders — drop failed/interrupted runs and reclaim their disk.

RAPTOR uses the SAME simple no-reuse layout as ReadAgent/RLM/CodeAct: each task run is ONE
self-contained folder ``logs/{benchmark}/raptor/{run_tag}/`` holding the ``manifest.json``
(success) OR ``error.json`` (failure), ``calls.json``, the built ``tree.json``, a
live ``progress.json``, and — for the offline-embed phase — the ``embed.json`` + ``leaves.pkl``
checkpoint. There is **no ``inferences/`` subfolder and no shared ``_indices/`` store**; the runner
**resumes** (skips tasks already done for the config) but never reuses a tree.

So this cleaner removes JUNK run folders — a ``ChildCrash`` / empty-or-unparseable ``error.json`` /
incomplete (neither manifest nor error, e.g. a tree build the child was killed mid-way) /
transient-error run — while KEEPING successful runs (``manifest.json``) and the genuine model
failures (``ContextWindowExceededError``). Removing a folder takes its ``tree.json`` / ``leaves.pkl``
with it; clearing a junk/transient folder also lets resumption re-run that task.

**Two raptor-specific phase-split safety rules** (raptor is the one baseline here with a
phase split, so these two rules live only in this cleaner):

1. A phase-1 ``--phase embed`` checkpoint (``embed.json`` + ``leaves.pkl``, no manifest/error yet)
   would classify as ``"incomplete"`` junk by file presence — a ``keep_if`` rescue KEEPS it so
   cleaning between phases never wipes the (heavy) offline leaf-embedding work that ``--phase build``
   is waiting to consume.
2. When ``--phase build`` fails it writes ``error.json`` into the SAME folder that holds the phase-1
   ``embed.json`` + ``leaves.pkl``. Deleting that folder wholesale would throw away the expensive
   embed work, so such a build failure is **reverted, not deleted**: only its ``error.json`` is
   removed, leaving a clean phase-1 checkpoint ``--phase build`` re-picks-up (reusing its
   ``run_tag``) — the embed work survives and the failed build still retries
   (``_revert_failed_builds``). A build failure with NO checkpoint (e.g. ``--phase all``) is deleted
   normally.

The per-folder classification is the SAME ``scripts._clean_common.classify_inference`` the other
cleaners use. ``--all-errors`` also removes the genuine-failure folders; ``--dry-run`` previews.

    python scripts/clean_raptor_logs.py                  # all raptor benchmarks
    python scripts/clean_raptor_logs.py --benchmark loong
    python scripts/clean_raptor_logs.py --dry-run        # preview, delete nothing
    python scripts/clean_raptor_logs.py --all-errors     # also clear real errors
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct invocation as a path script (`python scripts/clean_raptor_logs.py`),
# where sys.path[0] is scripts/ not the repo root, to import the sibling
# `scripts._clean_common`. Harmless under `-m` / pytest (root already importable).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clean_common import classify_inference, clean_inference_dirs  # noqa: E402
from evals.baselines.raptor.run import _EMBED_FILE  # noqa: E402  (the phase-1 checkpoint marker)
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "raptor"

# A build-phase failure leaves an error.json beside the phase-1 embed.json (+ leaves.pkl). These are
# the deletion categories such a failure produces — a transient vLLM error, a hard child crash, or a
# half-written error record. (NOT "incomplete": a checkpoint with no error is the awaiting-build case
# the keep_if rescue already handles; NOT "real_error": a ContextWindowExceeded build is kept anyway.)
_BUILD_FAILURE_CATEGORIES = ("transient", "crash", "broken")


def _revert_failed_builds(base_dir: Path, *, all_errors: bool, dry_run: bool) -> list[Path]:
    """Revert checkpoint-backed build failures instead of deleting them.

    When ``--phase build`` fails it writes ``error.json`` into the SAME run folder that already holds
    the phase-1 ``embed.json`` + ``leaves.pkl``. The default transient/crash policy would ``rmtree``
    the whole folder — discarding the expensive leaf-embedding work for that one task. Instead, for
    such a folder, remove ONLY the ``error.json``: that reverts it to a clean phase-1 checkpoint
    ``--phase build`` re-picks-up (reusing its ``run_tag``), so the embed work survives and the failed
    build still retries.

    Only touches folders with an ``embed.json`` and no ``manifest.json`` whose error classifies as a
    build failure (``_BUILD_FAILURE_CATEGORIES``). Returns the reverted folders; removes nothing under
    ``dry_run``. A build failure with NO checkpoint (e.g. ``--phase all``) has no embed work to protect
    and is left to normal deletion.
    """
    reverted: list[Path] = []
    for folder in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        if (folder / "manifest.json").exists():
            continue  # a successful build — never touched
        if not (folder / _EMBED_FILE).exists():
            continue  # no checkpoint to protect — leave to normal cleanup
        if classify_inference(folder, all_errors) in _BUILD_FAILURE_CATEGORIES:
            reverted.append(folder)
            if not dry_run:
                (folder / "error.json").unlink()
    return reverted


def clean_raptor_logs(
    logs_dir: Path,
    *,
    benchmark: str | None = None,
    all_errors: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove junk RAPTOR run folders (and the tree/checkpoint each holds) under ``logs_dir``.

    Scans ``logs_dir/{benchmark}/raptor/{run_tag}/`` (every benchmark when ``benchmark`` is None).
    Like ReadAgent/RLM/CodeAct there is no ``inferences/`` level: the run folders are the baseline
    dir's direct children, so the shared ``clean_inference_dirs`` is pointed straight at ``base_dir``
    — each run folder is classified by its ``manifest.json`` / ``error.json`` exactly like an
    inference folder, and ``rmtree``-ing a junk one removes its ``tree.json`` too. Phase-1 ``embed``
    checkpoints are rescued/reverted (see module docstring). Returns ``{deleted: [(Path, category)],
    reverted, kept, by_category, dry_run}``; ``dry_run`` removes nothing.
    """
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    deleted: list[tuple[Path, str]] = []
    reverted: list[Path] = []
    by_category: Counter[str] = Counter()
    kept = 0

    for base_dir in sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}")):
        if not base_dir.is_dir():
            continue
        total = sum(1 for p in base_dir.iterdir() if p.is_dir())
        # First preserve the embed work of any checkpoint-backed build failure (drop only its
        # error.json), so the whole-folder cleanup below never discards it.
        run_reverted = _revert_failed_builds(base_dir, all_errors=all_errors, dry_run=dry_run)
        run_deleted, _ = clean_inference_dirs(
            base_dir, all_errors=all_errors, dry_run=dry_run,
            # A phase-1 `embed.json` checkpoint (awaiting `--phase build`) has no manifest/error yet —
            # it would classify as "incomplete" junk. Rescue it so cleaning between the two phases
            # never wipes the (expensive) offline leaf-embedding work.
            keep_if=lambda d: (d / _EMBED_FILE).exists(),
        )
        # Under --dry-run the error.json is still on disk, so clean_inference_dirs would have listed a
        # reverted folder as a would-delete transient/crash; drop those so it's reported as reverted,
        # not deleted (in a real run its error.json is already gone, so this is a no-op there).
        reverted_set = set(run_reverted)
        run_deleted = [(d, c) for d, c in run_deleted if d not in reverted_set]
        deleted.extend(run_deleted)
        reverted.extend(run_reverted)
        for _, category in run_deleted:
            by_category[category] += 1
        kept += total - len(run_deleted) - len(run_reverted)

    return {
        "deleted": deleted,
        "reverted": reverted,
        "kept": kept,
        "by_category": dict(by_category),
        "dry_run": dry_run,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    """Per-benchmark one-liner. A run folder is ``{benchmark}/raptor/{run_tag}`` — there is no
    ``inferences/`` level, so ``parents[1].name`` is the benchmark (not ``parents[2]``)."""
    verb = "would delete" if summary["dry_run"] else "deleted"
    revert_verb = "would revert" if summary["dry_run"] else "reverted"
    per_bench: dict[str, Counter[str]] = defaultdict(Counter)
    for run_dir, category in summary["deleted"]:
        per_bench[run_dir.parents[1].name][category] += 1

    n_reverted = len(summary.get("reverted", []))
    revert_line = (
        f"{revert_verb} {n_reverted} build-failure(s) to their phase-1 checkpoint "
        f"(kept the embed work)" if n_reverted else None
    )

    if not summary["deleted"]:
        base = f"nothing to clean (kept {summary['kept']} run folder(s) intact)"
        return f"{revert_line}\n{base}" if revert_line else base

    lines: list[str] = []
    for bench in sorted(per_bench):
        cats = per_bench[bench]
        breakdown = ", ".join(f"{cats[c]} {c}" for c in sorted(cats, key=lambda c: -cats[c]))
        lines.append(f"{bench}: {verb} {sum(cats.values())} run folder(s) ({breakdown})")
    if revert_line:
        lines.append(revert_line)
    lines.append(f"kept {summary['kept']} run folder(s) intact (manifests + real errors)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/clean_raptor_logs.py",
        description="Delete junk RAPTOR run folders (crash / incomplete / transient) and the "
                    "tree each holds, reclaiming disk (keeps successful runs + real errors like "
                    "ContextWindowExceededError; preserves phase-1 embed checkpoints).",
    )
    parser.add_argument("--benchmark", default=None,
                        help="Only this benchmark (default: all raptor benchmarks).")
    parser.add_argument("--all-errors", action="store_true",
                        help="Also delete REAL error folders (e.g. ContextWindowExceededError) "
                             "for a full clean of every failed task.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — print what would be removed, delete nothing.")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    summary = clean_raptor_logs(
        logs_dir, benchmark=args.benchmark, all_errors=args.all_errors, dry_run=args.dry_run,
    )
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
