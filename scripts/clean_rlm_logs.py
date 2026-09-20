"""Clean RLM run folders — drop failed/interrupted runs and reclaim their disk.

RLM uses the SAME simple no-reuse layout as LinearRAG/ReadAgent: each task run is ONE
self-contained folder ``logs/{benchmark}/rlm/{run_tag}/`` holding the ``manifest.json``
(success) OR ``error.json`` (failure), ``calls.json``, ``score.json``, AND RLM's full
trajectory (the native ``RLMLogger`` ``.jsonl``, written live — every turn, code block, REPL
output, and sub-call). There is **no ``inferences/`` subfolder and no shared
``_indices/`` store**; the runner **resumes** (skips tasks already done for the config).

So this cleaner removes JUNK run folders — a ``ChildCrash`` / empty-or-unparseable ``error.json`` /
incomplete (neither manifest nor error, e.g. a run the child was killed mid-trajectory, or a hung
REPL cell SLURM-killed) / transient-error run — while KEEPING successful runs (``manifest.json``)
and the genuine model failures (``ContextWindowExceededError``). Removing a folder takes its
(sometimes large) trajectory with it; clearing a junk/transient folder also lets resumption re-run
that task.

The per-folder classification is the SAME ``scripts._clean_common.classify_inference`` the other
cleaners use. ``--all-errors`` also removes the genuine-failure folders; ``--max-iter`` ALSO
removes *successful* runs that hit the iteration cap (``n_iterations >= max_iterations`` — the
same judgment the analysis ⁺ view counts as a failure) so those capped tasks re-run; ``--dry-run``
previews.

    python scripts/clean_rlm_logs.py                  # all rlm benchmarks
    python scripts/clean_rlm_logs.py --benchmark loong
    python scripts/clean_rlm_logs.py --dry-run        # preview, delete nothing
    python scripts/clean_rlm_logs.py --all-errors     # also clear real errors
    python scripts/clean_rlm_logs.py --max-iter       # also clear runs that hit max_iterations
    python scripts/clean_rlm_logs.py --max-iter --dry-run   # preview which capped runs would go
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct invocation as a path script (`python scripts/clean_rlm_logs.py`),
# where sys.path[0] is scripts/ not the repo root, to import the sibling
# `scripts._clean_common`. Harmless under `-m` / pytest (root already importable).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clean_common import clean_inference_dirs  # noqa: E402
from evals.baselines._common import hit_step_cap  # noqa: E402
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "rlm"


def _capped_max_iter(run_dir: Path) -> str | None:
    """``extra_remove`` predicate for ``--max-iter``: return ``"max_iter"`` if this run's
    ``manifest.json`` shows the task EXHAUSTED its iteration budget
    (``n_iterations >= max_iterations``), else ``None``.

    Uses the SHARED ``_common.hit_step_cap`` — the SAME judgment the analysis ⁺ view
    counts as a failed prediction — so the cleaner and the report agree on what "capped"
    means. A folder with no/unreadable ``manifest.json`` (an ``error.json`` run) returns
    ``None`` (untouched here — the normal classification already handled it)."""
    mf = run_dir / "manifest.json"
    if not mf.exists():
        return None
    try:
        manifest = json.loads(mf.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return "max_iter" if hit_step_cap(manifest) else None


def clean_rlm_logs(
    logs_dir: Path,
    *,
    benchmark: str | None = None,
    all_errors: bool = False,
    dry_run: bool = False,
    max_iter: bool = False,
) -> dict[str, Any]:
    """Remove junk RLM run folders (and the trajectory each holds) under ``logs_dir``.

    Scans ``logs_dir/{benchmark}/rlm/{run_tag}/`` (every benchmark when ``benchmark`` is None).
    Like LinearRAG/ReadAgent there is no ``inferences/`` level: the run folders are the baseline
    dir's direct children, so the shared ``clean_inference_dirs`` is pointed straight at
    ``base_dir`` — each run folder is classified by its ``manifest.json`` / ``error.json`` exactly
    like an inference folder, and ``rmtree``-ing a junk one removes its trajectory too. Returns
    ``{deleted: [(Path, category)], kept, by_category, dry_run}``; ``dry_run`` removes nothing.

    ``max_iter`` ADDITIONALLY removes *successful* run folders whose task exhausted its
    iteration budget (``n_iterations >= max_iterations`` — category ``"max_iter"``), so those
    capped tasks re-run (e.g. after raising ``--max-iterations``). Off by default; the cap
    judgment is the shared ``_common.hit_step_cap`` the analysis ⁺ view uses.
    """
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    deleted: list[tuple[Path, str]] = []
    by_category: Counter[str] = Counter()
    kept = 0

    for base_dir in sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}")):
        if not base_dir.is_dir():
            continue
        run_deleted, run_kept = clean_inference_dirs(
            base_dir, all_errors=all_errors, dry_run=dry_run,
            extra_remove=_capped_max_iter if max_iter else None,
        )
        deleted.extend(run_deleted)
        for _, category in run_deleted:
            by_category[category] += 1
        kept += run_kept

    return {
        "deleted": deleted,
        "kept": kept,
        "by_category": dict(by_category),
        "dry_run": dry_run,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    """Per-benchmark one-liner. A run folder is ``{benchmark}/rlm/{run_tag}`` — there is no
    ``inferences/`` level, so ``parents[1].name`` is the benchmark (not ``parents[2]``)."""
    verb = "would delete" if summary["dry_run"] else "deleted"
    per_bench: dict[str, Counter[str]] = defaultdict(Counter)
    for run_dir, category in summary["deleted"]:
        per_bench[run_dir.parents[1].name][category] += 1

    if not summary["deleted"]:
        return f"nothing to clean (kept {summary['kept']} run folder(s) intact)"

    lines: list[str] = []
    for bench in sorted(per_bench):
        cats = per_bench[bench]
        breakdown = ", ".join(f"{cats[c]} {c}" for c in sorted(cats, key=lambda c: -cats[c]))
        lines.append(f"{bench}: {verb} {sum(cats.values())} run folder(s) ({breakdown})")
    lines.append(f"kept {summary['kept']} run folder(s) intact (manifests + real errors)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/clean_rlm_logs.py",
        description="Delete junk RLM run folders (crash / incomplete / transient) and the "
                    "trajectory each holds, reclaiming disk (keeps successful runs + real errors "
                    "like ContextWindowExceededError).",
    )
    parser.add_argument("--benchmark", default=None,
                        help="Only this benchmark (default: all rlm benchmarks).")
    parser.add_argument("--all-errors", action="store_true",
                        help="Also delete REAL error folders (e.g. ContextWindowExceededError) "
                             "for a full clean of every failed task.")
    parser.add_argument("--max-iter", action="store_true",
                        help="Also delete SUCCESSFUL runs whose task hit the iteration cap "
                             "(n_iterations >= max_iterations) so they re-run — e.g. before a "
                             "re-run at a higher --max-iterations. Honors --dry-run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — print what would be removed, delete nothing.")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    summary = clean_rlm_logs(
        logs_dir, benchmark=args.benchmark, all_errors=args.all_errors, dry_run=args.dry_run,
        max_iter=args.max_iter,
    )
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
