"""Clean HippoRAG 2 run folders — drop failed/interrupted runs and reclaim their disk.

HippoRAG uses the SAME simple no-reuse layout as ReadAgent/RLM/MemAgent: each task run is
ONE self-contained folder ``logs/{benchmark}/hipporag/{run_tag}/`` holding the
``manifest.json`` (success) OR ``error.json`` (failure), ``calls.json``,
AND the HippoRAG index itself (the OpenIE knowledge graph + embedding stores under
``index/``). There is **no ``inferences/`` subfolder and no shared index store**.

This cleaner tidies the dir and **reclaims disk** (the per-task ``index/`` graphs are the
bulk of it): it removes JUNK run folders — a ``ChildCrash`` / empty-or-unparseable
``error.json`` / incomplete (neither manifest nor error, e.g. a run killed mid-indexing) /
transient-error run — while KEEPING successful runs (``manifest.json``) and the genuine
model failures (``ContextWindowExceededError``, still counted as a failed prediction by
whatever grades these logs later). Removing a junk folder takes its ``index/`` with it.
``--all-errors`` also removes genuine-failure folders; ``--dry-run`` previews.

    python scripts/clean_hipporag_logs.py                  # all hipporag benchmarks
    python scripts/clean_hipporag_logs.py --benchmark loong
    python scripts/clean_hipporag_logs.py --dry-run        # preview, delete nothing
    python scripts/clean_hipporag_logs.py --all-errors     # also clear real errors
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct invocation as a path script, importing the sibling `scripts._clean_common`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clean_common import clean_inference_dirs  # noqa: E402
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "hipporag"


def clean_hipporag_logs(
    logs_dir: Path,
    *,
    benchmark: str | None = None,
    all_errors: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove junk HippoRAG run folders (and the ``index/`` graph each holds) under
    ``logs_dir``.

    Scans ``logs_dir/{benchmark}/hipporag/{run_tag}/`` (every benchmark when ``benchmark``
    is None). The run folders are the baseline dir's direct children (no ``inferences/``
    level), so the shared ``clean_inference_dirs`` is pointed straight at ``base_dir`` —
    each run folder is classified by its ``manifest.json`` / ``error.json`` exactly like an
    inference folder, and ``rmtree``-ing a junk one removes its ``index/`` too. Returns
    ``{deleted, kept, by_category, dry_run}``.
    """
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    deleted: list[tuple[Path, str]] = []
    by_category: Counter[str] = Counter()
    kept = 0

    for base_dir in sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}")):
        if not base_dir.is_dir():
            continue
        run_deleted, run_kept = clean_inference_dirs(
            base_dir, all_errors=all_errors, dry_run=dry_run
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
    """Per-benchmark one-liner. A run folder is ``{benchmark}/hipporag/{run_tag}`` — no
    ``inferences/`` level, so ``parents[1].name`` is the benchmark."""
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
        prog="python scripts/clean_hipporag_logs.py",
        description="Delete junk HippoRAG run folders (crash / incomplete / transient) and the "
                    "OpenIE-graph index each holds, reclaiming disk (keeps successful runs + real "
                    "errors like ContextWindowExceededError).",
    )
    parser.add_argument("--benchmark", default=None,
                        help="Only this benchmark (default: all hipporag benchmarks).")
    parser.add_argument("--all-errors", action="store_true",
                        help="Also delete REAL error folders (e.g. ContextWindowExceededError) "
                             "for a full clean of every failed task.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — print what would be removed, delete nothing.")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    summary = clean_hipporag_logs(
        logs_dir, benchmark=args.benchmark, all_errors=args.all_errors, dry_run=args.dry_run,
    )
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
