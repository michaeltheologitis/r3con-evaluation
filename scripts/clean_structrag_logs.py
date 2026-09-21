"""Clean structrag inference folders whose task should re-run.

An interrupted run or a transient error leaves an inference folder the resumption
scan still counts as "done" — so the task never re-runs:

- a ``ChildCrash`` ``error.json`` (the child died on a signal, e.g. SIGINT, mid
  request — the parent recorded it);
- an **empty / unparseable** ``error.json`` (a half-written file the interrupt
  caught mid-flush);
- a folder with **neither** ``manifest.json`` nor ``error.json`` (incomplete);
- a **transient** ``error.json`` — any ``error_type`` outside the
  ``FAILURE_ERROR_TYPES`` whitelist shared with whatever grades these logs later
  (e.g. a litellm ``Timeout``): retryable infra noise, not a model failure.

This removes those folders (so the runner re-dispatches just those tasks) while
KEEPING real results and genuine model failures: any folder with a
``manifest.json`` (an answer) and any error that counts as a genuine failed
prediction (``ContextWindowExceededError``) are left intact.
``--all-errors`` also removes those genuine-failure folders for a full retry.

structrag is an inference-time baseline with **no shared index store**, so this is
the simple inference-only cleaner (no ``_indices/`` handling — arag is the one
baseline that needs it, in ``scripts/clean_arag_logs.py``). The inference-folder
logic every baseline's cleaner shares lives in ``scripts/_clean_common.py``.

    python scripts/clean_structrag_logs.py                      # all structrag benchmarks
    python scripts/clean_structrag_logs.py --benchmark loong
    python scripts/clean_structrag_logs.py --dry-run            # preview, delete nothing
    python scripts/clean_structrag_logs.py --all-errors         # also clear real errors (full retry)
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct invocation as a path script (`python scripts/clean_structrag_logs.py`),
# where sys.path[0] is scripts/ not the repo root, to import the sibling
# `scripts._clean_common`. Harmless under `-m` / pytest (root already importable).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clean_common import clean_inference_dirs  # noqa: E402
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "structrag"


def clean_structrag_logs(
    logs_dir: Path,
    *,
    benchmark: str | None = None,
    all_errors: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove interrupted/crashed structrag inference folders under ``logs_dir``.

    Scans ``logs_dir/{benchmark}/structrag/inferences/{hash}/`` (every benchmark
    when ``benchmark`` is None). Returns a summary
    ``{deleted: [(Path, category)], kept: int, by_category: {cat: n}, dry_run}``;
    in ``dry_run`` nothing is removed but the same would-be-deleted list is returned.
    """
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    deleted: list[tuple[Path, str]] = []
    by_category: Counter[str] = Counter()
    kept = 0

    for base_dir in sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}")):
        inferences = base_dir / "inferences"
        if not inferences.is_dir():
            continue
        inf_deleted, inf_kept = clean_inference_dirs(
            inferences, all_errors=all_errors, dry_run=dry_run
        )
        deleted.extend(inf_deleted)
        for _, category in inf_deleted:
            by_category[category] += 1
        kept += inf_kept

    return {
        "deleted": deleted,
        "kept": kept,
        "by_category": dict(by_category),
        "dry_run": dry_run,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    """Per-benchmark one-liner: '<benchmark>: deleted N (...by category...), kept M'."""
    verb = "would delete" if summary["dry_run"] else "deleted"
    # Group deleted dirs by benchmark (folder is {benchmark}/structrag/inferences/{hash}).
    per_bench: dict[str, Counter[str]] = defaultdict(Counter)
    for inf_dir, category in summary["deleted"]:
        per_bench[inf_dir.parents[2].name][category] += 1

    if not summary["deleted"]:
        return f"nothing to clean (kept {summary['kept']} intact)"

    lines: list[str] = []
    for bench in sorted(per_bench):
        cats = per_bench[bench]
        breakdown = ", ".join(f"{cats[c]} {c}" for c in sorted(cats, key=lambda c: -cats[c]))
        lines.append(f"{bench}: {verb} {sum(cats.values())} ({breakdown})")
    lines.append(f"kept {summary['kept']} intact (manifests + real errors)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/clean_structrag_logs.py",
        description="Delete interrupted/crashed structrag inference folders so they re-run "
                    "(keeps manifests + real errors like ContextWindowExceededError).",
    )
    parser.add_argument("--benchmark", default=None,
                        help="Only this benchmark (default: all structrag benchmarks).")
    parser.add_argument("--all-errors", action="store_true",
                        help="Also delete REAL error folders (e.g. ContextWindowExceededError) "
                             "for a full retry of every failed task.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — print what would be removed, delete nothing.")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    summary = clean_structrag_logs(
        logs_dir, benchmark=args.benchmark, all_errors=args.all_errors, dry_run=args.dry_run,
    )
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
