"""Clean interrupted / crashed A-RAG logs so they re-run.

The A-RAG analogue of ``clean_graphrag_logs`` — A-RAG, like graphrag, has a
**content-addressed ``_indices/`` store** (per-task: ``chunks.json`` +
``sentence_index.pkl`` + receipts) that direct-llm / structrag don't, so it needs
the index-store layer on top of the shared inference-folder cleanup. Two layers
(both shared via ``scripts._clean_common``):

1. **Inference folders** (``{benchmark}/arag/inferences/{hash}/``) — identical to
   the other cleaners: ChildCrash / empty / incomplete folders AND transient errors
   (any ``error_type`` outside ``FAILURE_ERROR_TYPES``, e.g. ``Timeout`` /
   ``NotFoundError``) are removed so the runner re-dispatches just those tasks, while
   ``manifest.json`` and the genuine model failures (``ContextWindowExceededError``)
   are kept (``--all-errors`` also clears those).

2. **Index store** (``{benchmark}/arag/_indices/{index_hash}/``):

   - **referenced** — a surviving ``manifest.json`` points at it via ``index_ref``:
     **always kept**. NOTE A-RAG's index is **LLM-independent** (``index_hash`` =
     hash of the doc-set + embedding model + chunker/index versions — NOT the
     completion model, seed, or sampling), so ONE index is typically referenced by
     manifests from *different completion models* (e.g. an OpenAI run and a vLLM-Qwen
     run share it). The "never delete a referenced index" invariant covers that — and
     this tool never deletes a ``manifest.json`` anyway.
   - **half-built** — unreferenced AND missing the build's completion receipt
     (``index_meta.json`` / ``index_usage.json``, written LAST): an interrupted build.
     **Removed by default** (the runner self-heals these too: ``_ensure_index`` clears
     receipt-less leftovers before rebuilding).
   - **orphan** — unreferenced but COMPLETE: a valid, reusable index whose referencing
     inference didn't survive. **Kept by default**; removed only with
     ``--prune-orphan-indices``.

    python scripts/clean_arag_logs.py                        # all arag benchmarks
    python scripts/clean_arag_logs.py --benchmark loong
    python scripts/clean_arag_logs.py --dry-run              # preview, delete nothing
    python scripts/clean_arag_logs.py --all-errors           # also clear real errors
    python scripts/clean_arag_logs.py --prune-orphan-indices # also drop orphaned indices
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# Allow direct invocation as a path script (`python scripts/clean_arag_logs.py`),
# where sys.path[0] is scripts/ not the repo root, to import the sibling
# `scripts._clean_common`. Harmless under `-m` / pytest (root already importable).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._clean_common import (  # noqa: E402
    INDEX_CATEGORIES,
    clean_index_store,
    clean_inference_dirs,
    referenced_index_hashes,
)
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "arag"


def clean_arag_logs(
    logs_dir: Path,
    *,
    benchmark: str | None = None,
    all_errors: bool = False,
    prune_orphan_indices: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Remove interrupted A-RAG inference folders + dangling indices under ``logs_dir``.

    Scans ``logs_dir/{benchmark}/arag/`` (every benchmark when ``benchmark`` is None):
    cleans inference folders (shared logic), then — using the index references of the
    SURVIVING manifests — cleans the ``_indices/`` store. Returns a summary
    ``{deleted: [(Path, category)], by_category, kept_inferences, kept_indices,
    n_referenced_indices, dry_run}``. In ``dry_run`` nothing is removed but the same
    would-be-deleted list is returned (the reference set is computed from manifests,
    which this tool never deletes, so the preview is exact).
    """
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    deleted: list[tuple[Path, str]] = []
    by_category: Counter[str] = Counter()
    kept_inferences = 0
    kept_indices = 0
    n_referenced = 0

    for base_dir in sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}")):
        inferences = base_dir / "inferences"
        if inferences.is_dir():
            inf_deleted, inf_kept = clean_inference_dirs(
                inferences, all_errors=all_errors, dry_run=dry_run
            )
            deleted.extend(inf_deleted)
            for _, category in inf_deleted:
                by_category[category] += 1
            kept_inferences += inf_kept

        # Index references of the surviving manifests (this tool never deletes a
        # manifest). The ``_indices/`` cleanup is shared with the graphrag cleaner.
        referenced = referenced_index_hashes(inferences)
        idx_deleted, idx_kept, n_ref = clean_index_store(
            base_dir / "_indices", referenced,
            prune_orphans=prune_orphan_indices, dry_run=dry_run,
        )
        deleted.extend(idx_deleted)
        for _, category in idx_deleted:
            by_category[category] += 1
        kept_indices += idx_kept
        n_referenced += n_ref

    return {
        "deleted": deleted,
        "by_category": dict(by_category),
        "kept_inferences": kept_inferences,
        "kept_indices": kept_indices,
        "n_referenced_indices": n_referenced,
        "dry_run": dry_run,
    }


# Categories that are inference folders vs index dirs (for the summary split).
_INFERENCE_CATS = ("crash", "broken", "incomplete", "transient", "real_error")
_INDEX_CATS = INDEX_CATEGORIES


def _format_summary(summary: dict[str, Any]) -> str:
    """Per-benchmark one-liners: an inference line and an index line where each
    applies. Folder is ``{benchmark}/arag/{inferences|_indices}/{hash}``, so
    ``parents[2].name`` is the benchmark for both."""
    verb = "would delete" if summary["dry_run"] else "deleted"
    inf_by_bench: dict[str, Counter[str]] = defaultdict(Counter)
    idx_by_bench: dict[str, Counter[str]] = defaultdict(Counter)
    for path, category in summary["deleted"]:
        bench = path.parents[2].name
        if category in _INDEX_CATS:
            idx_by_bench[bench][category] += 1
        else:
            inf_by_bench[bench][category] += 1

    if not summary["deleted"]:
        return (f"nothing to clean (kept {summary['kept_inferences']} inferences, "
                f"{summary['kept_indices']} indices intact)")

    lines: list[str] = []
    for bench in sorted(set(inf_by_bench) | set(idx_by_bench)):
        if bench in inf_by_bench:
            cats = inf_by_bench[bench]
            breakdown = ", ".join(f"{cats[c]} {c}" for c in sorted(cats, key=lambda c: -cats[c]))
            lines.append(f"{bench} inferences: {verb} {sum(cats.values())} ({breakdown})")
        if bench in idx_by_bench:
            cats = idx_by_bench[bench]
            breakdown = ", ".join(f"{cats[c]} {c}" for c in sorted(cats, key=lambda c: -cats[c]))
            lines.append(f"{bench} indices: {verb} {sum(cats.values())} ({breakdown})")
    lines.append(f"kept {summary['kept_inferences']} inferences + {summary['kept_indices']} indices "
                 f"intact ({summary['n_referenced_indices']} indices referenced)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/clean_arag_logs.py",
        description="Delete interrupted/crashed A-RAG inference folders + dangling indices so "
                    "they re-run (keeps manifests, real errors, and every referenced index).",
    )
    parser.add_argument("--benchmark", default=None,
                        help="Only this benchmark (default: all arag benchmarks).")
    parser.add_argument("--all-errors", action="store_true",
                        help="Also delete REAL error folders (e.g. ContextWindowExceededError) "
                             "for a full retry of every failed task.")
    parser.add_argument("--prune-orphan-indices", action="store_true",
                        help="Also delete COMPLETE indices that no surviving manifest references "
                             "(reclaims disk; they'd otherwise be reused for free on a re-run).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview only — print what would be removed, delete nothing.")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)

    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    summary = clean_arag_logs(
        logs_dir,
        benchmark=args.benchmark,
        all_errors=args.all_errors,
        prune_orphan_indices=args.prune_orphan_indices,
        dry_run=args.dry_run,
    )
    print(_format_summary(summary))


if __name__ == "__main__":
    main()
