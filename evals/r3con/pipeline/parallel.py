"""Bounded, order-preserving parallel map for per-document fan-out.

The summaries-pipeline runs the same LLM call across *every document in a
collection in parallel* — once per document per summary round (stage 1) and
once per document for extraction (stage 2). Both want the same thing: run up to
``max_workers`` calls concurrently, preserve input order in the results, and
surface the first exception. This is that one helper.

It is deliberately tiny and generic (no R3Con types) so both stages share one
code path. The concurrency *budget* (``R3CON_DOC_WORKERS``) lives in
:func:`evals.r3con.pipeline.settings.active_doc_workers`; callers pass the resolved number
in as ``max_workers`` so the eval's ``--workers`` (tasks) × doc fan-out stays
under the served endpoint's concurrency cap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[int, T], R], items: Sequence[T], *, max_workers: int
) -> list[R]:
    """Apply ``func(index, item)`` across ``items`` concurrently; return results in
    **input order**.

    - ``max_workers <= 1`` (or a single item) runs sequentially with no thread
      pool — the deterministic path used by tests and the degenerate one-doc case.
    - Otherwise up to ``min(max_workers, len(items))`` calls run at once.
    - The first exception raised by any call propagates out (the pool's context
      manager still waits for the already-running calls to finish before the
      exception surfaces).

    ``func`` receives the item's index so a caller can tag its work (e.g. the
    log ``kind`` per document) without threading position through the result.
    """
    n = len(items)
    if n == 0:
        return []
    if max_workers <= 1 or n == 1:
        return [func(i, item) for i, item in enumerate(items)]

    results: list[R] = [None] * n  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as ex:
        futures = {ex.submit(func, i, item): i for i, item in enumerate(items)}
        for fut, idx in futures.items():
            results[idx] = fut.result()
    return results
