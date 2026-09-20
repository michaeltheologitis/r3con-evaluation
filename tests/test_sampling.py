"""Tests for the shared deterministic task-id ordering (``evals.benchmarks._sampling``).

Every benchmark's ``get_task_ids`` routes its result through ``order_task_ids``:
a deterministic shuffle (fixed, non-exposed seed) + optional ``limit``. These pin
the pure-logic contract; the per-benchmark test files assert each loader actually
routes through it.
"""
from __future__ import annotations

from evals.benchmarks._sampling import order_task_ids

IDS = [f"t{i}" for i in range(50)]


def test_deterministic_same_input_same_order() -> None:
    assert order_task_ids(IDS) == order_task_ids(IDS)


def test_actually_shuffles_but_preserves_the_set() -> None:
    out = order_task_ids(IDS)
    assert set(out) == set(IDS)   # same elements
    assert out != IDS             # but reordered (not insertion order)


def test_does_not_mutate_input() -> None:
    src = list(IDS)
    order_task_ids(src)
    assert src == IDS


def test_limit_is_a_prefix_of_the_full_shuffled_order() -> None:
    full = order_task_ids(IDS)
    assert order_task_ids(IDS, limit=10) == full[:10]


def test_limit_none_returns_all() -> None:
    assert order_task_ids(IDS, limit=None) == order_task_ids(IDS)
    assert len(order_task_ids(IDS, limit=None)) == len(IDS)


def test_limit_zero_returns_empty() -> None:
    assert order_task_ids(IDS, limit=0) == []


def test_limit_exceeding_length_returns_all() -> None:
    assert order_task_ids(IDS, limit=10**9) == order_task_ids(IDS)


def test_empty_input() -> None:
    assert order_task_ids([]) == []
    assert order_task_ids([], limit=5) == []
