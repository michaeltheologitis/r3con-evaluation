"""Tests for `evals.r3con.harness.loong` — the Loong adapter (offline).

The upstream loader module (``loong._mod()``) is monkeypatched with a fake so
these stay offline.

Run with:  uv run python tests/unit/test_loong.py
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Iterator

from evals.r3con.harness import loong


@contextlib.contextmanager
def _patched_mod(fake: SimpleNamespace) -> Iterator[None]:
    original = loong._mod
    loong._mod = lambda: fake  # type: ignore[assignment]
    try:
        yield
    finally:
        loong._mod = original  # type: ignore[assignment]


def test_load_composes_instruction_and_question() -> None:
    fake = SimpleNamespace(get_task=lambda tid: ("INSTRUCTION", "QUESTION", ["d1", "d2"]))
    with _patched_mod(fake):
        ti = loong.load("x")
    assert ti.task == "INSTRUCTION\n\nQUESTION"
    assert ti.documents == ["d1", "d2"]


def test_load_empty_question_uses_instruction_only() -> None:
    fake = SimpleNamespace(get_task=lambda tid: ("INSTR-ONLY", "", ["d"]))
    with _patched_mod(fake):
        ti = loong.load("x")
    assert ti.task == "INSTR-ONLY"


def test_list_task_ids_forwards_filters() -> None:
    seen = {}

    def get_task_ids(*, sets, tasks, limit):
        seen.update(sets=sets, tasks=tasks, limit=limit)
        return ["a", "b"]

    with _patched_mod(SimpleNamespace(get_task_ids=get_task_ids)):
        ids = loong.list_task_ids(sets=[1], tasks=[3], limit=2)
    assert ids == ["a", "b"]
    assert seen == {"sets": [1], "tasks": [3], "limit": 2}


def test_gold_and_metadata_passthrough() -> None:
    fake = SimpleNamespace(
        get_task_answer=lambda tid: f"gold-{tid}",
        get_task_metadata=lambda tid: {"domain": "finance", "set": 1, "task": 3, "task_name": "Clustering"},
    )
    with _patched_mod(fake):
        assert loong.gold("t") == "gold-t"
        assert loong.metadata("t")["domain"] == "finance"
