"""Unit tests for `evals.r3con.harness.corpusqa` — the pure, no-I/O bits of the adapter.

Deliberately **no mocking of the upstream loader**: the only thing worth a fast,
deterministic unit test here is the task composition (which is pure string logic), and
that's tested directly via ``_compose_task`` — including the empty-part branches that
real data never exercises. Everything that actually talks to the upstream CorpusQA loader
(real downloads, real shapes) and the real ORM judge is covered against **real data** in
``tests/live/test_corpusqa.py`` — that's where the adapter is verified end-to-end, not
behind a fake.

Run with:  uv run python tests/unit/test_corpusqa.py
"""

from __future__ import annotations

from evals.r3con.harness import corpusqa
from evals.r3con.harness.corpusqa import _compose_task


def test_constants() -> None:
    assert corpusqa.NAME == "CorpusQA"


def test_compose_task_question_then_instruction() -> None:
    # Upstream get_task -> (instruction, question, docs); the model gets the question first,
    # then the instruction (the output-requirements block the gold depends on).
    assert _compose_task("INSTRUCTION", "QUESTION") == "QUESTION\n\nINSTRUCTION"


def test_compose_task_empty_question_uses_instruction_only() -> None:
    assert _compose_task("INSTR-ONLY", "") == "INSTR-ONLY"
    assert _compose_task("INSTR-ONLY", "   ") == "INSTR-ONLY"  # blank counts as empty


def test_compose_task_empty_instruction_uses_question_only() -> None:
    assert _compose_task("", "Q-ONLY") == "Q-ONLY"
