"""Shared scoring-result shape returned by every benchmark's ``score_details``.

The benchmark's ``score(task_id, raw_answer)`` is the single public way to grade a
model's **raw** answer and returns just the metric (0/1 for the label and
LooGLE/CorpusQA judge benchmarks, 1–100 for Loong). ``score_details`` returns this
richer record so a caller — and the scoreboard, which persists a self-contained
``score.json`` — can see the grade's provenance: *what* was extracted (label
benchmarks), *why* it was graded so (judge benchmarks), and *which model* graded it.

This is a **plain dict** at runtime (a ``TypedDict`` — so the keys are documented +
statically checkable, but it serializes straight to JSON with no conversion). It is a
shared *shape*, not a shared *scorer*: each benchmark owns its own mechanism
(parse-then-match vs LLM judge) and fills only the fields that apply, leaving the
others ``None``.
"""
from __future__ import annotations

from typing import TypedDict


class ScoreResult(TypedDict):
    """One graded answer — a dict with these keys:

    - ``score`` — the metric: 0/1 for the label benchmarks (longbenchv2, casefacts)
      and the LooGLE / CorpusQA judges, the 1–100 rating for Loong's judge.
    - ``parsed`` — label benchmarks only: the letter / verdict ``parse`` extracted
      from the raw answer (e.g. ``"D."`` / ``"Supported"`` / ``"N/A"``). ``None`` for
      the free-form judge benchmarks (nothing is parsed).
    - ``rationale`` — judge benchmarks only: the grader's reasoning (LooGLE's verdict
      reasoning, Loong's rating rationale, CorpusQA's equivalence explanation) —
      exactly what the deep-dive discipline reads. ``None`` for the label benchmarks
      and for the empty-answer short-circuit (no judge call was made).
    - ``model`` — the **canonical slug** of the grader model that produced this row
      (e.g. ``"gpt-5-4-mini"`` — the judge for the free-form benchmarks, the parse
      model for the label ones). Provenance: which model graded this answer.
    """

    score: int
    parsed: str | None
    rationale: str | None
    model: str
