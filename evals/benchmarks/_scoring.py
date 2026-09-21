"""Shared scoring-result shape returned by every benchmark's ``score_details``.

The benchmark's ``score(task_id, raw_answer)`` is the single public way to grade a
model's **raw** answer and returns just the metric (0/1 for CorpusQA, a 0/1
strict+lenient pair for Dracula, 1–100 for Loong). ``score_details`` returns this
richer record so a caller — no runner here invokes grading; that happens outside
this repo, over the saved ``logs/`` — can see the grade's provenance: *why* it was
graded so, any extra verdict the benchmark carries alongside the metric, and
*which model* graded it.

This is a **plain dict** at runtime (a ``TypedDict`` — so the keys are documented +
statically checkable, but it serializes straight to JSON with no conversion). It is a
shared *shape*, not a shared *scorer*: all three benchmarks grade with an LLM judge,
but each owns its own rubric and metric and fills only the fields that apply, leaving
the others ``None``.
"""
from __future__ import annotations

from typing import TypedDict


class ScoreResult(TypedDict):
    """One graded answer — a dict with these keys:

    - ``score`` — the metric: Loong's 1–100 rating, CorpusQA's 0/1 answer
      equivalence, Dracula's 0/1 **strict** verdict.
    - ``parsed`` — a benchmark-specific extra riding alongside the metric. Only
      Dracula uses it: the strict/lenient verdict PAIR, encoded as
      ``"strict=correct, lenient=incorrect"`` (``dracula/judge.py``'s
      ``encode_verdicts`` / ``read_verdicts``). ``None`` for Loong and CorpusQA.
    - ``rationale`` — the grader's reasoning (Loong's rating rationale, CorpusQA's
      equivalence explanation, Dracula's two labelled verdict walkthroughs) —
      exactly what a failure deep-dive reads. ``None`` for the empty-answer
      short-circuit (no judge call was made).
    - ``model`` — the **canonical slug** of the judge model that produced this row
      (e.g. ``"gpt-5-4-mini"``). Provenance: which model graded this answer.
    """

    score: int
    parsed: str | None
    rationale: str | None
    model: str
