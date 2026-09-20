"""Loong LLM judge — a 1–100 rating, structured-output edition.

Loong is judge-only (no exact match). A judge model rates the model's answer
**1–100** against the gold answer on two rubric criteria — *Accuracy &
Hallucinations* and *Completeness*. Downstream metrics (computed by the analysis
CLI) are **Avg Score** (mean rating) and **Perfect Rate** (fraction == 100).

This mirrors ``loogle/judge.py``'s mechanics:
- the rubric is delivered to the project's ``JUDGE_MODEL`` (``openai/gpt-5.4-mini``,
  see ``evals/settings.py``) — NOT upstream's GPT-4;
- structured output via ``chat.py``'s ``schema=`` (a Pydantic ``Rating``) means
  there is **no output parsing** (upstream regex-extracts ``[[score]]``);
- ``rationale`` is generated *before* ``rating`` so the judge commits to a
  reasoning first;
- ``score`` / ``score_batch`` fan out across a thread pool;
- an empty answer is scored the worst (the 1–100 floor) without a paid call.

================ What changed from upstream's judge prompt ================
The rubric below — the two criteria, the 1-to-100 scale, and the "fully meet →
full marks (100)" rule — is VERBATIM from upstream
``src/utils/prompt.py:get_evaluate_prompts``. Three changes:
  1. The upstream output-format block (``PLEASE OUTPUT WITH THE FOLLOWING
     FORMAT … "[[score]]" … <start output> … Rating: [[score]] … <end output>``,
     incl. its ``Then, output a line …`` / ``Now, start your evaluation:``
     plumbing lines) is DELETED and replaced by an instruction to return a
     ``rationale`` (≤100 words) and an integer ``rating`` (1–100) as
     **structured fields** (the ``Rating`` schema), so no text parsing is needed.
  2. Upstream leaves the literal ``{docs}`` slot in the judge's question text for
     the ``paper`` domain (a quirk of its ``doc_type != "paper"`` guard); we drop
     ``{docs}`` for all domains — the judge grades against the gold answer, not
     the source documents, so the documents are never shown to it.
  3. Upstream sends ONE user message: the data block first, then the ``[System]``
     rubric ("… displayed above"). We send the rubric as a SYSTEM message that
     PRECEDES the user data block, and "displayed above" became "displayed
     below" accordingly.
Everything else (criteria wording, scale, full-marks rule, the bracketed
[Question]/[Gold Answer]/[Predicted Answer] framing) is upstream verbatim.
==========================================================================
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from evals.benchmarks._scoring import ScoreResult
from evals.benchmarks.loong.loader import _load
from evals.llm.chat import litellm_chat_completion
from evals.settings import JUDGE_MODEL, canonical_model_id

# Fixed for reproducibility — judging is best-effort deterministic.
_JUDGE_SEED = 0

# Identity + VERSION of this benchmark's scoring mechanism — recorded in
# score.json's ``scorer`` and part of its cache key. BUMP it (e.g.
# ``loong-judge-v2``) whenever the grading mechanism changes in a way that can
# alter scores — the judge ``SYSTEM_PROMPT`` rubric or the ``Rating`` schema — so
# cached score.json files auto-invalidate (same discipline as graphrag's
# ``_INDEX_VERSION``). A grader-MODEL change is caught separately via
# ``GRADER_MODEL`` → score.json's ``scored_with``.
SCORER = "loong-judge"

# The grader model score.json caches against (its ``scored_with``). Changing it
# invalidates the cache — re-scoring re-pays the judge LLM.
GRADER_MODEL = JUDGE_MODEL

# Canonical slug of the grader model, stamped on each ScoreResult's ``model`` field.
_GRADER_SLUG = canonical_model_id(GRADER_MODEL)

# The 1–100 scale floor. An empty / missing answer earns this without a paid
# judge call (it cannot meet any criterion). Upstream simply fails to parse a
# score for empty predictions and drops them; we assign the minimum instead so
# every instance contributes to Avg Score / Perfect Rate.
_WORST_SCORE = 1

# Public marker: Loong scores on a continuous 1–100 rating, NOT 0/1 correctness.
# The analysis CLI keys on this attribute to report Avg Score (mean rating) +
# Perfect Rate (fraction == PERFECT_SCORE) instead of accuracy. 100 == "fully
# meets the rubric" (the upstream full-marks rule).
PERFECT_SCORE = 100


# The judge rubric — see the module docstring for exactly what was changed from
# upstream. The model fills the `Rating` schema (rationale first, then the int
# score) instead of emitting the upstream `[[score]]` text format.
SYSTEM_PROMPT = (
    "We would like to request your feedback on the performance of the AI "
    "assistant in response to the user question displayed below according to the "
    "gold answer. Please use the following listed aspects and their descriptions "
    "as evaluation criteria:\n"
    "    - Accuracy and Hallucinations: The assistant's answer is semantically "
    "consistent with the gold answer; The numerical value and order need to be "
    "accurate, and there should be no hallucinations.\n"
    "    - Completeness: Referring to the reference answers, the assistant's "
    "answer should contain all the key points needed to answer the user's "
    "question; further elaboration on these key points can be omitted.\n"
    "Please rate whether this answer is suitable for the question. Please note "
    "that the gold answer can be considered as a correct answer to the question.\n\n"

    "The assistant receives an overall score on a scale of 1 to 100, where a "
    "higher score indicates better overall performance.\n"
    "Please note that if the assistant's answer and the gold answer fully meet "
    "the above criteria, its overall rating should be the full marks (100).\n"
    "Please first provide a comprehensive explanation of your evaluation, "
    "avoiding any potential bias.\n\n"

    "Return your evaluation as structured fields: a `rationale` (your evaluation "
    "explanation, no more than 100 words) and an integer `rating` on the 1-to-100 "
    "scale."
)


class Rating(BaseModel):
    # `rationale` is generated before `rating` so the judge commits to an
    # explanation first. `score()` returns only the rating (1–100); `score_details`
    # also surfaces this rationale into score.json's `rationale` (the deep-dive).
    rationale: str
    rating: int


def _format_question(prompt_template: str, instruction: str, question: str) -> str:
    """Build the judge's ``[Question]`` text from the instance's components.

    Mirrors upstream ``get_evaluate_prompts``: fill the per-instance
    ``prompt_template``'s ``{instruction}`` / ``{question}`` slots. The ``{docs}``
    slot is dropped (see the module docstring — the judge grades against the gold,
    not the documents).
    """
    return (
        prompt_template
        .replace("{docs}", "")
        .replace("{question}", question)
        .replace("{instruction}", instruction)
    )


def _format_gold(gold) -> str:
    """Serialize a gold answer to text for the judge prompt.

    Loong golds are sometimes a JSON object / list (e.g. the citation task's
    ``{"Reference": [...], "Citation": [...]}``), not a string. Strings pass
    through; everything else is ``json.dumps``-ed (faithful + readable — upstream
    just ``str()``-ed it via ``.format``)."""
    if isinstance(gold, str):
        return gold
    return json.dumps(gold, ensure_ascii=False, indent=2)


def _user_prompt(question: str, gold: str, predict: str) -> str:
    """The bracketed data block — [Question] / [Gold Answer] / [Predicted Answer]
    — verbatim from upstream's judge prompt."""
    return (
        "[Question]\n"
        f"{question}\n\n"
        "[Gold Answer]\n"
        f"{gold}\n\n"
        "[The Start of Assistant's Predicted Answer]\n"
        f"{predict}\n"
        "[The End of Assistant's Predicted Answer]"
    )


def _judge(question: str, gold: str, predict: str) -> ScoreResult:
    """Judge one answer → ``ScoreResult`` (1–100 rating + the judge's rationale)."""
    # An empty answer cannot meet any criterion — assign the floor, skip the call.
    if not predict.strip():
        return ScoreResult(score=_WORST_SCORE, parsed=None, rationale=None, model=_GRADER_SLUG)
    result = litellm_chat_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(question, gold, predict),
        model=JUDGE_MODEL,
        schema=Rating,
        seed=_JUDGE_SEED,
    )
    return ScoreResult(
        score=result.rating, parsed=None, rationale=result.rationale, model=_GRADER_SLUG
    )


def _gather(task_ids: list[str]) -> dict[str, tuple[str, str]]:
    """Resolves ``{task_id -> (judge_question, gold_text)}`` in a single pass.

    Done once per batch so the worker pool only issues judge LLM calls and never
    touches the dataset. Raises ``KeyError`` if any task id is unknown.
    """
    rows = _load()
    found: dict[str, tuple[str, str]] = {}
    missing: list[str] = []
    for tid in task_ids:
        row = rows.get(tid)
        if row is None:
            missing.append(tid)
            continue
        question = _format_question(row["prompt_template"], row["instruction"], row["question"])
        found[tid] = (question, _format_gold(row["answer"]))
    if missing:
        raise KeyError(f"Unknown Loong task id(s): {sorted(set(missing))}")
    return found


def score_details(
    task_ids: list[str], answers: list[str], max_workers: int = 50
) -> list[ScoreResult]:
    """Judge many free-form answers in parallel → rating + rationale per item, in order.

    Resolves the judge question + gold for every task in a single dataset pass,
    then fans the judge LLM calls out across a thread pool (default 50 workers).
    Each call hits ``JUDGE_MODEL`` (``openai/gpt-5.4-mini``) — this costs money.
    Returns one :class:`ScoreResult` per input (the scoreboard's entry, carrying
    the judge's rationale).

    Raises ``ValueError`` if ``task_ids`` and ``answers`` differ in length, and
    ``KeyError`` if any task id is unknown.
    """
    if len(task_ids) != len(answers):
        raise ValueError(
            f"task_ids and answers length mismatch: "
            f"{len(task_ids)} vs {len(answers)}"
        )
    if not task_ids:
        return []

    resolved = _gather(task_ids)
    payloads = [
        (resolved[tid][0], resolved[tid][1], answer)
        for tid, answer in zip(task_ids, answers)
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda p: _judge(*p), payloads))


def score_batch(
    task_ids: list[str], answers: list[str], max_workers: int = 50
) -> list[int]:
    """Judges many free-form answers in parallel. Returns a 1–100 rating per item,
    in input order."""
    return [r["score"] for r in score_details(task_ids, answers, max_workers=max_workers)]


def score(task_id: str, answer: str) -> int:
    """Returns the LLM judge's 1–100 rating of ``answer`` for ``task_id``.

    NOT a 0/1 correctness flag — the analysis CLI averages these (Avg Score) and
    counts the fraction == 100 (Perfect Rate). Single-item convenience wrapper
    over :func:`score_details`. Hits a real LLM.
    """
    return score_details([task_id], [answer])[0]["score"]
