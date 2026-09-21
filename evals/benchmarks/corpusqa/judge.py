"""CorpusQA LLM judge — answer-equivalence (ORM), 0/1.

CorpusQA is judge-only (no exact match): a judge model decides whether the model's
answer is **equivalent** to the programmatically-computed gold (same value/meaning,
tolerating ``282`` vs ``282.0``, formatting, units, etc.) → 1/0. This reproduces
upstream ``src/eval.py``'s "ORM" (Output Reward Model) equivalence judge. The
verdict is the metric — this benchmark exposes ``score`` / ``score_batch`` /
``score_details`` + ``SCORER`` + ``GRADER_MODEL``, extracts nothing from the answer
beyond upstream's :func:`extract_answer`, and (unlike Loong) declares **no
``PERFECT_SCORE``**, so plain accuracy is the metric to report.

The upstream prompt is sent with the ``The answer is: xxx`` output contract intact
(it is part of each row's baked instruction block — see ``loader._unbake``), so the
judge first runs upstream's :func:`extract_answer` on the model's raw reply, exactly
as ``eval.py`` does, before the equivalence check.

================ What changed from upstream's ORM judge ================
The equivalence rubric (the first four lines of upstream ``GENERAL_ORM_PROMPT``) and
the ``Problem / Answer 1 / Answer 2`` data block (``ORM_USER_TEMPLATE``) are
VERBATIM from ``eval.py``. Two faithful re-seams, the same as Loong's judge swaps:
  1. Transport: routed through the project's ``JUDGE_MODEL`` (``openai/gpt-5.4-mini``,
     ``evals/settings.py``) via ``evals/llm/chat.py`` — NOT upstream's
     deepseek-v3/DashScope.
  2. Output format: upstream's *"provide your final answer in the form of:
     ``[[YES]]`` or ``[[NO]]``"* block (then string-parsed) is replaced by a
     structured ``Equivalence`` schema (an ``explanation`` then a boolean
     ``equivalent``), so there is no brittle text parsing. ``extract_answer``,
     ``answer_1 = model`` / ``answer_2 = gold`` ordering, and the rubric wording are
     unchanged.
=======================================================================
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from evals.benchmarks._scoring import ScoreResult
from evals.benchmarks.corpusqa import loader
from evals.llm.chat import litellm_chat_completion
from evals.settings import JUDGE_MODEL, canonical_model_id

# Fixed for reproducibility — judging is best-effort deterministic.
_JUDGE_SEED = 0

# Identity + VERSION of this benchmark's scoring mechanism — exported so whatever
# grades these logs can record it beside each grade and key its cache on it. BUMP it
# (e.g. ``corpusqa-orm-judge-v2``) whenever the grading mechanism changes in a way
# that can alter scores — the rubric, the ``Equivalence`` schema, or
# ``extract_answer`` — so already-cached grades auto-invalidate. A grader-MODEL
# change is caught separately via ``GRADER_MODEL``.
SCORER = "corpusqa-orm-judge"

# The grader model a cached grade is keyed against. Changing it invalidates that
# cache — re-scoring re-pays the judge LLM.
GRADER_MODEL = JUDGE_MODEL

# Canonical slug of the grader model, stamped on each ScoreResult's ``model`` field.
_GRADER_SLUG = canonical_model_id(GRADER_MODEL)

# Upstream ``extract_answer`` regex: capture what follows "The answer is: " (the
# enforced output contract); no DOTALL, so it takes the final-answer line.
_ANSWER_RE = re.compile(r"The answer is: (.*)")


def extract_answer(response: str) -> str:
    """The model's final answer — ported VERBATIM from upstream ``eval.py``.

    Returns the text after ``"The answer is: "`` (the output contract every row's
    instruction block enforces), else the whole stripped response.
    """
    match = _ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


# The ORM equivalence rubric — first four lines VERBATIM from upstream
# ``GENERAL_ORM_PROMPT``; the ``[[YES]]/[[NO]]`` output-format block is replaced by
# the structured ``Equivalence`` schema (see the module docstring).
SYSTEM_PROMPT = (
    "You are an expert in verifying if two answers are the same.\n"
    "Your input is a problem and two answers, Answer 1 and Answer 2. You need to "
    "check if they are equivalent.\n"
    "Your task is to determine if two answers are equivalent, without attempting to "
    "solve the original problem.\n"
    "Compare the answers to verify they represent identical values or meaning, even "
    "when written in different forms or notations.\n\n"
    "First provide an explanation for why the answers are equivalent or not, then "
    "give your verdict as the boolean `equivalent` field."
)


class Equivalence(BaseModel):
    # `explanation` is generated before `equivalent` so the judge commits to a
    # rationale first. `score()` returns only 0/1; `score_details` also surfaces
    # this explanation in the ScoreResult's `rationale` (the deep-dive).
    explanation: str
    equivalent: bool


def _user_prompt(problem: str, answer_1: str, answer_2: str) -> str:
    """Upstream ``ORM_USER_TEMPLATE`` verbatim — ``answer_1`` = the model's
    (extracted) answer, ``answer_2`` = the gold."""
    return (
        "\n"
        f"Problem: {problem}\n"
        f"Answer 1: {answer_1}\n"
        f"Answer 2: {answer_2}\n"
    )


def _format_gold(gold) -> str:
    """Serialize a gold answer to text for the judge — ``str(gold)``, faithful to
    upstream's ``ORM_USER_TEMPLATE.format(answer_2=gold)``. Handles the native
    number / string / list golds, including the intentional empty list (``str([])``
    → ``"[]"``)."""
    return str(gold)


def _judge(question: str, gold, raw_answer: str) -> ScoreResult:
    """Judge one raw answer → ``ScoreResult`` (0/1 + the judge's explanation).

    Extracts the final answer first (upstream ``extract_answer``); an empty
    extracted answer is unambiguously wrong (an empty model answer never matches a
    gold, including a ``[]`` gold) → 0 with NO paid call.
    """
    extracted = extract_answer(raw_answer)
    if not extracted.strip():
        return ScoreResult(score=0, parsed=None, rationale=None, model=_GRADER_SLUG)
    result = litellm_chat_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(question, extracted, _format_gold(gold)),
        model=JUDGE_MODEL,
        schema=Equivalence,
        seed=_JUDGE_SEED,
    )
    return ScoreResult(
        score=int(result.equivalent), parsed=None,
        rationale=result.explanation, model=_GRADER_SLUG,
    )


def _gather(task_ids: list[str]) -> dict[str, tuple[str, object]]:
    """Resolves ``{task_id -> (question, gold)}`` from the cheap offset index (no
    prompt read). Raises ``KeyError`` if any task id is unknown."""
    found: dict[str, tuple[str, object]] = {}
    missing: list[str] = []
    for tid in task_ids:
        try:
            _tier, rec = loader._locate(tid)
        except KeyError:
            missing.append(tid)
            continue
        found[tid] = (rec["question"], rec["answer"])
    if missing:
        raise KeyError(f"Unknown CorpusQA task id(s): {sorted(set(missing))}")
    return found


def score_details(
    task_ids: list[str], answers: list[str], max_workers: int = 50
) -> list[ScoreResult]:
    """Judge many RAW answers in parallel → 0/1 + explanation per item, in order.

    Resolves the question + gold for every task in a single index pass, then fans
    the judge LLM calls out across a thread pool (default 50 workers). Each call hits
    ``JUDGE_MODEL`` (``openai/gpt-5.4-mini``) — this costs money. Returns one
    :class:`ScoreResult` per input, carrying the judge's explanation as its
    rationale.

    Raises ``ValueError`` if ``task_ids`` and ``answers`` differ in length, and
    ``KeyError`` if any task id is unknown.
    """
    if len(task_ids) != len(answers):
        raise ValueError(
            f"task_ids and answers length mismatch: {len(task_ids)} vs {len(answers)}"
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
    """Judges many raw answers in parallel. Returns 0/1 per item, in input order."""
    return [r["score"] for r in score_details(task_ids, answers, max_workers=max_workers)]


def score(task_id: str, answer: str) -> int:
    """Returns 1 if the LLM judge rules ``answer`` equivalent to the gold for
    ``task_id``, else 0. Single-item wrapper over :func:`score_details`. Hits a real
    LLM."""
    return score_details([task_id], [answer])[0]["score"]
