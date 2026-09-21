"""Dracula mini-benchmark LLM judge — DUAL scoring (strict + lenient), 0/1 each.

Judges free-form answers to the questions in ``questions.json`` (the Dracula
showcase corpus — see ``split.py`` beside this file) against their gold answers.
Built on the same pattern as the other judges here (structured output, a reasoning
field generated before the verdict — here ``reasoning`` then ``verdict``; an empty
answer short-circuits to the floor score without a paid call), with ONE difference —
this benchmark grades every answer **twice**:

* **strict**  — against ``questions.json``'s ``answer`` (the full 13-victim roster).
* **lenient** — Mr. Swales may be omitted. Judged with the **same prompt** against
  the row's ``answer_lenient`` (the 12-victim roster), then combined in code as
  ``lenient = max(strict, alt)`` so lenient is always a SUPERSET of strict (an
  answer that names Swales stays correct — it satisfies the strict gold).

Why two golds rather than a hand-written "relaxed" prompt: the strict prompt's
rigor (the total must match the gold's, the golden list is exhaustive, no
additions) then applies UNCHANGED to both calls. A softened prompt was measured
to leak leniency into the crew-count check — passing answers that never commit to
nine — which the two-gold construction cannot do.

**Everything here is local to this benchmark; no generic harness code changes.**
``score_details`` returns the SHARED ``ScoreResult`` shape unchanged, using its
existing fields:

* ``score``    — the **strict** 0/1. A report reads this, so its headline
  accuracy for dracula is the strict metric.
* ``parsed``   — the pair of verdicts, e.g. ``"strict=correct, lenient=correct"``
  (the other two benchmarks leave ``parsed`` as ``None``, and anything that
  persists a ScoreResult carries it verbatim, so the lenient grade is cached for
  free). Decode it with :func:`read_verdicts`.
* ``rationale``— both judges' reasoning, labelled by which gold each ran against.

``score`` / ``score_batch`` return the **pair** ``(strict, lenient)``.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from evals.benchmarks._scoring import ScoreResult
from evals.llm.chat import litellm_chat_completion
from evals.settings import JUDGE_MODEL, canonical_model_id

from .loader import _load

# Fixed for reproducibility — judging is best-effort deterministic.
_JUDGE_SEED = 0

# Identity + VERSION of this benchmark's scoring mechanism — exported so whatever
# grades these logs can record it beside each grade and key its cache on it. BUMP it
# whenever the grading mechanism changes in a way that can alter scores. v6: dual
# strict+lenient grading (two calls per answer, same prompt, two golds). v7: counts
# must be STATED, not inferred by the grader. v8: v7 was over-strict — a qualifier
# around the CORRECT number ("at least 13") is fine. v9: the accumulated rule-list
# was replaced by ONE rule + six worked examples (see SYSTEM_PROMPT).
SCORER = "dracula-judge-v9"

# The grader model a cached grade is keyed against.
GRADER_MODEL = JUDGE_MODEL

# Canonical slug of the grader model, stamped on each ScoreResult's ``model`` field.
_GRADER_SLUG = canonical_model_id(GRADER_MODEL)

# Drives BOTH calls (strict + lenient) — only the golden answer differs.
#
# STEERED BY EXAMPLES, NOT RULES. v6→v8 accumulated an ever-longer rule list, and the
# judge kept rationalizing around it — most damningly on an answer that said "The
# Entire Crew of the Demeter" with no size and no total, which it passed while
# asserting "it explicitly names the crew as nine" (the word never appears in that
# answer). That item flipped on every rescore: 0,1,0,1. v9 replaces the rule list
# with ONE rule — every number in the golden answer must be quotable from the
# model's answer, never supplied by the grader — plus six worked examples that show
# the shape: a qualifier round the right number is fine; a group named but never
# sized is not; a wrong number is not; naming people in order to exclude them is
# fine; a missing item and an added item are both fatal.
#
# The examples deliberately use an UNRELATED scenario (a polar expedition, not the
# Dracula roster). Reusing the real golds would invite the grader to copy an
# example's verdict instead of grading the answer in front of it — the examples must
# teach the FUNCTION, which is also why the same prompt serves both the 13-victim
# strict gold and the 12-victim lenient one. `test_dracula.py` pins the isolation.
SYSTEM_PROMPT = """\
You grade a model's answer against a golden answer. Your input always has this form:

## Question
## Golden (reference) answer
## Model's answer to judge

You reply with two fields: `reasoning` first, then `verdict`, which is exactly
"correct" or "incorrect".

Correct means the model's answer says the same thing as the golden answer. Wording,
formatting, ordering and extra true detail do not matter.

Be strict about one thing: every number in the golden answer must actually appear in
the model's answer. Never work a number out for yourself — if you cannot quote it from
the model's own words, it is not there.

In `reasoning`, walk the golden answer's items and numbers one at a time, quoting where
each appears in the model's answer or noting that it does not. Then commit.

The worked examples below are from an UNRELATED question. They show only how to apply
the rule — never carry their content or their verdict over to the answer you grade.

═══ EXAMPLE 1 ═══
## Question
Across the expedition and the winter that followed, how many of the party died, and who were they?
## Golden (reference) answer
8 — (1) the four sled drivers, (2) Dr. Hallam, (3) Vance, (4) Ibarra, (5) Captain Reyes
## Model's answer to judge
At least 8 of the party died. The four sled drivers were lost in the crevasse field (among
them Okada and Bell); Dr. Hallam, Vance, Ibarra and Captain Reyes died over the winter.

reasoning: The golden answer needs the total 8 and the group size four. The model says "At
least 8", so 8 appears, and "the four sled drivers", so four appears. Hallam, Vance, Ibarra
and Captain Reyes are each named. Okada and Bell are named as members of the sled drivers —
detail about that group, not extra deaths. "At least" around the correct number is fine.
verdict: correct

═══ EXAMPLE 2 ═══
## Question
(same)
## Golden (reference) answer
(same)
## Model's answer to judge
The whole sled team was lost in the crevasse field — Okada, Bell and the others — and
Dr. Hallam, Vance, Ibarra and Captain Reyes died during the winter.

reasoning: Hallam, Vance, Ibarra and Captain Reyes are all present. But the model says "The
whole sled team" and never gives its size, and it states no total; I cannot quote either
number from its answer. The people are right and the numbers are absent, so I must not
supply four or 8 myself.
verdict: incorrect

═══ EXAMPLE 3 ═══
## Question
(same)
## Golden (reference) answer
(same)
## Model's answer to judge
7 died: the three sled drivers, Dr. Hallam, Vance, Ibarra and Captain Reyes.

reasoning: The model states "7" and "three sled drivers"; the golden answer is 8 and four.
Both numbers are present but wrong.
verdict: incorrect

═══ EXAMPLE 4 ═══
## Question
(same)
## Golden (reference) answer
(same)
## Model's answer to judge
8 — the four sled drivers, Dr. Hallam, Vance, Ibarra and Captain Reyes. Not counted: Petrov,
who turned back before the expedition, and Nurse Adeyemi, who survived her frostbite.

reasoning: "8" and "four sled drivers" both appear, and all four named people are present.
Petrov and Adeyemi are named only in order to exclude them, which is fine.
verdict: correct

═══ EXAMPLE 5 ═══
## Question
(same)
## Golden (reference) answer
(same)
## Model's answer to judge
8 — the four sled drivers, Dr. Hallam, Vance and Ibarra.

reasoning: The four sled drivers, Hallam, Vance and Ibarra appear, but Captain Reyes is
missing. The stated total "8" also does not match the seven people the answer lists.
verdict: incorrect

═══ EXAMPLE 6 ═══
## Question
(same)
## Golden (reference) answer
(same)
## Model's answer to judge
9 — the four sled drivers, Dr. Hallam, Vance, Ibarra, Captain Reyes, and Petrov.

reasoning: Every golden item appears, but Petrov is added as a death and the total is given
as 9 rather than 8.
verdict: incorrect"""


class Verdict(BaseModel):
    # `reasoning` is generated before `verdict` so the judge commits to a
    # rationale first. `score()` returns only the verdicts (0/1 each);
    # `score_details` also surfaces this reasoning into the ScoreResult's
    # `rationale`.
    reasoning: str
    verdict: Literal["correct", "incorrect"]


def _user_prompt(question: str, gold_answer: str, model_answer: str) -> str:
    return (
        "## Question\n\n"
        f"{question}\n\n"
        "## Golden (reference) answer\n\n"
        f"{gold_answer}\n\n"
        "## Model's answer to judge\n\n"
        f"{model_answer}"
    )


# --- the strict/lenient pair, encoded into the shared ScoreResult's `parsed` ----

def encode_verdicts(strict: int, lenient: int) -> str:
    """``(1, 1)`` → ``"strict=correct, lenient=correct"`` — what lands in ``parsed``."""
    word = {1: "correct", 0: "incorrect"}
    return f"strict={word[int(bool(strict))]}, lenient={word[int(bool(lenient))]}"


def read_verdicts(parsed: str | None) -> tuple[int, int] | None:
    """Decode a ``parsed`` string (fresh or cached) → ``(strict, lenient)``.

    Returns None when ``parsed`` is absent or not in this benchmark's format (e.g. a
    grade cached by an older SCORER).
    """
    if not parsed:
        return None
    out: dict[str, int] = {}
    for part in parsed.split(","):
        key, _, value = part.strip().partition("=")
        if key in ("strict", "lenient") and value in ("correct", "incorrect"):
            out[key] = int(value == "correct")
    if "strict" not in out or "lenient" not in out:
        return None
    return out["strict"], out["lenient"]


def _merge_rationales(strict_reasoning: str, lenient_reasoning: str) -> str:
    """Both judges' reasoning, labelled by which gold each ran against."""
    return (
        f"[strict gold — full 13-victim roster]\n{strict_reasoning}\n\n"
        f"[lenient gold — Mr. Swales omitted]\n{lenient_reasoning}"
    )


def _judge_one(question: str, gold_answer: str, model_answer: str) -> tuple[int, str]:
    """ONE judge call against ONE golden answer → ``(0/1, reasoning)``."""
    result = litellm_chat_completion(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=_user_prompt(question, gold_answer, model_answer),
        model=JUDGE_MODEL,
        schema=Verdict,
        seed=_JUDGE_SEED,
    )
    return int(result.verdict == "correct"), result.reasoning


def _empty_result() -> ScoreResult:
    """An empty answer is unambiguously incorrect under BOTH golds — no paid call."""
    return ScoreResult(
        score=0, parsed=encode_verdicts(0, 0), rationale=None, model=_GRADER_SLUG
    )


def score_details(
    task_ids: list[str], answers: list[str], max_workers: int = 50
) -> list[ScoreResult]:
    """Judge many free-form answers in parallel → one ScoreResult per item, in order.

    Each non-empty answer costs **two** ``JUDGE_MODEL`` calls (strict gold + lenient
    gold). Both calls of every item are fanned out into the SAME pool (2N jobs), so
    adding the lenient metric does not halve throughput.

    Raises ``ValueError`` on a length mismatch and ``KeyError`` on an unknown task id.
    """
    if len(task_ids) != len(answers):
        raise ValueError(
            f"task_ids and answers length mismatch: {len(task_ids)} vs {len(answers)}"
        )
    if not task_ids:
        return []
    rows = _load()
    missing = [tid for tid in task_ids if tid not in rows]
    if missing:
        raise KeyError(f"Unknown Dracula task id(s): {sorted(set(missing))}")

    results: list[ScoreResult | None] = [None] * len(task_ids)
    jobs: list[tuple[int, str, str, str, str]] = []  # (idx, slot, question, gold, answer)
    for idx, (tid, answer) in enumerate(zip(task_ids, answers)):
        if not answer.strip():
            results[idx] = _empty_result()
            continue
        row = rows[tid]
        # A question with no lenient gold falls back to the strict one → lenient
        # collapses to strict (harmless: max(s, s) == s).
        gold_lenient = row.get("answer_lenient") or row["answer"]
        jobs.append((idx, "strict", row["question"], row["answer"], answer))
        jobs.append((idx, "lenient", row["question"], gold_lenient, answer))

    if jobs:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            judged = list(
                pool.map(lambda j: (j[0], j[1], _judge_one(j[2], j[3], j[4])), jobs)
            )
        by_idx: dict[int, dict[str, tuple[int, str]]] = defaultdict(dict)
        for idx, slot, outcome in judged:
            by_idx[idx][slot] = outcome
        for idx, slots in by_idx.items():
            strict, strict_reasoning = slots["strict"]
            alt, alt_reasoning = slots["lenient"]
            # The OR: an answer correct under EITHER gold is lenient-correct, so
            # lenient is always a superset of strict.
            lenient = max(strict, alt)
            results[idx] = ScoreResult(
                score=strict,
                parsed=encode_verdicts(strict, lenient),
                rationale=_merge_rationales(strict_reasoning, alt_reasoning),
                model=_GRADER_SLUG,
            )

    return [r for r in results if r is not None]


def score_batch(
    task_ids: list[str], answers: list[str], max_workers: int = 50
) -> list[tuple[int, int]]:
    """Judges many free-form answers in parallel → ``(strict, lenient)`` per item, in order."""
    return [
        read_verdicts(r["parsed"]) or (r["score"], r["score"])
        for r in score_details(task_ids, answers, max_workers=max_workers)
    ]


def score(task_id: str, answer: str) -> tuple[int, int]:
    """Returns ``(strict, lenient)`` — each 1 if the LLM judge ruled ``answer`` correct.

    Unlike the other benchmarks, dracula returns a PAIR: the strict metric (the full
    13-victim roster) and the lenient one (Mr. Swales optional). ``lenient >= strict``
    always. Single-item convenience wrapper over :func:`score_details`; hits a real LLM.
    """
    return score_batch([task_id], [answer])[0]
