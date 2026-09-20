"""Tests for ``evals.r3con.pipeline.stages.summaries.summarize_collection`` — the iterative
cross-document summaries stage.

Covers the load-bearing invariants of the design:
- ``rounds`` synchronous rounds, each re-summarizing every document;
- **others-only conditioning**: in round k ≥ 2 a document is conditioned on the
  OTHER documents' frozen round-(k-1) summaries, never its own;
- the final (last-round) summaries are returned aligned to the input documents,
  with every round kept for debugging;
- ``rounds=1`` degenerates to independent round-1 summaries;
- empty / one-document edge cases;
- the prompt-render contract (task + other-docs block in the system prompt; the
  document in the user message; round 1 omits the other-docs block).

The LLM call is monkeypatched two ways: the conditioning/round tests patch
``summaries.summarize_one`` (capturing the ``other_summaries`` each doc sees),
and the prompt-contract test patches ``summaries.litellm_chat_completion`` to
inspect the rendered messages.

Run with:  uv run python tests/unit/test_summaries.py
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable, Iterator
from typing import Any

from evals.r3con.pipeline.stages import summaries


@contextlib.contextmanager
def _patched_summarize_one(
    fake: Callable[..., str],
) -> Iterator[list[dict[str, Any]]]:
    """Swap ``summarize_one`` for a scripted fake; yield the captured kwargs."""
    original = summaries.summarize_one
    calls: list[dict[str, Any]] = []

    def wrapper(**kwargs: Any) -> str:
        calls.append({k: (list(v) if k == "other_summaries" else v) for k, v in kwargs.items()})
        return fake(**kwargs)

    summaries.summarize_one = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        summaries.summarize_one = original  # type: ignore[assignment]


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., str]) -> Iterator[list[dict[str, Any]]]:
    """Swap the raw LLM call for a scripted fake; yield the captured kwargs."""
    original = summaries.litellm_chat_completion
    calls: list[dict[str, Any]] = []

    def wrapper(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    summaries.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        summaries.litellm_chat_completion = original  # type: ignore[assignment]


def _tagged(**kwargs: Any) -> str:
    """Fake summarize_one: return a summary that encodes its (round, doc) from kind."""
    m = re.match(r"summary-r(\d+)-d(\d+)", kwargs["kind"])
    return f"S(r={m.group(1)},d={m.group(2)})"


def _by_kind(calls: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [c for c in calls if c["kind"] == kind]
    assert len(matches) == 1, f"expected exactly one call with kind={kind!r}, got {len(matches)}"
    return matches[0]


# --------------------------------------------------------------------------- #
# Rounds + others-only conditioning
# --------------------------------------------------------------------------- #


def test_returns_final_round_summaries_aligned_to_docs() -> None:
    docs = ["DOC0", "DOC1", "DOC2"]
    with _patched_summarize_one(_tagged):
        result = summaries.summarize_collection(
            task="q", documents=docs, model="m", prompt_version="v3", rounds=3, workers=4
        )
    assert result.final == ["S(r=3,d=0)", "S(r=3,d=1)", "S(r=3,d=2)"]
    # Every round is kept for debugging; the last IS the returned summaries.
    assert len(result.rounds) == 3
    assert result.rounds[-1] == result.final
    assert result.rounds[0] == ["S(r=1,d=0)", "S(r=1,d=1)", "S(r=1,d=2)"]


def test_every_doc_summarized_each_round() -> None:
    docs = ["A", "B", "C"]
    with _patched_summarize_one(_tagged) as calls:
        summaries.summarize_collection(task="q", documents=docs, model="m", prompt_version="v3", rounds=3, workers=4)
    # 3 rounds × 3 docs = 9 calls, one per (round, doc).
    kinds = sorted(c["kind"] for c in calls)
    assert kinds == sorted(f"summary-r{r}-d{d}" for r in (1, 2, 3) for d in (0, 1, 2))


def test_round1_has_no_other_summaries() -> None:
    docs = ["A", "B", "C"]
    with _patched_summarize_one(_tagged) as calls:
        summaries.summarize_collection(task="q", documents=docs, model="m", prompt_version="v3", rounds=2, workers=4)
    for d in range(3):
        assert _by_kind(calls, f"summary-r1-d{d}")["other_summaries"] == []


def test_round2_conditions_on_others_only_frozen_prior() -> None:
    """Round 2, doc d sees exactly the OTHER docs' round-1 summaries — never its own."""
    docs = ["A", "B", "C"]
    with _patched_summarize_one(_tagged) as calls:
        summaries.summarize_collection(task="q", documents=docs, model="m", prompt_version="v3", rounds=2, workers=4)
    # doc 1 in round 2 sees round-1 of docs 0 and 2, not its own (d=1).
    others_d1 = _by_kind(calls, "summary-r2-d1")["other_summaries"]
    assert others_d1 == ["S(r=1,d=0)", "S(r=1,d=2)"]
    assert "S(r=1,d=1)" not in others_d1
    # doc 0 sees docs 1 and 2.
    assert _by_kind(calls, "summary-r2-d0")["other_summaries"] == ["S(r=1,d=1)", "S(r=1,d=2)"]


def test_round3_conditions_on_round2_not_round1() -> None:
    docs = ["A", "B"]
    with _patched_summarize_one(_tagged) as calls:
        summaries.summarize_collection(task="q", documents=docs, model="m", prompt_version="v3", rounds=3, workers=4)
    # round 3 doc 0 sees round-2 of doc 1 (the frozen previous round), not round 1.
    others = _by_kind(calls, "summary-r3-d0")["other_summaries"]
    assert others == ["S(r=2,d=1)"]


def test_rounds_one_degenerates_to_independent() -> None:
    docs = ["A", "B"]
    with _patched_summarize_one(_tagged) as calls:
        result = summaries.summarize_collection(task="q", documents=docs, model="m", prompt_version="v3", rounds=1, workers=4)
    assert result.final == ["S(r=1,d=0)", "S(r=1,d=1)"]
    assert len(result.rounds) == 1
    # No round produced an other-summaries context.
    assert all(c["other_summaries"] == [] for c in calls)


# --------------------------------------------------------------------------- #
# Edge cases
# --------------------------------------------------------------------------- #


def test_empty_documents() -> None:
    with _patched_summarize_one(_tagged) as calls:
        result = summaries.summarize_collection(task="q", documents=[], model="m", prompt_version="v3", rounds=3)
    assert result.final == []
    assert result.rounds == []
    assert calls == []


def test_single_document_has_empty_others_every_round() -> None:
    with _patched_summarize_one(_tagged) as calls:
        result = summaries.summarize_collection(task="q", documents=["only"], model="m", prompt_version="v3", rounds=3, workers=4)
    assert result.final == ["S(r=3,d=0)"]
    assert len(result.rounds) == 3
    # Only document → no others, ever.
    assert all(c["other_summaries"] == [] for c in calls)


def test_rounds_zero_means_no_summaries() -> None:
    # sr=0 is a deliberate "no summaries" run: empty result AND not a single
    # summarize_one call — round 1 must be skipped, not merely the later rounds.
    with _patched_summarize_one(_tagged) as calls:
        result = summaries.summarize_collection(task="q", documents=["a", "b"], model="m", prompt_version="v3", rounds=0)
    assert result.final == []
    assert result.rounds == []
    assert calls == []


# --------------------------------------------------------------------------- #
# Prompt-render contract (real template, faked transport)
# --------------------------------------------------------------------------- #


def test_render_summaries_empty() -> None:
    assert summaries.render_summaries(None) == ""
    assert summaries.render_summaries([]) == ""


def test_render_summaries_labels_and_content() -> None:
    out = summaries.render_summaries(["first doc summary", "second doc summary"])
    assert "### Document 1\nfirst doc summary" in out
    assert "### Document 2\nsecond doc summary" in out


def test_render_summaries_doc_ids_label() -> None:
    out = summaries.render_summaries(["s0", "s1"], doc_ids=["10-K", "10-Q"])
    assert "### 10-K\ns0" in out
    assert "### 10-Q\ns1" in out


def test_prompt_contract_task_and_document_placement() -> None:
    """System prompt carries the task + (round ≥ 2) the others block; the document
    is the user message; round 1 omits the others block."""
    def fake(*, system_prompt: str, user_prompt: str, **_: Any) -> str:
        return f"<summary of {user_prompt}>"

    with _patched_llm(fake) as calls:
        summaries.summarize_collection(
            task="Which filing reports higher revenue?",
            documents=["AAA", "BBB"],
            model="m",
            prompt_version="v3",
            rounds=2,
            workers=1,  # deterministic ordering for this contract test
        )

    def find(kind: str) -> dict[str, Any]:
        m = [c for c in calls if c["kind"] == kind]
        assert len(m) == 1, f"{kind}: {len(m)}"
        return m[0]

    # Round 1, doc 0: task present, NO other-docs block, document is the user message.
    r1d0 = find("summary-r1-d0")
    assert "Which filing reports higher revenue?" in r1d0["system_prompt"]
    assert "<task>" in r1d0["system_prompt"]
    assert "summaries of the other documents" not in r1d0["system_prompt"]
    assert r1d0["user_prompt"] == "AAA"

    # Round 2, doc 0: other-docs block present, carries doc 1's round-1 summary,
    # NOT its own; document is still the user message.
    r2d0 = find("summary-r2-d0")
    assert "summaries of the other documents" in r2d0["system_prompt"]
    assert "<summary of BBB>" in r2d0["system_prompt"]
    assert "<summary of AAA>" not in r2d0["system_prompt"]
    assert r2d0["user_prompt"] == "AAA"
