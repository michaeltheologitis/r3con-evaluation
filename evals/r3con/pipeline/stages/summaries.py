"""Stage 1: task-conditioned, cross-document summaries (the long-text mechanism).

These are **not** generic summaries. Each one is written *for a specific task* and
is conditioned on it — a note of what one document contributes toward the task, read
in light of the rest of the collection. The point is cross-document reasoning: to use
one document for the task you often need to know what the others say. (We still call
them "summaries" for brevity, but everywhere they are task-conditioned.)

Documents are assumed to each fit in context (no chunking). Cross-collection awareness
is built over ``rounds`` **synchronous** rounds:

- **Round 1** — each document's summary is written from the task and the document
  alone (no other document is visible).
- **Round k ≥ 2** — each document's summary is rewritten given the task and the
  **previous round's** summaries of the OTHER documents (never the document's own
  prior summary — it is itself in the user message). Only the immediately-previous
  round is fed in, not the whole history.

Round k reads the *frozen* round-(k-1) set, so within a round the per-document calls
are independent and fan out in parallel (bounded by
:func:`evals.r3con.pipeline.settings.active_doc_workers`). Downstream stages consume the
**final-round** per-document summaries (``SummariesResult.final``); earlier rounds are
kept only for inspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from evals.r3con.pipeline.logging_setup import get_logger
from evals.r3con.pipeline.parallel import parallel_map
from evals.r3con.pipeline.prompts import load_prompt
from evals.r3con.pipeline.runs import StageRun
from evals.r3con.pipeline.runtime.llm import litellm_chat_completion
from evals.r3con.pipeline.settings import active_doc_workers

_log = get_logger("summaries")

# Separator between the other documents' summaries in the cross-conditioning block.
_OTHER_SEP = "\n\n---\n\n"


@dataclass
class SummariesResult:
    """Output of :func:`summarize_collection`.

    - ``final`` — the final-round per-document summaries, aligned 1:1 with the input
      ``documents`` (``final[d]`` is document ``d``'s task-conditioned summary). This
      is what downstream stages consume.
    - ``rounds`` — every round's per-document summaries, in order: ``rounds[k]`` is
      round ``k+1`` (so ``rounds[-1] is final``). Kept for inspection of the
      refinement; not fed downstream.
    """

    final: list[str]
    rounds: list[list[str]]


def render_summaries(summaries: list[str] | None, doc_ids: list[str] | None = None) -> str:
    """Render the final per-document task-conditioned summaries into a labeled block
    for a downstream stage's "## Task-conditioned document summaries" section.

    Each document gets a ``### Document N`` (or its ``doc_ids`` label) heading. A
    document with nothing relevant renders as ``(no relevant summary for this task)``.
    An empty / ``None`` list renders to ``""`` so the consuming prompt omits the block.
    """
    if not summaries:
        return ""
    parts: list[str] = []
    for i, s in enumerate(summaries):
        text = (s or "").strip()
        label = doc_ids[i] if (doc_ids and i < len(doc_ids)) else f"Document {i + 1}"
        parts.append(f"### {label}\n{text}" if text else f"### {label}\n(no relevant summary for this task)")
    return "\n\n".join(parts)


def _render_others(other_summaries: list[str]) -> str:
    """Concatenate the OTHER documents' summaries for the cross-conditioning block.
    Empty / whitespace-only entries are dropped; an empty list renders to ``""`` so
    the prompt omits the block (round 1, or a one-document collection)."""
    parts = [s.strip() for s in other_summaries if s and s.strip()]
    return _OTHER_SEP.join(parts)


def summarize_one(
    *,
    task: str,
    document: str,
    other_summaries: list[str],
    model: str,
    prompt_version: str,
    run: StageRun | None = None,
    kind: str = "summary",
    **llm_kwargs: Any,
) -> str:
    """Write the task-conditioned summary of one ``document``, given the OTHER
    documents' previous-round summaries (empty list in round 1).

    The task + the (possibly empty) other-documents block live in the system prompt;
    the document is the user message. Returns the summary text, stripped — an empty
    string is allowed (the document contributes nothing relevant to the task).
    """
    system_prompt = load_prompt(
        "summaries", version=prompt_version, task=task, other_summaries=_render_others(other_summaries)
    )
    return cast(
        str,
        litellm_chat_completion(
            system_prompt=system_prompt, user_prompt=document,
            model=model, run=run, kind=kind, **llm_kwargs,
        ),
    ).strip()


def summarize_collection(
    *,
    task: str,
    documents: list[str],
    model: str,
    prompt_version: str,
    rounds: int = 2,
    run: StageRun | None = None,
    workers: int | None = None,
    **llm_kwargs: Any,
) -> SummariesResult:
    """Produce final task-conditioned per-document summaries over ``rounds`` rounds.

    Round 1 summarizes each document from the task + document alone; each later round
    rewrites every document given the *previous round's* summaries of the OTHER
    documents (never its own). Returns a :class:`SummariesResult` (`.final` =
    last-round per-document, `.rounds` = all rounds). ``rounds=1`` = independent
    round-1 only; ``rounds < 1`` = no summaries at all (`.final` / `.rounds` empty —
    a deliberate "no-summaries" run). An empty ``documents`` → empty results.
    """
    # rounds < 1 = "no summaries at all" (a deliberate sr=0 re-run); an empty document
    # collection is likewise empty. Both short-circuit BEFORE round 1 runs (the
    # unconditional run_round(None, 1) below) — a `return`, not a fall-through.
    if rounds < 1 or not documents:
        return SummariesResult(final=[], rounds=[])

    max_workers = workers if workers is not None else active_doc_workers()

    def run_round(prev: list[str] | None, round_idx: int) -> list[str]:
        """Re-summarize every document in parallel. ``prev`` is the frozen
        previous-round summary set (``None`` in round 1)."""
        _log.info("round %d/%d · %d doc(s) (≤%d parallel)", round_idx, rounds, len(documents), max_workers)

        def one(i: int, doc: str) -> str:
            # Others-only: document i sees the previous round's summaries of the OTHER
            # documents (j != i), never its own (the document itself is the user message).
            others = [] if prev is None else [s for j, s in enumerate(prev) if j != i]
            return summarize_one(
                task=task, document=doc, other_summaries=others, model=model,
                prompt_version=prompt_version, run=run, kind=f"summary-r{round_idx}-d{i}", **llm_kwargs,
            )

        result = parallel_map(one, documents, max_workers=max_workers)
        n_nonempty = sum(1 for s in result if s and s.strip())
        _log.info("round %d/%d done · %d/%d doc(s) had a relevant summary",
                  round_idx, rounds, n_nonempty, len(documents))
        return result

    all_rounds: list[list[str]] = [run_round(None, 1)]
    for r in range(2, rounds + 1):
        all_rounds.append(run_round(all_rounds[-1], r))  # only the previous round feeds in

    return SummariesResult(final=all_rounds[-1], rounds=all_rounds)
