"""CorpusQA benchmark adapter.

CorpusQA bundles a set of table-rich documents with a computation-heavy analytical
question (filter / rank / aggregate / cross-document arithmetic) spanning a large fraction
of them — deliberately anti-RAG. Only the ``1m`` tier is wired (~1M tokens of documents per
instance).

The composed ``task`` is the ``question`` followed by the ``instruction`` — the "Output
requirements" block, which carries the answer-format and conflict rules, so it has to reach
the model. The adapter's only job is to hand the pipeline a ``(task, documents)`` pair; it
consumes the repo's own ``evals.benchmarks.corpusqa`` surface, imported lazily (the 1m tier
is a ~1 GB download).
"""

from __future__ import annotations

from dataclasses import dataclass

NAME = "CorpusQA"


@dataclass(frozen=True)
class TaskInput:
    """What the R3Con pipeline consumes for one task."""

    task: str
    documents: list[str]


def _mod():  # noqa: ANN202 — the lazily-imported loader module
    from evals.benchmarks import corpusqa

    return corpusqa


def list_task_ids(*, limit: int | None = None) -> list[str]:
    """Task ids (all 1m instances; CorpusQA has no other wired tier). The upstream order
    is deterministically shuffled, so ``limit=N`` is a stable representative sample —
    handy for cheap validation before a full run. (Domains exist upstream but we don't
    filter by them.)"""
    return list(_mod().get_task_ids(limit=limit))


def _compose_task(instruction: str, question: str) -> str:
    """Compose the pipeline ``task`` from CorpusQA's ``(instruction, question)``: the
    ``question`` first, then the ``instruction`` (the "Output requirements" block — the
    answer-format + conflict rules the gold depends on), mirroring how upstream feeds the
    model. Either part may be empty/blank (dropped). Pure (no I/O) so it's unit-testable
    without touching the upstream loader."""
    parts = [p for p in (question, instruction) if p and p.strip()]
    return "\n\n".join(parts)


def load(task_id: str) -> TaskInput:
    """The composed ``task`` + the multi-document bundle for one task.

    Upstream feeds the model ``question`` then the ``instruction`` (the "Output
    requirements" block — answer-format + conflict rules the gold depends on, so it must
    reach the model). We compose the task the same way so the pipeline's downstream
    stages — the inference stage especially — see the output contract."""
    instruction, question, docs = _mod().get_task(task_id)
    return TaskInput(task=_compose_task(instruction, question), documents=list(docs))


def gold(task_id: str):  # noqa: ANN201 — gold may be a number / string / list (incl. [])
    """The programmatically-computed gold (not always a string — can be a number or an
    intentional empty list ``[]``)."""
    return _mod().get_task_answer(task_id)


def metadata(task_id: str) -> dict:
    """Per-task metadata (``domain`` / ``set`` / ``language`` / ``n_docs``)."""
    return _mod().get_task_metadata(task_id)


