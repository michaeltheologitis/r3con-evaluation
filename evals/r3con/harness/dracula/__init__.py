"""Dracula benchmark adapter — the showcase mini-benchmark.

Bram Stoker's *Dracula* (1897, public domain — Project Gutenberg #345) decompiled back into
the in-world documents the novel pretends to be compiled from: journals, letters, telegrams,
newspaper cuttings, the ship's log, memoranda. The questions are hand-curated and
**compositional** — no single passage answers one; the evidence has to be assembled across
many documents. Every question shares the same full corpus, returned in a fixed-seed
shuffled order.

The ``task`` is just the question (no instruction or options block), and there are no
filters — the simplest adapter of the three. The corpus is vendored in the repo's own
``evals.benchmarks.dracula``, so nothing is downloaded; the import stays lazy.
"""

from __future__ import annotations

from dataclasses import dataclass

NAME = "Dracula"


@dataclass(frozen=True)
class TaskInput:
    """What the R3Con pipeline consumes for one task."""

    task: str
    documents: list[str]


def _mod():  # noqa: ANN202 — the lazily-imported loader module
    from evals.benchmarks import dracula

    return dracula


def list_task_ids(*, limit: int | None = None) -> list[str]:
    """All question ids (in ``questions.json`` order — currently just ``death_toll``), optionally
    truncated to the first ``limit``. Unlike the other benchmarks the upstream order is NOT shuffled
    (the pool is tiny), so ``limit=N`` is simply the first N questions — enough to keep the family's
    ``--limit`` idiom working as questions are added."""
    ids = list(_mod().get_task_ids())
    return ids[:limit] if limit is not None else ids


def load(task_id: str) -> TaskInput:
    """The ``question`` + the full 46-document corpus for one task.

    There is no instruction/output-requirements or options block — the question is self-contained,
    so the composed ``task`` is just the question. ``documents`` is the whole corpus (every Dracula
    question reasons over all 46 in-world documents), returned in the loader's fixed-seed order."""
    question, docs = _mod().get_task(task_id)
    return TaskInput(task=question, documents=list(docs))


def gold(task_id: str) -> str:
    """The gold answer (a string, e.g. ``"15"``)."""
    return _mod().get_task_answer(task_id)


def metadata(task_id: str) -> dict:
    """Per-task metadata (empty for Dracula; kept for surface parity with the other adapters)."""
    return _mod().get_task_metadata(task_id)


