"""Loong benchmark adapter.

Loong is extended multi-document QA: each task's evidence is scattered across a
per-instance bundle of full documents (SEC filings or arXiv papers), so the answer needs
most of the bundle rather than one retrievable passage. The composed ``task`` is the
per-instance ``instruction`` plus the ``question`` (the question is empty for the
citation-chain instances, where the whole task lives in the instruction).

The adapter's only job is to hand the pipeline a ``(task, documents)`` pair. It consumes
the repo's own ``evals.benchmarks.loong`` surface; the import is lazy so this module stays
importable without the benchmark data on disk.
"""

from __future__ import annotations

from dataclasses import dataclass

NAME = "Loong"


@dataclass(frozen=True)
class TaskInput:
    """What the R3Con pipeline consumes for one task."""

    task: str
    documents: list[str]


def _mod():  # noqa: ANN202 — the lazily-imported loader module
    from evals.benchmarks import loong

    return loong


def list_task_ids(*, sets=None, tasks=None, limit=None) -> list[str]:
    """Task ids matching the filters (all if none). The upstream order is
    deterministically shuffled, so ``limit=N`` is a stable representative sample."""
    return list(_mod().get_task_ids(sets=sets, tasks=tasks, limit=limit))


def load(task_id: str) -> TaskInput:
    """The composed ``task`` + the multi-document bundle for one task."""
    instruction, question, docs = _mod().get_task(task_id)
    task = instruction if not question else f"{instruction}\n\n{question}"
    return TaskInput(task=task, documents=list(docs))


def gold(task_id: str) -> str:
    return _mod().get_task_answer(task_id)


def metadata(task_id: str) -> dict:
    """Per-task metadata (``domain`` / ``set`` / ``task`` / ``task_name`` / …)."""
    return _mod().get_task_metadata(task_id)


