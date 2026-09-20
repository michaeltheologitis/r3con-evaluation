"""CorpusQA benchmark loader (corpus-level analytical reasoning, EN + ZH).

Like LooGLE, CorpusQA is free-form generation graded by an LLM judge — answer
equivalence against a programmatically-computed gold → 0/1 — so this module omits
``get_task_choices`` and ``parse``. Scoring is the ORM equivalence judge (see
``judge.py``); the analysis CLI reports plain accuracy (no ``PERFECT_SCORE``). It is
multi-document and per-instance (no shared corpus → no ``get_corpus``), so the RAG
baselines treat it per-task automatically. CorpusQA is **1m-only** (the single wired
tier — 329 tasks, ~1M tokens each); see ``loader.py`` for the offset-index design that
keeps the multi-GB file out of memory.
"""

from . import download_data  # the data-fetch script (1m tier), exposed as a package attr
from .judge import GRADER_MODEL, SCORER, score, score_batch, score_details
from .loader import (
    DOMAINS,
    SETS,
    STARTER_FILTER,
    get_documents,
    get_task,
    get_task_answer,
    get_task_ids,
    get_task_metadata,
)

__all__ = [
    "DOMAINS",
    "SETS",
    "STARTER_FILTER",
    "download_data",
    "get_task_ids",
    "get_task",
    "get_documents",
    "get_task_answer",
    "get_task_metadata",
    "score",
    "score_batch",
    "score_details",
    "SCORER",
    "GRADER_MODEL",
]
