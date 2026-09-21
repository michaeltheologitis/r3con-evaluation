"""CorpusQA benchmark loader (corpus-level analytical reasoning, EN + ZH).

CorpusQA is free-form generation graded by an LLM judge — answer equivalence
against a programmatically-computed gold → 0/1 — so there are no answer choices to
expose and no ``parse`` step onto a fixed answer vocabulary; the judge lifts the
final answer out with upstream's ``extract_answer`` and then judges equivalence.
Scoring is the ORM equivalence judge (see ``judge.py``); with no ``PERFECT_SCORE``
declared, plain accuracy is the metric to report. It is multi-document and
per-instance (no shared corpus → no ``get_corpus``), so the RAG baselines treat it
per-task automatically. CorpusQA is **1m-only** (the single wired tier — 329 tasks,
~1M tokens each); see ``loader.py`` for the offset-index design that keeps the
multi-GB file out of memory.
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
