"""Loong benchmark loader (EN + ZH, extended multi-doc QA).

Loong QA is free-form generation graded by an LLM judge: there are no answer
choices to expose and nothing to parse out of an answer. Scoring is the judge's
1–100 rating — see `score` / `score_batch` (and `PERFECT_SCORE`, the marker telling
whatever grades these logs to report Avg Score + Perfect Rate instead of accuracy).
"""

from . import download_docs  # the doc-pool fetch script, exposed as `loong.download_docs`
from .judge import GRADER_MODEL, PERFECT_SCORE, SCORER, score, score_batch, score_details
from .loader import (
    ANALYSIS_HIDE_AXES,
    ANALYSIS_VALUE_LABELS,
    LANGUAGES,
    SETS,
    STARTER_FILTER,
    TASKS,
    get_documents,
    get_prompt_template,
    get_task,
    get_task_answer,
    get_task_ids,
    get_task_metadata,
)

__all__ = [
    "SETS",
    "TASKS",
    "LANGUAGES",
    "STARTER_FILTER",
    "ANALYSIS_HIDE_AXES",
    "ANALYSIS_VALUE_LABELS",
    "PERFECT_SCORE",
    "download_docs",
    "get_task_ids",
    "get_task",
    "get_documents",
    "get_prompt_template",
    "get_task_answer",
    "get_task_metadata",
    "score",
    "score_batch",
    "score_details",
    "SCORER",
    "GRADER_MODEL",
]
