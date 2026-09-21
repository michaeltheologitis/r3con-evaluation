"""Per-task benchmark metadata for the scoreboards — from vendored snapshots.

The boards slice Loong by context-length tier and reasoning category, and CorpusQA by
domain; none of that is in a run's manifest, it is in the benchmark's own task files. Those
files are large (CorpusQA's are multi-GB), so the small per-task facts we need — plus each
task's document length in exact served-model tokens — are snapshotted once into
``data/{loong,corpusqa}_meta.json`` and read from there.
"""
from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pandas as pd

# Loong's four task types (level 1–4) and four context-length tiers (set 1–4).
TASK_NAMES = {1: "Spotlight Locating", 2: "Comparison", 3: "Clustering", 4: "Chain of Reasoning"}
# DISPLAY order, deliberately NOT level order: Chain of Reasoning (4) is shown before Clustering
# (3) — owner. This is the single source of truth for the reasoning-category column order; the
# crosstab (via `report._DEFAULT_ORDERS`) and every notebook board / LaTeX export read it, so a
# swap here moves them all together. `TASK_NAMES` keeps the benchmark's own level → name mapping.
TASK_ORDER = [TASK_NAMES[i] for i in (1, 2, 4, 3)]

_META_PATH = Path(__file__).resolve().parent / "data" / "loong_meta.json"
_CORPUSQA_META_PATH = Path(__file__).resolve().parent / "data" / "corpusqa_meta.json"


@cache
def loong_meta() -> dict[str, dict]:
    """``task_id -> {set, task, task_name, domain, language, length, input_tokens}``."""
    if not _META_PATH.is_file():
        raise FileNotFoundError(
            f"{_META_PATH} missing — the snapshot is missing."
        )
    return json.loads(_META_PATH.read_text(encoding="utf-8"))


def attach_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``set`` / ``task`` / ``task_name`` / ``domain`` / ``language`` columns by joining
    the Loong snapshot on ``task_id``. Non-Loong rows (e.g. corpusqa) get NaNs — harmless,
    the loong scoreboards filter to ``benchmark == "loong"``."""
    meta = pd.DataFrame.from_dict(loong_meta(), orient="index")
    meta.index.name = "task_id"
    cols = ["set", "task", "task_name", "domain", "language"]
    return df.merge(meta[cols].reset_index(), on="task_id", how="left")


@cache
def corpusqa_meta() -> dict[str, dict]:
    """``task_id -> {domain, set, language, n_docs}`` (cached). See
    the notes in :mod:`scoreboard.meta` for why this is snapshotted."""
    if not _CORPUSQA_META_PATH.is_file():
        raise FileNotFoundError(
            f"{_CORPUSQA_META_PATH} missing — the snapshot is missing."
        )
    return json.loads(_CORPUSQA_META_PATH.read_text(encoding="utf-8"))


def attach_corpusqa_meta(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``domain`` / ``language`` / ``n_docs`` / ``set`` (the context-length tier, e.g.
    ``"1m"``) by joining the CorpusQA snapshot on ``task_id``. The CorpusQA twin of
    :func:`attach_meta` — used by the corpusqa notebook; non-corpusqa rows get NaNs."""
    meta = pd.DataFrame.from_dict(corpusqa_meta(), orient="index")
    meta.index.name = "task_id"
    cols = ["domain", "language", "n_docs", "set"]
    return df.merge(meta[cols].reset_index(), on="task_id", how="left")

