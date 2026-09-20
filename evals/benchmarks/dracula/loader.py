"""Dracula mini-benchmark loader — the showcase corpus + question set.

Bram Stoker's *Dracula* (1897, public domain — Project Gutenberg #345) decompiled
back into the in-world documents the novel pretends to be compiled from
(``split.py``), plus a small set of hand-curated compositional questions
(``questions.json``) whose answers require piecing together evidence across many
of those documents.

The corpus is the **46 documents** under ``docs/``: the five running
journals/diaries (Jonathan Harker, Mina Murray, Mina Harker, Lucy Westenra,
Dr. Seward) pooled into one document each, and every letter, telegram, newspaper
cutting, memorandum, report, the ship's log, the phonograph message, and both
notes as individual documents. One in-world artifact per file, and every file
whole: where one document is printed inside another -- the log of the *Demeter*
inside the Dailygraph cutting that transcribes it -- both are documents and the
container is rejoined around it (see ``split.py``). Every question shares the SAME full corpus,
returned in a fixed-seed shuffled order (``_CORPUS_ORDER_SEED`` — see ``_corpus``).

Exposes the standard benchmark surface (``get_task_ids`` / ``get_task`` /
``get_documents`` / ``get_task_answer`` / ``get_task_metadata`` +
``STARTER_FILTER``); scoring lives in ``judge.py`` (0/1 LLM judge, the LooGLE
pattern).
"""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path

# Order of the corpus. `get_documents` returns the documents shuffled ONCE with this
# fixed seed, so document position carries no signal from the novel's own arrangement and
# every consumer sees the same stream.
#
# The VALUE is chosen, not arbitrary. A method that reads the corpus as one linear stream
# — a recurrent memory, a gist pass, an agent scanning in order — meets each document with
# however much of the corpus it has already absorbed, so a document's POSITION decides how
# much context precedes it. Book order buries the load-bearing evidence: because the five
# journals are pooled one-per-source, Harker's and Seward's (105k of 161k words) come
# first and the log of the *Demeter* does not begin until ~75% in.
#
# 163740 places the **log of the "Demeter" 6th of 46**, behind three short letters and two
# telegrams (502 words) — so the muster the `death_toll` gold turns on ("five hands ...
# two mates, cook, and myself") is read EARLY, against almost no prior context. That makes
# "what does a method do with the nine deaths when it meets them cold, before the novel
# has told it who Dracula is?" a measurable question rather than an untestable one.
#
# CHANGING THIS RE-ORDERS THE CORPUS FOR EVERY BASELINE: chunk ids, retrieval indices and
# gist pages all shift, so existing `logs/dracula/` runs become runs against a different
# stream. `tests/test_dracula.py` pins what the value buys.
_CORPUS_ORDER_SEED = 163740

_HERE = Path(__file__).parent
_QUESTIONS_FILE = _HERE / "questions.json"
_CORPUS_DIR = _HERE / "docs"

# The default measured-on subset (`get_task_ids(**STARTER_FILTER)`): everything.
STARTER_FILTER: dict = {}


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """The questions, keyed by ``id``."""
    with open(_QUESTIONS_FILE, encoding="utf-8") as f:
        return {row["id"]: row for row in json.load(f)}


def _row(task_id: str) -> dict:
    row = _load().get(task_id)
    if row is None:
        raise KeyError(f"Unknown Dracula task id: {task_id!r}")
    return row


@lru_cache(maxsize=1)
def _corpus() -> tuple[str, ...]:
    """The corpus documents, book order shuffled ONCE with ``_CORPUS_ORDER_SEED`` —
    deterministic across calls, runs and baselines. See the constant for why that value."""
    files = sorted(_CORPUS_DIR.glob("*.txt"))
    if not files:
        raise FileNotFoundError(
            f"No corpus documents under {_CORPUS_DIR} — run split.py."
        )
    docs = [f.read_text(encoding="utf-8") for f in files]
    random.Random(_CORPUS_ORDER_SEED).shuffle(docs)
    return tuple(docs)


def get_task_ids() -> list[str]:
    """All question ids, in ``questions.json`` order."""
    return list(_load().keys())


def get_task(task_id: str) -> tuple[str, list[str]]:
    """Returns ``(question, docs)`` — everything the task needs.

    ``docs`` is the full 46-document bundle (every Dracula question reasons over
    the whole corpus), in the fixed shuffled order.
    """
    return _row(task_id)["question"], get_documents(task_id)


def get_documents(task_id: str) -> list[str]:
    """The documents this task reasons over — the WHOLE corpus, for every task.

    All questions share one document set (like a per-task multi-doc benchmark
    whose bundles happen to coincide — byte-identical doc-sets, so an
    index-building baseline resolves them to one shared index).
    """
    _row(task_id)  # validate the id
    return list(_corpus())


def get_task_answer(task_id: str) -> str:
    """The gold answer for a task id."""
    return _row(task_id)["answer"]


def get_task_metadata(task_id: str) -> dict:
    """Metadata — no axes yet; kept for surface parity."""
    _row(task_id)
    return {}
