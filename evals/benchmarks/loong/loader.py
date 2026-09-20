"""Loong benchmark loader (EN + ZH, extended multi-doc QA).

Loong (EMNLP 2024, Alibaba; arXiv 2406.17419) is a long-context QA benchmark
whose evidence is *scattered across multiple relevant documents* — "leave no
document behind": ignore any document and the answer is wrong. Each instance
ships a list of document FILENAMES (resolved against a separate ~34 MB document
pool) plus a per-instance ``prompt_template`` with ``{docs}``/``{instruction}``/
``{question}`` slots.

This loader exposes **all 1,600 instances** — English (695) AND Chinese (905);
``get_task_ids(languages=…)`` is the selector (default = both). Three domains:
``paper`` (EN only, 400), ``financial`` (EN 295 + ZH 405), ``legal`` (ZH only,
500). Task types (``level`` 1–4) and length sets (``set`` 1–4) span both
languages. What we changed from the upstream benchmark — including a real bug we
fixed in the legal domain — is recorded in ``CHANGES.md`` beside this file.

Axes, all filterable in ``get_task_ids``:
- ``languages`` — ``"en"`` / ``"zh"`` (``loong.LANGUAGES``); default = both.
- ``tasks`` — the task type, the upstream ``level`` (1 Spotlight Locating,
  2 Comparison, 3 Clustering, 4 Chain of Reasoning), of rising difficulty.
- ``sets`` — the context-length bucket, the upstream ``set`` (1 ~10–50K …
  4 ~200–250K), dialed by adding *more documents*, not noise.

Scoring is judge-only (no exact match): see ``judge.py`` (1–100 rating, mean =
Avg Score, fraction==100 = Perfect Rate). This module deliberately omits
``get_task_choices`` and ``parse`` — like LooGLE, Loong is free-form generation.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, TypeVar

_T = TypeVar("_T")

from evals.benchmarks._sampling import order_task_ids
from evals.settings import settings


# The four task types (upstream ``level``), four length buckets (upstream
# ``set``), and two languages. Exposed so callers can pass ``tasks=`` / ``sets=`` /
# ``languages=`` to ``get_task_ids`` without typos. Tasks/sets span 1–4 in both langs.
TASKS = frozenset({1, 2, 3, 4})
SETS = frozenset({1, 2, 3, 4})
LANGUAGES = frozenset({"en", "zh"})

# The opening of the level-4 legal "match each judgment document to its verdict"
# question. When it's present the verdict (a doc's ``result``) must be withheld or
# the answer leaks into the input. Upstream gates this on the marker being in the
# ``instruction``, but it actually lives in the ``question`` — so upstream never
# hides the verdict; we fix that here (see CHANGES.md).
_LEGAL_VERDICT_MARKER = "阅读以上判决文书，我将给你若干份判决结果："

# The default measured-on subset for a runner (`get_task_ids(**STARTER_FILTER)`):
# all languages (EN + ZH), all sets, all task types. Lives with the benchmark.
STARTER_FILTER: dict = {}

# Human-readable task-type names (the upstream ``level`` → name mapping, README
# "Things To Know"). Surfaced in metadata for legible per-axis breakdowns.
_TASK_NAMES = {
    1: "Spotlight Locating",
    2: "Comparison",
    3: "Clustering",
    4: "Chain of Reasoning",
}

# Analysis-report display config, read generically by `evals.analysis.aggregate`
# via getattr (so the analysis layer stays benchmark-agnostic — no name branching).
# - Hide the redundant breakdown axes: `task_name` duplicates `task` (whose integer
#   values the labels below annotate with the name), and `length` is the raw token
#   count — a per-task continuous value that's just a finer-grained `set` (the
#   context-length bucket).
# - Label the small integer axes so a reader needn't memorize the 1–4 encoding:
#   `set` is the context-length tier (Loong's four buckets, arXiv 2406.17419 — the
#   loader docstring's "1 ~10–50K … 4 ~200–250K"), `task` is the level → task name.
ANALYSIS_HIDE_AXES = frozenset({"task_name", "length"})
ANALYSIS_VALUE_LABELS: dict[str, dict[str, str]] = {
    "set": {
        "1": "1 · 10–50K tok",
        "2": "2 · 50–100K tok",
        "3": "3 · 100–200K tok",
        "4": "4 · 200–250K tok",
    },
    "task": {str(level): f"{level} · {name}" for level, name in _TASK_NAMES.items()},
}


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """Loads ALL Loong instances (EN + ZH; 1,600), indexed by task id (``id``).

    Language is a *filter axis* (``get_task_ids(languages=…)``), not a load-time
    cut — so the full benchmark is available and the vendored file stays a faithful
    copy of upstream. Task ids (UUID strings) are globally unique.
    """
    rows: dict[str, dict] = {}
    with open(settings.LOONG_DIR / "loong.jsonl", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def _validate_filter(
    name: str, values: Iterable[_T] | None, allowed: frozenset[_T]
) -> set[_T] | None:
    """Returns the validated filter as a set, or None when no filter is requested."""
    if values is None:
        return None
    values_set = set(values)
    unknown = values_set - allowed
    if unknown:
        raise ValueError(
            f"Unknown {name}(s): {sorted(unknown)}. Valid: {sorted(allowed)}"
        )
    return values_set


def get_task_ids(
    sets: Iterable[int] | None = None,
    tasks: Iterable[int] | None = None,
    languages: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Returns task ids (the upstream ``id`` UUIDs), optionally filtered.

    Filters combine with **AND** semantics: a task is included only if it matches
    every provided filter. ``None`` for a filter means "don't filter on this axis"
    — so the default returns ALL 1,600 ids (EN + ZH).

    Args:
        sets: If provided, only ids whose length bucket (upstream ``set``) is in
            this iterable. Each value must be in ``loong.SETS`` (= ``{1, 2, 3,
            4}``). Raises ``ValueError`` on any unknown value.
        tasks: If provided, only ids whose task type (upstream ``level``) is in
            this iterable. Each value must be in ``loong.TASKS`` (= ``{1, 2, 3,
            4}``). Raises ``ValueError`` on any unknown value.
        languages: If provided, only ids whose ``language`` is in this iterable.
            Each value must be in ``loong.LANGUAGES`` (= ``{"en", "zh"}``). Raises
            ``ValueError`` on any unknown value.
        limit: If provided, return at most this many ids — a stable, representative
            sample (see the ordering note below).

    The result is **deterministically shuffled** (a fixed, non-exposed seed) before
    returning, so ``limit`` yields a representative spread rather than the first N
    in file order, and the order is reproducible across runs
    (``evals.benchmarks._sampling``).
    """
    rows = _load()
    set_filter = _validate_filter("set", sets, SETS)
    task_filter = _validate_filter("task", tasks, TASKS)
    lang_filter = _validate_filter("language", languages, LANGUAGES)

    if set_filter is None and task_filter is None and lang_filter is None:
        ids = list(rows.keys())
    else:
        ids = [
            tid
            for tid, row in rows.items()
            if (set_filter is None or row["set"] in set_filter)
            and (task_filter is None or row["level"] in task_filter)
            and (lang_filter is None or row["language"] in lang_filter)
        ]
    return order_task_ids(ids, limit)


def get_task(task_id: str) -> tuple[str, str, list[str]]:
    """Returns ``(instruction, question, docs)`` — everything needed for the task.

    ``docs`` is the instance's resolved multi-doc bundle (a list of document
    strings, NOT concatenated). Pass the pieces straight to the model, e.g.
    ``model(system_prompt=instruction, user_prompt=question, docs=docs)``.

    Note ``question`` is empty for some instances (e.g. paper Chain-of-Reasoning)
    where the whole task lives in ``instruction``. Resolving ``docs`` downloads the
    ~34 MB document pool on first use (see :func:`get_documents`).
    """
    row = _load()[task_id]
    return row["instruction"], row["question"], get_documents(task_id)


def get_prompt_template(task_id: str) -> str:
    """The instance's upstream Loong ``prompt_template`` — the raw template string
    with ``{instruction}`` / ``{question}`` / ``{docs}`` slots.

    Exposed for baselines that reproduce Loong's *own* prompt assembly rather than
    the harness's component composition. The StructRAG baseline uses it to build
    its query exactly as upstream StructRAG does
    (``prompt_template.format(instruction=…, question=…, docs="......")``). Not part
    of the core surface — ``get_task`` already returns the resolved components — so
    it's a thin extra accessor a baseline can opt into.
    """
    return _load()[task_id]["prompt_template"]


def _ensure_doc_dir() -> Path:
    """Return the extracted document pool, downloading it on first use.

    Delegates to :mod:`evals.benchmarks.loong.download_docs` (the single source
    of truth for acquiring the docs) — the same code path you can run explicitly
    via ``python -m evals.benchmarks.loong.download_docs``. The ~34 MB pool is
    fetched + unzipped into ``settings.LOONG_DIR/doc/`` (gitignored) and reused on
    subsequent calls.
    """
    from evals.benchmarks.loong.download_docs import ensure

    return ensure()


@lru_cache(maxsize=1)
def _load_legal_json(legal_dir: str) -> dict:
    """The ZH legal corpus: ``doc/legal/legal.json`` (629 cases keyed by case name;
    each value carries ``content`` (case body) + ``result`` (verdict)). Cached — it
    is ~4.5 MB and every legal-doc resolution indexes into it."""
    with open(Path(legal_dir) / "legal.json", encoding="utf-8") as f:
        return json.load(f)


def _resolve_doc(doc_dir: Path, row: dict, idx: int, doc_name: str) -> str:
    """Resolve one filename → formatted document text, per domain.

    Mirrors upstream ``src/utils/prompt.py:get_content`` for all three domains.
    ``idx`` is the document's position in the instance's ``doc`` list (only legal
    uses it, for its positional title).

    - **financial** (EN + ZH): the file is located by glob — ``*2024-{doc_name}*.txt``
      (``doc_name`` a company name, e.g. ``"AUDDIA INC."`` / ``"英力特"``), except
      ``level == 4`` (chain-of-reasoning over a company's multiple annual reports)
      where ``doc_name`` already carries the year so the glob is ``*{doc_name}*.txt``.
      Prefixed with ``《{title}》`` where ``title`` is the file stem's last
      ``-``-segment — VERBATIM upstream, incl. odd headings like ``《j》``.
    - **paper** (EN only): the file is ``{doc_name}`` (an arXiv id like
      ``2402.01739.md``) read directly; prefixed with its first-line title (leading
      ``#`` stripped).
    - **legal** (ZH only): ``doc_name`` is a KEY into ``doc/legal/legal.json``; the
      doc is ``content + result`` (case body + verdict), titled positionally
      ``《判决文书{idx+1}》``. EXCEPTION — the level-4 "match each document to its
      verdict" task (``_LEGAL_VERDICT_MARKER`` present) emits ``content`` ONLY so the
      verdict isn't leaked into the input. **This is the one place we FIX an upstream
      bug**: upstream gates that on the marker being in ``instruction``, but it lives
      in ``question``, so upstream never hides the verdict (CHANGES.md).

    Upstream takes ``glob(...)[0]`` in arbitrary order; we ``sorted`` first for
    determinism (matters only on the rare multi-match glob).
    """
    doc_type = row["type"]
    type_dir = doc_dir / doc_type
    if doc_type == "financial":
        level = row["level"]
        pattern = f"*{doc_name}*.txt" if str(level).strip() == "4" else f"*2024-{doc_name}*.txt"
        matches = sorted(type_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No Loong financial document matching {pattern!r} under {type_dir}"
            )
        path = matches[0]
        title = path.stem.split("-")[-1]
        return f"《{title}》\n" + path.read_text(encoding="utf-8") + "\n\n"
    if doc_type == "paper":
        path = type_dir / doc_name
        content = path.read_text(encoding="utf-8")
        title = content.split("\n", 1)[0].strip("#").strip()
        return f"{title}\n" + content + "\n\n"
    if doc_type == "legal":
        entry = _load_legal_json(str(type_dir))[doc_name]
        hide_verdict = row["level"] == 4 and _LEGAL_VERDICT_MARKER in row.get("question", "")
        content = entry["content"] if hide_verdict else entry["content"] + entry["result"]
        return f"《判决文书{idx + 1}》\n" + content + "\n\n"
    raise ValueError(
        f"Unsupported Loong doc_type {doc_type!r} "
        f"(expected 'financial', 'paper', or 'legal')."
    )


def get_documents(task_id: str) -> list[str]:
    """The documents this task reasons over — the resolved text of each filename
    in the instance's ``doc`` list.

    Graphrag-facing accessor (the index is content-addressed by this document
    SET). Loong is the motivating **per-instance multi-doc** case: each instance
    has its own bundle, so distinct instances usually get distinct indexes, while
    instances sharing a doc-set reuse one index automatically. No ``get_corpus``,
    so graphrag treats Loong as per-task (indexes on demand in ``run_one``).

    Documents are returned in the instance's ``doc`` order. Upstream optionally
    shuffles + length-truncates when building the prompt string; we don't —
    deterministic order is reproducible, and graphrag's index identity is
    order-independent anyway (see ``graphrag/run.py:_content_fingerprint``). The
    ``doc`` order is also what legal's positional title ``《判决文书{idx+1}》`` is keyed
    to, and what the legal gold answers reference — so it must be preserved.

    First call downloads + extracts the ~34 MB document pool (see
    :func:`_ensure_doc_dir`).
    """
    row = _load()[task_id]
    doc_dir = _ensure_doc_dir()
    return [_resolve_doc(doc_dir, row, idx, name) for idx, name in enumerate(row["doc"])]


def get_task_answer(task_id: str):
    """Returns the gold answer for a task id.

    Unlike the other loaders this is **not always a string**: Loong answers are
    sometimes a JSON object (e.g. the paper citation task's
    ``{"Reference": [...], "Citation": [...]}``) or a list, returned here as the
    native ``dict`` / ``list`` / ``str``. The judge (``judge.py``) serializes
    non-string golds before grading.
    """
    return _load()[task_id]["answer"]


def get_task_metadata(task_id: str) -> dict:
    """Returns metadata: ``set``, ``task``, ``task_name``, ``domain``, ``language``,
    ``length``.

    - ``set`` — context-length bucket (upstream ``set``, 1–4).
    - ``task`` — task type number (upstream ``level``, 1–4).
    - ``task_name`` — the human-readable task type (e.g. ``"Clustering"``).
    - ``domain`` — document domain (upstream ``type``: ``"paper"`` / ``"financial"`` /
      ``"legal"``).
    - ``language`` — ``"en"`` / ``"zh"``.
    - ``length`` — the instance's token length (upstream ``length``).
    """
    row = _load()[task_id]
    return {
        "set": row["set"],
        "task": row["level"],
        "task_name": _TASK_NAMES[row["level"]],
        "domain": row["type"],
        "language": row["language"],
        "length": row["length"],
    }
