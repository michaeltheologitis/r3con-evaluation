"""CorpusQA benchmark loader (corpus-level analytical reasoning, EN + ZH).

CorpusQA (Alibaba / Tongyi-Zhiwen, Dec 2025; MIT) — *"A 10 Million Token Benchmark
for Corpus-Level Analysis and Reasoning"*. Each instance ships a per-instance
bundle of real documents (PDF→markdown, rich with statistical tables) plus a
**computation-heavy** analytical question (filter / rank / statistical aggregation
/ cross-document arithmetic) whose answer requires reading a *large fraction* of
the documents — deliberately anti-RAG (evidence is dispersed). The gold answer is
**programmatically computed** (schema-extract → global table → NL2SQL), so it is a
native number / string / list (sometimes an empty list ``[]`` meaning "no rows
matched", by design). Scoring is an LLM equivalence judge → 0/1 (see ``judge.py``,
which reproduces upstream ``src/eval.py``'s ORM judge). It is Loong's bigger,
computation-heavy cousin — Loong was this loader's template.

Axes (all metadata):
- **domain** (4): ``financial_zh``, ``financial_en``, ``education_en``,
  ``real_estate_en`` (only ``financial_zh`` is Chinese; the judge is
  language-agnostic, proven on Loong).
- **set** = context-length tier: **always ``1m``** (a 1m instance is ~1M tokens of
  documents). CorpusQA is deliberately 1m-only — that's the tier we measure on, so it is
  the only wired tier and ``get_task_ids()`` always returns it. (128k/4m/10m exist
  upstream but are NOT wired: 128k is too small to be interesting, 4m/10m fit no window
  and Hugging Face — the sole mirror — doesn't host them.) NOTE: the tier is the FILE —
  upstream ships every row with ``set == "128k"`` even in the 1m file, so the loader keys
  task ids on the file (``@1m``), not that field (see ``_index_record``).
- **language**: ``en`` / ``zh`` — DERIVED from the domain (only ``financial_zh`` is
  ``zh``), so it is a metadata axis, not a separate filter (use ``domains=``).

Upstream's per-row ``type`` field (intended difficulty: Easy/Medium/Hard) is the
**empty string in every released row** (verified across all four domains at 128k),
so it carries no signal — this loader does NOT surface it as a filter or axis.

The crucial data detail — un-baking the frozen prompt
=====================================================
Each row's ``prompt`` is NOT raw data: it is the **fully-assembled chat request**
upstream sent (a ``[system, user]`` message list). The ``user`` message glues
together (1) the documents, each behind a ``# Document N:`` header; (2) a
``# Question:`` marker + the NL question; (3) a **required "Output requirements:"
instruction block** carrying the answer-format + conflict-resolution rules the
NL2SQL gold was computed under (e.g. *"if multiple files conflict, use the value
from the latest file"*; the ``The answer is: xxx`` output contract). This loader
un-bakes that into the harness's ``(instruction, question, docs)`` components:
``get_documents`` splits on the ``# Document N:`` headers; ``instruction`` is the
output-requirements block (it MUST reach the model — it is task semantics); the
question is the row's own ``question`` field. The ``The answer is:`` contract is
kept, so the judge ports upstream's ``extract_answer`` (see ``judge.py``).

Scale + the offset index
=========================
The 1m tier is ONE huge file (``1m_4domains.jsonl`` ≈ 1 GB). Loading it wholesale per
process — as the runner fans out one subprocess per task — would OOM. So on first use
this loader builds a small **byte-offset index** over the file
(``{tier}_index_<v>.jsonl``: everything but the giant ``prompt``, plus the
row's byte offset). Task-id enumeration, metadata, gold answers, and judge scoring
all read the cheap index; only ``get_task`` / ``get_documents`` seek+read the one
row's prompt to un-bake its documents. The raw tier files are fetched lazily (see
``download_data.py``) and are gitignored — nothing multi-GB is vendored.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, TypeVar

_T = TypeVar("_T")

from evals.benchmarks import _sampling
from evals.benchmarks.corpusqa import download_data
from evals.settings import settings

# The four document domains. Only ``financial_zh`` is Chinese.
DOMAINS = frozenset({"financial_zh", "financial_en", "education_en", "real_estate_en"})

# The wired context-length tiers — owned by the acquisition module (it knows which
# tiers it can fetch), re-exported here as the benchmark's public ``corpusqa.SETS``.
SETS = download_data.SETS

# CorpusQA is 1m-only (``SETS == {"1m"}``), so there is nothing to filter: a bare
# ``get_task_ids()`` already returns the whole benchmark — the 1m tier, all 4 domains
# (329 tasks). The starter subset is therefore EMPTY, exactly like loong/longbenchv2:
# ``get_task_ids(**STARTER_FILTER)`` == ``get_task_ids()`` == all 329 1m tasks.
STARTER_FILTER: dict = {}

# Bump when the offset-index RECORD shape changes — the version is in the index
# filename, so a bump rebuilds rather than mis-reads an old index.
# v2: task_id / `set` now key on the FILE tier, not the row's (always-"128k") `set`
#     field — so 1m tasks are `@1m` and actually resolve to the 1m file (fixes the v1
#     bug where every task id was `@128k` and 1m/4m silently served 128k content).
_INDEX_VERSION = "v2"

# Structural markers in the baked ``user`` prompt (verified byte-stable across all
# four domains, including the Chinese ``financial_zh``, at 128k):
#   …# Document 1:\n<doc1>\n# Document 2:\n<doc2>…\n\n# Question:\n<question>\n\n<instr>
_DOC_HEADER_RE = re.compile(r"# Document \d+:\n")
_QUESTION_SEP = "\n# Question:\n"

# Language is fully derived from the domain — only ``financial_zh`` is Chinese.
_ZH_DOMAINS = frozenset({"financial_zh"})

# Report/scoreboard display names for the ``domain`` axis (what ``get_task_metadata``
# returns + what a report groups "by domain" on): the bilingual ``financial``
# split (``financial_en`` + ``financial_zh``) MERGES into one ``financial`` bucket and the
# ``_en``/``_zh`` suffix is dropped → three buckets, sorting to ``education`` / ``financial``
# / ``real estate``. This is ONLY the report grouping — the raw 4-way domain still drives
# :data:`DOMAINS` and ``get_task_ids(domains=…)``, and ``language`` stays its own axis, so
# ``financial_en`` vs ``financial_zh`` is still recoverable (by ``language``, or by ``n_docs``:
# financial_en=25 vs financial_zh=90).
_DOMAIN_DISPLAY = {
    "education_en": "education",
    "financial_en": "financial",
    "financial_zh": "financial",
    "real_estate_en": "real estate",
}


def _index_file(tier: str) -> Path:
    """The persisted offset index for one tier (versioned in the name)."""
    return settings.CORPUSQA_DIR / f"{tier}_index_{_INDEX_VERSION}.jsonl"


def _user_content(prompt: list[dict]) -> str:
    """The ``user`` message content from a row's baked ``prompt`` (the docs +
    question + instruction live here; the ``system`` message is just the persona)."""
    for msg in reversed(prompt):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    raise ValueError("CorpusQA prompt has no 'user' message")


def _index_record(row: dict, offset: int, length: int, tier: str) -> dict:
    """One offset-index entry: everything the cheap paths need, minus the giant
    ``prompt`` (re-read on demand via ``offset``/``length``).

    The task id folds in the **tier** (the FILE the row came from, passed in as
    ``tier``). **We use the file tier, NOT the row's own ``set`` field**: upstream ships
    every row with ``set == "128k"`` even inside the 1m file, so keying on ``row["set"]``
    would make every 1m task id ``@128k`` — and ``@128k`` is not a wired tier
    (``SETS == {"1m"}``), so every lookup would then fail (and historically, when 128k was
    also wired, it silently served 128k content for 1m tasks). The tier IS the filename.
    Validates the ``# Document N:`` header count against ``doc_files`` here, at build time,
    so a malformed row fails loudly during prep rather than mid-run.
    """
    rid = row["id"]
    doc_files = row.get("doc_files", [])
    n_headers = len(_DOC_HEADER_RE.findall(_user_content(row["prompt"])))
    if n_headers != len(doc_files):
        raise ValueError(
            f"CorpusQA row {rid!r} (tier {tier!r}): {n_headers} '# Document N:' headers "
            f"but {len(doc_files)} doc_files — un-baking would mismatch."
        )
    return {
        "task_id": f"{rid}@{tier}",
        "id": rid,
        "domain": row["domain"],
        "set": tier,
        "question": row["question"],
        "answer": row["answer"],
        "doc_files": doc_files,
        "n_docs": len(doc_files),
        "offset": offset,
        "length": length,
    }


def _build_index(tier: str) -> None:
    """Build the tier's offset index if missing (download the tier file first).

    Streams the (multi-GB) tier file ONCE in binary mode, recording each row's byte
    offset + a small metadata record, and atomically publishes the index. An
    exclusive per-tier file lock + the atomic publish make this safe under the
    runner's parallel children (exactly one builds; the rest block then reuse).
    """
    download_data.ensure(tier)  # the raw file (its own lock); idempotent
    index_path = _index_file(tier)
    if index_path.exists():
        return

    settings.CORPUSQA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = settings.CORPUSQA_DIR / f".{tier}.index.lock"
    with open(lock_path, "w") as lock_file:
        _acquire_lock(lock_file)
        if index_path.exists():  # a peer built it while we blocked
            return
        records: list[dict] = []
        with open(download_data.tier_file(tier), "rb") as f:
            offset = 0
            for raw in f:
                line = raw.rstrip(b"\r\n")
                if line.strip():
                    records.append(_index_record(json.loads(line), offset, len(line), tier))
                offset += len(raw)
        fd, tmp = tempfile.mkstemp(dir=settings.CORPUSQA_DIR, suffix=".index.part")
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for rec in records:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(tmp, index_path)  # atomic publish


def _acquire_lock(handle) -> None:
    """Best-effort exclusive cross-process lock (POSIX ``flock``); no-op off POSIX."""
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(handle, fcntl.LOCK_EX)


@lru_cache(maxsize=None)
def _tier_index(tier: str) -> dict[str, dict]:
    """The tier's offset index, keyed by task id — built + cached once per process.

    Building it downloads the 1m file from Hugging Face (~1 GB) on first use (the cluster
    pre-fetches via ``download_data``). CorpusQA is 1m-only, so this only ever builds the
    1m index.
    """
    if tier not in SETS:
        raise ValueError(f"Unknown tier {tier!r}. Wired tiers: {sorted(SETS)}.")
    _build_index(tier)
    out: dict[str, dict] = {}
    with open(_index_file(tier), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                out[rec["task_id"]] = rec
    return out


def rebuild_index(tier: str) -> None:
    """Delete + drop the cached offset index for a tier (forces a rebuild on next
    use). Called after a ``--force`` re-download, whose new file invalidates the
    offsets the old index recorded."""
    index_path = _index_file(tier)
    if index_path.exists():
        index_path.unlink()
    _tier_index.cache_clear()


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
    domains: Iterable[str] | None = None,
    sets: Iterable[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Returns task ids (``{id}@{set}``), optionally filtered.

    Filters combine with **AND**: ``domains`` (each in :data:`DOMAINS`) and ``sets``
    (each in :data:`SETS` — only ``1m`` is wired). ``None`` means "don't filter this
    axis" — so a bare ``get_task_ids()`` returns ALL 329 1m instances (lazily downloading
    the ~1 GB file on first use).

    The result is **deterministically shuffled** (a fixed, non-exposed seed) before
    returning — so ``limit`` yields a representative spread across domains rather than the
    first N in file order, and the order is reproducible across runs
    (``evals.benchmarks._sampling``).

    Raises ``ValueError`` on any unknown domain / set.
    """
    dom_filter = _validate_filter("domain", domains, DOMAINS)
    set_filter = _validate_filter("set", sets, SETS)
    tiers = sorted(set_filter) if set_filter is not None else sorted(SETS)
    ids: list[str] = []
    for tier in tiers:
        for tid, rec in _tier_index(tier).items():
            if dom_filter is None or rec["domain"] in dom_filter:
                ids.append(tid)
    return _sampling.order_task_ids(ids, limit)


def _locate(task_id: str) -> tuple[str, dict]:
    """Returns ``(tier, index_record)`` for a task id. The tier is encoded in the id
    (``{id}@{set}``); an id with an unknown/absent tier raises ``KeyError`` without
    touching the network."""
    tier = task_id.rpartition("@")[2]
    if tier not in SETS:
        raise KeyError(f"Unknown CorpusQA task id: {task_id!r}")
    rec = _tier_index(tier).get(task_id)
    if rec is None:
        raise KeyError(f"Unknown CorpusQA task id: {task_id!r}")
    return tier, rec


def _read_row(tier: str, rec: dict) -> dict:
    """Read ONE row (the full record, including the giant ``prompt``) by seeking to
    its recorded byte offset — never loads the whole tier file."""
    with open(download_data.tier_file(tier), "rb") as f:
        f.seek(rec["offset"])
        raw = f.read(rec["length"])
    return json.loads(raw.decode("utf-8"))


def _unbake(row: dict) -> tuple[str, str, list[str]]:
    """Un-bake a row's frozen prompt into ``(instruction, question, docs)``.

    Splits the ``user`` message on the LAST ``# Question:`` marker into the document
    blob and the ``question + instruction`` tail; splits the blob on the
    ``# Document N:`` headers into the document list; takes the question from the
    row's own field and the instruction (the "Output requirements" block) as the
    remainder of the tail.
    """
    user = _user_content(row["prompt"])
    body, sep, tail = user.rpartition(_QUESTION_SEP)
    if not sep:
        raise ValueError("CorpusQA prompt missing the '# Question:' marker")
    docs = [chunk.strip() for chunk in _DOC_HEADER_RE.split(body)[1:]]
    question = row["question"]
    if tail.startswith(question):
        instruction = tail[len(question):].strip()
    else:
        # Fallback: the instruction is the block after the blank line that follows
        # the question (the structural separator in every observed row).
        instruction = tail.partition("\n\n")[2].strip() or tail.strip()
    return instruction, question, docs


def get_task(task_id: str) -> tuple[str, str, list[str]]:
    """Returns ``(instruction, question, docs)`` — everything needed for the task.

    - ``instruction`` — the "Output requirements" block (answer-format + conflict
      rules the gold was computed under; it MUST reach the model).
    - ``question`` — the natural-language analytical query.
    - ``docs`` — the resolved per-instance document bundle (a list, NOT
      concatenated), identical to :func:`get_documents`.

    Reads (and un-bakes) the one row's prompt via the offset index — see the module
    docstring. Resolving the tier downloads its file on first use.
    """
    tier, rec = _locate(task_id)
    return _unbake(_read_row(tier, rec))


def get_documents(task_id: str) -> list[str]:
    """The documents this task reasons over — the per-instance bundle, un-baked from
    the frozen prompt's ``# Document N:`` sections.

    Graphrag/arag-facing accessor (the index is content-addressed by this document
    SET). CorpusQA is per-instance multi-doc (no shared corpus → no ``get_corpus``),
    so each instance usually gets its own index. Documents are returned in the
    prompt's order.
    """
    tier, rec = _locate(task_id)
    docs = _unbake(_read_row(tier, rec))[2]
    if len(docs) != rec["n_docs"]:
        raise ValueError(
            f"CorpusQA {task_id!r}: un-baked {len(docs)} docs but index expected "
            f"{rec['n_docs']} (doc_files count) — prompt structure changed."
        )
    return docs


def get_task_answer(task_id: str) -> Any:
    """Returns the programmatically-computed gold answer for a task id.

    Like Loong, this is **not always a string**: CorpusQA golds are a native number
    / string / list (including the intentional empty list ``[]`` = "no rows
    matched"). The judge (``judge.py``) serializes non-string golds before grading.
    """
    return _locate(task_id)[1]["answer"]


def get_task_metadata(task_id: str) -> dict:
    """Returns metadata: ``domain`` (the REPORT grouping — ``education`` / ``financial`` /
    ``real estate``, with the bilingual ``financial`` en+zh MERGED and the language suffix
    dropped; NOT the raw 4-way filter domain — see :data:`_DOMAIN_DISPLAY`), ``set``
    (context-length tier), ``language`` (``en``/``zh``, derived from the RAW domain),
    ``n_docs`` (the per-instance document count)."""
    rec = _locate(task_id)[1]
    raw = rec["domain"]
    return {
        "domain": _DOMAIN_DISPLAY.get(raw, raw),
        "set": rec["set"],
        "language": "zh" if raw in _ZH_DOMAINS else "en",
        "n_docs": rec["n_docs"],
    }
