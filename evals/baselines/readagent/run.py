"""ReadAgent baseline pipeline — connects the gist-memory method to the harness.

ReadAgent (Google DeepMind, ICML 2024; arXiv 2402.09727) is an LLM-ONLY long-context
reading method (no embeddings, no retriever): paginate the document(s) into pages,
compress each page into a gist, then for a question look up a few pages' full text from
the gist memory and answer. We wire **ReadAgent-P** (one batched look-up call) — the only
look-up variant upstream implements in code (PROVENANCE.md D5); there is no variant flag.

SIMPLE, NO-INDEX-REUSE logging (the flat per-run-folder layout): each task
run gets its OWN folder ``logs/{benchmark}/readagent/{run_tag}/`` holding EVERYTHING — the
gist memory (``gist_memory.json``), the ``manifest.json`` (with the **TOTAL** cost —
pagination + gisting + look-up + answer, one number), ``calls.json``, a live
``progress.json`` (the task's current stage + per-stage counts while it runs — readagent
is slow, so this lets you see where a task is mid-flight), and (at score
time) ``score.json``. Nothing is content-addressed and no gist memory is shared — a
pending task always rebuilds its gist memory from scratch. The runner DOES resume, though
(it skips tasks already completed for the same config — see ``runner.py``); resumption is
about not re-RUNNING finished tasks, not about reusing any index.

SUPPORTED_BENCHMARKS = {loong, corpusqa, longhealth, dracula} — per-instance multi-doc bundles (no
shared corpus). Full deviation ledger: evals/baselines/readagent/PROVENANCE.md
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from evals.baselines.readagent import gist

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# ── Per-benchmark pagination/gist regime — selected AUTOMATICALLY by benchmark (PROVENANCE D10) ──
# ReadAgent uses a fixed page SIZE (variable page count). The ORIGINAL regime (600/280/350, no gist
# length clause) is right for docs in ReadAgent's validated range: Loong (~52–76K words → ~90–126
# pages) AND LongHealth (~1.5K-word docs → a few pages). CorpusQA's ~370–420K-word instances are NOT
# — at 600 they paginate to ~600–1,300 pages (~2,500 reasoning calls/task), so they get ×10 pages +
# a 640-token gist hint (~125 pages). This used to be a manual pre-run flip of gist.py/prompts.py +
# ``_RUN_VERSION``; it is now chosen per benchmark below (no editing, no footgun).
_ORIGINAL_REGIME = gist.Regime(word_limit=600, start_threshold=280, short_page_words=350,
                               gist_token_hint=None)   # Loong (v2) + LongHealth
_CORPUSQA_REGIME = gist.Regime(word_limit=6000, start_threshold=2800, short_page_words=3500,
                               gist_token_hint=640)    # CorpusQA (v4)
_REGIMES: dict[str, gist.Regime] = {
    "loong": _ORIGINAL_REGIME,
    "corpusqa": _CORPUSQA_REGIME,
    # dracula: Loong's regime — matching token profile (211k pooled / 81k max doc
    # vs Loong's median ~90k), per the STATUS TODO.
    "dracula": _ORIGINAL_REGIME,
}

# ``run_version`` LABELS which regime a run used — it lives in the manifest ``config`` and resumption
# keys on it PER BENCHMARK (so runs of different regimes never mix). loong stays "v2" (its existing
# runs used the original regime — they MUST stay valid) and corpusqa stays "v4" (its existing runs
# used the CorpusQA regime — likewise). longhealth moves off "v4" (its old runs wrongly used the
# CorpusQA regime — the 6000-word cap ≫ its ~1.5K-word docs collapsed every doc to ~1 page) onto "v2"
# (the original regime it should have used) → those wrong runs are SUPERSEDED and re-run.
_RUN_VERSIONS: dict[str, str] = {"loong": "v2", "corpusqa": "v4", "dracula": "v2"}


def _regime_for(benchmark_name: str) -> gist.Regime:
    """The pagination/gist regime for a benchmark (raises on an unsupported one)."""
    return _REGIMES[benchmark_name]


def run_version_for(benchmark_name: str) -> str:
    """The ``run_version`` label a benchmark's runs carry (resumption keys on it)."""
    return _RUN_VERSIONS[benchmark_name]

# The gist memory is persisted here inside the run folder (pooled pages + their gists),
# so a deep-dive can read exactly what was paginated/gisted for the task.
_GIST_FILE = "gist_memory.json"

# Live progress for a long-running task — readagent writes the gist memory + manifest only at the
# END, so a mid-flight task's folder would otherwise be empty. This small file is updated as the
# task moves through its stages so you can see where it is + how far (see ``_Progress``).
_PROGRESS_FILE = "progress.json"


class _Progress:
    """Writes/updates a tiny live ``progress.json`` in the run folder so a long readagent task's
    stage + counts are visible WHILE it runs (debug long batches; estimate remaining work).

    Duck-typed for ``gist.build_gist_memory``: ``stage(name, **extra)`` marks a phase, and
    ``page_done()`` / ``gist_done()`` tick the per-page loops. Each update records the running
    ``llm_calls`` (from the seam) + ``elapsed_s``, written atomically (tmp + replace) so a reader
    never sees a half-written file. Cheap: one write per stage + per page (~30s apart with thinking).

    Thread-safe: with ``--gist-workers > 1`` the per-page ticks fire from worker threads (and all
    share ONE ``progress.json.tmp`` path), so a lock serializes the counter bumps + the atomic
    write — without it concurrent writers would race on the temp file."""

    def __init__(self, path: Path, llm: Any, task_id: str, n_docs: int):
        self._path = Path(path)
        self._llm = llm
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "task_id": str(task_id), "n_docs": n_docs, "stage": "starting",
            "pages_paginated": 0, "n_pages": None, "pages_gisted": 0,
            "started_at": round(time.time(), 1),
        }
        self._flush()  # single-threaded at construction

    def stage(self, name: str, **extra: Any) -> None:
        with self._lock:
            self._state["stage"] = name
            self._state.update(extra)
            self._flush()

    def page_done(self) -> None:
        with self._lock:
            self._state["pages_paginated"] += 1
            self._flush()

    def gist_done(self) -> None:
        with self._lock:
            self._state["pages_gisted"] += 1
            self._flush()

    def _flush(self) -> None:
        # The caller holds ``self._lock`` (except ``__init__``, which runs single-threaded).
        now = time.time()
        self._state["llm_calls"] = len(self._llm.full_calls)
        self._state["updated_at"] = round(now, 1)
        self._state["elapsed_s"] = round(now - self._state["started_at"], 1)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))
        tmp.replace(self._path)  # atomic — readers see a complete file


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _build_task(benchmark, task_id: str) -> str:
    """The question posed to look-up + answer — the SAME components the other baselines
    pose (the documents are the gist memory, dropped from the question itself).

    - **loong**: ``instruction`` (the whole task when ``question`` is empty, e.g. paper
      Chain-of-Reasoning) else ``instruction\\n\\nquestion``.
    - **corpusqa**: ``question\\n\\n{output-requirements}`` — the output-requirements
      block (with the ``The answer is:`` contract the judge extracts) MUST reach the model.
    """
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 45-doc corpus becomes the gist memory.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"readagent has no task assembly for benchmark {name!r}")


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task ReadAgent pipeline, ALL inside ``run_dir``: build the gist memory
    (paginate the pooled documents → gist each page), look up pages for the task, then
    answer. Returns the harness record — ``raw_answer`` + the **TOTAL** ``usage`` (every
    stage's LLM calls) + ``calls_full`` + a light ``trace``. Nothing is reused.

    The pagination/gist regime is selected AUTOMATICALLY from the benchmark (``_regime_for``) —
    the original 600-word/no-clause regime for loong + longhealth, the ×10/640-token one for
    corpusqa (PROVENANCE D10). No global to flip."""
    from evals.baselines.readagent.llm import ReadAgentLLM

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = benchmark.get_documents(task_id)
    task = _build_task(benchmark, task_id)
    regime = _regime_for(_benchmark_name(benchmark))

    # ONE LLM seam for the whole task → its usage IS the total (pagination + gisting +
    # look-up + answer), and calls.json holds every call in order.
    llm = ReadAgentLLM(
        litellm_kwargs,
        seed=run_config.get("seed"),
        completion_params=run_config.get("completion_params"),
    )

    # Live progress.json so a long task's stage + counts are visible while it runs.
    progress = _Progress(run_dir / _PROGRESS_FILE, llm, task_id, len(documents))

    # Inner-task concurrency for pagination (across docs) + gisting (across pages). It rides in
    # litellm_kwargs (a runtime channel) — NOT run_config — on purpose: it's a pure throughput knob
    # (output is order-preserved + identical), so it must NOT split the config identity that
    # resumption + the analysis group by, and existing sequential runs stay resumable under it.
    gist_workers = int(litellm_kwargs.get("gist_workers", 1))

    # (1)+(2) Build the gist memory (pooled multi-doc pagination → gisting), persist it.
    progress.stage("paginating")
    memory = gist.build_gist_memory(documents, llm, regime=regime, progress=progress,
                                    gist_workers=gist_workers)
    (run_dir / _GIST_FILE).write_text(json.dumps(memory, ensure_ascii=False, indent=2))
    gists, pages = memory["gists"], memory["pages"]

    # (3) ReadAgent-P look-up: one call names the pages to re-read; expand, answer.
    progress.stage("lookup", n_pages=len(pages))
    page_ids, lookup_reply = gist.parallel_lookup(gists, task, llm)
    progress.stage("answering", looked_up_pages=page_ids)
    expanded_article = gist.assemble_expanded(pages, gists, page_ids)
    raw_answer = gist.answer(expanded_article, task, llm)
    progress.stage("answered")

    return {
        "raw_answer": raw_answer,
        "usage": llm.usage,                # TOTAL: pagination + gisting + look-up + answer
        "calls_full": llm.full_calls,      # → calls.json (every internal call, in order)
        "trace": {
            "task": task,
            "n_docs": memory["n_docs"],
            "n_pages": len(pages),
            "regime": {                     # the auto-selected pagination/gist sizes (provenance)
                "word_limit": regime.word_limit,
                "start_threshold": regime.start_threshold,
                "short_page_words": regime.short_page_words,
                "gist_token_hint": regime.gist_token_hint,
            },
            "gist_workers": gist_workers,   # inner-task concurrency used (provenance; NOT config identity)
            "looked_up_pages": page_ids,
            "n_looked_up": len(page_ids),
            "lookup_reply": lookup_reply,
        },
    }
