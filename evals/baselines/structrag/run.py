"""StructRAG baseline pipeline — connects the vendored upstream modules to the
harness's per-task contract.

This file is the harness-side CONNECTOR (everything here is a documented
deviation from / addition to upstream; the vendored pipeline lives under
``upstream/``). It fills the roles upstream's ``main.py`` played, mapped onto the
harness:

  upstream main.py                         this module
  ---------------------------------------  -----------------------------------
  load loong_process.jsonl, shard, loop    the runner fans out one run_one/task
  data['docs'] (marker string)             _to_structrag_docs(get_documents(...))
  query = prompt_template.format(...)       _build_query(...)
  per-item route→structurize→utilize body   run_one(...)  (main.py:75-111)
  intermediate_results/<type>_kb/ tree      an EPHEMERAL per-task scratch dir
  utils/qwenapi.QwenAPI                      structrag/llm.StructRAGLLM

DEVIATIONS (vs upstream main.py):
  • No ``setup()`` and no cross-task state: StructRAG restructures documents per
    task, so each ``run_one`` is independent (no shared/persistent index).
  • The structurizer/utilizer on-disk KB round-trip is KEPT (faithful to how
    upstream runs) but lands in an ephemeral per-task ``TemporaryDirectory``
    instead of a persistent ``intermediate_results/`` tree (the runner owns
    logging + resumption).
  • ``data_id`` (only ever a KB-scratch filename token upstream) is a constant —
    the per-task scratch dir already isolates tasks, and it dodges
    filesystem-unsafe task ids.

SUPPORTED_BENCHMARKS = {loong, corpusqa, dracula} — all three benchmarks in this
repo. StructRAG restructures EVERY document per task, so a benchmark only fits
when a task's document set is small enough to rebuild per question: loong and
corpusqa are per-instance multi-doc bundles, and dracula's shared corpus is 46
documents. (A large shared corpus would not fit — restructuring it per question
is infeasible and has no upstream analogue.) The dispatch here is name-based,
like arag, so adding a benchmark later is a branch in ``_build_query`` (+ a title
convention in ``_split_title_content``).

Full deviation ledger: evals/baselines/structrag/PROVENANCE.md
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from evals.baselines.structrag.llm import StructRAGLLM
from evals.baselines.structrag.upstream.router import Router
from evals.baselines.structrag.upstream.structurizer import Structurizer
from evals.baselines.structrag.upstream.utilizer import Utilizer

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Upstream title-marker tokens — StructRAG's loong_process.jsonl doc-string format.
# The vendored ``structurizer.split_content_and_tile`` parses exactly these, so the
# Loong connector re-wraps our loader's docs into this shape and the vendored
# module runs UNMODIFIED. (Glyphs are upstream's 标题起始符 / 标题终止符 / doc终止符.)
_TITLE_START = "<标题起始符>"
_TITLE_END = "<标题终止符>"
_DOC_END = "<doc终止符>"

# The 5 per-type KB scratch dirs Structurizer/Utilizer expect (upstream main.py:47-56),
# in the order of their ``__init__`` signature: chunk, graph, table, algorithm, catalogue.
_KB_SUBDIRS = ("chunk_kb", "graph_kb", "table_kb", "algorithm_kb", "catalogue_kb")

# Upstream keyed scratch KB files by ``data['id']``; the per-task scratch dir already
# isolates tasks, so a constant token is enough (and avoids unsafe task ids in paths).
_DATA_ID = "task"


def _benchmark_name(benchmark) -> str:
    """``evals.benchmarks.<name>`` → ``<name>`` (the dispatch key, like arag)."""
    return benchmark.__name__.rsplit(".", 1)[-1]


def _split_title_content(name: str, documents: list[str]) -> list[tuple[str, str]]:
    """``(title, content)`` per document, per benchmark.

    Upstream consumed Loong docs that already carried a title — its ``get_content``
    emits ``《title》\\ncontent`` (financial) / ``title\\ncontent`` (paper), exactly
    what our loong loader reproduces — so for loong we split on the first newline
    (the title, incl. the ``《》`` for financial, is preserved verbatim); dracula's
    docs carry a title line of their own, so they split the same way. corpusqa has
    NO title convention (and no upstream StructRAG reference — it is a per-instance
    multi-doc bundle), so we synthesize ``Document N``.
    """
    if name in ("loong", "dracula"):
        # dracula docs open with their in-world source header ("DR. SEWARD'S DIARY",
        # "_Letter, Mina Harker to Lucy Westenra._") — a real title line, like loong's.
        out: list[tuple[str, str]] = []
        for doc in documents:
            head, _, body = doc.partition("\n")
            out.append((head.strip(), body.strip()))
        return out
    return [(f"Document {i + 1}", doc.strip()) for i, doc in enumerate(documents)]


def _to_structrag_docs(name: str, documents: list[str]) -> str:
    """Re-wrap our ``get_documents`` list into StructRAG's title-marker doc STRING.

    Reproduces upstream's loong_process.jsonl doc format
    (``<标题起始符>{title}<标题终止符>\\n{content}<doc终止符>\\n\\n`` per doc) so the
    vendored ``structurizer.split_content_and_tile`` runs byte-for-byte. See
    PROVENANCE.md ("Loong input connector").
    """
    parts = []
    for title, content in _split_title_content(name, documents):
        parts.append(f"{_TITLE_START}{title}{_TITLE_END}\n{content}{_DOC_END}\n\n")
    return "".join(parts)


def _build_query(benchmark, task_id) -> str:
    """The StructRAG ``query`` (fed to route / decompose / merge), per benchmark.

    - **loong**: FAITHFUL to upstream main.py — the instance's ``prompt_template``
      with instruction/question filled and ``docs`` as the ``"......"`` placeholder
      (the documents become the structurized knowledge, not part of the query).
    - **corpusqa / dracula**: no upstream StructRAG reference; mirror the
      harness's other baselines (e.g. ``arag/run.py``'s ``_build_query``) so the
      question posed is identical across baselines.
    """
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        template = benchmark.get_prompt_template(task_id)
        return template.format(instruction=instruction, question=question, docs="......")
    if name == "corpusqa":
        # No upstream StructRAG reference (corpusqa is our addition); mirror the
        # other baselines — question THEN the output-requirements block. Each doc
        # is restructured, so the docs become the knowledge, not the query.
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 46 docs become the structurized knowledge.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"structrag has no query assembly for benchmark {name!r}")


def _make_kb_dirs(scratch: Path) -> list[str]:
    """Create the 5 per-type KB scratch dirs under ``scratch``; return their paths
    in the upstream ``__init__`` order (chunk, graph, table, algorithm, catalogue)."""
    paths = []
    for sub in _KB_SUBDIRS:
        p = scratch / sub
        p.mkdir(parents=True, exist_ok=True)
        paths.append(str(p))
    return paths


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    base_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task StructRAG pipeline (route → structurize → utilize).

    A faithful re-host of upstream main.py's per-item body (main.py:75-111): the
    only changes are the LLM seam (``StructRAGLLM``), the input (our loader via the
    connectors above), and an ephemeral per-task scratch dir for the KB round-trip.
    Returns the harness record: ``{raw_answer, usage, trace}`` — purely the model's
    output (nothing in this repo grades it; the separate scoring repo reads these
    logs). ``base_dir`` is unused (no index store), kept for the ``run_one``
    contract.
    """
    name = _benchmark_name(benchmark)
    llm = StructRAGLLM(
        litellm_kwargs,
        seed=run_config.get("seed"),
        completion_params=run_config.get("completion_params"),
    )
    documents = benchmark.get_documents(task_id)
    docs = _to_structrag_docs(name, documents)
    query = _build_query(benchmark, task_id)

    with tempfile.TemporaryDirectory(prefix="structrag-kb-") as scratch:
        kb_paths = _make_kb_dirs(Path(scratch))
        router = Router(llm)
        structurizer = Structurizer(llm, *kb_paths)
        utilizer = Utilizer(llm, *kb_paths)

        # main.py:80-81 — titles → core_content for the router.
        _docs, titles = structurizer.split_content_and_tile(docs)
        core_content = "The titles of the docs are: " + "\n".join(list(set(titles)))

        chosen = router.do_route(query, core_content, _DATA_ID)                       # main.py:84
        instruction, kb_info = structurizer.construct(query, chosen, docs, _DATA_ID)  # main.py:89
        subqueries = utilizer.do_decompose(query, kb_info, _DATA_ID)                  # main.py:94
        subknowledges = utilizer.do_extract(query, subqueries, chosen, _DATA_ID)      # main.py:97
        answer, _, _ = utilizer.do_merge(query, subqueries, subknowledges, chosen, _DATA_ID)  # main.py:100

    return {
        "raw_answer": answer,
        "usage": llm.usage,
        # The full request/response of EVERY internal call → written to `calls.json` by the
        # runner. The per-doc structured knowledge + the extracted evidence live there (in
        # the structurize/extract call responses), so the manifest's `trace` stays a light
        # summary (the route decision, the query, the decomposition).
        "calls_full": llm.full_calls,
        "trace": {
            "chosen": chosen,
            "query": query,
            "kb_info": kb_info,
            "subqueries": subqueries,
        },
    }
