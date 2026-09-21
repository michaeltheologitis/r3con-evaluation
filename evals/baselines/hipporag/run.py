"""``hipporag`` baseline: HippoRAG 2 (OpenIE knowledge graph + Personalized PageRank).

Per task: chunk the document bundle into passages (our own by-token chunker — HippoRAG
ships none), build a HippoRAG 2 index over them (per-passage OpenIE → a phrase+passage
knowledge graph with synonym/context edges), then answer the composed query via
HippoRAG's own retrieve→read front door (query→triple linking + recognition-memory
filter + PPR over the graph → top-``qa_top_k`` passages → a reader LLM). ``raw_answer``
is the reader's output; nothing here grades it — scoring happens outside this repo.

Supports **loong / corpusqa / dracula** (multi-doc bundles; loong/corpusqa per instance,
dracula's 46-doc corpus shared by every question). The vendored upstream
(``upstream/hipporag`` @ ``ad30fc3``, MIT) runs unmodified; all LLM
calls (OpenIE + filter + reader) route through the litellm seam (``llm.HippoRAGLLM``)
and embeddings through the OpenAI seam (``embedding.HippoRAGOpenAIEmbedder``), injected
by monkeypatching the two upstream factories. Deviation ledger: PROVENANCE.md.

Cost note: indexing is a per-passage OpenIE pass (2 LLM calls/passage), so this is an
expensive baseline — run a stratified subset, not full sets. Expected to be a retrieval
FOIL on these leave-no-document-behind benchmarks (top-k passages ≠ the whole corpus).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.baselines.hipporag.chunker import CHUNK_SIZES, chunk_documents
from evals.baselines.hipporag.embedding import HippoRAGOpenAIEmbedder
from evals.baselines.hipporag.llm import HippoRAGLLM

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Bump when the connector changes in a way that alters a run's output (chunker sizes,
# query assembly, the seams, or the vendored pipeline). Folded into the run config so a
# bump re-runs the inferences instead of counting stale ones. v1: initial wiring.
_RUN_VERSION = "v1"

# Embeddings are ALWAYS OpenAI text-embedding-3-small (the harness rule); recorded in
# the run config so whatever prices these logs later can attribute the index-build
# embedding cost.
EMBEDDING_MODEL = "openai/text-embedding-3-small"


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _merge_usage(*records) -> dict[str, Any]:
    """Combine several ``{total, calls}`` usage records into one — per-model totals
    summed (so the completion model AND the embedding model each keep their own key),
    call lists concatenated. ``None`` records are skipped."""
    from evals.llm.usage import _merge_numeric
    total: dict[str, Any] = {}
    calls: list[Any] = []
    for rec in records:
        if not rec:
            continue
        for model, bucket in rec.get("total", {}).items():
            _merge_numeric(total.setdefault(model, {}), bucket)
        calls.extend(rec.get("calls", []))
    return {"total": total, "calls": calls}


def _build_query(benchmark, task_id) -> str:
    """The single-string query posed to HippoRAG — the SAME composed task the other
    baselines pose (structrag is the exception: for loong it renders the instance's
    ``prompt_template``). The docs are the index here, dropped from the query."""
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        # Some Loong instances (paper Chain-of-Reasoning) have an empty question —
        # the whole task is the instruction, so that string alone is the query.
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        # question THEN the output-requirements block (answer-format + conflict rules).
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 46-doc corpus is the index (get_task → (question, docs)).
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"hipporag has no query assembly for benchmark {name!r}")


def _build_config(index_dir: Path, model: str, run_config: dict[str, Any]):
    """A vendored ``BaseConfig`` for a single per-task index build.

    online OpenIE (per-passage, via the litellm seam), OpenAI embeddings, and a fresh
    build every time (no reuse within the flat run folder). ``dataset=None`` makes the
    QA reader fall back to HippoRAG's MuSiQue prompt template (its front door)."""
    from evals.baselines.hipporag.upstream.hipporag.utils.config_utils import BaseConfig
    return BaseConfig(
        save_dir=str(index_dir),
        llm_name=model,
        embedding_model_name="text-embedding-3-small",
        openie_mode="online",
        save_openie=False,
        force_index_from_scratch=True,
        force_openie_from_scratch=True,
        dataset=None,
        seed=run_config.get("seed", 0),
    )


def _build_hipporag(index_dir: Path, model: str, run_config: dict[str, Any], llm: HippoRAGLLM):
    """Construct a vendored ``HippoRAG`` with our seams injected via the two upstream
    factories. Injecting through the factories (rather than post-construction attribute
    surgery) means OpenIE, the rerank filter, and the embedding stores all pick up the
    seams during ``__init__`` — the vendored code runs unmodified."""
    import evals.baselines.hipporag.upstream.hipporag.HippoRAG as hr_module

    # The names live in HippoRAG.py's namespace (it did `from .llm import
    # _get_llm_class`), so patch THEM, not the source modules.
    hr_module._get_llm_class = lambda _config: llm
    hr_module._get_embedding_model_class = lambda embedding_model_name=None: HippoRAGOpenAIEmbedder

    config = _build_config(index_dir, model, run_config)
    return hr_module.HippoRAG(global_config=config)


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task work: chunk → index → retrieve+read → raw_answer.

    Writes the HippoRAG index under ``run_dir/index/``. Returns the flat-layout record:
    ``raw_answer`` + the TOTAL ``usage`` (OpenIE + filter + reader, one number) +
    ``calls_full`` (every internal call) + a ``trace``. Nothing here grades: the
    benchmark's judge is run later, by whatever consumes these logs.
    """
    name = _benchmark_name(benchmark)
    if name not in CHUNK_SIZES:
        raise ValueError(f"hipporag has no chunk size for benchmark {name!r}")

    documents = benchmark.get_documents(task_id)
    passages = chunk_documents(documents, CHUNK_SIZES[name])
    query = _build_query(benchmark, task_id)

    llm = HippoRAGLLM(
        litellm_kwargs,
        seed=run_config.get("seed"),
        completion_params=run_config.get("completion_params"),
    )
    hipporag = _build_hipporag(Path(run_dir) / "index", litellm_kwargs["model"], run_config, llm)

    hipporag.index(docs=passages)
    _solutions, responses, _metadata = hipporag.rag_qa(queries=[query])
    raw_answer = responses[0] if responses else ""

    # TOTAL cost = the LLM calls (OpenIE + filter + reader) AND the index-build
    # embeddings (passage/entity/fact) — folded into one {total, calls} record so the
    # embedding model appears as its own model key and no cost is dropped.
    embedder = getattr(hipporag, "embedding_model", None)
    embed_usage = embedder.usage if embedder is not None and hasattr(embedder, "usage") else None

    return {
        "raw_answer": raw_answer,
        "usage": _merge_usage(llm.usage, embed_usage),
        "calls_full": llm.full_calls,
        "trace": {
            "query": query,
            "n_documents": len(documents),
            "n_passages": len(passages),
            "chunk_size": CHUNK_SIZES[name],
        },
    }
