"""A-RAG baseline pipeline — connects the vendored agent + tools to the harness's
per-task contract.

This file is the harness-side CONNECTOR (everything here is a documented deviation
from / addition to upstream; the vendored agent/tools live under ``upstream/``). It
fills the roles upstream's ``scripts/{batch_runner,build_index}.py`` played, mapped
onto the harness:

  upstream                                  this module / connector
  ----------------------------------------  -----------------------------------------
  batch_runner.py (loop, agent assembly)    the runner fans out one run_one/task
  scripts/build_index.py (chunks→index)     _ensure_index → chunker + embedding seam
  pre-chunked chunks.json (their corpus)    chunker.build_chunks(get_documents(...))
  core/llm.LLMClient (requests)             llm.AragLLM (litellm)
  Qwen3-Embedding-0.6B (sentence-transf.)   embedding.OpenAIEmbedder (project rule)
  prompts/default.txt + max_loops/budget    faithful config (default.txt; max_loops=15;
                                            DYNAMIC per-model budget — model_budget)

SUPPORTED_BENCHMARKS = {longbenchv2, loong, corpusqa, longhealth, dracula} — all per-task doc-sets (no
shared corpus), so there is no ``setup()``; each task's documents are chunked + indexed
on demand into a content-addressed ``_indices/{index_hash}/`` (a content-addressed index store). The
index is LLM-INDEPENDENT (chunking + embedding are deterministic, no completion calls),
so ``index_hash`` folds only the doc-set fingerprint + embedding model + chunker /
index versions — NOT the completion model, seed, or sampling (those vary the inference,
which reuses the same index).

Full deviation ledger: evals/baselines/arag/PROVENANCE.md
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Bump when the index BUILD changes in a way that alters index content (chunking,
# embedding, sentence split). Folded into both the index hash AND (via the runner)
# the inference identity, so a bump rebuilds indices AND re-runs inferences.
_INDEX_VERSION = "v1"

# A-RAG's recommended agent config (configs/example.yaml). max_token_budget is NOT
# fixed here — it's resolved per-model in model_budget (the sanctioned D2 deviation).
_MAX_LOOPS = 15

_CHUNKS_FILE = "chunks.json"        # A-RAG corpus: ["0:text", "1:text", …]
_INDEX_SUBDIR = "index"             # holds sentence_index.pkl
_INDEX_USAGE_FILE = "index_usage.json"   # build (embedding) token cost — {total, calls}
_INDEX_META_FILE = "index_meta.json"     # human-readable description of the index


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    base_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task work: resolve/build the index, run the A-RAG agent, return the record.

    Returns the harness record — purely the model's output + index_ref + usage +
    the full call trace (``calls_full`` → ``calls.json``) + a light ``trace``. Grading
    (the Loong 1–100 judge / longbenchv2 parse-then-match) is deferred to score time.
    """
    from evals.baselines.arag.chunker import CHUNKER_VERSION
    from evals.baselines.arag.embedding import OpenAIEmbedder, OpenAISemanticSearchTool
    from evals.baselines.arag.llm import AragLLM
    from evals.baselines.arag.model_budget import (
        AragAgent, context_budget, resolve_context_window, resolve_token_counter,
    )
    from evals.baselines.arag.upstream.tools.keyword_search import KeywordSearchTool
    from evals.baselines.arag.upstream.tools.read_chunk import ReadChunkTool
    from evals.baselines.arag.upstream.tools.registry import ToolRegistry

    documents = benchmark.get_documents(task_id)
    index_hash = _ensure_index(base_dir, documents, run_config, litellm_kwargs)
    index_dir = _index_dir(base_dir, index_hash)

    chunks_file = str(index_dir / _CHUNKS_FILE)
    # Embeddings route to OpenAI by the model's own provider (env key) — independent of
    # the completion's litellm_kwargs (which may point at a local vLLM endpoint).
    query_embedder = OpenAIEmbedder(run_config["embedding_model"])

    tools = ToolRegistry()
    tools.register(KeywordSearchTool(chunks_file=chunks_file))
    tools.register(OpenAISemanticSearchTool(chunks_file, str(index_dir / _INDEX_SUBDIR), query_embedder))
    tools.register(ReadChunkTool(chunks_file=chunks_file))

    llm = AragLLM(
        litellm_kwargs,
        seed=run_config.get("seed"),
        completion_params=run_config.get("completion_params"),
    )
    model = litellm_kwargs["model"]
    agent = AragAgent(
        llm, tools, system_prompt=_system_prompt(),
        token_counter=resolve_token_counter(model),
        max_token_budget=context_budget(resolve_context_window(model)),
        max_loops=_MAX_LOOPS,
    )

    query = _build_query(benchmark, task_id)
    result = agent.run(query)

    # Inference usage = the agent's completion calls (AragLLM) + this task's query
    # embedding calls (query_embedder). The index BUILD embedding cost is separate —
    # it lives in the index store (index_usage.json) and travels on reuse.
    usage = _merge_usage(llm.usage, query_embedder.usage)
    return {
        "raw_answer": result["answer"],
        "index_ref": index_hash,
        "usage": usage,
        # Full request/response of every agent LLM call (incl. the message history,
        # which carries the retrieved tool results) → written to calls.json.
        "calls_full": llm.full_calls,
        "trace": {
            "query": query,
            "loops": result.get("loops"),
            "total_retrieved_tokens": result.get("total_retrieved_tokens"),
            "chunks_read_ids": result.get("chunks_read_ids"),
            "token_budget_exceeded": result.get("token_budget_exceeded", False),
            "max_loops_exceeded": result.get("max_loops_exceeded", False),
            # Light tool-call sequence (tool_result text dropped — it's in calls.json).
            "tool_calls": [
                {"loop": t.get("loop"), "tool_name": t.get("tool_name"),
                 "arguments": t.get("arguments"), "retrieved_tokens": t.get("retrieved_tokens"),
                 "chunks_found": t.get("chunks_found")}
                for t in result.get("trajectory", [])
            ],
        },
    }


# ============================================================
# Query assembly (mirrors graphrag/structrag _build_query exactly)
# ============================================================


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _build_query(benchmark, task_id) -> str:
    """The single-string question handed to ``agent.run`` — the SAME components the
    other baselines pose (the documents are the index, dropped from the query)."""
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        # Some Loong instances have an empty question (the task is the instruction) —
        # fall back to the instruction alone (no dangling "\n\n"), matching the other baselines.
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        # question THEN the output-requirements block (answer-format + conflict rules),
        # matching the other baselines. The documents are chunked into the index.
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 45-doc corpus is chunked into the index.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"arag has no query assembly for benchmark {name!r}")


def _system_prompt() -> str:
    """A-RAG's vendored ReAct system prompt (``upstream/agent/prompts/default.txt``) —
    the one ``batch_runner`` loads upstream. Part of the method, used verbatim."""
    return (Path(__file__).parent / "upstream" / "agent" / "prompts" / "default.txt").read_text()


# ============================================================
# Content-addressed per-task index store (mirrors graphrag)
# ============================================================


def _content_fingerprint(documents: list[str]) -> str:
    """Order-independent content fingerprint of the document SET — the index identity."""
    import hashlib
    per_doc = sorted(hashlib.sha256(d.encode("utf-8")).hexdigest() for d in documents)
    return hashlib.sha256("\n".join(per_doc).encode("utf-8")).hexdigest()[:16]


def _index_hash(doc_key: str, run_config: dict[str, Any]) -> str:
    """Content hash of what determines the index: the doc-set fingerprint, the embedding
    model, and the chunker / index versions. NOT the completion model / seed / sampling —
    the index (chunks + sentence embeddings) is independent of the agent's LLM."""
    from evals.baselines.arag.chunker import CHUNKER_VERSION
    from evals.baselines._common import compute_config_hash
    return compute_config_hash({
        "doc_key": doc_key,
        "embedding_model": run_config["embedding_model"],
        "chunker_version": run_config.get("chunker_version", CHUNKER_VERSION),
        "index_version": run_config.get("index_version", _INDEX_VERSION),
    })


def _index_dir(base_dir: Path, index_hash: str) -> Path:
    return base_dir / "_indices" / index_hash


def _index_is_built(index_dir: Path) -> bool:
    """Built TO COMPLETION = the two receipt files (written LAST) both exist — the same
    completion sentinel graphrag uses, so a build that died mid-way is never reused."""
    return all((index_dir / name).exists() for name in (_INDEX_USAGE_FILE, _INDEX_META_FILE))


def _ensure_index(base_dir, documents, run_config, litellm_kwargs) -> str:
    """Build the per-task index if absent (chunks.json + sentence_index.pkl), return its
    ``index_hash``. Concurrent children sharing a doc-set serialize on a per-index flock
    (exactly one builds; the rest reuse) — a content-addressed store's mechanics."""
    from evals.baselines.arag.chunker import build_chunks
    from evals.baselines.arag.embedding import OpenAIEmbedder, build_sentence_index

    doc_key = _content_fingerprint(documents)
    index_hash = _index_hash(doc_key, run_config)
    index_dir = _index_dir(base_dir, index_hash)
    if _index_is_built(index_dir):
        return index_hash

    index_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(index_dir.parent / f"{index_hash}.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if _index_is_built(index_dir):
            return index_hash  # another child built it while we waited
        if index_dir.exists():
            shutil.rmtree(index_dir)  # receipts absent → interrupted-build leftovers
        index_dir.mkdir(parents=True)

        rows = build_chunks(documents)
        (index_dir / _CHUNKS_FILE).write_text(json.dumps(rows, ensure_ascii=False))
        build_embedder = OpenAIEmbedder(run_config["embedding_model"])  # OpenAI via env key
        build_sentence_index(rows, build_embedder, index_dir / _INDEX_SUBDIR)

        _write_json_atomic(index_dir / _INDEX_USAGE_FILE, build_embedder.usage)
        from evals.baselines.arag.chunker import CHUNKER_VERSION
        _write_json_atomic(index_dir / _INDEX_META_FILE, {
            "doc_key": doc_key,
            "index_hash": index_hash,
            "embedding_model": run_config["embedding_model"],
            "chunker_version": CHUNKER_VERSION,
            "index_version": _INDEX_VERSION,
            "n_docs": len(documents),
            "n_chunks": len(rows),
        })
    return index_hash


def _write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON via a temp file + atomic rename (so the lock-free fast path never
    reads a torn receipt)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _merge_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """Merge several ``{total, calls}`` envelopes into one (per-model numeric sum +
    concatenated call list)."""
    from evals.llm.usage import _merge_numeric
    total: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    for usage in usages:
        for model, bucket in (usage.get("total") or {}).items():
            _merge_numeric(total.setdefault(model, {}), bucket)
        calls.extend(usage.get("calls") or [])
    return {"total": total, "calls": calls}
