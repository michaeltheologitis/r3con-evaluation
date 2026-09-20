"""Tests for the A-RAG baseline connector.

A-RAG (github.com/Ayanami0730/arag @ a44de6b) is vendored byte-for-byte under
``evals/baselines/arag/upstream/`` (only the intra-package import prefix is
rewritten — D1). The harness-side connectors are tested here:

- ``chunker``     — the ~1000-token sentence-aligned corpus-prep A-RAG doesn't ship.
- ``model_budget``— the dynamic per-model token counter + context window + AragAgent
                    (the sanctioned deviation replacing upstream's hardcoded gpt-4o / 128k).
- ``llm``         — AragLLM, the tool-calling LLM seam over litellm (fakes litellm).
- ``embedding``   — the OpenAI sentence-embedding index build + OpenAISemanticSearchTool.
- ``run``         — run_one's call graph + per-task content-addressed index store (fakes).

The ``arag`` marker gates a real end-to-end run against gpt-5.4-nano (deselected by
default; see pyproject addopts).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.baselines.arag import chunker


# ============================================================
# chunker — ~1000-token, sentence-aligned, sequential ids
# ============================================================


import tiktoken  # noqa: E402


# ============================================================
# _build_query — dracula (bare question; docs are the index)
# ============================================================


def test_build_query_dracula_bare_question() -> None:
    from types import SimpleNamespace

    from evals.baselines.arag import run as arag_run
    b = SimpleNamespace(__name__="evals.benchmarks.dracula",
                        get_task=lambda t: ("HOW MANY?", ["d1", "d2"]))
    assert arag_run._build_query(b, "t") == "HOW MANY?"
    assert "dracula" in arag_run.SUPPORTED_BENCHMARKS

_GPT4O = tiktoken.encoding_for_model("gpt-4o")


def _ntok(text: str) -> int:
    return len(_GPT4O.encode(text))


def _sentences(n: int, words: int = 20) -> str:
    """``n`` distinct sentences of ``words`` words each (period-terminated)."""
    return " ".join(" ".join(f"word{j}" for j in range(words)) + f" end{i}." for i in range(n))


def test_chunk_text_short_doc_is_one_chunk() -> None:
    assert chunker.chunk_text("One short sentence. Another one.", target_tokens=1000) == \
        ["One short sentence. Another one."]


def test_chunk_text_respects_token_capacity() -> None:
    text = _sentences(400)  # well over one chunk
    chunks = chunker.chunk_text(text, target_tokens=1000)
    assert len(chunks) > 1
    # semantic-text-splitter fills each chunk UP TO the capacity (counted with gpt-4o);
    # none exceeds it, and the bulk land near it ("approximately 1,000").
    sizes = [_ntok(c) for c in chunks]
    assert max(sizes) <= 1000
    assert sum(s >= 800 for s in sizes) >= len(sizes) - 1  # only the tail chunk may be small


def test_chunk_text_preserves_all_content() -> None:
    # The splitter PARTITIONS the text (no content lost or duplicated): every sentence
    # marker appears in exactly one chunk. (Sentence-boundary alignment is the library's
    # job — we don't over-pin its exact break heuristic.)
    chunks = chunker.chunk_text(_sentences(300), target_tokens=1000)
    for i in range(300):
        assert sum(f"end{i}." in c for c in chunks) == 1


def test_build_chunks_sequential_ids_id_text_format() -> None:
    docs = [_sentences(120), _sentences(120)]  # each doc → ≥2 chunks
    rows = chunker.build_chunks(docs, target_tokens=1000)
    ids = [r.split(":", 1)[0] for r in rows]
    # Contiguous integer ids 0..N-1, in document order (so read_chunk's ±1 stays meaningful).
    assert ids == [str(i) for i in range(len(rows))]
    assert all(":" in r and r.split(":", 1)[1].strip() for r in rows)


def test_build_chunks_does_not_span_documents() -> None:
    # A chunk is built per-document, so doc A's tail and doc B's head never merge
    # into one chunk (each row's text comes from exactly one source doc).
    rows = chunker.build_chunks(["Alpha sentence one. Alpha two.", "Beta sentence one."],
                                target_tokens=1000)
    texts = [r.split(":", 1)[1] for r in rows]
    assert any("Alpha" in t for t in texts) and any("Beta" in t for t in texts)
    assert not any("Alpha" in t and "Beta" in t for t in texts)


# ============================================================
# model_budget — DYNAMIC per-model token counter + context window + AragAgent
# (the sanctioned deviation replacing upstream's hardcoded gpt-4o tokenizer / 128k)
# ============================================================

from evals.baselines.arag import model_budget  # noqa: E402
from evals.baselines.arag.model_budget import AragAgent  # noqa: E402
from evals.baselines.arag.upstream.tools.registry import ToolRegistry  # noqa: E402


def test_token_counter_openai_uses_tiktoken() -> None:
    counter = model_budget.resolve_token_counter("openai/gpt-4o")
    import tiktoken
    expected = len(tiktoken.encoding_for_model("gpt-4o").encode("hello world, this is a test"))
    assert counter("hello world, this is a test") == expected


def test_token_counter_unknown_openai_model_falls_back_to_o200k() -> None:
    # gpt-5.x isn't in tiktoken's model map; the counter must still work (o200k_base).
    counter = model_budget.resolve_token_counter("openai/gpt-5.4-nano")
    assert counter("some text") > 0


def test_token_counter_open_source_without_hf_uses_char_estimate(monkeypatch) -> None:
    # No HF tokenizer available (transformers absent / id unresolvable) → char/4 estimate,
    # NOT a crash and NOT silently gpt-4o.
    monkeypatch.setattr(model_budget, "_hf_tokenizer", lambda model: None)
    counter = model_budget.resolve_token_counter("hosted_vllm/Org/Some-Model")
    text = "x" * 400
    assert counter(text) == pytest.approx(100, abs=1)  # ~chars/4


def test_context_window_known_model_from_litellm() -> None:
    # gpt-4o is in litellm's registry → a real positive window (not the fallback).
    assert model_budget.resolve_context_window("openai/gpt-4o") > 10000


def test_context_window_falls_back_when_unknown(monkeypatch) -> None:
    monkeypatch.setattr(model_budget, "_litellm_model_info", lambda model: None)
    monkeypatch.setattr(model_budget, "_hf_window", lambda model: None)
    assert model_budget.resolve_context_window("hosted_vllm/Org/Mystery") == model_budget._FALLBACK_WINDOW


def test_context_budget_leaves_generation_room() -> None:
    win = 262144
    budget = model_budget.context_budget(win)
    assert 0 < budget < win  # room reserved for the answer


class _FakeLLM:
    """Duck-typed A-RAG client: answers immediately (no tool call)."""
    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, temperature=None, max_tokens=None):
        self.calls += 1
        return {"message": {"role": "assistant", "content": "the answer"}, "cost": 0.0}


def test_arag_agent_uses_injected_counter_and_budget() -> None:
    agent = AragAgent(_FakeLLM(), ToolRegistry(), system_prompt="sys",
                      token_counter=lambda s: len(s.split()), max_token_budget=500, max_loops=5)
    # The overridden counter is word-count (the injected one), NOT tiktoken gpt-4o.
    assert agent._calculate_message_tokens([{"content": "one two three"}]) == 1 + 3  # "sys"=1 word
    assert agent.max_token_budget == 500


def test_arag_agent_force_answers_when_budget_exceeded() -> None:
    # A tiny budget → the agent force-answers on the first loop instead of looping.
    llm = _FakeLLM()
    agent = AragAgent(llm, ToolRegistry(), system_prompt="sys",
                      token_counter=lambda s: 10_000, max_token_budget=5, max_loops=5)
    out = agent.run("question")
    assert out.get("token_budget_exceeded") is True
    assert out["answer"] == "the answer"


# ============================================================
# llm — AragLLM: the tool-calling seam over litellm (replaces upstream LLMClient)
# ============================================================

from types import SimpleNamespace  # noqa: E402

from evals.baselines.arag import llm as arag_llm  # noqa: E402


class _FakeMessage:
    """Mimics a litellm ``Message``: has ``.content`` + ``model_dump()``."""
    def __init__(self, content, tool_calls=None):
        self.content = content
        self._tool_calls = tool_calls

    def model_dump(self):
        return {"role": "assistant", "content": self.content,
                "function_call": None, "tool_calls": self._tool_calls}


def _fake_response(content, tool_calls=None, *, model="gpt-5.4-nano", ptoks=11, ctoks=7):
    usage = SimpleNamespace(
        prompt_tokens=ptoks, completion_tokens=ctoks, total_tokens=ptoks + ctoks,
        model_dump=lambda: {"prompt_tokens": ptoks, "completion_tokens": ctoks,
                            "total_tokens": ptoks + ctoks},
    )
    msg = _FakeMessage(content, tool_calls)
    resp = SimpleNamespace(model=model, usage=usage,
                           choices=[SimpleNamespace(message=msg)],
                           _hidden_params={"response_cost": 0.0009})
    return resp


def _patch_litellm(monkeypatch, responses):
    """Patch litellm.completion to pop from ``responses`` and record the request kwargs."""
    captured = []
    it = iter(responses)

    def fake_completion(**kwargs):
        captured.append(kwargs)
        return next(it)

    monkeypatch.setattr(arag_llm.litellm, "completion", fake_completion)
    return captured


def test_aragllm_chat_builds_tool_request_and_returns_message(monkeypatch) -> None:
    tcs = [{"id": "c1", "type": "function", "function": {"name": "keyword_search", "arguments": "{}"}}]
    captured = _patch_litellm(monkeypatch, [_fake_response(None, tcs)])
    llm = arag_llm.AragLLM(
        {"model": "openai/gpt-5.4-nano", "api_base": "http://x/v1", "api_key": "k"},
        seed=42, completion_params={"temperature": 0.0},
    )
    out = llm.chat(messages=[{"role": "user", "content": "q"}], tools=[{"type": "function"}])

    req = captured[0]
    assert req["model"] == "openai/gpt-5.4-nano"
    assert req["messages"] == [{"role": "user", "content": "q"}]
    assert req["tools"] == [{"type": "function"}] and req["tool_choice"] == "auto"
    assert req["seed"] == 42 and req["api_base"] == "http://x/v1" and req["api_key"] == "k"
    assert req["temperature"] == 0.0 and req["num_retries"] == arag_llm._NUM_RETRIES
    # The returned message is a clean OpenAI dict the agent can append + re-send.
    assert out["message"]["tool_calls"] == tcs
    assert out["cost"] == 0.0009


def test_aragllm_force_answer_call_has_no_tools(monkeypatch) -> None:
    captured = _patch_litellm(monkeypatch, [_fake_response("final")])
    llm = arag_llm.AragLLM({"model": "openai/gpt-5.4-nano"})
    out = llm.chat(messages=[{"role": "user", "content": "q"}], tools=None, temperature=0.0)
    req = captured[0]
    assert "tools" not in req and "tool_choice" not in req
    assert req["temperature"] == 0.0
    # A clean answer message drops null tool_calls so the next request stays valid.
    assert out["message"]["content"] == "final"
    assert "tool_calls" not in out["message"]


def test_aragllm_accumulates_usage_and_full_calls(monkeypatch) -> None:
    _patch_litellm(monkeypatch, [_fake_response("a", ptoks=10, ctoks=5),
                                 _fake_response("b", ptoks=20, ctoks=8)])
    llm = arag_llm.AragLLM({"model": "openai/gpt-5.4-nano"})
    llm.chat(messages=[{"role": "user", "content": "1"}])
    llm.chat(messages=[{"role": "user", "content": "2"}])
    usage = llm.usage
    # Deterministic {total, calls} accumulation (model key = response.model).
    assert usage["total"]["gpt-5.4-nano"]["num_calls"] == 2
    assert usage["total"]["gpt-5.4-nano"]["total_tokens"] == (15 + 28)
    assert len(usage["calls"]) == 2
    # Full per-call trace for calls.json (content surfaced explicitly).
    assert [c["content"] for c in llm.full_calls] == ["a", "b"]


# ============================================================
# embedding — OpenAI sentence-embedding index + OpenAISemanticSearchTool
# ============================================================

import numpy as np  # noqa: E402

from evals.baselines.arag import embedding as arag_embed  # noqa: E402


class _FakeEmbedder:
    """Deterministic, offline stand-in for the OpenAI embedder (no network)."""
    model = "fake-embed"

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encoded: list[list[str]] = []

    def encode(self, texts, normalize_embeddings: bool = True, **kw):
        if isinstance(texts, str):
            texts = [texts]
        self.encoded.append(list(texts))
        # A deterministic vector from each text's character profile.
        arr = np.array(
            [[(sum(ord(c) for c in t) + i) % 13 for i in range(self.dim)] for t in texts],
            dtype=np.float32,
        )
        if normalize_embeddings and len(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    @property
    def usage(self):
        return {"total": {}, "calls": []}


def test_split_sentences_matches_upstream_regex() -> None:
    # Upstream build_index splits on [.!?\n]+ and drops fragments <= 10 chars.
    out = arag_embed._split_sentences("A short. This is a long enough sentence here.\nTiny.")
    assert "This is a long enough sentence here" in out
    assert all(len(s) > 10 for s in out)  # "A short." (8) and "Tiny." (5) dropped


def test_build_sentence_index_writes_expected_pickle(tmp_path) -> None:
    import pickle
    rows = ["0:Cats are small animals. They like to sleep a lot every day.",
            "1:Dogs are loyal companions. They enjoy long walks outside daily."]
    embedder = _FakeEmbedder(dim=16)
    arag_embed.build_sentence_index(rows, embedder, tmp_path)

    index = pickle.loads((tmp_path / "sentence_index.pkl").read_bytes())
    assert set(index) == {"sentences", "embeddings", "sentence_to_chunk", "chunks", "model_name"}
    assert index["model_name"] == "fake-embed"
    # Every sentence maps back to a chunk id; embeddings line up 1:1 with sentences.
    assert len(index["sentences"]) == len(index["sentence_to_chunk"]) == index["embeddings"].shape[0]
    assert set(index["sentence_to_chunk"]) <= {"0", "1"}
    assert index["chunks"]["0"]["text"].startswith("Cats are small")


def test_openai_semantic_search_tool_runs_vendored_execute(tmp_path) -> None:
    # The vendored SemanticSearchTool.execute() runs unchanged with our OpenAI embedder
    # injected (no SentenceTransformer): builds the index, then a query returns chunks.
    rows = ["0:The capital of France is Paris, a large European city.",
            "1:Photosynthesis lets plants convert sunlight into chemical energy."]
    embedder = _FakeEmbedder(dim=16)
    chunks_file = tmp_path / "chunks.json"
    chunks_file.write_text(json.dumps(rows))
    arag_embed.build_sentence_index(rows, embedder, tmp_path)

    tool = arag_embed.OpenAISemanticSearchTool(str(chunks_file), str(tmp_path), embedder)
    assert tool.name == "semantic_search"
    from evals.baselines.arag.upstream.core.context import AgentContext
    result, log = tool.execute(AgentContext(), query="What is the capital of France?", top_k=1)
    assert "Chunk ID:" in result and log["chunks_found"] >= 1


def test_openai_embedder_encode_normalizes_batches_and_records_usage(monkeypatch) -> None:
    calls = []

    def fake_embedding(**kwargs):
        calls.append(kwargs)
        n = len(kwargs["input"])
        data = [{"embedding": [float(i + 1), 0.0, 0.0]} for i in range(n)]
        return SimpleNamespace(
            model="text-embedding-3-small", data=data,
            usage=SimpleNamespace(prompt_tokens=3 * n, total_tokens=3 * n,
                                  model_dump=lambda: {"prompt_tokens": 3 * n, "total_tokens": 3 * n}),
        )

    monkeypatch.setattr(arag_embed.litellm, "embedding", fake_embedding)
    embedder = arag_embed.OpenAIEmbedder("openai/text-embedding-3-small")
    vecs = embedder.encode(["a", "b", "c"], normalize_embeddings=True, batch_size=2)
    assert vecs.shape == (3, 3)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)  # L2-normalized for cosine dot
    assert len(calls) == 2  # 3 inputs, batch_size 2 → 2 calls
    assert embedder.usage["total"]["text-embedding-3-small"]["num_calls"] == 2
    # The embedding routes by its OWN provider (openai/ + env key), never a passed
    # completion endpoint — so no api_base / api_key is forwarded.
    assert all("api_base" not in c and "api_key" not in c for c in calls)


# ============================================================
# run — run_one call graph + per-task content-addressed index store (fakes)
# ============================================================

from evals.baselines.arag import run as arag_run  # noqa: E402


def _fake_benchmark():
    """A dracula-shaped fake benchmark module (bare question + the doc bundle)."""
    return SimpleNamespace(
        __name__="evals.benchmarks.dracula",
        get_documents=lambda tid: [
            "Paris is the capital of France. It is a large city on the river Seine.",
            "Berlin is the capital of Germany. It sits on the river Spree in Europe.",
        ],
        get_task=lambda tid: ("What is the capital of France?", ["d1", "d2"]),
    )


def _fake_embedding(**kwargs):
    n = len(kwargs["input"])
    data = [{"embedding": [float(len(kwargs["input"][i]) % 9 + 1), 1.0, 0.5]} for i in range(n)]
    return SimpleNamespace(
        model="text-embedding-3-small", data=data,
        usage=SimpleNamespace(prompt_tokens=2 * n, total_tokens=2 * n,
                              model_dump=lambda: {"prompt_tokens": 2 * n, "total_tokens": 2 * n}),
    )


def _arag_run_config():
    return {"benchmark": "dracula", "baseline": "arag", "model": "gpt-5-4-nano", "seed": 42,
            "embedding_model": "openai/text-embedding-3-small",
            "index_version": arag_run._INDEX_VERSION, "chunker_version": chunker.CHUNKER_VERSION}


def test_run_one_builds_index_runs_agent_returns_record(tmp_path, monkeypatch) -> None:
    import litellm
    # Agent: turn 1 calls semantic_search, turn 2 answers.
    tcs = [{"id": "c1", "type": "function",
            "function": {"name": "semantic_search", "arguments": '{"query": "capital of France"}'}}]
    completions = iter([_fake_response(None, tcs), _fake_response("Paris (A).")])
    monkeypatch.setattr(litellm, "completion", lambda **kw: next(completions))
    embed_calls: list = []

    def capturing_embedding(**kw):
        embed_calls.append(kw)
        return _fake_embedding(**kw)

    monkeypatch.setattr(litellm, "embedding", capturing_embedding)

    # Regression for the live vLLM bug: the completion runs against a vLLM endpoint,
    # but OpenAI embeddings must NOT inherit that endpoint (they route to OpenAI via env).
    rec = arag_run.run_one(_fake_benchmark(), "task1", _arag_run_config(), tmp_path,
                           {"model": "openai/gpt-5.4-nano", "api_base": "http://vllm:8555/v1", "api_key": "k"})
    assert embed_calls and all("api_base" not in c for c in embed_calls)

    assert rec["raw_answer"] == "Paris (A)."
    assert rec["index_ref"]
    # Per-task index store: chunks.json + sentence_index.pkl + the two receipts.
    idx = tmp_path / "_indices" / rec["index_ref"]
    assert (idx / "chunks.json").exists()
    assert (idx / "index" / "sentence_index.pkl").exists()
    assert (idx / "index_usage.json").exists() and (idx / "index_meta.json").exists()
    # Usage merges agent completions + query embeddings; calls.json has both agent calls.
    assert rec["usage"]["total"]
    assert len(rec["calls_full"]) == 2
    # The agent's light trajectory records the semantic_search call.
    assert any(t["tool_name"] == "semantic_search" for t in rec["trace"]["tool_calls"])


def test_run_one_reuses_existing_index(tmp_path, monkeypatch) -> None:
    import litellm
    from evals.baselines.arag import embedding as _embed

    monkeypatch.setattr(litellm, "embedding", _fake_embedding)
    builds = {"n": 0}
    real_build = _embed.build_sentence_index

    def counting_build(*a, **k):
        builds["n"] += 1
        return real_build(*a, **k)

    monkeypatch.setattr(_embed, "build_sentence_index", counting_build)

    bench, cfg = _fake_benchmark(), _arag_run_config()
    kw = {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": "k"}
    for _ in range(2):
        monkeypatch.setattr(litellm, "completion", lambda **k: _fake_response("answer"))
        arag_run.run_one(bench, "task1", cfg, tmp_path, kw)

    # Same doc-set → same index_hash → built ONCE, reused the second time.
    assert builds["n"] == 1
    index_dirs = [p for p in (tmp_path / "_indices").iterdir() if p.is_dir()]
    assert len(index_dirs) == 1  # one index dir (a sibling {hash}.lock file is expected)


# ============================================================
# Live wire-test (marker-gated) — real gpt-5.4-nano, 1 dracula task
# ============================================================


@pytest.mark.arag
def test_arag_end_to_end_dracula_real(tmp_path) -> None:
    """Drive the REAL A-RAG agent (tool-calling + OpenAI embeddings) on one
    dracula task against gpt-5.4-nano. Asserts the pipeline produces a non-empty
    answer, captures usage for both the completion model + the embedder, and builds
    the per-task index store. Costs a few cents; deselected by default (``-m arag``)."""
    from evals.baselines import _common
    from evals.baselines.arag.run import run_one
    from evals.settings import DEFAULT_COMPLETION_MODEL, DEFAULT_EMBEDDING_MODEL

    benchmark = _common.load_benchmark_module("dracula")
    task_id = benchmark.get_task_ids(**benchmark.STARTER_FILTER)[0]
    run_config = {
        "benchmark": "dracula", "baseline": "arag",
        "model": _common.canonical_model_id(DEFAULT_COMPLETION_MODEL), "seed": 42,
        "embedding_model": DEFAULT_EMBEDDING_MODEL,
    }
    litellm_kwargs = {"model": _common.with_provider_prefix(DEFAULT_COMPLETION_MODEL),
                      "api_base": None, "api_key": None}

    rec = run_one(benchmark, task_id, run_config, tmp_path, litellm_kwargs)

    assert isinstance(rec["raw_answer"], str) and rec["raw_answer"].strip()
    assert rec["index_ref"] and (tmp_path / "_indices" / rec["index_ref"] / "chunks.json").exists()
    # Usage captured for both the agent's completions and the query embedding.
    assert any("nano" in m for m in rec["usage"]["total"])
    assert rec["calls_full"]  # the agent's LLM calls were recorded for calls.json
