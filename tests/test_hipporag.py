"""Tests for the HippoRAG 2 baseline connector.

Fake-driven: ``litellm.completion`` is patched and the OpenAI embedder is stubbed, so
``run_one`` drives the REAL vendored HippoRAG pipeline (chunk → per-passage OpenIE →
query→triple linking → recognition filter → PPR → reader) with no network. Pins the
chunker, the per-benchmark query assembly, the LLM/embedding seams, and ``run_one``'s
per-passage OpenIE call graph + TOTAL-cost record. No marker — fake-driven, runs by
default (like the linearrag/raptor tests)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import litellm

from evals.baselines.hipporag import run as hr_run
from evals.baselines.hipporag.chunker import CHUNK_SIZES, chunk_documents
from evals.baselines.hipporag.embedding import HippoRAGOpenAIEmbedder
from evals.baselines.hipporag.llm import HippoRAGLLM


# ---------------------------------------------------------------------------
# Fake litellm response + a prompt-kind-aware fake completion
# ---------------------------------------------------------------------------
class _Resp:
    def __init__(self, content, model="openai/gpt-5.4-nano", pt=10, ct=5):
        msg = SimpleNamespace(content=content, reasoning_content=None)
        self.choices = [SimpleNamespace(message=msg, finish_reason="stop")]
        self.usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        self.model = model

    def model_dump(self):
        return {"model": self.model, "usage": {"total_tokens": self.usage.total_tokens}}


def _classify(messages) -> str:
    text = "\n".join(m.get("content", "") for m in messages if isinstance(m, dict)).lower()
    if "fact_before_filter" in text:
        return "filter"
    if "thought:" in text and "question:" in text:
        return "qa"
    if "triples" in text:
        return "triples"
    return "ner"


def _install_fakes(monkeypatch, counter=None, capture=None):
    """Patch litellm.completion (kind-aware) + the embedder to deterministic vectors."""
    def fake_completion(**kw):
        kind = _classify(kw["messages"])
        if counter is not None:
            counter[kind] = counter.get(kind, 0) + 1
        if capture is not None:
            capture.append(kw)
        if kind == "ner":
            return _Resp('{"named_entities": ["Aspirin", "Dr. Smith", "Anna"]}')
        if kind == "triples":
            return _Resp('{"triples": [["Dr. Smith", "prescribed", "Aspirin"]]}')
        if kind == "filter":
            return _Resp('[[ ## fact_after_filter ## ]]\n'
                         '{"fact": [["Dr. Smith", "prescribed", "Aspirin"]]}\n\n[[ ## completed ## ]]')
        return _Resp("Thought: The doctor prescribed Aspirin.\nAnswer: Aspirin")

    def fake_encode(self, texts):
        out = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            out.append(np.random.default_rng(seed).standard_normal(1536).astype(np.float32))
        return np.array(out, dtype=np.float32)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(HippoRAGOpenAIEmbedder, "encode", fake_encode)


def _fake_benchmark(name, get_task):
    return SimpleNamespace(__name__=f"evals.benchmarks.{name}", get_task=get_task,
                           get_documents=lambda tid: get_task(tid)[-1])


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------
def test_chunk_sizes_are_the_greenlit_per_benchmark_values() -> None:
    assert CHUNK_SIZES == {"corpusqa": 8000, "loong": 3000, "dracula": 3000}


def test_chunker_splits_by_token_and_never_spans_documents() -> None:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    # Two docs, each ~5 chunks at size 100.
    doc_a = " ".join(f"alpha{i}" for i in range(600))
    doc_b = " ".join(f"beta{i}" for i in range(300))
    passages = chunk_documents([doc_a, doc_b], chunk_size=100)
    # every passage ≤ 100 tokens
    assert all(len(enc.encode(p)) <= 100 for p in passages)
    # no passage mixes tokens from both docs (a passage never spans a doc boundary)
    assert all(("alpha" in p) != ("beta" in p) for p in passages)
    # a doc shorter than the chunk size stays one passage
    assert chunk_documents(["short doc"], chunk_size=1000) == ["short doc"]
    # empty/whitespace docs drop out
    assert chunk_documents(["   ", ""], chunk_size=100) == []


# ---------------------------------------------------------------------------
# Per-benchmark query assembly (same composed task the other baselines pose)
# ---------------------------------------------------------------------------
def test_build_query_loong_instruction_plus_question() -> None:
    b = _fake_benchmark("loong", lambda tid: ("DO IT", "WHICH?", ["d"]))
    assert hr_run._build_query(b, "t") == "DO IT\n\nWHICH?"


def test_build_query_loong_empty_question_is_instruction_only() -> None:
    b = _fake_benchmark("loong", lambda tid: ("SUMMARIZE.", "  ", ["d"]))
    assert hr_run._build_query(b, "t") == "SUMMARIZE."


def test_build_query_corpusqa_question_then_output_requirements() -> None:
    b = _fake_benchmark("corpusqa", lambda tid: ("OUTPUT REQS", "HOW MANY?", ["d"]))
    assert hr_run._build_query(b, "t") == "HOW MANY?\n\nOUTPUT REQS"


def test_build_query_dracula_bare_question() -> None:
    b = SimpleNamespace(__name__="evals.benchmarks.dracula", get_task=lambda t: ("HOW MANY?", ["d1", "d2"]))
    assert hr_run._build_query(b, "t") == "HOW MANY?"
# ---------------------------------------------------------------------------
# LLM seam
# ---------------------------------------------------------------------------
def test_llm_seam_returns_tuple_drops_caps_and_captures_usage(monkeypatch) -> None:
    cap: list = []
    _install_fakes(monkeypatch, capture=cap)
    seam = HippoRAGLLM({"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None}, seed=7)
    content, meta, cache_hit = seam.infer(
        [{"role": "user", "content": "hi"}], max_completion_tokens=512, max_tokens=99,
    )
    assert isinstance(content, str) and cache_hit is False
    assert meta["prompt_tokens"] == 10 and meta["completion_tokens"] == 5
    # D5: no token cap is forwarded to litellm even though the caller asked for one
    assert "max_completion_tokens" not in cap[0] and "max_tokens" not in cap[0]
    assert cap[0]["seed"] == 7
    # usage + calls captured
    assert seam.usage["total"]["openai/gpt-5.4-nano"]["num_calls"] == 1
    assert len(seam.full_calls) == 1 and seam.full_calls[0]["content"] == content


def test_llm_seam_usage_is_thread_safe(monkeypatch) -> None:
    # HippoRAG's OpenIE fans NER + triple calls across a ThreadPoolExecutor, so the seam's
    # usage accumulation must not lose counts under concurrency (a racy read-modify-write
    # on the totals would undercount tokens).
    from concurrent.futures import ThreadPoolExecutor
    _install_fakes(monkeypatch)
    seam = HippoRAGLLM({"model": "openai/gpt-5.4-nano"})
    n = 500
    with ThreadPoolExecutor(max_workers=32) as pool:
        list(pool.map(lambda _: seam.infer([{"role": "user", "content": "triples: x"}]), range(n)))
    assert seam.usage["total"]["openai/gpt-5.4-nano"]["num_calls"] == n
    assert seam.usage["total"]["openai/gpt-5.4-nano"]["prompt_tokens"] == n * 10
    assert len(seam.usage["calls"]) == n and len(seam.full_calls) == n


def test_llm_seam_config_sampling_applies_to_every_call(monkeypatch) -> None:
    cap: list = []
    _install_fakes(monkeypatch, capture=cap)
    seam = HippoRAGLLM({"model": "openai/x"}, completion_params={"temperature": 0.7})
    seam.infer([{"role": "user", "content": "hi"}])
    assert cap[0]["temperature"] == 0.7


# ---------------------------------------------------------------------------
# Embedding seam
# ---------------------------------------------------------------------------
def test_embedding_seam_normalizes_and_shapes(monkeypatch) -> None:
    _install_fakes(monkeypatch)
    emb = HippoRAGOpenAIEmbedder(embedding_model_name="text-embedding-3-small")
    vecs = emb.batch_encode(["hello world", "foo"])
    assert vecs.shape == (2, 1536)
    # L2-normalized (HippoRAG scores by cosine)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_embedding_seam_records_usage_for_index_cost() -> None:
    # encode() folds the embeddings response's usage into {total, calls} so the run's
    # TOTAL cost includes the index-build embeddings, not just the LLM calls.
    emb = HippoRAGOpenAIEmbedder(embedding_model_name="text-embedding-3-small")
    emb._record(_Resp("x", model="text-embedding-3-small", pt=100, ct=0))
    emb._record(_Resp("y", model="text-embedding-3-small", pt=40, ct=0))
    u = emb.usage
    assert u["total"]["text-embedding-3-small"]["num_calls"] == 2
    assert u["total"]["text-embedding-3-small"]["prompt_tokens"] == 140


def test_merge_usage_keeps_separate_model_keys() -> None:
    llm_u = {"total": {"gpt-5.4-nano": {"num_calls": 3, "total_tokens": 30}}, "calls": [1, 2, 3]}
    emb_u = {"total": {"text-embedding-3-small": {"num_calls": 2, "total_tokens": 10}}, "calls": [4, 5]}
    merged = hr_run._merge_usage(llm_u, emb_u, None)
    assert set(merged["total"]) == {"gpt-5.4-nano", "text-embedding-3-small"}
    assert merged["total"]["gpt-5.4-nano"]["total_tokens"] == 30
    assert merged["total"]["text-embedding-3-small"]["num_calls"] == 2
    assert len(merged["calls"]) == 5


def test_embedding_seam_splits_batch_on_token_limit_error() -> None:
    # The API's own token count can exceed our tiktoken estimate on dense text; a
    # per-request-token-cap 400 must trigger split-and-retry, not fail the task.
    from evals.baselines.hipporag import embedding as emb_mod

    class _TokenLimit(Exception):
        message = "Requested 394566 tokens, max 300000 tokens per request (max_tokens_per_request)"

    emb = HippoRAGOpenAIEmbedder(embedding_model_name="text-embedding-3-small")
    seen_batch_sizes = []

    class _FakeEmbeds:
        def create(self, input, model):
            seen_batch_sizes.append(len(input))
            if len(input) > 1:  # reject any multi-item batch (forces full split to singletons)
                raise _TokenLimit()
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.0] * 1536)],
                usage=SimpleNamespace(prompt_tokens=5, total_tokens=5),
                model="text-embedding-3-small",
                model_dump=lambda: {},
            )

    emb._client = SimpleNamespace(embeddings=_FakeEmbeds())
    vecs = emb._embed_request(["a", "b", "c", "d"])
    assert vecs.shape == (4, 1536)              # all four embedded via recursive split
    assert seen_batch_sizes[0] == 4 and min(seen_batch_sizes) == 1  # split down to singletons
    assert emb_mod._is_token_limit_error(_TokenLimit())


def test_embedding_seam_truncates_over_cap_without_ipdb() -> None:
    emb = HippoRAGOpenAIEmbedder(embedding_model_name="text-embedding-3-small")
    huge = "word " * 20000  # ~20K tokens, over the 8,191 cap
    clipped = emb._truncate(huge)
    import tiktoken
    n = len(tiktoken.get_encoding("cl100k_base").encode(clipped))
    assert n <= 8000  # D9: truncated under the cap (no crash, no ipdb)
    assert emb._truncate("") == " "  # empty guarded


# ---------------------------------------------------------------------------
# run_one — drives the REAL vendored pipeline end-to-end
# ---------------------------------------------------------------------------
def test_run_one_drives_real_pipeline_per_passage_openie_and_total_cost(monkeypatch, tmp_path) -> None:
    counter: dict = {}
    _install_fakes(monkeypatch, counter=counter)
    docs = [
        "Anna was admitted. Dr. Smith treated Anna for a headache.",
        "Dr. Smith reviewed the chart and prescribed Aspirin 500mg.",
    ]
    bench = _fake_benchmark("loong", lambda tid: ("Answer from the docs.", "Which drug?", docs))
    run_dir = tmp_path / "run_abc"
    run_dir.mkdir()

    record = hr_run.run_one(
        benchmark=bench, task_id="t1", run_config={"seed": 0},
        run_dir=run_dir, litellm_kwargs={"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None},
    )

    # OpenIE ran PER PASSAGE (2 docs ≤ chunk_size → 2 passages → 2 NER + 2 triple calls),
    # then the recognition filter, then exactly one reader call.
    assert counter["ner"] == 2 and counter["triples"] == 2
    assert counter["filter"] >= 1 and counter["qa"] == 1
    # the reader's answer is the raw_answer
    assert "Aspirin" in record["raw_answer"]
    # TOTAL cost = every internal call folded into one usage record + calls.json
    total_calls = sum(counter.values())
    assert record["usage"]["total"]["openai/gpt-5.4-nano"]["num_calls"] == total_calls
    assert len(record["calls_full"]) == total_calls
    # trace + the index written inside the run folder
    assert record["trace"]["n_passages"] == 2 and record["trace"]["chunk_size"] == 3000
    assert (run_dir / "index").exists()


def test_supported_benchmarks() -> None:
    assert hr_run.SUPPORTED_BENCHMARKS == frozenset({"loong", "corpusqa", "dracula"})


def test_tasks_flag_filters_id_variants_parent_only() -> None:
    from evals.baselines.hipporag import runner
    ids = ["financial_en_1@1m", "financial_en_2@4m", "real_estate_en_3@1m"]
    assert runner._filter_task_variants(ids, ["4m"]) == ["financial_en_2@4m"]
    assert runner._filter_task_variants(ids, ["1m"]) == ["financial_en_1@1m", "real_estate_en_3@1m"]
    # a parent-mode selection filter only — NOT part of a task's identity or the child cmd
    args = runner.build_arg_parser().parse_args(["--benchmark", "corpusqa", "--tasks", "1m"])
    assert "tasks" not in runner.build_run_config(args)
    assert "--tasks" not in runner.build_child_cmd(args, "t", "r")


def test_run_one_rejects_unsupported_benchmark(monkeypatch, tmp_path) -> None:
    _install_fakes(monkeypatch)
    b = _fake_benchmark("nosuchbench", lambda tid: ("c", "q", ["d"]))
    with pytest.raises(ValueError, match="no chunk size"):
        hr_run.run_one(benchmark=b, task_id="t", run_config={}, run_dir=tmp_path,
                       litellm_kwargs={"model": "openai/x"})
