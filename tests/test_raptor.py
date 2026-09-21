"""Tests for the RAPTOR baseline connectors.

Fast, fake-driven (no real LLM / embeddings): the two LLM seams + the embedder are driven with a
patched ``litellm``, the connector helpers are pinned directly, and ``run_one`` drives the REAL
vendored RAPTOR pipeline (tree build + collapse-tree retrieval + QA) with fake litellm responses —
on a SMALL document so RAPTOR's clustering guard (``len(nodes) <= reduction_dimension+1``) skips
UMAP/GMM, keeping the test deterministic + fast while still exercising leaf creation, retrieval,
and the TOTAL-cost record. Requires the ``evals[raptor]`` extra installed.
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import litellm  # patched per-test (the seams + the vendored pipeline call litellm.{completion,embedding})

from evals.baselines.raptor import run as raptor_run
from evals.baselines.raptor.embedding import RaptorEmbeddingModel
from evals.baselines.raptor.llm import RaptorQAModel, RaptorSummarizationModel
from evals.baselines.raptor.runner import build_arg_parser, build_run_config


# ============================================================
# Fake litellm responses (mirror the other baselines' tests)
# ============================================================


class _FakeCompletion:
    def __init__(self, content, model="gpt-5.4-nano", pt=10, ct=5):
        msg = SimpleNamespace(content=content, reasoning_content=None)
        self.choices = [SimpleNamespace(message=msg)]
        self.usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct)
        self.model = model

    def model_dump(self):
        return {"model": self.model, "usage": {"total_tokens": self.usage.total_tokens}}


class _FakeEmbedding:
    def __init__(self, vec, model="text-embedding-3-small", pt=3):
        self.data = [SimpleNamespace(embedding=vec)]
        self.usage = SimpleNamespace(prompt_tokens=pt, completion_tokens=0, total_tokens=pt)
        self.model = model

    def model_dump(self):
        return {"model": self.model, "usage": {"total_tokens": self.usage.total_tokens}}


def _vec_for(text: str, dim: int = 8) -> list[float]:
    """A deterministic, text-dependent unit-ish vector so cosine distances are meaningful."""
    h = abs(hash(text))
    return [((h >> (i * 3)) & 7) + 1.0 for i in range(dim)]


def _patch(monkeypatch, *, completion_content="ANSWER", capture=None):
    def fake_completion(**kw):
        if capture is not None:
            capture.append(kw)
        return _FakeCompletion(completion_content)

    def fake_embedding(model, input, **kw):
        return _FakeEmbedding(_vec_for(input[0]), model=model.split("/")[-1])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "embedding", fake_embedding)


_KW = {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None}


# ============================================================
# LLM seams — verbatim prompts, usage, calls.json, max_tokens
# ============================================================


def test_summarization_seam_prompt_hint_and_no_cap(monkeypatch) -> None:
    """D10/D12: upstream's summary sentence is kept verbatim, the summarization_length becomes a PROMPT
    length hint (not the cap), and the call sends NO max_tokens (v4) so a reasoning model uses the full
    window. The incoming `max_tokens` (77 here) drives the hint, NOT the call."""
    cap: list = []
    _patch(monkeypatch, completion_content="a summary", capture=cap)
    s = RaptorSummarizationModel(_KW, seed=42)
    out = s.summarize("CONTEXT TEXT", max_tokens=77)
    assert out == "a summary"
    msgs = cap[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are a helpful assistant."}
    # upstream sentence VERBATIM as the prefix + the length hint derived from summarization_length
    assert msgs[1]["content"].startswith(
        "Write a summary of the following, including as many key details as possible: CONTEXT TEXT:")
    assert msgs[1]["content"].endswith("IMPORTANT: keep your answer below 77 tokens.")
    # NO max_tokens is sent (v4/D12 — the cap would be eaten by reasoning); one attempt (content non-empty)
    assert "max_tokens" not in cap[0] and cap[0]["seed"] == 42
    assert len(cap) == 1
    assert s.usage["total"]["gpt-5.4-nano"]["total_tokens"] == 15
    assert s.full_calls[0]["role"] == "summarize" and s.full_calls[0]["content"] == "a summary"


def test_summarization_retries_on_empty_then_succeeds(monkeypatch) -> None:
    """Runaway reasoning hits the budget → empty content; the seam retries with a varied seed and
    returns the first non-empty summary (all attempts are still cost-counted)."""
    scripted = ["", "  ", "real summary"]  # two empties, then a real one
    cap: list = []

    def fake_completion(**kw):
        cap.append(kw)
        return _FakeCompletion(scripted[len(cap) - 1])

    monkeypatch.setattr(litellm, "completion", fake_completion)
    s = RaptorSummarizationModel(_KW, seed=10)
    out = s.summarize("CTX", max_tokens=100)
    assert out == "real summary"
    assert len(cap) == 3                       # retried past the two empties
    assert [c["seed"] for c in cap] == [10, 11, 12]   # seed varied per attempt
    assert s.usage["total"]["gpt-5.4-nano"]["num_calls"] == 3  # every attempt cost-counted


def test_qa_seam_verbatim_prompt_temp0_strip_no_max_tokens(monkeypatch) -> None:
    cap: list = []
    _patch(monkeypatch, completion_content="  the answer  ", capture=cap)
    qa = RaptorQAModel(_KW, seed=7)
    out = qa.answer_question("CTX", "QUESTION?")
    assert out == "the answer"  # .strip()ped (upstream behavior)
    msgs = cap[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "You are Question Answering Portal"}
    assert msgs[1]["content"] == "Given Context: CTX Give the best full answer amongst the option to question QUESTION?"
    assert cap[0]["temperature"] == 0          # upstream pins temp=0
    assert "max_tokens" not in cap[0]          # upstream QA sends NO max_tokens — we keep that
    assert qa.full_calls[0]["role"] == "qa"


def test_qa_config_temperature_overrides_default_zero(monkeypatch) -> None:
    cap: list = []
    _patch(monkeypatch, capture=cap)
    qa = RaptorQAModel(_KW, completion_params={"temperature": 0.9})
    qa.answer_question("c", "q")
    assert cap[0]["temperature"] == 0.9  # --config preset wins; we don't force 0


def test_embedding_seam_replaces_newlines_returns_vector_and_records(monkeypatch) -> None:
    seen: list = []

    def fake_embedding(model, input, **kw):
        seen.append(input[0])
        return _FakeEmbedding([1.0, 2.0, 3.0])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    e = RaptorEmbeddingModel("openai/text-embedding-3-small")
    vec = e.create_embedding("line one\nline two")
    assert vec == [1.0, 2.0, 3.0]
    assert seen == ["line one line two"]  # verbatim upstream \n→space
    assert e.usage["total"]["text-embedding-3-small"]["total_tokens"] == 3


def test_embedding_seam_guards_empty_string(monkeypatch) -> None:
    """An empty node text (a thinking model that emitted only reasoning) must NOT reach OpenAI as
    ``""`` (it 400s and crashes the build) — the embedder sends a single space instead."""
    seen: list = []

    def fake_embedding(model, input, **kw):
        seen.append(input[0])
        return _FakeEmbedding([0.0])

    monkeypatch.setattr(litellm, "embedding", fake_embedding)
    e = RaptorEmbeddingModel("openai/text-embedding-3-small")
    e.create_embedding("")        # empty summary
    assert seen == [" "]          # embedded a space, not "" → no 400


def test_summarization_seam_thread_safe(monkeypatch) -> None:
    _patch(monkeypatch, completion_content="s")
    s = RaptorSummarizationModel(_KW)
    threads = [threading.Thread(target=lambda: s.summarize(f"ctx{i}", max_tokens=10)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(s.full_calls) == 16  # no records lost to the race (RAPTOR builds multithreaded)
    assert s.usage["total"]["gpt-5.4-nano"]["num_calls"] == 16


# ============================================================
# Connector helpers
# ============================================================


def _loong_bench(instruction="INSTR", question="Q", docs=("d1", "d2")):
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_task=lambda t: (instruction, question, list(docs)),
        get_documents=lambda t: list(docs),
    )


def _corpusqa_bench(instruction="OUTPUT REQS", question="Q?", docs=("d1", "d2")):
    return SimpleNamespace(
        __name__="evals.benchmarks.corpusqa",
        get_task=lambda t: (instruction, question, list(docs)),
        get_documents=lambda t: list(docs),
    )


def test_build_query_loong_joins_instruction_and_question() -> None:
    assert raptor_run._build_query(_loong_bench("I", "Qq"), "t") == "I\n\nQq"


def test_build_query_loong_empty_question_falls_back_to_instruction() -> None:
    assert raptor_run._build_query(_loong_bench("only instruction", ""), "t") == "only instruction"


def test_build_query_dracula_bare_question() -> None:
    from types import SimpleNamespace
    b = SimpleNamespace(__name__="evals.benchmarks.dracula",
                        get_task=lambda t: ("HOW MANY?", ["d1", "d2"]))
    assert raptor_run._build_query(b, "t") == "HOW MANY?"
    assert "dracula" in raptor_run.SUPPORTED_BENCHMARKS


def test_build_query_corpusqa_question_then_output_requirements() -> None:
    # corpusqa poses `question\n\n{output-requirements}` so the answer-format contract reaches the model
    assert raptor_run._build_query(_corpusqa_bench("OUT REQS", "How many?"), "t") == "How many?\n\nOUT REQS"


def test_pool_documents_joins_bundle() -> None:
    assert raptor_run._pool_documents(["a", "b", "c"]) == "a\n\nb\n\nc"


def test_connector_tree_builder_enables_summary_multithreading(monkeypatch) -> None:
    """D8: ``build_from_text`` calls ``construct_tree`` with NO ``use_multithreading`` (→ sequential
    default). Our subclass must default the flag to True and delegate, so the per-cluster summaries
    parallelize."""
    import evals.baselines.raptor.upstream.cluster_tree_builder as ctb
    seen: dict = {}

    class _Stub:
        def __init__(self, cfg):
            pass

        def construct_tree(self, current, all_nodes, layers, use_multithreading=False):
            seen["mt"] = use_multithreading

    monkeypatch.setattr(ctb, "ClusterTreeBuilder", _Stub)
    builder = raptor_run._connector_tree_builder("dummy-config")
    builder.construct_tree({}, {}, {})   # exactly how build_from_text calls it (no mt arg)
    assert seen["mt"] is True            # our subclass defaults it True → summaries parallelize


def test_cjk_chunker_byte_identical_to_upstream_on_ascii() -> None:
    """D9 superset claim: the CJK-aware splitter MUST be byte-identical to upstream `split_text` on
    ASCII text (English contains none of the full-width delimiters), so EN trees are unchanged."""
    import tiktoken

    from evals.baselines.raptor.chunker import split_text as cjk
    from evals.baselines.raptor.upstream.utils import split_text as upstream
    tok = tiktoken.get_encoding("cl100k_base")
    text = ("Retrieval augmented generation has seven failure points. Missing content is one! "
            "Is the top doc missed? Yes, sometimes; also format errors, wrong specificity: many cases.")
    for cap in (5, 20, 100):
        assert cjk(text, tok, cap) == upstream(text, tok, cap)


def test_cjk_chunker_splits_chinese_prose_upstream_does_not() -> None:
    """D9 fix: a long run of Chinese prose (full-width 。！？，；： only, no ASCII delimiters) is ONE
    oversized chunk under upstream but splits into <=cap-token leaves under the CJK-aware splitter."""
    import tiktoken

    from evals.baselines.raptor.chunker import split_text as cjk
    from evals.baselines.raptor.upstream.utils import split_text as upstream
    tok = tiktoken.get_encoding("cl100k_base")
    # A long run of Chinese prose: many short sentences joined by full-width 。 — NO ASCII
    # delimiter anywhere (this is what Loong ZH `legal` looks like).
    text = "。".join(["分家契约系各方真实意思表示父母在现场主持"] * 40) + "。"
    cap = 50
    up = upstream(text, tok, cap)
    cj = cjk(text, tok, cap)
    up_max = max(len(tok.encode(c)) for c in up)
    cj_max = max(len(tok.encode(c)) for c in cj)
    # upstream sees no recognized delimiter → one (or few) giant oversized chunk(s)
    assert up_max > 3 * cap
    # CJK-aware → many chunks, each bounded near the cap → NO oversized leaf (this is the fix:
    # an oversized leaf is what makes a 2-4-node cluster blow past 3500 and crash UMAP)
    assert len(cj) > len(up)
    assert cj_max < up_max and cj_max <= 2 * cap


def test_merge_usage_sums_models_and_concatenates_calls() -> None:
    u1 = {"total": {"m": {"total_tokens": 3, "num_calls": 1}}, "calls": [{"model": "m"}]}
    u2 = {"total": {"m": {"total_tokens": 4, "num_calls": 1}, "e": {"total_tokens": 9, "num_calls": 2}},
          "calls": [{"model": "e"}]}
    merged = raptor_run._merge_usage(u1, u2)
    assert merged["total"]["m"]["total_tokens"] == 7 and merged["total"]["m"]["num_calls"] == 2
    assert merged["total"]["e"]["total_tokens"] == 9
    assert len(merged["calls"]) == 2


def test_tree_summary_shape() -> None:
    leaf = SimpleNamespace(index=0, text="leaf", children=set())
    root = SimpleNamespace(index=1, text="summary", children={0})
    tree = SimpleNamespace(
        all_nodes={0: leaf, 1: root}, leaf_nodes={0: leaf}, root_nodes={1: root},
        num_layers=1, layer_to_nodes={0: [leaf], 1: [root]},
    )
    s = raptor_run._tree_summary(tree)
    assert s["n_nodes"] == 2 and s["n_leaf_nodes"] == 1 and s["num_layers"] == 1
    assert s["nodes"]["1"]["children"] == [0] and s["nodes"]["0"]["text"] == "leaf"


# ============================================================
# run_one — drives the REAL RAPTOR pipeline with fakes
# ============================================================


def test_run_one_builds_tree_in_folder_with_total_cost(tmp_path, monkeypatch) -> None:
    _patch(monkeypatch, completion_content="The answer is 42")
    # A small multi-doc bundle → few leaf chunks → RAPTOR skips clustering (no UMAP), tree = leaves.
    docs = [
        "Alpha facts. The first document discusses alpha topics in some detail here.",
        "Beta facts. The second document covers beta material and more context follows.",
    ]
    bench = _loong_bench("Summarize the docs.", "What is the answer?", docs)
    run_config = {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano",
                  "seed": 42, "embedding_model": "openai/text-embedding-3-small", "run_version": "v1"}
    run_dir = tmp_path / "run1"

    record = raptor_run.run_one(bench, "task-x", run_config, run_dir, _KW)

    # the record shape (purely model output + TOTAL usage + calls + trace)
    assert record["raw_answer"] == "The answer is 42"
    total = record["usage"]["total"]
    assert "gpt-5.4-nano" in total and "text-embedding-3-small" in total   # QA + embeddings both counted
    assert total["gpt-5.4-nano"]["num_calls"] == 1                          # one QA call (no clusters → no summaries)
    assert total["text-embedding-3-small"]["num_calls"] >= 2               # ≥ each leaf + the query
    # the readable tree was written inside the run folder
    tree = json.loads((run_dir / raptor_run._TREE_FILE).read_text())
    assert tree["n_nodes"] >= 1 and record["trace"]["num_tree_nodes"] == tree["n_nodes"]
    assert record["trace"]["n_docs"] == 2 and record["trace"]["query"] == "Summarize the docs.\n\nWhat is the answer?"
    assert "retrieved_layer_information" in record["trace"]
    # calls_full carries the QA call (verbatim request/response) for calls.json
    assert any(c["role"] == "qa" for c in record["calls_full"])
    # live progress.json was written and ended at the "done" stage with the tree size
    prog = json.loads((run_dir / raptor_run._PROGRESS_FILE).read_text())
    assert prog["stage"] == "done" and prog["n_docs"] == 2
    assert prog["n_tree_nodes"] == tree["n_nodes"] and "elapsed_s" in prog


# ============================================================
# Phase split — embed (offline leaves) / build / save+load
# ============================================================


def test_embed_one_embeds_leaves_no_completion(tmp_path, monkeypatch) -> None:
    """PHASE 1 makes ZERO completion calls — only OpenAI leaf embeddings — and checkpoints the leaf
    nodes + the embed usage. The leaf-embed count == the embed_usage num_calls."""
    comp_calls: list = []

    def fake_completion(**kw):
        comp_calls.append(kw)
        return _FakeCompletion("should-not-be-called")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "embedding",
                        lambda model, input, **kw: _FakeEmbedding(_vec_for(input[0]), model=model.split("/")[-1]))
    docs = ["Alpha facts here. The first document discusses alpha topics in detail.",
            "Beta facts here. The second document covers beta material and context."]
    bench = _loong_bench("Summarize.", "What?", docs)
    run_config = {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano", "seed": 42,
                  "embedding_model": "openai/text-embedding-3-small", "run_version": "v3"}

    ckpt = raptor_run.embed_one(bench, "task-x", run_config, tmp_path / "r")

    assert comp_calls == []                       # NO completion LLM in phase 1
    assert ckpt["n_leaves"] >= 1 and "leaf_nodes" in ckpt and ckpt["query"] == "Summarize.\n\nWhat?"
    assert ckpt["embed_usage"]["total"]["text-embedding-3-small"]["num_calls"] == ckpt["n_leaves"]
    # progress.json shows the offline embed stage finished
    prog = json.loads((tmp_path / "r" / raptor_run._PROGRESS_FILE).read_text())
    assert prog["stage"] == "embedded" and prog["n_leaves"] == ckpt["n_leaves"]


def test_save_load_embed_round_trip(tmp_path) -> None:
    """``save_embed`` writes the SMALL ``embed.json`` marker (task_id + config, NO vectors) + the big
    ``leaves.pkl``; ``load_embed`` reconstructs the leaf nodes. ``_progress`` is never serialized."""
    from evals.baselines.raptor.upstream.tree_structures import Node
    leaf_nodes = {0: Node("leaf zero", 0, set(), {"EMB": [1.0, 2.0]}),
                  1: Node("leaf one", 1, set(), {"EMB": [3.0, 4.0]})}
    ckpt = {"query": "Q", "n_docs": 2, "n_leaves": 2, "pooled_chars": 99,
            "leaf_nodes": leaf_nodes, "embed_usage": {"total": {}, "calls": []},
            "_progress": object()}
    cfg = {"benchmark": "loong", "run_version": "v3"}
    rd = tmp_path / "run"

    ckpt["embed_usage"] = {"total": {"text-embedding-3-small": {"num_calls": 2}}, "calls": [{"a": 1}]}
    raptor_run.save_embed(rd, "task-x", cfg, ckpt)

    meta = json.loads((rd / raptor_run._EMBED_FILE).read_text())
    assert meta["task_id"] == "task-x" and meta["config"] == cfg
    # the bulky per-call embed list is NOT in the small marker — only the rollup total (for the scan)
    assert "leaf_nodes" not in meta and "_progress" not in meta and "embed_usage" not in meta
    assert meta["embed_usage_total"] == {"text-embedding-3-small": {"num_calls": 2}}
    assert (rd / raptor_run._LEAVES_FILE).exists()

    loaded = raptor_run.load_embed(rd)
    assert loaded["query"] == "Q" and loaded["n_docs"] == 2 and set(loaded["leaf_nodes"]) == {0, 1}
    assert loaded["leaf_nodes"][0].text == "leaf zero"
    assert loaded["leaf_nodes"][0].embeddings["EMB"] == [1.0, 2.0]
    # the FULL embed usage (incl. the per-call list) round-trips via leaves.pkl for the manifest merge
    assert loaded["embed_usage"]["calls"] == [{"a": 1}]


def test_build_one_from_checkpoint_merges_embed_usage(tmp_path, monkeypatch) -> None:
    """PHASE 2 reconstructs the tree from the checkpoint's leaves + answers; the TOTAL usage merges
    the phase-1 leaf embeddings (from the checkpoint) with the QA + the query embedding."""
    _patch(monkeypatch, completion_content="The answer is 7")
    from evals.baselines.raptor.upstream.tree_structures import Node
    # 3 leaf nodes (< reduction_dimension+1 → no clustering, tree == leaves), each pre-embedded.
    leaf_nodes = {i: Node(f"leaf {i} content text here", i, set(), {"EMB": _vec_for(f"leaf{i}")})
                  for i in range(3)}
    embed_usage = {"total": {"text-embedding-3-small": {"num_calls": 3, "total_tokens": 9}},
                   "calls": [{"model": "text-embedding-3-small", "usage": {"total_tokens": 3}}] * 3}
    ckpt = {"query": "Summarize.\n\nWhat?", "n_docs": 2, "n_leaves": 3, "pooled_chars": 123,
            "leaf_nodes": leaf_nodes, "embed_usage": embed_usage}
    run_config = {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano", "seed": 42,
                  "embedding_model": "openai/text-embedding-3-small", "run_version": "v3"}

    record = raptor_run.build_one(_loong_bench(), "task-x", run_config, tmp_path / "r", _KW, ckpt)

    assert record["raw_answer"] == "The answer is 7"
    total = record["usage"]["total"]
    assert total["gpt-5.4-nano"]["num_calls"] == 1                      # one QA call (no clusters)
    assert total["text-embedding-3-small"]["num_calls"] == 3 + 1       # phase-1 leaves(3) + query(1)
    assert record["trace"]["pooled_chars"] == 123 and record["trace"]["n_docs"] == 2
    tree = json.loads((tmp_path / "r" / raptor_run._TREE_FILE).read_text())
    assert tree["n_leaf_nodes"] == 3


def test_run_one_all_writes_no_checkpoint(tmp_path, monkeypatch) -> None:
    """``--phase all`` (run_one) keeps the embed→build handoff in-memory: it writes NO
    embed.json/leaves.pkl, but produces the full manifest record + tree (byte-compatible single pass)."""
    _patch(monkeypatch, completion_content="The answer is 42")
    docs = ["Alpha facts. The first document discusses alpha topics here.",
            "Beta facts. The second document covers beta material here."]
    bench = _loong_bench("Summarize the docs.", "What is the answer?", docs)
    run_config = {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano", "seed": 42,
                  "embedding_model": "openai/text-embedding-3-small", "run_version": "v3"}
    run_dir = tmp_path / "run"

    record = raptor_run.run_one(bench, "task-x", run_config, run_dir, _KW)

    assert record["raw_answer"] == "The answer is 42"
    assert not (run_dir / raptor_run._EMBED_FILE).exists()       # all-phase never persists the checkpoint
    assert not (run_dir / raptor_run._LEAVES_FILE).exists()
    assert (run_dir / raptor_run._TREE_FILE).exists()
    assert "text-embedding-3-small" in record["usage"]["total"]


# ============================================================
# D11 — tiktoken special-token strings in document text
# ============================================================


def test_lenient_tiktoken_allows_special_token_text() -> None:
    """D11: tiktoken raises on literal special-token strings by default; inside _lenient_tiktoken they
    encode as normal text, and the patch is restored afterwards."""
    import pytest
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    with pytest.raises(ValueError):
        enc.encode("hello <|endoftext|> world")          # default: raises
    with raptor_run._lenient_tiktoken():
        toks = enc.encode("hello <|endoftext|> world")    # patched: encoded as normal text
    assert len(toks) > 3
    with pytest.raises(ValueError):
        enc.encode("hello <|endoftext|> world")          # restored


def test_embed_one_handles_special_token_docs(tmp_path, monkeypatch) -> None:
    """A doc literally containing tiktoken markers must chunk+embed without raising (D11)."""
    monkeypatch.setattr(litellm, "embedding",
                        lambda model, input, **kw: _FakeEmbedding(_vec_for(input[0]), model=model.split("/")[-1]))
    docs = ["Intro. The model emits <|endoftext|> then stops. Also <|endofprompt|> appears here."]
    bench = _loong_bench("Sum.", "Q?", docs)
    run_config = {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano", "seed": 42,
                  "embedding_model": "openai/text-embedding-3-small", "run_version": "v3"}
    ckpt = raptor_run.embed_one(bench, "t", run_config, tmp_path / "r")   # must NOT raise
    assert ckpt["n_leaves"] >= 1


# ============================================================
# Runner — config identity + child cmd + phase resumption
# ============================================================


def _args(**over):
    base = dict(benchmark="loong", model="openai/gpt-5.4-nano", config=None, seed=42,
                base_url=None, api_key=None, embedding_model="openai/text-embedding-3-small",
                phase="all", embed_workers=None, summary_workers=None, max_workers=8,
                paper_hparams=False, limit=None, task_id=None, run_tag=None)
    base.update(over)
    return SimpleNamespace(**base)


def test_build_run_config_identity() -> None:
    cfg = build_run_config(_args())
    assert cfg == {"benchmark": "loong", "baseline": "raptor", "model": "gpt-5-4-nano",
                   "seed": 42, "embedding_model": "openai/text-embedding-3-small", "run_version": "v4"}
    assert "config_name" not in cfg and "completion_params" not in cfg


def test_build_run_config_with_sampling_preset() -> None:
    from evals.model_sampling import MODEL_SAMPLING_CONFIG
    name = sorted(MODEL_SAMPLING_CONFIG)[0]
    cfg = build_run_config(_args(config=name))
    assert cfg["config_name"] == name and cfg["completion_params"] == MODEL_SAMPLING_CONFIG[name]


def test_paper_hparams_flag_selects_values_and_folds_into_config() -> None:
    from evals.baselines.raptor.runner import build_child_cmd
    # OFF (default): the D12 re-scale, and NO config key → byte-identical to before.
    assert raptor_run._hparams({}) == raptor_run._D12_HPARAMS
    assert raptor_run._hparams({})["chunk_tokens"] == 2000
    assert "paper_hparams" not in build_run_config(_args())
    assert "--paper-hparams" not in build_child_cmd(_args(), "TID", "TAG")
    # ON: RAPTOR's published values, folded into the run identity + threaded to the child cmd.
    assert raptor_run._hparams({"paper_hparams": True}) == raptor_run._PAPER_HPARAMS == {
        "chunk_tokens": 100, "recluster_threshold": 3500, "summary_length": 100,
        "retrieval_top_k": 10, "retrieval_max_tokens": 3500}
    assert build_run_config(_args(paper_hparams=True))["paper_hparams"] is True
    assert "--paper-hparams" in build_child_cmd(_args(paper_hparams=True), "TID", "TAG")


def test_tasks_filter_selects_variants_and_stays_out_of_identity() -> None:
    from evals.baselines.raptor.runner import _filter_task_variants, build_child_cmd
    ids = ["financial_en_1@1m", "financial_en_2@4m", "real_estate_en_3@1m", "uuid", "bareid"]
    assert _filter_task_variants(ids, ["1m"]) == ["financial_en_1@1m", "real_estate_en_3@1m"]
    assert _filter_task_variants(ids, ["1m", "4m"]) == ids[:3]
    assert _filter_task_variants(ids, ["10m"]) == []   # typo → empty (main turns this into a clear error)
    # --tasks parses, but it's a parent-mode selection filter — NOT in the run identity or the child cmd
    args = build_arg_parser().parse_args(["--benchmark", "corpusqa", "--tasks", "1m", "4m"])
    assert args.tasks == ["1m", "4m"]
    assert "tasks" not in build_run_config(_args())
    assert "--tasks" not in build_child_cmd(_args(), "TID", "TAG")


def test_summary_pool_size_caps_restores_and_is_noop_when_absent() -> None:
    """--summary-workers (D8): None leaves RAPTOR's vendored ThreadPoolExecutor byte-for-byte
    untouched (the no-flag contract); a value injects max_workers for the scope, then restores."""
    from evals.baselines.raptor import run as raptor_run
    from evals.baselines.raptor.upstream import cluster_tree_builder as ctb
    orig = ctb.ThreadPoolExecutor
    with raptor_run._summary_pool_size(None):       # no flag → vendored pool untouched
        assert ctb.ThreadPoolExecutor is orig
    with raptor_run._summary_pool_size(48):         # flag → injected max_workers
        ex = ctb.ThreadPoolExecutor()
        assert ex._max_workers == 48
        ex.shutdown()
    assert ctb.ThreadPoolExecutor is orig           # restored after the scope


def test_summary_workers_threads_to_child_cmd_not_into_config() -> None:
    from evals.baselines.raptor.runner import build_child_cmd
    cmd = build_child_cmd(_args(summary_workers=64), "TID", "TAG")
    assert "--summary-workers" in cmd and cmd[cmd.index("--summary-workers") + 1] == "64"
    assert "--summary-workers" not in build_child_cmd(_args(), "TID", "TAG")   # absent → not passed
    assert "summary_workers" not in build_run_config(_args(summary_workers=64))  # never run identity


def test_arg_parser_accepts_loong_and_corpusqa() -> None:
    p = build_arg_parser()
    assert p.parse_args(["--benchmark", "loong"]).benchmark == "loong"
    assert p.parse_args(["--benchmark", "corpusqa"]).benchmark == "corpusqa"
    # an unsupported benchmark is still rejected by argparse choices
    import pytest
    with pytest.raises(SystemExit):
        p.parse_args(["--benchmark", "nosuchbench"])


def test_arg_parser_phase_and_embed_workers() -> None:
    p = build_arg_parser()
    assert p.parse_args(["--benchmark", "loong"]).phase == "all"  # default
    for ph in ("embed", "build", "all"):
        assert p.parse_args(["--benchmark", "loong", "--phase", ph]).phase == ph
    assert p.parse_args(["--benchmark", "loong", "--embed-workers", "8"]).embed_workers == 8
    import pytest
    with pytest.raises(SystemExit):
        p.parse_args(["--benchmark", "loong", "--phase", "retrieve"])  # not a raptor phase


def test_child_cmd_threads_phase_and_embed_workers() -> None:
    from evals.baselines.raptor.runner import build_child_cmd
    cmd = build_child_cmd(_args(phase="embed", embed_workers=4), "t1", "tagX")
    assert "--phase" in cmd and cmd[cmd.index("--phase") + 1] == "embed"
    assert "--embed-workers" in cmd and cmd[cmd.index("--embed-workers") + 1] == "4"
    assert "--task-id" in cmd and "--run-tag" in cmd
    # embed_workers omitted when None (default ThreadPool)
    assert "--embed-workers" not in build_child_cmd(_args(phase="all"), "t1", "tagX")


def _write(folder, name, payload):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(payload))


def test_scan_phase_state_and_pending(tmp_path) -> None:
    """The phase-aware resumption: an ``embed.json``-only folder is an awaiting-build checkpoint
    (reused by ``--phase build``); a manifest/error folder is done; config mismatch is ignored."""
    from evals.baselines.raptor import runner
    base = tmp_path
    cfg = {"benchmark": "loong", "baseline": "raptor", "model": "m", "seed": 42,
           "embedding_model": "e", "run_version": "v3"}
    _write(base / "tagA", raptor_run._EMBED_FILE, {"task_id": "A", "config": cfg})   # awaiting build
    _write(base / "tagB", "manifest.json", {"task_id": "B", "config": cfg})          # done
    _write(base / "tagC", raptor_run._EMBED_FILE, {"task_id": "C", "config": {"seed": 99}})  # other config

    embedded, completed = runner._scan_phase_state(base, cfg)
    assert embedded == {"A": "tagA"} and completed == {"B"}   # tagC's config doesn't match → ignored

    all_ids = ["A", "B", "C", "D"]
    # build: only the existing checkpoint A (not B=done), reusing its run_tag
    assert runner._pending_for_phase(_args(phase="build"), all_ids, embedded, completed) == [("A", "tagA")]
    # embed: tasks with no checkpoint and not done → C, D (A has a checkpoint, B is done)
    assert [t for t, _ in runner._pending_for_phase(_args(phase="embed"), all_ids, embedded, completed)] == ["C", "D"]
    # all: every not-done task → A, C, D (B done); fresh tags
    assert [t for t, _ in runner._pending_for_phase(_args(phase="all"), all_ids, embedded, completed)] == ["A", "C", "D"]
