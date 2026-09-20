"""Tests for `evals.baselines.structrag` (the StructRAG inference-time baseline).

StructRAG vendors its pipeline (`upstream/router|structurizer|utilizer` + prompts)
byte-for-byte and connects it via thin adapters: `StructRAGLLM` (the LLM seam over
the harness LiteLLM wrapper), and `run.py`'s connectors (`_to_structrag_docs` /
`_build_query`) + `run_one`. See `evals/baselines/structrag/PROVENANCE.md`.

Coverage:
- SUPPORTED_BENCHMARKS = {loong, corpusqa, dracula}.
- `_to_structrag_docs` round-trips through the VENDORED `split_content_and_tile`
  (the marker format upstream expects) for loong; single-doc benchmarks get a
  synthetic `Document N` title.
- `_build_query`: loong faithful to upstream (`prompt_template.format(... docs=
  "......")`); corpusqa/dracula mirror the other baselines.
- `StructRAGLLM`: user-only message, seed + max_tokens + `--config` params passed
  through, usage accumulated across calls into `{total, calls}`.
- `run_one`: drives the real vendored pipeline with a FAKE llm, asserting the exact
  per-task call graph (route → structurize → decompose → extract → merge) and the
  returned `{raw_answer, usage, trace}`.
- runner argparse / run_config / child_cmd (an unsupported benchmark is rejected).
- Real-LLM smoke test (`@pytest.mark.structrag`, deselected by default): the full
  pipeline against gpt-5.4-nano on a tiny multi-doc input. Run: `pytest -m structrag`.
"""
from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from evals.baselines.structrag import SUPPORTED_BENCHMARKS, run
from evals.baselines.structrag import runner  # imports _common → loads .env for the live test
from evals.baselines.structrag.upstream.structurizer import Structurizer


def _fake_completion(text: str = "ans", *, prompt=10, completion=2, total=12, model="M",
                     reasoning=None):
    """Minimal mock of the LiteLLM response shape the seam reads (.choices/.usage/.model).
    ``reasoning`` mimics a thinking model's ``message.reasoning_content``."""
    usage_dict = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}
    message = SimpleNamespace(content=text)
    if reasoning is not None:
        message.reasoning_content = reasoning
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        model=model,
        usage=SimpleNamespace(model_dump=lambda: usage_dict, **usage_dict),
    )


def _split(marker: str):
    """Run the VENDORED split on a marker string (proves our connector feeds it right)."""
    return Structurizer.split_content_and_tile(Structurizer.__new__(Structurizer), marker)


# ============================================================
# SUPPORTED_BENCHMARKS
# ============================================================


def test_supported_benchmarks() -> None:
    assert SUPPORTED_BENCHMARKS == frozenset({"loong", "corpusqa", "dracula"})


# ============================================================
# _to_structrag_docs — re-wrap our docs into upstream's marker string
# ============================================================


def test_loong_docs_roundtrip_through_vendored_split() -> None:
    """Our loong docs (`《title》\\ncontent`) → marker string → the UNMODIFIED upstream
    splitter recovers the same titles + content."""
    docs = ["《CompanyA》\nRevenue was 100.\n\n", "《CompanyB》\nRevenue was 200.\n\n"]
    marker = run._to_structrag_docs("loong", docs)
    assert marker.startswith(run._TITLE_START + "《CompanyA》" + run._TITLE_END)
    recovered, titles = _split(marker)
    assert titles == ["《CompanyA》", "《CompanyB》"]
    assert recovered[0]["title"] == "《CompanyA》"
    assert "Revenue was 100." in recovered[0]["document"]


def test_single_doc_benchmarks_get_synthetic_titles() -> None:
    marker = run._to_structrag_docs("corpusqa", ["a long context with no title line"])
    _, titles = _split(marker)
    assert titles == ["Document 1"]


def test_build_query_and_titles_dracula() -> None:
    from types import SimpleNamespace
    b = SimpleNamespace(__name__="evals.benchmarks.dracula",
                        get_task=lambda t: ("HOW MANY?", ["JONATHAN HARKER'S JOURNAL\nbody"]))
    assert run._build_query(b, "t") == "HOW MANY?"                       # bare question
    assert run._split_title_content("dracula", ["HDR\nbody\nmore"]) == [("HDR", "body\nmore")]


def test_split_title_content_loong_vs_single_doc() -> None:
    assert run._split_title_content("loong", ["T\nbody\nmore"]) == [("T", "body\nmore")]
    assert run._split_title_content("corpusqa", ["ctx1", "ctx2"]) == [
        ("Document 1", "ctx1"), ("Document 2", "ctx2")
    ]


# ============================================================
# _build_query — per benchmark
# ============================================================


def _fake_module(name: str, **attrs) -> ModuleType:
    mod = ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


def test_build_query_loong_is_faithful_prompt_template() -> None:
    """loong: upstream main.py's `prompt_template.format(instruction, question,
    docs="......")` — docs is a placeholder (the docs become the structured KB)."""
    bench = _fake_module(
        "evals.benchmarks.loong",
        get_task=lambda tid: ("INSTR", "QUESTION", ["d"]),
        get_prompt_template=lambda tid: "{docs}\n\n{instruction}\n\n{question}",
    )
    assert run._build_query(bench, "t") == "......\n\nINSTR\n\nQUESTION"


def test_build_query_loong_empty_question_template() -> None:
    bench = _fake_module(
        "evals.benchmarks.loong",
        get_task=lambda tid: ("INSTR", "", ["d"]),
        get_prompt_template=lambda tid: "{instruction}\n\n#Papers Provided:\n{docs}",
    )
    assert run._build_query(bench, "t") == "INSTR\n\n#Papers Provided:\n......"


def test_build_query_unknown_benchmark_raises() -> None:
    bench = _fake_module("evals.benchmarks.nosuchbench", get_task=lambda tid: ("a", "b", []))
    with pytest.raises(ValueError, match="no query assembly"):
        run._build_query(bench, "t")


# ============================================================
# loong.get_prompt_template accessor
# ============================================================


def test_loong_get_prompt_template_is_format_safe() -> None:
    from evals.benchmarks import loong

    tid = loong.get_task_ids(limit=1)[0]
    template = loong.get_prompt_template(tid)
    assert isinstance(template, str) and "{docs}" in template
    instruction, question, _docs = loong.get_task(tid)
    # The faithful StructRAG query construction must not raise on real templates.
    out = template.format(instruction=instruction, question=question, docs="......")
    assert "......" in out


# ============================================================
# StructRAGLLM — the LLM seam
# ============================================================


def test_structrag_llm_user_only_message_and_accumulates_usage() -> None:
    from evals.baselines.structrag.llm import StructRAGLLM

    llm = StructRAGLLM({"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None}, seed=42)
    with patch("evals.llm.chat.litellm.completion",
               side_effect=[_fake_completion("r1", total=12), _fake_completion("r2", total=23)]) as mock:
        assert llm.response("hello") == "r1"
        assert llm.response("world", max_new_tokens=100) == "r2"
    req1 = mock.call_args_list[0].kwargs
    assert req1["messages"] == [{"role": "user", "content": "hello"}]  # no system message
    assert req1["seed"] == 42 and req1["max_tokens"] == 32768  # completion budget (thinking models; not upstream's 4096)
    assert mock.call_args_list[1].kwargs["max_tokens"] == 100   # explicit override still honoured
    # usage accumulated across the two calls into the {total, calls} shape
    assert llm.usage["total"]["M"]["num_calls"] == 2
    assert llm.usage["total"]["M"]["total_tokens"] == 35
    assert len(llm.usage["calls"]) == 2


def test_structrag_llm_forwards_completion_params() -> None:
    from evals.baselines.structrag.llm import StructRAGLLM

    llm = StructRAGLLM({"model": "x", "api_base": None, "api_key": None}, seed=7,
                       completion_params={"temperature": 0.5, "extra_body": {"top_k": 20}})
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion("a")) as mock:
        llm.response("hi")
    req = mock.call_args.kwargs
    assert req["temperature"] == 0.5 and req["extra_body"] == {"top_k": 20}
    assert req["seed"] == 7


# ============================================================
# StructRAGLLM — REACTIVE truncate-to-fit (clip ONLY on a server length error)
# ============================================================


class _FakeTokenizer:
    """1 token per character (token count == len(text)) — for the precise-clip path."""

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=list(range(len(text))))


def test_parse_window_extracts_the_context_length() -> None:
    from evals.baselines.structrag.llm import _parse_window

    msg = ("This model's maximum context length is 262144 tokens. However, you requested "
           "65536 output tokens and your prompt contains at least 196609 input tokens.")
    assert _parse_window(msg) == 262144            # the ONLY thing we need from the message
    assert _parse_window("some unrelated error") is None


def test_no_proactive_clip_normal_prompt_sent_whole() -> None:
    """Reactive-only: a normal prompt is sent as-is, in ONE call, untouched."""
    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None})
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion("ok")) as mock:
        assert llm.response("a normal prompt") == "ok"
    assert len(mock.call_args_list) == 1
    assert mock.call_args.kwargs["messages"][-1]["content"] == "a normal prompt"


def test_reactive_clips_and_retries_on_length_error() -> None:
    """On a context-length rejection, clip the prompt (window read from the error) and
    retry — answering rather than erroring. Tokenizer-free (conservative char) path."""
    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None})
    overflow = Exception("This model's maximum context length is 300000 tokens. However, you "
                         "requested 32768 output tokens and your prompt contains at least 267000 input tokens.")
    with patch("evals.llm.chat.litellm.completion",
               side_effect=[overflow, _fake_completion("ok")]) as mock:
        assert llm.response("y" * 4000) == "ok"
    assert len(mock.call_args_list) == 2  # sent whole, rejected, then retried
    first = mock.call_args_list[0].kwargs["messages"][-1]["content"]
    second = mock.call_args_list[1].kwargs["messages"][-1]["content"]
    assert len(first) == 4000 and len(second) < len(first)  # untouched first, clipped on retry


def test_reactive_clip_is_precise_with_a_tokenizer() -> None:
    """When a tokenizer is loaded, the reactive clip targets window − max_tokens − margin."""
    from evals.baselines.structrag.llm import _CONTEXT_MARGIN

    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None})
    llm._tokenizer = _FakeTokenizer()  # 1 token/char
    overflow = Exception("This model's maximum context length is 10000 tokens. you requested too much.")
    with patch("evals.llm.chat.litellm.completion",
               side_effect=[overflow, _fake_completion("ok")]) as mock:
        assert llm.response("z" * 9000, max_new_tokens=2000) == "ok"
    second = mock.call_args_list[1].kwargs["messages"][-1]["content"]
    assert len(second) <= 10000 - 2000 - _CONTEXT_MARGIN  # clipped to window − max_tokens − margin


def test_non_context_error_is_reraised() -> None:
    """A non-length error is re-raised → recorded as error.json, not clipped/retried."""
    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None})
    with patch("evals.llm.chat.litellm.completion", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            llm.response("hi")


def test_full_calls_captured_for_calls_json() -> None:
    """Every call's full request + response is captured (→ calls.json)."""
    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None}, seed=42)
    with patch("evals.llm.chat.litellm.completion",
               side_effect=[_fake_completion("a"), _fake_completion("b")]):
        llm.response("first")
        llm.response("second")
    fc = llm.full_calls
    assert len(fc) == 2
    assert fc[0]["request"]["user_prompt"] == "first"
    assert fc[0]["request"]["max_tokens"] == 32768 and fc[0]["request"]["seed"] == 42
    assert "response" in fc[0]


def test_full_calls_surface_content_and_reasoning_explicitly() -> None:
    """Each calls.json entry carries the generated content as an explicit field (a
    reader shouldn't dig through the response dump); a thinking model's
    reasoning_content appears when present, and the field is omitted otherwise."""
    llm = run.StructRAGLLM({"model": "x", "api_base": None, "api_key": None})
    with patch("evals.llm.chat.litellm.completion",
               side_effect=[_fake_completion("plain answer"),
                            _fake_completion("final", reasoning="thinking step by step")]):
        llm.response("p1")
        llm.response("p2")
    fc = llm.full_calls
    assert fc[0]["content"] == "plain answer"
    assert "reasoning_content" not in fc[0]  # no noise field for non-thinking calls
    assert fc[1]["content"] == "final"
    assert fc[1]["reasoning_content"] == "thinking step by step"


# ============================================================
# run_one — drive the REAL vendored pipeline with a fake llm
# ============================================================


def _fake_llm_class(reply: str, calls: list):
    class FakeLLM:
        def __init__(self, litellm_kwargs, *, seed=None, completion_params=None, max_context_tokens=None):
            self.seed = seed

        def response(self, input_text, max_new_tokens=32768):
            calls.append(input_text)
            return reply

        @property
        def usage(self):
            return {"total": {"fake-model": {"num_calls": len(calls)}},
                    "calls": [{"model": "fake-model", "usage": {}} for _ in calls]}

        @property
        def full_calls(self):
            return [{"request": {"user_prompt": p}, "response": {"content": reply}} for p in calls]

    return FakeLLM


def _fake_loong(ndocs: int = 2) -> ModuleType:
    docs = [f"《Co{i}》\nbody text {i}\n\n" for i in range(ndocs)]
    return _fake_module(
        "evals.benchmarks.loong",
        get_documents=lambda tid: list(docs),
        get_task=lambda tid: ("compare the companies", "which is higher?", list(docs)),
        get_prompt_template=lambda tid: "{docs}\n\n{instruction}\n\n{question}",
    )


def test_run_one_graph_path_call_graph_and_record(tmp_path, monkeypatch) -> None:
    """reply='graph' → router picks graph → structurize is one call PER DOC. Asserts the
    exact per-task call count: 1 route + N_docs structurize + 1 decompose + 1 extract
    (one subquery) + 1 merge."""
    calls: list[str] = []
    monkeypatch.setattr(run, "StructRAGLLM", _fake_llm_class("graph", calls))
    bench = _fake_loong(ndocs=2)
    record = run.run_one(bench, "t1", {"seed": 42}, tmp_path,
                         {"model": "x", "api_base": None, "api_key": None})
    assert record["raw_answer"] == "graph"
    tr = record["trace"]
    assert tr["chosen"] == "graph"
    assert tr["query"] == "......\n\ncompare the companies\n\nwhich is higher?"
    assert len(calls) == 1 + 2 + 1 + 1 + 1  # route + 2 structurize + decompose + extract + merge
    assert record["usage"]["total"]["fake-model"]["num_calls"] == len(calls)
    assert tr["subqueries"] == ["graph"]
    # full per-call request/response captured for calls.json (one entry per LLM call)
    assert len(record["calls_full"]) == len(calls)
    assert record["calls_full"][0]["request"]["user_prompt"]  # non-empty


def test_run_one_chunk_path_no_structurize_calls(tmp_path, monkeypatch) -> None:
    """reply='chunk' → chunk structurize is NO LLM (just splits); extract is one call
    PER CHUNK/doc. 1 route + 0 structurize + 1 decompose + N_docs extract + 1 merge."""
    calls: list[str] = []
    monkeypatch.setattr(run, "StructRAGLLM", _fake_llm_class("chunk", calls))
    bench = _fake_loong(ndocs=3)
    record = run.run_one(bench, "t1", {"seed": 42}, tmp_path,
                         {"model": "x", "api_base": None, "api_key": None})
    assert record["raw_answer"] == "chunk"
    assert record["trace"]["chosen"] == "chunk"
    assert len(calls) == 1 + 0 + 1 + 3 + 1  # route + decompose + 3 extract + merge
    assert len(record["calls_full"]) == len(calls)  # one calls.json entry per LLM call


def test_run_one_uses_run_config_seed(tmp_path, monkeypatch) -> None:
    captured = {}

    def fake_cls(litellm_kwargs, *, seed=None, completion_params=None):
        captured["seed"] = seed
        captured["params"] = completion_params
        return _fake_llm_class("chunk", [])(litellm_kwargs, seed=seed, completion_params=completion_params)

    monkeypatch.setattr(run, "StructRAGLLM", fake_cls)
    run.run_one(_fake_loong(1), "t1", {"seed": 99, "completion_params": {"temperature": 0.3}},
                tmp_path, {"model": "x", "api_base": None, "api_key": None})
    assert captured["seed"] == 99
    assert captured["params"] == {"temperature": 0.3}


# ============================================================
# runner — argparse / run_config / child_cmd
# ============================================================


def _args(**over):
    argv = ["--benchmark", over.pop("benchmark", "loong")]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    return runner.build_arg_parser().parse_args(argv)


def test_argparse_rejects_unsupported_accepts_loong() -> None:
    assert runner.build_arg_parser().parse_args(["--benchmark", "loong"]).benchmark == "loong"
    with pytest.raises(SystemExit):
        runner.build_arg_parser().parse_args(["--benchmark", "nosuchbench"])


def test_build_run_config_shape_no_index_knobs() -> None:
    cfg = runner.build_run_config(_args(model="Qwen/Qwen3-1.7B", seed=7))
    assert cfg == {"benchmark": "loong", "baseline": "structrag", "model": "qwen3-1-7b", "seed": 7}


def test_build_run_config_with_config_folds_params() -> None:
    from evals.model_sampling import MODEL_SAMPLING_CONFIG

    name = next(iter(MODEL_SAMPLING_CONFIG))
    cfg = runner.build_run_config(_args(model="m", config=name))
    assert cfg["config_name"] == name
    assert cfg["completion_params"] == MODEL_SAMPLING_CONFIG[name]


def test_build_child_cmd_targets_structrag_module() -> None:
    cmd = runner.build_child_cmd(_args(model="m"), "task-42")
    assert "evals.baselines.structrag" in cmd
    assert cmd[cmd.index("--task-id") + 1] == "task-42"
    assert cmd[cmd.index("--benchmark") + 1] == "loong"
    assert "--search-method" not in cmd and "--embedding-model" not in cmd


def test_child_writes_calls_json_beside_manifest(tmp_path, monkeypatch) -> None:
    """`calls_full` is popped from the record and written to calls.json (NOT the
    manifest); the manifest keeps task_id/config/usage/raw_answer/trace."""
    import json as _json
    from types import SimpleNamespace

    monkeypatch.setattr(runner._common, "load_benchmark_module", lambda name: SimpleNamespace())
    monkeypatch.setattr(runner, "run_one", lambda **k: {
        "raw_answer": "x", "trace": {"chosen": "graph"},
        "usage": {"total": {}, "calls": []},
        "calls_full": [{"request": {"user_prompt": "the prompt"}, "response": {"content": "x"}}],
    })
    args = _args()
    rc = runner.build_run_config(args)
    runner._run_one_task(args, rc, tmp_path, "t1")

    inf = runner._common.inference_dir(tmp_path, runner._common.compute_inference_hash(rc, "t1"))
    manifest = _json.loads((inf / "manifest.json").read_text())
    assert "calls_full" not in manifest and manifest["raw_answer"] == "x"
    calls = _json.loads((inf / "calls.json").read_text())
    assert calls[0]["request"]["user_prompt"] == "the prompt"


# ============================================================
# Real-LLM smoke test — gpt-5.4-nano, deselected by default (`pytest -m structrag`)
# ============================================================


@pytest.mark.structrag
def test_smoke_structrag_pipeline_real(tmp_path) -> None:
    """The full vendored pipeline end-to-end against gpt-5.4-nano on a tiny multi-doc
    loong-shaped input. Asserts a non-empty answer + a multi-call usage record."""
    bench = _fake_module(
        "evals.benchmarks.loong",
        get_documents=lambda tid: [
            "《Acme Corp 2024》\nAcme Corp total revenue in 2024 was $500 million.\n\n",
            "《Globex Corp 2024》\nGlobex Corp total revenue in 2024 was $300 million.\n\n",
        ],
        get_task=lambda tid: (
            "Compare the two companies using the provided financial reports.",
            "Which company had higher total revenue in 2024?",
            None,
        ),
        get_prompt_template=lambda tid: "{docs}\n\n{instruction}\n\n{question}",
    )
    record = run.run_one(
        bench, "smoke-1", {"seed": 42}, tmp_path,
        {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None},
    )
    assert isinstance(record["raw_answer"], str) and record["raw_answer"].strip()
    assert record["trace"]["chosen"] in {"table", "graph", "algorithm", "catalogue", "chunk"}
    n_calls = sum(b.get("num_calls", 0) for b in record["usage"]["total"].values())
    assert n_calls >= 4, f"expected the multi-call pipeline, got {n_calls} calls"
    # full per-call request/response captured for calls.json (one entry per call)
    assert len(record["calls_full"]) == n_calls
    assert all("user_prompt" in c["request"] and "response" in c for c in record["calls_full"])
    assert len(record["usage"]["calls"]) == n_calls
