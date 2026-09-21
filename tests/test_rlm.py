"""Tests for the RLM (Recursive Language Models) baseline connector.

RLM (the ``rlms`` PyPI package) is a normal dependency (the ``evals[rlms]`` extra), not vendored.
The harness-side pieces are tested here with fakes:

- ``run``  — _strip_provider / _build_context / _build_task, and a run_one that drives the REAL RLM
             pipeline (REPL loop + ``answer`` protocol) with only the LLM call faked.
- ``llm``  — RLMUsageCapture, the client-level monkeypatch that counts EVERY call's tokens + logs.
- ``runner`` — the no-OpenAI guard, the config shape, and resumption.

The fake LLM emits a ``repl`` code block that submits the answer via ``answer["content"]`` +
``answer["ready"] = True`` (RLM's real completion protocol), so the real loop/REPL/parser run. The
``rlm`` marker gates a real end-to-end run against a live vLLM endpoint (deselected by default).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from evals.baselines.rlm import run as rlm_run


# ============================================================
# run — pure helpers
# ============================================================


def test_strip_provider() -> None:
    assert rlm_run._strip_provider("hosted_vllm/Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"
    assert rlm_run._strip_provider("openai/gpt-5.4-nano") == "gpt-5.4-nano"
    assert rlm_run._strip_provider("Qwen/Qwen3.5-35B-A3B") == "Qwen/Qwen3.5-35B-A3B"  # bare unchanged


def test_build_context_single_and_multi() -> None:
    assert rlm_run._build_context(["only doc"]) == "only doc"   # single doc verbatim
    multi = rlm_run._build_context(["A", "B"])
    assert "=== Document 1 ===\nA" in multi and "=== Document 2 ===\nB" in multi


def test_build_task_loong_and_corpusqa() -> None:
    loong = SimpleNamespace(__name__="evals.benchmarks.loong", get_task=lambda t: ("INSTR", "Q", ["d"]))
    loong_empty = SimpleNamespace(__name__="evals.benchmarks.loong",
                                  get_task=lambda t: ("INSTR-ONLY", "  ", ["d"]))
    corpusqa = SimpleNamespace(__name__="evals.benchmarks.corpusqa",
                               get_task=lambda t: ("OUTPUT-REQS", "Q2", ["d"]))
    assert rlm_run._build_task(loong, "t") == "INSTR\n\nQ"
    assert rlm_run._build_task(loong_empty, "t") == "INSTR-ONLY"
    assert rlm_run._build_task(corpusqa, "t") == "Q2\n\nOUTPUT-REQS"


def test_build_task_dracula() -> None:
    dracula = SimpleNamespace(__name__="dracula", get_task=lambda t: ("Q-DRAC", ["d1", "d2"]))
    assert rlm_run._build_task(dracula, "t") == "Q-DRAC"  # bare question; docs go to the REPL context


def test_backend_kwargs_injects_seed_and_config() -> None:
    lk = {"model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B",
          "api_base": "http://localhost:8555/v1", "api_key": "k"}
    # Default path: seed injected into generation (reproducibility); provider prefix stripped for vLLM.
    kw = rlm_run._backend_kwargs(lk, {"seed": 42})
    assert kw["model_name"] == "Qwen/Qwen3.5-35B-A3B"
    assert kw["base_url"] == "http://localhost:8555/v1" and kw["api_key"] == "k"
    assert kw["sampling_args"] == {"seed": 42}
    # seed defaults to 42 when run_config doesn't carry one.
    assert rlm_run._backend_kwargs(lk, {})["sampling_args"]["seed"] == 42
    # run-config completion_params merge with the seed.
    kw2 = rlm_run._backend_kwargs(lk, {"seed": 7, "completion_params": {"temperature": 0.7}})
    assert kw2["sampling_args"] == {"temperature": 0.7, "seed": 7}


# ============================================================
# llm — RLMUsageCapture (client-level token + call capture)
# ============================================================

from evals.baselines.rlm import llm as rlm_llm  # noqa: E402


def _fake_response(content, *, ptoks=100, ctoks=20, reasoning=None):
    usage = SimpleNamespace(prompt_tokens=ptoks, completion_tokens=ctoks, total_tokens=ptoks + ctoks)
    msg = SimpleNamespace(content=content, reasoning_content=reasoning, tool_calls=None)
    return SimpleNamespace(usage=usage, choices=[SimpleNamespace(message=msg)], model="Qwen/Q")


def test_usage_capture_records_every_call() -> None:
    from rlm.clients.openai import OpenAIClient

    # A real client (no network until completion); we replace completion with a fake that calls
    # the real _track_cost (so RLMUsageCapture's wrapper fires) and returns the content.
    client = OpenAIClient(model_name="Qwen/Q", base_url="http://localhost:8555/v1", api_key="x")

    def fake_completion(self, prompt, model=None):
        resp = _fake_response("hello", ptoks=100, ctoks=20, reasoning="thinking")
        self._track_cost(resp, model or self.model_name)
        return resp.choices[0].message.content

    orig = OpenAIClient.completion
    OpenAIClient.completion = fake_completion
    try:
        with rlm_llm.RLMUsageCapture() as cap:
            client.completion("q1")
            client.completion("q2")
        usage = cap.usage
        assert usage["total"]["Qwen/Q"]["num_calls"] == 2
        assert usage["total"]["Qwen/Q"]["prompt_tokens"] == 200
        assert usage["total"]["Qwen/Q"]["completion_tokens"] == 40
        assert len(cap.full_calls) == 2
        assert cap.full_calls[0]["content"] == "hello"
        assert cap.full_calls[0]["reasoning_content"] == "thinking"
        assert cap.full_calls[0]["request"]["messages"] == [{"role": "user", "content": "q1"}]
    finally:
        OpenAIClient.completion = orig


def test_usage_capture_snapshots_messages_not_reference() -> None:
    # RLM mutates its message_history list in place across turns; each captured call must record
    # the prompt AT THAT CALL, not a reference that later shows the final history.
    from rlm.clients.openai import OpenAIClient

    client = OpenAIClient(model_name="Qwen/Q", base_url="http://localhost:8555/v1", api_key="x")
    history = [{"role": "system", "content": "sys"}, {"role": "user", "content": "turn1"}]

    def fake_completion(self, prompt, model=None):
        resp = _fake_response("ok")
        self._track_cost(resp, model or self.model_name)
        return resp.choices[0].message.content

    orig = OpenAIClient.completion
    OpenAIClient.completion = fake_completion
    try:
        with rlm_llm.RLMUsageCapture() as cap:
            client.completion(history)                                  # call 1: 2 messages
            history.append({"role": "assistant", "content": "a1"})      # RLM-style in-place growth
            history.append({"role": "user", "content": "turn2"})
            client.completion(history)                                  # call 2: 4 messages
        assert len(cap.full_calls[0]["request"]["messages"]) == 2       # frozen at call time
        assert len(cap.full_calls[1]["request"]["messages"]) == 4
        assert cap.full_calls[0]["request"]["messages"][-1]["content"] == "turn1"
    finally:
        OpenAIClient.completion = orig


# ============================================================
# run — run_one over the REAL RLM pipeline (only the LLM faked)
# ============================================================

# The model's turn: a repl block that submits the final answer (RLM's real completion protocol).
_ANSWER_TURN = (
    'I will read the context and answer.\n'
    '```repl\n'
    'assert "context" in dir()  # the documents are in the `context` variable\n'
    'answer["content"] = "PARIS"\n'
    'answer["ready"] = True\n'
    '```'
)


def _loong_benchmark():
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: ["Paris is the capital of France.",
                                   "Berlin is the capital of Germany."],
        get_task=lambda tid: ("Answer using the documents.", "What is the capital of France?", []),
    )


def test_run_one_drives_real_pipeline(tmp_path) -> None:
    from rlm.clients.openai import OpenAIClient

    def fake_completion(self, prompt, model=None):
        resp = _fake_response(_ANSWER_TURN, ptoks=300, ctoks=40)
        self._track_cost(resp, model or self.model_name)
        return resp.choices[0].message.content

    orig = OpenAIClient.completion
    OpenAIClient.completion = fake_completion
    try:
        run_config = {"benchmark": "loong", "baseline": "rlm",
                      "model": "qwen-q", "seed": 42, "max_iterations": 4, "max_depth": 1,
                      "max_timeout": None, "run_version": rlm_run._RUN_VERSION}
        litellm_kwargs = {"model": "Qwen/Qwen3.5-35B-A3B",
                          "api_base": "http://localhost:8555/v1", "api_key": "x"}
        rec = rlm_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path, litellm_kwargs)
    finally:
        OpenAIClient.completion = orig

    assert rec["raw_answer"] == "PARIS"                       # the REPL `answer` protocol worked
    assert "index_ref" not in rec                             # flat layout, no content-addressed index
    # Complete token capture (the bare vLLM model id, not the litellm-prefixed one).
    assert rec["usage"]["total"]["Qwen/Qwen3.5-35B-A3B"]["prompt_tokens"] == 300
    assert rec["usage"]["calls"] and rec["calls_full"]
    # RLM's full trajectory is persisted LIVE as the native RLMLogger jsonl; we no longer dump a
    # redundant rlm_trajectory.json (it was a byte-for-byte second copy of the same trajectory).
    assert list(tmp_path.glob("rlm_*.jsonl"))
    assert not (tmp_path / "rlm_trajectory.json").exists()
    assert rec["trace"]["n_docs"] == 2 and "capital of France" in rec["trace"]["task"]


# ============================================================
# logger — trajectory without the REPL-variable snapshots
# ============================================================

from evals.baselines.rlm import logger as rlm_logger  # noqa: E402


def test_drop_repl_locals_keeps_trajectory_and_recurses() -> None:
    child = {"code_blocks": [{"code": "c", "result": {"stdout": "cs", "locals": {"context": "BIG"}}}]}
    entry = {
        "type": "iteration", "prompt": [{"role": "user", "content": "p"}], "response": "r",
        "code_blocks": [{"code": "x = 1", "result": {
            "stdout": "out", "stderr": "", "locals": {"context": "BIG", "x": 1},
            "execution_time": 0.1, "final_answer": None,
            "rlm_calls": [{"prompt": "sub", "response": "ans", "metadata": {"iterations": [child]}}],
        }}],
        "final_answer": None,
    }
    rlm_logger.drop_repl_locals(entry)
    result = entry["code_blocks"][0]["result"]
    assert "locals" not in result and "locals" not in child["code_blocks"][0]["result"]
    assert result["stdout"] == "out" and result["rlm_calls"][0]["response"] == "ans"
    assert entry["prompt"] and entry["response"] == "r" and entry["code_blocks"][0]["code"] == "x = 1"


def test_run_one_trajectory_has_no_locals_and_model_input_is_unchanged(tmp_path) -> None:
    # Two turns over the REAL loop: turn 1 creates a variable and prints, turn 2 submits. The logged
    # trajectory must drop `locals` but keep code/stdout — and what the model SEES on turn 2 (built
    # by RLM from the live REPLResult, incl. the variable names) must be untouched by the logger.
    from rlm.clients.openai import OpenAIClient

    turns = iter([
        "```repl\nx = 41\nprint(x + 1)\n```",
        '```repl\nanswer["content"] = "42"\nanswer["ready"] = True\n```',
    ])

    def fake_completion(self, prompt, model=None):
        resp = _fake_response(next(turns))
        self._track_cost(resp, model or self.model_name)
        return resp.choices[0].message.content

    orig = OpenAIClient.completion
    OpenAIClient.completion = fake_completion
    try:
        run_config = {"benchmark": "loong", "seed": 42, "max_iterations": 4, "max_depth": 1}
        litellm_kwargs = {"model": "Qwen/Q", "api_base": "http://localhost:8555/v1", "api_key": "x"}
        rec = rlm_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path, litellm_kwargs)
    finally:
        OpenAIClient.completion = orig

    assert rec["raw_answer"] == "42" and rec["trace"]["n_iterations"] == 2
    lines = [json.loads(l) for l in next(tmp_path.glob("rlm_*.jsonl")).read_text().splitlines()]
    assert lines[0]["type"] == "metadata"
    first = lines[1]["code_blocks"][0]
    assert first["code"] == "x = 41\nprint(x + 1)" and first["result"]["stdout"] == "42\n"
    assert all("locals" not in b["result"] for it in lines[1:] for b in it["code_blocks"])
    turn2_input = rec["calls_full"][1]["request"]["messages"][-2]["content"]
    assert "42" in turn2_input and "REPL variables:" in turn2_input and "'x'" in turn2_input


def test_run_one_rejects_openai_endpoint(tmp_path) -> None:
    with pytest.raises(ValueError, match="non-OpenAI"):
        rlm_run.run_one(_loong_benchmark(), "t1", {"benchmark": "loong"}, tmp_path,
                        {"model": "gpt", "api_base": "https://api.openai.com/v1", "api_key": "x"})


# ============================================================
# runner — no-OpenAI guard, config shape, resumption
# ============================================================

from evals.baselines.rlm import runner as rlm_runner  # noqa: E402


def test_runner_rejects_openai_and_missing_base_url() -> None:
    p = rlm_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "loong", "--model", "Qwen/Q", "--base-url", "https://api.openai.com/v1"])
    with pytest.raises(SystemExit):
        rlm_runner._validate_endpoint(a)
    # --base-url is required by argparse (no default).
    with pytest.raises(SystemExit):
        p.parse_args(["--benchmark", "loong", "--model", "Qwen/Q"])


def test_build_run_config_shape() -> None:
    p = rlm_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "corpusqa", "--model", "Qwen/Q",
                      "--base-url", "http://localhost:8555/v1", "--seed", "7", "--max-iterations", "12"])
    cfg = rlm_runner.build_run_config(a)
    assert cfg["benchmark"] == "corpusqa" and cfg["baseline"] == "rlm"
    assert cfg["seed"] == 7 and cfg["max_iterations"] == 12 and cfg["max_depth"] == 1
    assert cfg["run_version"] == rlm_run._RUN_VERSION


def test_runner_resumes_and_does_not_rerun_completed(tmp_path, monkeypatch) -> None:
    from evals.baselines import _common as common
    fake = SimpleNamespace(get_task_ids=lambda **k: ["t1", "t2", "t3"], STARTER_FILTER={})
    monkeypatch.setattr(common, "load_benchmark_module", lambda n: fake)
    monkeypatch.setattr(common, "base_dir", lambda b, bl: tmp_path)
    dispatched: list[str] = []

    def fake_dispatch(args, run_config, base, tid, run_tag):
        dispatched.append(tid)
        d = base / run_tag
        d.mkdir(parents=True, exist_ok=True)
        common.write_manifest(d, {"task_id": tid, "config": run_config})
        return "ok"

    monkeypatch.setattr(rlm_runner, "_dispatch", fake_dispatch)
    argv = ["--benchmark", "loong", "--model", "Qwen/Q", "--base-url", "http://localhost:8555/v1"]
    rlm_runner.main(argv)
    assert sorted(dispatched) == ["t1", "t2", "t3"]   # first run: all 3
    dispatched.clear()
    rlm_runner.main(argv)
    assert dispatched == []                            # re-run: all done → nothing re-run


# ============================================================
# Live wire-test (marker-gated) — real vLLM, 1 loong task
# ============================================================


@pytest.mark.rlm
def test_rlm_end_to_end_loong_real(tmp_path) -> None:
    """Drive the REAL RLM pipeline (REPL code-agent + recursive sub-calls) on one Loong task against
    a live vLLM endpoint (default localhost:8555, override via RLM_BASE_URL/RLM_MODEL/RLM_API_KEY).
    Token-heavy; deselected by default (``-m rlm``)."""
    from evals.baselines import _common
    from evals.baselines.rlm.run import run_one

    base_url = os.getenv("RLM_BASE_URL", "http://localhost:8555/v1")
    model = os.getenv("RLM_MODEL", "Qwen/Qwen3.5-35B-A3B")
    api_key = os.getenv("RLM_API_KEY", "your_secret")

    benchmark = _common.load_benchmark_module("loong")
    task_id = benchmark.get_task_ids(languages=["en"], sets=[1], limit=1)[0]
    run_config = {"benchmark": "loong", "baseline": "rlm",
                  "model": _common.canonical_model_id(model), "seed": 42,
                  "max_iterations": 12, "max_depth": 1, "max_timeout": 600, "run_version": "test"}
    litellm_kwargs = {"model": model, "api_base": base_url, "api_key": api_key}

    rec = run_one(benchmark, task_id, run_config, tmp_path, litellm_kwargs)

    assert isinstance(rec["raw_answer"], str) and rec["raw_answer"].strip()
    assert list(tmp_path.glob("rlm_*.jsonl"))   # native trajectory; no redundant rlm_trajectory.json
    assert rec["usage"]["total"] and rec["calls_full"]
