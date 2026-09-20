"""Tests for the MemAgent baseline connector.

MemAgent (BytedTsinghua-SIA / Seed, arXiv 2507.02259) ships its inference as a demo
(``quickstart.py``, vendored under ``evals/baselines/memagent/upstream/``); the connector
REPRODUCES the recurrent-memory loop faithfully. Harness-side pieces tested with fakes:

- ``prompts``  — the verbatim memory-update / final-answer templates + brace-safe builders.
- ``chunker``  — the fixed token-window split (+ head+tail clip), via a fake tokenizer.
- ``llm``      — MemAgentLLM, the single-user-message seam over litellm (usage + calls),
                 which KEEPS the ``max_tokens`` cap (load-bearing — bounds the memory).
- ``run``      — run_one's recurrent loop + the per-run-folder (TOTAL-cost, no-reuse) logging.

The ``memagent`` marker gates a real end-to-end run against a served ``RL-MemoryAgent-14B``
vLLM endpoint (``$MEMAGENT_BASE_URL``); it SKIPS when that isn't set (the model isn't on
OpenAI), so the default suite is fully fake-driven.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from evals.baselines.memagent import chunker, prompts


# ============================================================
# prompts — verbatim templates + brace-safe builders
# ============================================================


def test_memory_update_prompt_fills_all_slots() -> None:
    p = prompts.memory_update_prompt("PROBLEM", "MEMORY", "CHUNK")
    assert "PROBLEM" in p and "MEMORY" in p and "CHUNK" in p
    assert "<section>" in p and "Updated memory:" in p        # the memory-update template
    assert "update the memory with the new information" in p  # verbatim wording


def test_final_answer_prompt_keeps_boxed_and_fills_slots() -> None:
    p = prompts.final_answer_prompt("PROBLEM", "MEMORY")
    assert "PROBLEM" in p and "MEMORY" in p
    assert "\\boxed{}" in p and "Your answer:" in p           # verbatim \boxed{} contract
    assert "<section>" not in p                                # final template has no section


def test_prompt_builders_dont_break_on_braces_in_content() -> None:
    # Document / memory text may contain braces (CorpusQA tables, JSON golds). str.replace
    # substitution preserves them literally.
    problem = "compute f({x})"
    memory = "revenue = {a: 1, b: 2}"
    chunk = "table {rows: [...]}"
    p = prompts.memory_update_prompt(problem, memory, chunk)
    assert "f({x})" in p and "{a: 1, b: 2}" in p and "{rows: [...]}" in p
    f = prompts.final_answer_prompt(problem, memory)
    assert "f({x})" in f and "{a: 1, b: 2}" in f and "\\boxed{}" in f  # \boxed{} still literal


# ============================================================
# chunker — fixed token-window split (fake tokenizer)
# ============================================================


class _FakeTokenizer:
    """Char-level tokenizer: encode → code points, decode → chars. Exact round-trip, so
    ``chunk_size`` is in characters (deterministic for the tests)."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_split_into_chunks_windows_by_chunk_size() -> None:
    tok = _FakeTokenizer()
    assert chunker.split_into_chunks("abcdefghij", tok, chunk_size=4) == ["abcd", "efgh", "ij"]
    assert chunker.split_into_chunks("abc", tok, chunk_size=10) == ["abc"]     # one chunk
    assert chunker.split_into_chunks("", tok, chunk_size=4) == []              # empty → no chunks


def test_split_into_chunks_head_tail_clip() -> None:
    tok = _FakeTokenizer()
    # 10 chars, max_context_len=6 → head=3, keep ids[:3] + ids[-3:] = "abc" + "hij", THEN chunk.
    out = chunker.split_into_chunks("abcdefghij", tok, chunk_size=100, max_context_len=6)
    assert out == ["abchij"]
    # max_context_len=0 (default) → no clip.
    assert chunker.split_into_chunks("abcdefghij", tok, chunk_size=100, max_context_len=0) == ["abcdefghij"]


# ============================================================
# llm — MemAgentLLM seam over litellm
# ============================================================

from evals.baselines.memagent import llm as ma_llm  # noqa: E402


def _fake_response(content, *, model="rl-memoryagent-14b", ptoks=11, ctoks=7, reasoning=None):
    usage = SimpleNamespace(
        prompt_tokens=ptoks, completion_tokens=ctoks, total_tokens=ptoks + ctoks,
        model_dump=lambda: {"prompt_tokens": ptoks, "completion_tokens": ctoks,
                            "total_tokens": ptoks + ctoks},
    )
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        model=model, usage=usage, choices=[SimpleNamespace(message=msg)],
        model_dump=lambda: {"model": model, "choices": [{"message": {"content": content}}]},
    )


def test_memagentllm_sends_max_tokens_and_one_user_message(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    llm = ma_llm.MemAgentLLM(
        {"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B", "api_base": "http://x/v1", "api_key": "k"},
        seed=42, completion_params={"temperature": 0.3}, max_new=777,
    )
    assert llm.complete("hello") == "ok"
    # One user message (no system role); the max_tokens cap IS sent (load-bearing — bounds memory).
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["max_tokens"] == 777
    assert captured["seed"] == 42 and captured["temperature"] == 0.3
    assert captured["api_base"] == "http://x/v1" and captured["api_key"] == "k"


def test_memagentllm_accumulates_total_usage_and_full_calls(monkeypatch) -> None:
    import litellm
    responses = iter([_fake_response("a", ptoks=10, ctoks=5),
                      _fake_response("b", ptoks=20, ctoks=8, reasoning="thinking...")])
    monkeypatch.setattr(litellm, "completion", lambda **kw: next(responses))
    llm = ma_llm.MemAgentLLM({"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B"})
    llm.complete("p1")
    llm.complete("p2")
    usage = llm.usage
    assert usage["total"]["rl-memoryagent-14b"]["num_calls"] == 2
    assert usage["total"]["rl-memoryagent-14b"]["total_tokens"] == (15 + 28)
    assert [c["content"] for c in llm.full_calls] == ["a", "b"]
    assert llm.full_calls[1]["reasoning_content"] == "thinking..."
    assert llm.full_calls[0]["request"]["prompt"] == "p1"
    assert llm.full_calls[0]["request"]["max_tokens"] == ma_llm._MAX_NEW


# ============================================================
# run — run_one recurrent loop + per-run-folder logging
# ============================================================

from evals.baselines.memagent import run as ma_run  # noqa: E402


def _loong_benchmark():
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: ["Paris is the capital of France.",
                                    "Berlin is the capital of Germany."],
        get_task=lambda tid: ("Answer using the documents.", "What is the capital of France?", []),
    )


def _corpusqa_benchmark():
    return SimpleNamespace(
        __name__="evals.benchmarks.corpusqa",
        get_documents=lambda tid: ["doc one", "doc two"],
        get_task=lambda tid: ("Output requirements: The answer is: xxx", "How many?", []),
    )


class _StageCompletion:
    """Stage-aware fake litellm.completion: 'memory' for each memory-update call (has
    ``<section>``), 'answer' for the final call."""

    def __init__(self):
        self.stages: list[str] = []

    def __call__(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "<section>" in prompt:
            stage, content = "memory", "MEM"
        else:
            stage, content = "answer", "\\boxed{Paris}"
        self.stages.append(stage)
        return _fake_response(content)


def _run_config(**over):
    cfg = {"benchmark": "loong", "baseline": "memagent", "model": "rl-memoryagent-14b",
           "seed": 42, "chunk_size": 40, "max_new": 1024, "max_context_len": 0,
           "run_version": ma_run._RUN_VERSION}
    cfg.update(over)
    return cfg


def test_run_one_recurrent_loop_and_per_run_folder_logging(tmp_path, monkeypatch) -> None:
    import litellm
    stages = _StageCompletion()
    monkeypatch.setattr(litellm, "completion", stages)
    monkeypatch.setattr(ma_run, "_load_tokenizer", lambda mid: _FakeTokenizer())

    litellm_kwargs = {"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B",
                      "api_base": "http://x/v1", "api_key": "k"}
    rec = ma_run.run_one(_loong_benchmark(), "t1", _run_config(), tmp_path, litellm_kwargs)

    # Pooled context = 66 chars, chunk_size 40 → 2 chunks → 2 memory updates + 1 answer.
    assert stages.stages == ["memory", "memory", "answer"]
    assert rec["raw_answer"] == "\\boxed{Paris}"
    assert "index_ref" not in rec                              # no content-addressed index
    assert rec["trace"]["n_chunks"] == 2 and rec["trace"]["n_docs"] == 2
    assert "capital of France" in rec["trace"]["problem"]      # loong composition
    # TOTAL usage (every memory update + the answer) + full call trace.
    assert rec["usage"]["total"] and len(rec["calls_full"]) == 3
    # The memory trajectory is persisted INSIDE the run folder (every intermediate memory).
    traj = json.loads((tmp_path / "memory_trajectory.json").read_text())
    assert traj["n_chunks"] == 2 and len(traj["memories"]) == 2
    assert traj["final_memory"] == "MEM" and traj["chunk_size"] == 40
    # Live progress.json reached the final stage with per-chunk counts.
    prog = json.loads((tmp_path / "progress.json").read_text())
    assert prog["stage"] == "answered" and prog["task_id"] == "t1"
    assert prog["n_chunks"] == 2 and prog["chunks_done"] == 2 and "elapsed_s" in prog


def test_run_one_corpusqa_problem_puts_question_then_requirements(tmp_path, monkeypatch) -> None:
    import litellm
    monkeypatch.setattr(litellm, "completion", _StageCompletion())
    monkeypatch.setattr(ma_run, "_load_tokenizer", lambda mid: _FakeTokenizer())
    rec = ma_run.run_one(
        _corpusqa_benchmark(), "t1", _run_config(benchmark="corpusqa"), tmp_path,
        {"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B", "api_base": "http://x/v1", "api_key": "k"},
    )
    problem = rec["trace"]["problem"]
    assert problem.startswith("How many?")                    # question first
    assert "The answer is: xxx" in problem                     # output-requirements block reaches the model


def test_run_one_dracula_problem_is_bare_question(tmp_path, monkeypatch) -> None:
    import litellm
    monkeypatch.setattr(litellm, "completion", _StageCompletion())
    monkeypatch.setattr(ma_run, "_load_tokenizer", lambda mid: _FakeTokenizer())
    dracula = SimpleNamespace(
        __name__="evals.benchmarks.dracula",
        get_documents=lambda tid: ["doc one", "doc two"],
        get_task=lambda tid: ("How many people died?", []),
    )
    rec = ma_run.run_one(
        dracula, "death_toll", _run_config(benchmark="dracula"), tmp_path,
        {"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B", "api_base": "http://x/v1", "api_key": "k"},
    )
    assert rec["trace"]["problem"] == "How many people died?"  # bare question, no docs in the problem


def test_run_one_requires_base_url_for_vllm_model(tmp_path, monkeypatch) -> None:
    # The RL default model (hosted_vllm/) needs a --base-url; run_one refuses without one,
    # BEFORE any tokenizer load or LLM call.
    monkeypatch.setattr(ma_run, "_load_tokenizer", lambda mid: pytest.fail("tokenizer loaded"))
    with pytest.raises(ValueError, match="vLLM endpoint"):
        ma_run.run_one(_loong_benchmark(), "t1", _run_config(),
                       tmp_path, {"model": "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B", "api_base": None})


def test_strip_provider_gives_hf_repo_id() -> None:
    assert ma_run._strip_provider("hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B") == \
        "BytedTsinghua-SIA/RL-MemoryAgent-14B"
    assert ma_run._strip_provider("openai/gpt-4o") == "gpt-4o"
    assert ma_run._strip_provider("BytedTsinghua-SIA/RL-MemoryAgent-14B") == \
        "BytedTsinghua-SIA/RL-MemoryAgent-14B"


# ============================================================
# runner — config shape + chunk knobs in the identity
# ============================================================

from evals.baselines.memagent import runner as ma_runner  # noqa: E402


def test_build_run_config_includes_chunk_knobs_and_default_model() -> None:
    args = ma_runner.build_arg_parser().parse_args(["--benchmark", "loong"])
    assert args.model == ma_runner.DEFAULT_MODEL                      # the RL-MemoryAgent-14B default
    cfg = ma_runner.build_run_config(args)
    assert cfg["baseline"] == "memagent" and cfg["model"] == "rl-memoryagent-14b"
    # chunk_size / max_new / max_context_len ARE part of the run identity (change output → new run).
    assert cfg["chunk_size"] == 5000 and cfg["max_new"] == 1024 and cfg["max_context_len"] == 0
    assert cfg["run_version"] == ma_run._RUN_VERSION


def test_child_cmd_forwards_chunk_knobs() -> None:
    args = ma_runner.build_arg_parser().parse_args(
        ["--benchmark", "corpusqa", "--chunk-size", "8000", "--max-new", "2048",
         "--base-url", "http://x/v1"])
    cmd = ma_runner.build_child_cmd(args, "t1", "tag")
    assert cmd[cmd.index("--chunk-size") + 1] == "8000"
    assert cmd[cmd.index("--max-new") + 1] == "2048"
    assert cmd[cmd.index("--base-url") + 1] == "http://x/v1"
    assert ma_runner._litellm_kwargs(args)["model"] == "hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B"


# --- resumption: a re-run must NOT redo finished tasks ---


def _patch_runner_for_resumption(tmp_path, monkeypatch):
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

    monkeypatch.setattr(ma_runner, "_dispatch", fake_dispatch)
    return dispatched


def test_runner_resumes_and_does_not_rerun_completed(tmp_path, monkeypatch) -> None:
    dispatched = _patch_runner_for_resumption(tmp_path, monkeypatch)
    ma_runner.main(["--benchmark", "loong", "--base-url", "http://x/v1"])
    assert sorted(dispatched) == ["t1", "t2", "t3"]
    dispatched.clear()
    ma_runner.main(["--benchmark", "loong", "--base-url", "http://x/v1"])
    assert dispatched == []   # all done for this config → nothing re-run


# ============================================================
# Live wire-test (marker-gated) — real RL-MemoryAgent-14B vLLM, 1 loong task
# ============================================================


@pytest.mark.memagent
def test_memagent_end_to_end_loong_real(tmp_path) -> None:
    """Drive the REAL recurrent-memory loop on one Loong task against a served
    ``RL-MemoryAgent-14B`` vLLM endpoint. SKIPS unless ``$MEMAGENT_BASE_URL`` is set (the RL
    model isn't on OpenAI) — run once the operator's SSH tunnel to vLLM is up:

        MEMAGENT_BASE_URL=http://localhost:8000/v1 MEMAGENT_API_KEY=... pytest -m memagent
    """
    base_url = os.getenv("MEMAGENT_BASE_URL")
    if not base_url:
        pytest.skip("set $MEMAGENT_BASE_URL (a served RL-MemoryAgent-14B vLLM) to run this")
    from evals.baselines import _common
    from evals.baselines.memagent.run import run_one
    from evals.baselines.memagent.runner import DEFAULT_MODEL, _DEFAULT_CHUNK_SIZE, _DEFAULT_MAX_NEW

    model = os.getenv("MEMAGENT_MODEL", DEFAULT_MODEL)
    benchmark = _common.load_benchmark_module("loong")
    task_id = benchmark.get_task_ids(languages=["en"], limit=1)[0]
    run_config = {"benchmark": "loong", "baseline": "memagent",
                  "model": _common.canonical_model_id(model), "seed": 42,
                  "chunk_size": _DEFAULT_CHUNK_SIZE, "max_new": _DEFAULT_MAX_NEW,
                  "max_context_len": 0, "run_version": run_one.__module__}
    litellm_kwargs = {"model": _common.with_provider_prefix(model),
                      "api_base": base_url, "api_key": os.getenv("MEMAGENT_API_KEY", "x")}

    rec = run_one(benchmark, task_id, run_config, tmp_path, litellm_kwargs)

    assert isinstance(rec["raw_answer"], str) and rec["raw_answer"].strip()
    assert (tmp_path / "memory_trajectory.json").exists()
    assert rec["usage"]["total"] and rec["calls_full"]
    assert rec["trace"]["n_chunks"] >= 1
