"""Tests for the CodeAct (smolagents ``CodeAgent``) baseline connector.

smolagents is a normal dependency (the ``evals[codeact]`` extra), not vendored. The harness-side
pieces are tested here with fakes:

- ``run``  — _build_task / _docs_note / _benchmark_name / _model_call_kwargs, and a run_one that
             drives the REAL smolagents CodeAgent loop (ReAct + local Python executor + final_answer)
             with only ``litellm.completion`` faked, so the real agent/executor/parsing run.
- ``llm``  — CodeActModel, the LiteLLMModel subclass that deterministically records EVERY
             completion's ``{total, calls}`` usage (incl. nested token details) + full calls.json.
- ``runner`` — the config shape (incl. max_steps) and resumption.

The fake model emits ``<code>…</code>`` blobs (smolagents' default code-block tags): step 1 reads the
offloaded ``documents`` variable, step 2 calls ``final_answer(...)`` — so the real loop/executor/parse
run. The ``codeact`` marker gates a real end-to-end run against gpt-5.4-nano (deselected by default).
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest

from evals.baselines.codeact import run as ca_run


# ============================================================
# run — pure helpers
# ============================================================


def test_build_task_loong_and_corpusqa() -> None:
    loong = SimpleNamespace(__name__="evals.benchmarks.loong", get_task=lambda t: ("INSTR", "Q", ["d"]))
    loong_empty = SimpleNamespace(__name__="evals.benchmarks.loong",
                                  get_task=lambda t: ("INSTR-ONLY", "  ", ["d"]))
    corpusqa = SimpleNamespace(__name__="evals.benchmarks.corpusqa",
                               get_task=lambda t: ("OUTPUT-REQS", "Q2", ["d"]))
    assert ca_run._build_task(loong, "t") == "INSTR\n\nQ"
    assert ca_run._build_task(loong_empty, "t") == "INSTR-ONLY"        # empty question → instruction only
    assert ca_run._build_task(corpusqa, "t") == "Q2\n\nOUTPUT-REQS"    # output-reqs after the question


def test_docs_note_points_at_variable_not_dump() -> None:
    note = ca_run._docs_note(7)
    assert "`documents`" in note and "7 source-document" in note
    # The note must NEVER carry doc contents (the whole point of the offload): it's a fixed pointer.
    assert "list of 7" in note


def test_build_task_dracula() -> None:
    dracula = SimpleNamespace(__name__="dracula", get_task=lambda t: ("Q-DRAC", ["d1", "d2"]))
    assert ca_run._build_task(dracula, "t") == "Q-DRAC"   # bare question; docs go to the sandbox var


def test_build_task_rejects_unsupported_benchmark() -> None:
    other = SimpleNamespace(__name__="evals.benchmarks.nosuchbench", get_task=lambda t: ("a", "b", "c"))
    with pytest.raises(ValueError, match="no task assembly"):
        ca_run._build_task(other, "t")


def test_model_call_kwargs_injects_seed_and_config() -> None:
    assert ca_run._model_call_kwargs({"seed": 42}) == {"seed": 42}
    assert ca_run._model_call_kwargs({})["seed"] == 42                  # default 42 when absent
    kw = ca_run._model_call_kwargs({"seed": 7, "completion_params": {"temperature": 0.7,
                                                                     "extra_body": {"top_k": 20}}})
    assert kw == {"temperature": 0.7, "extra_body": {"top_k": 20}, "seed": 7}


# ============================================================
# llm — CodeActModel (deterministic per-call usage + calls capture)
# ============================================================

from evals.baselines.codeact import llm as ca_llm  # noqa: E402


class _FakeUsage:
    """A litellm-usage-like object: attribute access (smolagents reads .prompt_tokens) AND a
    model_dump() carrying nested *_tokens_details (so we prove the rollup sums them)."""

    def __init__(self, p: int, c: int, reasoning: int = 0) -> None:
        self.prompt_tokens, self.completion_tokens, self.total_tokens = p, c, p + c
        self._reasoning = reasoning

    def model_dump(self) -> dict:
        return {"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "completion_tokens_details": {"reasoning_tokens": self._reasoning}}


def _fake_resp(content: str, usage: _FakeUsage, model: str = "gpt-fake", reasoning=None):
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=None, reasoning_content=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage, model=model,
                           model_dump=lambda: {})


class _FakeClient:
    """A drop-in for the litellm module's ``.completion`` (LiteLLMModel.create_client returns the
    litellm module; ApiModel accepts ``client=`` to bypass it)."""

    def __init__(self, scripts: list[tuple[str, _FakeUsage]]) -> None:
        self._scripts, self._i = scripts, 0

    def completion(self, **kwargs):  # noqa: ANN003
        content, usage = self._scripts[self._i]
        self._i += 1
        return _fake_resp(content, usage, reasoning="thinking" if self._i == 1 else None)


def test_codeact_model_captures_usage_and_calls() -> None:
    model = ca_llm.CodeActModel(
        model_id="openai/gpt-fake",
        client=_FakeClient([("ans1", _FakeUsage(100, 20, reasoning=5)),
                            ("ans2", _FakeUsage(50, 10, reasoning=3))]),
    )
    model.generate([{"role": "user", "content": "q1"}])
    model.generate([{"role": "user", "content": "q2"}])

    usage = model.usage
    bucket = usage["total"]["gpt-fake"]
    assert bucket["num_calls"] == 2
    assert bucket["prompt_tokens"] == 150 and bucket["completion_tokens"] == 30
    # nested *_tokens_details are summed by the same _merge_numeric the rest of the project uses
    assert bucket["completion_tokens_details"]["reasoning_tokens"] == 8
    assert len(usage["calls"]) == 2

    calls = model.full_calls
    assert calls[0]["content"] == "ans1" and calls[0]["reasoning_content"] == "thinking"
    assert calls[1]["content"] == "ans2" and calls[1]["reasoning_content"] is None
    # request messages are snapshotted (what THAT call sent)
    assert calls[0]["request"]["messages"] == [{"role": "user", "content": "q1"}]


# ============================================================
# run — run_one over the REAL CodeAgent (only litellm faked)
# ============================================================

# The model's two turns, in smolagents' default <code>…</code> blocks: (1) read the offloaded
# `documents` variable; (2) submit the final answer. The real agent loop + local executor run.
_TURN_READ = (
    "Thought: I'll inspect the provided documents.\n"
    "<code>\n"
    "assert isinstance(documents, list)  # the bundle is offloaded into the sandbox\n"
    "print(len(documents), documents[0][:20])\n"
    "</code>"
)
_TURN_ANSWER = (
    "Thought: The first document answers it.\n"
    "<code>\n"
    'final_answer("PARIS")\n'
    "</code>"
)


def _loong_benchmark():
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: ["Paris is the capital of France.",
                                   "Berlin is the capital of Germany."],
        get_task=lambda tid: ("Answer using the documents.", "What is the capital of France?", []),
    )


def test_run_one_drives_real_codeagent(tmp_path, monkeypatch) -> None:
    import litellm

    scripts = iter([(_TURN_READ, _FakeUsage(300, 40, reasoning=10)),
                    (_TURN_ANSWER, _FakeUsage(120, 8, reasoning=2))])

    def fake_completion(**kwargs):  # noqa: ANN003
        content, usage = next(scripts)
        return _fake_resp(content, usage)

    monkeypatch.setattr(litellm, "completion", fake_completion)

    run_config = {"benchmark": "loong", "baseline": "codeact", "model": "gpt-fake",
                  "seed": 42, "max_steps": 5, "run_version": ca_run._RUN_VERSION}
    litellm_kwargs = {"model": "openai/gpt-fake", "api_base": None, "api_key": "x"}

    rec = ca_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path, litellm_kwargs)

    assert rec["raw_answer"] == "PARIS"             # the CodeAct final_answer protocol worked
    assert "index_ref" not in rec                   # flat layout, no content-addressed index
    # Complete deterministic token capture across BOTH steps (keyed on the response model).
    total = rec["usage"]["total"]["gpt-fake"]
    assert total["num_calls"] == 2 and total["prompt_tokens"] == 420 and total["completion_tokens"] == 48
    assert total["completion_tokens_details"]["reasoning_tokens"] == 12
    assert len(rec["usage"]["calls"]) == 2 and len(rec["calls_full"]) == 2
    # The CodeAct trajectory is persisted inside the run folder (with the executed code).
    traj = json.loads((tmp_path / ca_run._TRAJECTORY_FILE).read_text())
    assert any("final_answer" in (step.get("code_action") or "") for step in traj)
    assert rec["trace"]["n_docs"] == 2 and rec["trace"]["state"] == "success"
    assert rec["trace"]["n_steps"] >= 2 and "capital of France" in rec["trace"]["task"]
    # the docs note is appended to the task, pointing at the variable (not dumping the docs)
    assert "`documents`" in rec["trace"]["task"]


def test_run_one_offloads_docs_into_sandbox_not_prompt(tmp_path, monkeypatch) -> None:
    """The documents must reach the executor variable but NOT the prompt (the offload). We assert
    the agent's code can read `documents`, and that the task text never contains the doc bodies."""
    import litellm

    secret = "ZEBRA-SENTINEL-42 is the magic phrase."
    scripts = iter([
        ("Thought: read it.\n<code>\nprint('LEN', len(documents[0]))\n</code>", _FakeUsage(10, 2)),
        ("Thought: done.\n<code>\nfinal_answer(documents[0])\n</code>", _FakeUsage(10, 2)),
    ])
    captured_prompts: list = []

    def fake_completion(**kwargs):  # noqa: ANN003
        captured_prompts.append(kwargs.get("messages"))
        content, usage = next(scripts)
        return _fake_resp(content, usage)

    monkeypatch.setattr(litellm, "completion", fake_completion)
    bench = SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: [secret],
        get_task=lambda tid: ("Repeat the magic phrase.", "", []),
    )
    rec = ca_run.run_one(bench, "t1", {"benchmark": "loong", "seed": 42, "max_steps": 4}, tmp_path,
                         {"model": "openai/gpt-fake", "api_base": None, "api_key": "x"})

    assert rec["raw_answer"] == secret              # the agent read the offloaded variable
    # The FIRST prompt (the task) must NOT contain the document body — only the pointer note.
    first_prompt_text = json.dumps(captured_prompts[0])
    assert "ZEBRA-SENTINEL" not in first_prompt_text
    assert "`documents`" in first_prompt_text


# ============================================================
# runner — config shape + resumption
# ============================================================

from evals.baselines.codeact import runner as ca_runner  # noqa: E402


def test_build_run_config_shape() -> None:
    p = ca_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "corpusqa", "--model", "openai/gpt-5.4-nano",
                      "--seed", "7", "--max-steps", "12"])
    cfg = ca_runner.build_run_config(a)
    assert cfg["benchmark"] == "corpusqa" and cfg["baseline"] == "codeact"
    assert cfg["seed"] == 7 and cfg["max_steps"] == 12
    assert cfg["run_version"] == ca_run._RUN_VERSION
    # Runs use the served model's own sampling; nothing pins generation params.
    assert "completion_params" not in cfg and "config_name" not in cfg


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

    monkeypatch.setattr(ca_runner, "_dispatch", fake_dispatch)
    argv = ["--benchmark", "loong", "--model", "openai/gpt-5.4-nano"]
    ca_runner.main(argv)
    assert sorted(dispatched) == ["t1", "t2", "t3"]   # first run: all 3
    dispatched.clear()
    ca_runner.main(argv)
    assert dispatched == []                            # re-run: all done → nothing re-run


# ============================================================
# Live wire-test (marker-gated) — real gpt-5.4-nano, 1 loong task
# ============================================================


@pytest.mark.codeact
def test_codeact_end_to_end_loong_real(tmp_path) -> None:
    """Drive the REAL CodeAgent on one short Loong EN task against gpt-5.4-nano (OpenAI). Costs a
    few cents; deselected by default (``-m codeact``)."""
    from evals.baselines import _common
    from evals.baselines.codeact.run import run_one

    model = os.getenv("CODEACT_MODEL", "openai/gpt-5.4-nano")
    benchmark = _common.load_benchmark_module("loong")
    task_id = benchmark.get_task_ids(languages=["en"], sets=[1], limit=1)[0]
    run_config = {"benchmark": "loong", "baseline": "codeact",
                  "model": _common.canonical_model_id(model), "seed": 42,
                  "max_steps": 8, "run_version": "test"}
    litellm_kwargs = {"model": _common.with_provider_prefix(model),
                      "api_base": os.getenv("CODEACT_BASE_URL"), "api_key": os.getenv("CODEACT_API_KEY")}

    rec = run_one(benchmark, task_id, run_config, tmp_path, litellm_kwargs)

    assert isinstance(rec["raw_answer"], str)
    assert (tmp_path / "trajectory.json").exists()
    assert rec["usage"]["total"] and rec["calls_full"]   # complete token capture
