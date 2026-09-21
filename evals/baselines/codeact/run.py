"""CodeAct baseline pipeline — connects smolagents' ``CodeAgent`` to the harness.

CodeAct = an agent that acts by writing & running Python code. smolagents' ``CodeAgent`` is its
reference implementation (a ReAct loop: the LM writes a code blob, a local executor runs it, the
agent observes stdout, repeat until ``final_answer(...)``). We pose a grounded-reasoning task like
RLM (the closest sibling): the document bundle is OFFLOADED into the code sandbox as a
``documents`` variable so the agent reads/slices/aggregates it with code, and the composed
task/question is the agent's task string.

The single channel for that offload is the agent's ``state`` dict, which ``CodeAgent.run`` forwards
to the executor (``python_executor.send_variables(self.state)``). We deliberately do NOT use
``run(additional_args=...)`` because smolagents ALSO stringifies ``additional_args`` INTO the prompt
(``str(additional_args)``) — which would dump the whole bundle into the context, defeating the
offload and overflowing the window. So we set ``agent.state["documents"]`` directly and add a short
note to the task pointing at it (mirroring smolagents' own additional-args note wording, minus the
value dump). See ``PROVENANCE.md``.

smolagents is a normal dependency (``evals[codeact]``), not vendored. The LLM routes through
smolagents' ``LiteLLMModel`` (subclassed in ``llm.py`` for deterministic, complete cost capture).
``environment`` = local (in-process executor, no Docker), like RLM.

SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets its OWN folder
``logs/{benchmark}/codeact/{run_tag}/`` holding the CodeAct trajectory (``trajectory.json`` — every
step's thought / code / observation), ``manifest.json`` (the TOTAL tokens across all steps) and
``calls.json``. Nothing here grades — a run records the raw answer and its cost, and the separate
scoring repo reads these logs.

SUPPORTED_BENCHMARKS = {loong, corpusqa, dracula}. Deviation ledger: evals/baselines/codeact/PROVENANCE.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Run-config version — bump on a change that alters a run's output (task framing, defaults).
_RUN_VERSION = "v1"

# The CodeAct trajectory (smolagents' per-step memory) dumped here inside the run folder.
_TRAJECTORY_FILE = "trajectory.json"

# smolagents' own CodeAgent/MultiStepAgent default (``max_steps=20``). Passed through unchanged
# (faithful — we run the method as shipped); exposed as ``--max-steps`` for an override.
_DEFAULT_MAX_STEPS = 20


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _build_task(benchmark, task_id: str) -> str:
    """The question handed to the CodeAgent — the SAME components every other baseline poses
    (documents dropped from the question; they go into the ``documents`` sandbox variable).

    - **loong**: ``instruction`` (the whole task when ``question`` is empty) else
      ``instruction\\n\\nquestion``.
    - **corpusqa**: ``question\\n\\n{output-requirements}`` (the output-requirements block, with the
      ``The answer is:`` contract the judge extracts, MUST reach the model).
    """
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the whole 46-document corpus goes into the
        # `documents` sandbox variable, dropped from the task string.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"codeact has no task assembly for benchmark {name!r}")


def _docs_note(n_docs: int) -> str:
    """The pointer that tells the CodeAgent WHERE the grounding lives — necessary plumbing (the
    agent sees only the task + smolagents' own system prompt, which doesn't know about our
    variable). Mirrors smolagents' own additional-args note wording, but describes ``documents``
    instead of dumping its (huge) value into the prompt — that's the whole point of the offload."""
    return (
        "\n\nYou have been provided with these additional arguments, that you can access "
        "directly using the keys as variables in your Python code:\n"
        f"`documents`: a list of {n_docs} source-document string(s) — the material to answer "
        "from. Read and analyze it with code; do not rely on prior knowledge."
    )


def _model_call_kwargs(run_config: dict[str, Any]) -> dict[str, Any]:
    """Generation kwargs applied to EVERY completion (smolagents merges model kwargs last, so they
    override per call): the run's ``seed`` (default 42 — the harness reproducibility convention
    every baseline follows), plus any ``completion_params`` the run config carries. Nothing sets
    those today, so in practice only ``seed`` is sent and generation follows the served
    model/provider defaults."""
    kw: dict[str, Any] = dict(run_config.get("completion_params") or {})
    kw.setdefault("seed", run_config.get("seed", 42))
    return kw


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task CodeAct pipeline, ALL inside ``run_dir``. Returns the harness record — ``raw_answer``
    + the **TOTAL** ``usage`` (every completion across all steps) + ``calls_full`` + a light
    ``trace`` — and writes the CodeAct trajectory to ``trajectory.json``."""
    from smolagents import CodeAgent, LogLevel

    from evals.baselines.codeact.llm import CodeActModel

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = benchmark.get_documents(task_id)
    task = _build_task(benchmark, task_id) + _docs_note(len(documents))

    # smolagents' LiteLLMModel, subclassed for complete deterministic cost capture. Same litellm
    # transport as every other baseline (so --model/--base-url/--api-key + provider prefixes work
    # identically); seed is a model kwarg applied to every completion.
    model = CodeActModel(
        model_id=litellm_kwargs["model"],
        api_base=litellm_kwargs.get("api_base"),
        api_key=litellm_kwargs.get("api_key"),
        **_model_call_kwargs(run_config),
    )

    max_steps = run_config.get("max_steps", _DEFAULT_MAX_STEPS)
    agent = CodeAgent(
        tools=[],                    # no external tools — the agent reads the docs via code (pure CodeAct)
        model=model,
        max_steps=max_steps,
        executor_type="local",       # in-process Python executor (no Docker), like RLM's environment="local"
        verbosity_level=LogLevel.OFF,  # quiet — this runs in batch on the cluster
        return_full_result=True,     # → RunResult (output + state + token_usage)
    )
    # Offload the documents into the code sandbox as `documents`. run() forwards self.state to the
    # executor (send_variables); we set it directly rather than via additional_args (which smolagents
    # would also stringify into the prompt — dumping the whole bundle). See the module docstring.
    agent.state["documents"] = list(documents)

    try:
        result = agent.run(task)
    finally:
        agent.cleanup()  # release the executor (a no-op for the local one, but the faithful teardown)

    raw_answer = "" if result.output is None else str(result.output)

    # Persist the full CodeAct trajectory (succinct steps: per-step model_output / code_action /
    # observations / error / token_usage). The verbatim per-call request+response lives in
    # calls.json, so we drop the redundant growing model_input_messages here.
    trajectory = agent.memory.get_succinct_steps()
    (run_dir / _TRAJECTORY_FILE).write_text(
        json.dumps(trajectory, ensure_ascii=False, indent=2, default=str)
    )
    n_steps = sum(1 for s in trajectory if "step_number" in s)

    tu = result.token_usage
    return {
        "raw_answer": raw_answer,
        "usage": model.usage,            # TOTAL across all steps (complete, deterministic)
        "calls_full": model.full_calls,  # → calls.json (per-call request + response + reasoning)
        "trace": {
            "task": task,
            "n_docs": len(documents),
            "state": result.state,       # "success" | "max_steps_error"
            "n_steps": n_steps,
            "max_steps": max_steps,
            "environment": "local",
            # smolagents' own (input/output only) token tally, for cross-checking the complete
            # per-call capture in `usage` (the latter also carries nested *_tokens_details).
            "smolagents_token_usage": (
                {"input_tokens": tu.input_tokens, "output_tokens": tu.output_tokens,
                 "total_tokens": tu.input_tokens + tu.output_tokens}
                if tu else None
            ),
        },
    }
