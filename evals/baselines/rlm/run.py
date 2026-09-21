"""RLM baseline pipeline — connects Recursive Language Models to the harness.

RLM (Recursive Language Models, alexzhang13/rlm, arXiv 2512.24601) replaces a plain
``llm.completion(prompt)`` with ``rlm.completion(prompt, root_prompt)``: the (long) context is
offloaded as a ``context`` **variable in a Python REPL**, and a root LM writes ``repl`` code
(CodeAct — not JSON tool-calling) to examine/decompose it and launch recursive sub-LM calls
(``llm_query`` / ``rlm_query``). We use RLM's canonical question-answering front door:

  - the document bundle → ``prompt`` (becomes the REPL ``context`` variable the agent explores),
  - the composed task/question → ``root_prompt`` (the small prompt the root LM sees directly).

This is the method's intended usage (its docstring describes exactly the "pass the question as the
root prompt" pattern) — we do NOT author a solution strategy, we pose the task.

RLM is a normal dependency (the ``evals[rlms]`` extra) — ``uv add``-ed, not vendored. Everything
routes to the **vLLM endpoint** via RLM's OpenAI-compatible client (``base_url``); **no OpenAI
cloud, no litellm**. Token/cost is captured completely at the client boundary (see ``llm.py``).

SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets its OWN folder
``logs/{benchmark}/rlm/{run_tag}/`` holding RLM's full trajectory — the native ``RLMLogger`` jsonl,
written LIVE (one object per iteration, crash-resilient; REPL-variable snapshots dropped, see
``logger.py``) — ``manifest.json`` (TOTAL tokens across all
depths/sub-calls) and ``calls.json``. Nothing here grades: the external scoring repo reads these
log folders and persists its own grades.

SUPPORTED_BENCHMARKS = {loong, corpusqa, dracula}. Deviation ledger: evals/baselines/rlm/PROVENANCE.md
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Run-config version — bump on a change that alters a run's output (context framing, defaults).
_RUN_VERSION = "v1"


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _strip_provider(model: str) -> str:
    """RLM's OpenAI client talks to vLLM directly, so it wants the BARE served model id
    (e.g. ``Qwen/Qwen3.5-35B-A3B``), not a litellm provider-prefixed one. Strip the prefixes
    the harness/litellm use (``hosted_vllm/``, ``openai/``, ``hosted_vllm/openai/``)."""
    for prefix in ("hosted_vllm/", "openai/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
    return model


def _build_task(benchmark, task_id: str) -> str:
    """The question handed to the root LM (RLM's ``root_prompt``) — the SAME components every other
    baseline poses (documents are dropped from the question; they go into the REPL ``context``).

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
        # The bare question; the whole 46-document corpus goes into the REPL
        # `context` variable, dropped from the question.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"rlm has no task assembly for benchmark {name!r}")


def _build_context(documents: list[str]) -> str:
    """The REPL ``context`` variable: the document bundle the agent programmatically explores.

    Multi-doc bundles (Loong/CorpusQA) are concatenated into one string with clear
    ``=== Document N ===`` markers so the agent can split/navigate them in code (the same
    per-document framing the other multi-doc baselines use). A single doc is passed as-is."""
    if len(documents) == 1:
        return documents[0]
    return "\n\n".join(f"=== Document {i + 1} ===\n{doc}" for i, doc in enumerate(documents))


def _backend_kwargs(litellm_kwargs: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    """RLM's OpenAI-compatible client config for the vLLM endpoint.

    We inject ``seed`` (the run's ``--seed``, **default 42** when not given) into the generation call
    for reproducibility — the harness convention every baseline follows — plus any opt-in ``--config``
    sampling preset (``completion_params``). RLM is otherwise left at its own defaults (no temperature
    pin, no max_tokens cap)."""
    kw: dict[str, Any] = {
        "model_name": _strip_provider(litellm_kwargs["model"]),
        "base_url": litellm_kwargs.get("api_base"),
        "api_key": litellm_kwargs.get("api_key"),
    }
    sampling_args: dict[str, Any] = dict(run_config.get("completion_params") or {})
    sampling_args.setdefault("seed", run_config.get("seed", 42))
    kw["sampling_args"] = sampling_args
    return kw


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task RLM pipeline, ALL inside ``run_dir``. Returns the harness record — ``raw_answer`` +
    the **TOTAL** ``usage`` (every LM call across all depths/sub-calls) + ``calls_full`` + a light
    ``trace``. Everything hits the vLLM endpoint; no OpenAI."""
    from rlm import RLM

    from evals.baselines.rlm.llm import RLMUsageCapture
    from evals.baselines.rlm.logger import TrajectoryLogger

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    base_url = litellm_kwargs.get("api_base")
    if not base_url or "api.openai.com" in base_url:
        # Hard guard: RLM is notoriously token-heavy and MUST run against the local vLLM, never
        # OpenAI. The runner enforces this too; this is belt-and-suspenders.
        raise ValueError(
            "rlm requires a non-OpenAI --base-url (the vLLM endpoint); got "
            f"{base_url!r}. RLM is token-heavy and must run on local vLLM."
        )

    documents = benchmark.get_documents(task_id)
    context = _build_context(documents)
    task = _build_task(benchmark, task_id)

    # vLLM via RLM's OpenAI-compatible client. NO OpenAI cloud, NO litellm. We inject `seed`
    # (default 42) for reproducibility + any --config preset; RLM is otherwise at its own defaults.
    backend_kwargs = _backend_kwargs(litellm_kwargs, run_config)

    # RLM's native trajectory jsonl, in the run folder — minus the REPL-variable snapshots (logger.py).
    logger = TrajectoryLogger(log_dir=str(run_dir))

    with RLMUsageCapture() as cap:
        rlm = RLM(
            backend="vllm",
            backend_kwargs=backend_kwargs,
            environment="local",  # in-process exec REPL (no Docker/cloud sandbox)
            logger=logger,
            # These three are RLM's OWN repo defaults (max_iterations=30, max_depth=1,
            # max_timeout=None) — passed through unchanged, just exposed as --max-* knobs. A
            # default run is identical to not setting them; we don't change RLM's defaults.
            max_iterations=run_config.get("max_iterations", 30),
            max_depth=run_config.get("max_depth", 1),
            max_timeout=run_config.get("max_timeout"),
            verbose=False,
        )
        result = rlm.completion(prompt=context, root_prompt=task)

    raw_answer = result.response or ""

    # n_iterations from RLM's returned metadata, IN-MEMORY only — the native RLMLogger jsonl (written
    # live during the run) already persists the full trajectory in the run folder, so we do NOT
    # re-dump result.metadata to disk (it was a byte-for-byte-redundant ~hundreds-of-MB second copy).
    trajectory = result.metadata or {}
    iterations = trajectory.get("iterations") if isinstance(trajectory, dict) else None
    n_iterations = len(iterations) if isinstance(iterations, list) else None

    return {
        "raw_answer": raw_answer,
        "usage": cap.usage,                 # TOTAL tokens across all depths (root + sub-calls)
        "calls_full": cap.full_calls,       # → calls.json (per-call request + response + reasoning)
        "trace": {
            "task": task,
            "n_docs": len(documents),
            "execution_time": result.execution_time,
            "n_iterations": n_iterations,
            "environment": "local",
            "max_iterations": run_config.get("max_iterations", 30),
            "max_depth": run_config.get("max_depth", 1),
            # RLM's own (root-handler-only) accounting, for cross-checking against the complete
            # client-level capture in `usage` (the latter also includes recursive sub-calls).
            "rlm_usage_summary": result.usage_summary.to_dict(),
        },
    }
