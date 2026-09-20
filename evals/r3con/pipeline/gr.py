"""End-to-end just-in-time solver: (task, documents) → answer.

Wires the four stages of the method:

1. **summaries** — iterative cross-document summaries (``stages.summaries``): the
   long-text mechanism. Each document is summarized over ``summary_rounds``
   synchronous rounds, each round conditioning on the *other* documents' previous
   summaries.
2. **propose** — a per-task Pydantic schema (``stages.proposer``), conditioned on
   the task + the final summaries.
3. **extract** — one call per whole document, in parallel (``stages.extractor``),
   each conditioned on the summaries.
4. **infer** — ``inference.infer_llm`` (single call over parse + summaries) and/or
   ``inference.infer_codeact`` (the multi-turn sandboxed loop).

Each stage writes into one flat run-folder ``logs/<run-folder>/`` (each run a new
timestamped folder; see :mod:`evals.r3con.pipeline.runs`). No coordinator class; the routing is
plain control flow here.
"""

from __future__ import annotations

import traceback
from typing import Any

from evals.r3con.pipeline.config import RunConfig
from evals.r3con.pipeline.logging_setup import get_logger
from evals.r3con.pipeline.runs import StageRun, TaskLogger
from evals.r3con.pipeline.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S
from evals.r3con.pipeline.settings import settings
from evals.r3con.pipeline.stages import inference
from evals.r3con.pipeline.stages.extractor import extract_text
from evals.r3con.pipeline.stages.proposer import propose_schema
from evals.r3con.pipeline.stages.summaries import summarize_collection

_log = get_logger("gr")


def _preview(text: str, n: int = 100) -> str:
    """One-line, truncated preview of an answer for the progress log."""
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "…"


def gr_answer(
    *,
    task: str,
    documents: list[str],
    config: RunConfig,
    strategies: tuple[str, ...] = ("codeact",),
    max_inference_turns: int = settings.INFERENCE_MAX_TURNS,
    inference_timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    task_logger: TaskLogger | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, tuple[str, str | None]]:
    """Run stages 1 → 4 for one ``(task, documents)`` pair under one ``config``.

    ``config`` (a :class:`evals.r3con.pipeline.config.RunConfig`) carries everything that shapes
    the output — model, seed, summary rounds, the prompt version of each stage, and
    the sampling params — so nothing output-affecting is threaded ad-hoc. ``api_base``
    / ``api_key`` are transport only (routing/auth, not part of the run identity).

    ``strategies`` selects which stage-4 inference(s) to run over the parse —
    ``"llm"`` and/or ``"codeact"``. Returns ``{strategy: (answer, error)}``; ``error``
    is ``None`` on success, and a failure in one strategy doesn't abort the other.

    If ``task_logger`` is provided, each stage's artifacts are written immediately
    after that stage succeeds, so a later failure still leaves earlier artifacts.
    """
    model = config.model
    rounds = config.summary_rounds
    seed = config.seed
    # The output knobs flow from the config; api_base/api_key are transport only.
    llm_kwargs: dict[str, Any] = {**config.sampling, "seed": seed}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key

    def _run(stage: str, model_for_log: str) -> StageRun | None:
        if task_logger is None:
            return None
        return StageRun(stage=stage, task_logger=task_logger, model=model_for_log, seed=seed)

    # --- Stage 1: task-conditioned cross-document summaries. ---
    _log.info("stage 1/4 · summaries (%d round(s), %d doc(s))", rounds, len(documents))
    summaries_run = _run("summaries", model)
    summ = summarize_collection(
        task=task, documents=documents, model=model, prompt_version=config.prompts["summaries"],
        rounds=rounds, run=summaries_run, **llm_kwargs,
    )
    summaries = summ.final  # the final-round per-document summaries feed downstream
    if summaries_run is not None:
        summaries_run.flush(write_transcript=False)
    if task_logger is not None:
        # Per-round, per-document — `round` is the refinement depth; the LAST round
        # is what downstream stages consume. summaries[doc_i] aligns to documents[i].
        task_logger.write_json(
            "summaries/result",
            {
                "n_rounds": rounds,
                "n_docs": len(documents),
                "rounds": [
                    {"round": k + 1, "summaries": per_doc}
                    for k, per_doc in enumerate(summ.rounds)
                ],
                "totals": summaries_run.compute_totals() if summaries_run else None,
            },
        )

    # --- Stage 2: propose the schema (conditioned on task + summaries). ---
    _log.info("stage 2/4 · proposing schema")
    proposer_run = _run("proposer", model)
    proposal = propose_schema(
        task=task, summaries=summaries, model=model, prompt_version=config.prompts["proposer"],
        run=proposer_run, **llm_kwargs,
    )
    if proposer_run is not None:
        proposer_run.flush()
    if task_logger is not None:
        task_logger.write_json(
            "proposer/result",
            {
                "schema_code": proposal.schema_code,
                "thought": proposal.attempts[-1].thought if proposal.attempts else None,
                "attempts": [
                    {"schema_code": a.schema_code, "error": a.error, "thought": a.thought}
                    for a in proposal.attempts
                ],
                "totals": proposer_run.compute_totals() if proposer_run else None,
            },
        )

    # --- Stage 3: extract every document in parallel (conditioned on summaries). ---
    _log.info("stage 3/4 · extracting %d doc(s)", len(documents))
    extractor_run = _run("extractor", model)
    extraction = extract_text(
        documents=documents, schema_code=proposal.schema_code, parse_cls=proposal.parse_cls,
        task=task, prompt_version=config.prompts["extractor"], summaries=summaries, model=model,
        run=extractor_run, **llm_kwargs,
    )
    parsed = extraction.parse
    if extractor_run is not None:
        extractor_run.flush(write_transcript=False)
    if task_logger is not None:
        task_logger.write_json(
            "extractor/result",
            {
                "parsed": parsed,
                "source_docs": extraction.source_docs,
                "totals": extractor_run.compute_totals() if extractor_run else None,
            },
        )

    # --- Stage 4: inference (llm and/or codeact over the same parse + summaries). ---
    return run_inference(
        task=task, schema_code=proposal.schema_code, parsed=parsed,
        source_docs=extraction.source_docs, summaries=summaries, config=config,
        strategies=strategies, max_inference_turns=max_inference_turns,
        inference_timeout_s=inference_timeout_s, task_logger=task_logger,
        api_base=api_base, api_key=api_key,
    )


def run_inference(
    *,
    task: str,
    schema_code: str,
    parsed: Any,
    source_docs: dict[str, list[int]] | None = None,
    summaries: list[str] | None = None,
    config: RunConfig,
    strategies: tuple[str, ...] = ("codeact",),
    max_inference_turns: int = settings.INFERENCE_MAX_TURNS,
    inference_timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    task_logger: TaskLogger | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
) -> dict[str, tuple[str, str | None]]:
    """Stage 4 only: run the requested inference strategies over an already-computed
    ``parsed`` + ``summaries`` (+ ``schema_code`` for codeact), writing native
    per-strategy logs via ``task_logger``.

    The output knobs come from ``config`` (model, seed, sampling, the per-stage
    inference prompt versions); ``api_base`` / ``api_key`` are transport only. This is
    the stage-4 core: :func:`gr_answer` runs it after the first three stages, and it
    can also be called directly over already-computed artifacts. ``strategies``
    failures are recorded per strategy without aborting the others (same contract as
    ``gr_answer``)."""
    model = config.model
    seed = config.seed
    llm_kwargs: dict[str, Any] = {**config.sampling, "seed": seed}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key

    def _run(stage: str, model_for_log: str) -> StageRun | None:
        if task_logger is None:
            return None
        return StageRun(stage=stage, task_logger=task_logger, model=model_for_log, seed=seed)

    _log.info("stage 4 · inference: %s", ", ".join(strategies))
    results: dict[str, tuple[str, str | None]] = {}
    if "llm" in strategies:
        results["llm"] = _run_llm(
            task=task, parsed=parsed, source_docs=source_docs, summaries=summaries,
            model=model, prompt_version=config.prompts["inference/llm"],
            task_logger=task_logger, make_run=_run, llm_kwargs=llm_kwargs,
        )
    if "codeact" in strategies:
        results["codeact"] = _run_codeact(
            task=task, schema_code=schema_code, parsed=parsed,
            source_docs=source_docs, summaries=summaries,
            model=model, prompt_version=config.prompts["inference/codeact"],
            max_turns=max_inference_turns, timeout_s=inference_timeout_s,
            task_logger=task_logger, make_run=_run, llm_kwargs=llm_kwargs,
        )
    return results


def _run_llm(*, task, parsed, source_docs, summaries, model, prompt_version, task_logger, make_run, llm_kwargs):
    run = make_run("inference/llm", model)
    try:
        answer = inference.infer_llm(
            task=task, parsed=parsed, source_docs=source_docs, summaries=summaries, model=model,
            prompt_version=prompt_version, run=run, **llm_kwargs,
        )
        _log.info("inference/llm done: %s", _preview(answer))
        if run is not None:
            run.flush()
        if task_logger is not None:
            task_logger.write_json(
                "inference/llm/result",
                {"answer": answer, "totals": run.compute_totals() if run else None},
            )
        return (answer, None)
    except Exception as e:  # noqa: BLE001 — record + keep going so codeact still runs
        _log.info("inference/llm failed: %s: %s", type(e).__name__, e)
        _record_error(task_logger, "inference/llm")
        return ("", f"{type(e).__name__}: {e}")


def _run_codeact(
    *, task, schema_code, parsed, source_docs, summaries, model, prompt_version, max_turns, timeout_s,
    task_logger, make_run, llm_kwargs,
):
    run = make_run("inference/codeact", model)
    try:
        result = inference.infer_codeact(
            task=task, schema_code=schema_code, parsed=parsed, source_docs=source_docs,
            summaries=summaries, model=model, prompt_version=prompt_version, max_turns=max_turns,
            timeout_s=timeout_s, run=run, **llm_kwargs,
        )
        _log.info("inference/codeact done (%s, %d turn(s)): %s",
                  result.terminated_by, len(result.turns), _preview(result.answer))
        if run is not None:
            run.flush()
        if task_logger is not None:
            task_logger.write_json(
                "inference/codeact/result",
                {
                    "answer": result.answer,
                    "terminated_by": result.terminated_by,
                    "n_turns": len(result.turns),
                    "turns": [
                        {
                            "response": t.response, "raw_response": t.raw_response,
                            "code": t.code, "observation": t.observation,
                            "error": t.error, "is_final_answer": t.is_final_answer,
                        }
                        for t in result.turns
                    ],
                    "totals": run.compute_totals() if run else None,
                },
            )
        return (result.answer, None)
    except Exception as e:  # noqa: BLE001 — record + keep going
        _log.info("inference/codeact failed: %s: %s", type(e).__name__, e)
        _record_error(task_logger, "inference/codeact")
        return ("", f"{type(e).__name__}: {e}")


def _record_error(task_logger: TaskLogger | None, stage: str) -> None:
    """Write ``error.txt`` to a strategy's folder so a per-strategy failure (e.g. a
    ``ContextWindowExceededError``) is discoverable on disk. Call from inside the
    ``except`` so ``traceback.format_exc()`` captures the active exception."""
    if task_logger is None:
        return
    err_dir = task_logger.dir / stage
    err_dir.mkdir(parents=True, exist_ok=True)
    (err_dir / "error.txt").write_text(traceback.format_exc())
