"""Per-task entrypoint: run the R3Con pipeline for ONE Loong task; write its artifacts.

The unit of work the subprocess launcher (:func:`evals.r3con.harness.runner.run`) spawns
once per task — one process per task gives true concurrency (each its own litellm
client) + crash isolation.
+ per-stage artifacts into one flat ``logs/<run-folder>/`` (the runner picks the
folder name) and returns. """

from __future__ import annotations

import datetime
import traceback
from dataclasses import dataclass, field
from typing import Any

from evals.r3con.pipeline.config import RunConfig, normalize_model_name
from evals.r3con.harness import loong
from evals.r3con.pipeline.gr import gr_answer
from evals.r3con.pipeline.runs import TaskLogger
from evals.r3con.pipeline.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S
from evals.r3con.pipeline.settings import settings_snapshot


def write_task_manifest(
    log: TaskLogger,
    task_id: str,
    ti: loong.TaskInput,
    *,
    config: RunConfig,
    strategies: tuple[str, ...],
    inference_timeout_s: float | None,
) -> dict[str, Any]:
    """Write ``<run-folder>/manifest.json`` — the task identity card AND the full
    parameter snapshot, so a result on disk records *everything* it depended on.
    Returns the dict it wrote, so the caller can re-write it with the run's token
    cost once the run is over.

    The ``config`` block is the resolved :class:`RunConfig` (everything that shapes
    the output); the run label is **recomputed** from it (``RunConfig.label()``) rather than persisted,
    so a label-format change never invalidates a folder on disk. The ``settings`` block is the runtime-only
    knobs. Written before any LLM call so a later crash still leaves the task
    discoverable. ``created`` is a sortable UTC stamp. The recorded ``config.model`` (and ``config.overrides.model``) are
    **sanitized** to the bare model name — the provider/route prefix is transport, not
    identity (the live ``config`` keeps the full string for routing).
    """
    config_block = config.model_dump()
    config_block["model"] = normalize_model_name(config_block.get("model"))
    overrides = config_block.get("overrides") or {}
    if "model" in overrides:
        overrides["model"] = normalize_model_name(overrides["model"])
    manifest: dict[str, Any] = {
        "benchmark": loong.NAME,
        "task_id": task_id,
        "run_folder": log.folder,
        "question": ti.task,
        "gold": loong.gold(task_id),
        "n_docs": len(ti.documents),
        "context_chars": sum(len(d) for d in ti.documents),
        "strategies": list(strategies),
        "inference_timeout_s": inference_timeout_s,
        # --- what shapes the output (the experiment identity) ---
        "config": config_block,
        # --- runtime knobs (parallelism / resilience) ---
        "settings": settings_snapshot(),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    log.write_json("manifest", manifest)
    return manifest


@dataclass
class RunTaskOutcome:
    """``results`` maps each strategy to ``(answer, error_or_None)``. ``failed`` is
    True iff R3Con hit a top-level exception (summaries/proposer/extractor raised before
    any answer) — a per-strategy inference error is recorded in ``results`` but does
    not set ``failed``."""

    results: dict[str, tuple[str, str | None]] = field(default_factory=dict)
    failed: bool = False


def run_task(
    task_id: str,
    *,
    config: RunConfig,
    run_folder: str,
    strategies: tuple[str, ...] = ("codeact",),
    api_base: str | None = None,
    api_key: str | None = None,
) -> RunTaskOutcome:
    """Run one Loong task end-to-end through R3Con under ``config``; write its artifacts
    into ``logs/<run_folder>/``.

    The manifest is written twice: once up front (so a crashed task is discoverable
    from t=0) and once at the end, with the task's complete token cost added."""
    ti = loong.load(task_id)
    log = TaskLogger(run_folder, task_id=task_id)

    manifest = write_task_manifest(
        log, task_id, ti, config=config, strategies=strategies,
        inference_timeout_s=DEFAULT_EXEC_TIMEOUT_S,
    )

    outcome = RunTaskOutcome()
    try:
        res = gr_answer(
            task=ti.task, documents=ti.documents, config=config, strategies=strategies,
            task_logger=log, api_base=api_base, api_key=api_key,
        )
        outcome.results = dict(res)
    except Exception as e:  # noqa: BLE001 — a top-level (summaries/proposer/extractor) failure
        errmsg = f"{type(e).__name__}: {e}"
        for strategy in strategies:
            err_dir = log.dir / "inference" / strategy
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(traceback.format_exc())
            outcome.results[strategy] = ("", errmsg)
        outcome.failed = True

    # Re-write the manifest now the run is over, adding `usage` — the task's whole
    # token cost, in the same shape the baselines record. Both paths land here, so a
    # crashed task still reports what it burned before it died. Accounting must never
    # sink a task that produced an answer, so a roll-up/write failure is recorded beside
    # the manifest and swallowed rather than raised out of `run_task`.
    try:
        manifest["usage"] = log.usage_rollup()
        log.write_json("manifest", manifest)
    except Exception:  # noqa: BLE001 — see above; the t=0 manifest is already on disk
        (log.dir / "usage_error.txt").write_text(traceback.format_exc())
    return outcome
