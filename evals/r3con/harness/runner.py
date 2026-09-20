"""Shared subprocess launcher: one per-task child (``scripts/r3con/<benchmark>/run_task.py``).

Benchmark-agnostic — the per-task child script is a parameter (``run_task_script``), so
all three benchmarks reuse this one launcher. One process per task gives **true
concurrency** (each child has its own litellm client / connection pool) and **crash
isolation** (an OOM-kill / segfault dies alone, logged + skipped, never collapsing the
run). Each child writes its answer and its per-stage artifacts into its own flat
``logs/r3con/<run-folder>/``. Nothing here grades anything.
"""

from __future__ import annotations

import argparse
import os
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from evals.r3con.pipeline.config import (
    RunConfig,
    available_configs,
    available_sampling_presets,
    load_config,
    normalize_model_name,
)
from evals.r3con.pipeline.runs import new_run_folder
from evals.r3con.pipeline.settings import active_doc_workers, settings

# Grace given to a child after SIGTERM before we SIGKILL it on interrupt. Children
# install no signal handler, so SIGTERM ends them near-instantly even mid-network-call;
# the grace only covers a child wedged in cleanup.
_TERMINATE_GRACE_S = 10.0


@dataclass
class LaunchSummary:
    n: int
    ok: int = 0
    failed: int = 0
    failures: list[tuple[str, int]] = field(default_factory=list)  # (task_id, returncode)


def _run_task_argv(
    *,
    task_id: str,
    config: RunConfig,
    strategies: tuple[str, ...],
    run_folder: str,
    api_base: str | None,
    api_key: str | None,
    run_task_script: str = "scripts/r3con/loong/run_task.py",
) -> list[str]:
    """Build the argv that spawns the per-task child for one task. The child re-loads
    ``--config`` + the same overrides (reconstructing the identical RunConfig) and writes
    into the parent-stamped ``--run-folder``. ``run_task_script`` is the per-task entry
    point relative to the repo root — Loong's by default, ``scripts/r3con/corpusqa/run_task.py``
    for CorpusQA (the only benchmark-specific bit of an otherwise generic launcher)."""
    inference_arg = "both" if set(strategies) == {"llm", "codeact"} else strategies[0]
    argv = [
        sys.executable,
        str(settings.ROOT / run_task_script),
        "--task-id", task_id,
        "--config", config.name,
        "--inference", inference_arg,
        "--run-folder", run_folder,
    ]
    for key, value in config.overrides.items():  # model / seed / summary_rounds
        argv += [f"--{key.replace('_', '-')}", str(value)]
    if api_base:
        argv += ["--base-url", api_base]
    if api_key:
        argv += ["--api-key", api_key]
    return argv


def run(
    task_ids: Sequence[str],
    config: RunConfig,
    *,
    strategies: tuple[str, ...] = ("codeact",),
    workers: int = 1,
    api_base: str | None = None,
    api_key: str | None = None,
    run_task_script: str = "scripts/r3con/loong/run_task.py",
    _popen: Any = None,
) -> LaunchSummary:
    """Spawn one ``run_task.py`` subprocess per task, ≤``workers`` at once.

    The parent stamps a fresh ``logs/<run-folder>/`` (see
    :func:`evals.r3con.pipeline.runs.new_run_folder`) per task and hands it to the child via
    ``--run-folder``, so the parent's ``run_task.log`` and the child's artifacts land
    in the same folder. Each child's stdout+stderr go to that log (a file, not a pipe
    — a pipe could fill and deadlock). ``_popen`` is the process factory (tests inject
    a fake). Returns launch counts only.
    """
    popen = _popen or subprocess.Popen
    n = len(task_ids)
    workers = max(1, workers)
    print(f"\n=== launching {n} task(s) · run: {config.label()} · ≤{workers} concurrent subprocess(es) ===")

    queue: list[str] = list(task_ids)
    inflight: dict[Any, tuple[str, str, Any]] = {}  # proc -> (task_id, run_folder, open_logfile)
    summary = LaunchSummary(n=n)
    done = 0

    def _launch(tid: str) -> None:
        folder = new_run_folder()
        log_dir = settings.LOGS_DIR / folder
        log_dir.mkdir(parents=True, exist_ok=True)
        logf = open(log_dir / "run_task.log", "w")
        argv = _run_task_argv(
            task_id=tid, config=config, strategies=strategies,
            run_folder=folder, api_base=api_base, api_key=api_key,
            run_task_script=run_task_script,
        )
        try:
            # start_new_session: each child gets its OWN session/process group, so a
            # terminal Ctrl-C does NOT reach it directly — the parent (below) is the sole
            # owner of the child's lifecycle and kills it explicitly on interrupt. Without
            # this, a child can swallow the terminal SIGINT mid-LLM-call and linger.
            proc = popen(
                argv, stdout=logf, stderr=subprocess.STDOUT, cwd=str(settings.ROOT),
                start_new_session=True,
            )
        except BaseException:
            logf.close()
            raise
        inflight[proc] = (tid, folder, logf)

    def _reap(proc: Any) -> None:
        nonlocal done
        tid, _folder, logf = inflight.pop(proc)
        logf.close()
        rc = int(proc.returncode)
        done += 1
        # One minimal line per finished task: ✓/✗ + id. The answer is in result.json and
        # full details are in the task's run_task.log — don't echo them here.
        if rc == 0:
            summary.ok += 1
            print(f"  ✓ {tid}")
        else:
            summary.failed += 1
            summary.failures.append((tid, rc))
            why = f"signal {-rc}" if rc < 0 else f"exit {rc}"
            print(f"  ✗ {tid} ({why})")

    def _terminate_inflight() -> int:
        """SIGTERM every still-running child (SIGKILL the stragglers after a grace), so
        an interrupt never leaves orphaned task subprocesses making LLM calls in the
        background. Returns how many children it had to stop."""
        live = [p for p in inflight if p.poll() is None]
        for p in live:
            try:
                p.terminate()
            except OSError:
                pass
        deadline = time.monotonic() + _TERMINATE_GRACE_S
        for p in live:
            try:
                p.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    p.kill()
                except OSError:
                    pass
            except (OSError, ValueError):
                pass
        return len(live)

    try:
        while queue or inflight:
            while queue and len(inflight) < workers:
                _launch(queue.pop(0))
            finished = [p for p in inflight if p.poll() is not None]
            for p in finished:
                _reap(p)
            if not finished:
                time.sleep(0.05)
    except KeyboardInterrupt:
        n_killed = _terminate_inflight()
        print(f"\n⚠  interrupted — stopped {n_killed} running task subprocess(es). "
              f"Their partial log folders are left as they are; re-running picks the tasks back up.")
        raise
    finally:
        for _tid, _folder, logf in inflight.values():
            try:
                logf.close()
            except OSError:
                pass

    return summary


def completed_task_ids(logs_root, run_label: str, strategies) -> set[str]:
    """Task ids already finished for ``run_label`` under EVERY requested strategy.

    The launcher skips these so a re-run only does what is missing — an interrupted
    1,600-task run resumes instead of paying for the whole thing again. A task counts as
    finished for a strategy when that strategy wrote a ``result.json`` (an answer) or a
    settled ``ContextWindowExceededError`` (re-running it under an identical config just
    fails the same way); any other error re-runs, since those are usually transient.

    Identity comes from ``RunConfig.label()``, recomputed from each folder's recorded
    ``config`` block — so changing the model, the seed, the summary rounds, the sampling
    or any prompt version yields a different label and correctly re-runs from scratch.
    """
    from evals.r3con.pipeline.config import RunConfig

    strats = list(strategies)
    if not strats or not Path(logs_root).is_dir():
        return set()
    done: dict[str, set[str]] = {s: set() for s in strats}
    for folder in sorted(p for p in Path(logs_root).iterdir() if p.is_dir()):
        try:
            manifest = json.loads((folder / "manifest.json").read_text())
            if RunConfig(**manifest["config"]).label() != run_label:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            continue  # no/!unreadable manifest, or a legacy config block -> not a match
        task_id = manifest.get("task_id")
        if not task_id:
            continue
        for strategy in strats:
            d = folder / "inference" / strategy
            if (d / "result.json").is_file():
                done[strategy].add(task_id)
            elif (d / "error.txt").is_file():
                try:
                    if "ContextWindowExceeded" in (d / "error.txt").read_text():
                        done[strategy].add(task_id)
                except OSError:
                    pass
    return set.intersection(*(done[s] for s in strats))


def print_launch_summary(summary: LaunchSummary) -> None:
    print("\n=== launch summary ===")
    print(
        f"  {summary.ok}/{summary.n} task(s) exited cleanly; "
        f"{summary.failed} failed (non-zero exit / signal death)."
    )
    for tid, rc in summary.failures[:20]:
        why = f"signal {-rc}" if rc < 0 else f"exit {rc}"
        print(f"    FAILED {tid} ({why})")
    if len(summary.failures) > 20:
        print(f"    … and {len(summary.failures) - 20} more")
    print(f"\nAnswers + per-stage artifacts: {settings.LOGS_DIR}/<run-folder>/")


# ---------- shared CLI scaffolding ----------


def build_common_argparser(description: str) -> argparse.ArgumentParser:
    """An argparser pre-loaded with the solver flags shared by every runner: a
    ``--config`` selector, the output overrides (model / seed / summary-rounds), and
    the transport / runtime flags."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--config", choices=available_configs(), default="default",
        help="Experiment config from configs/<name>.yaml — model, seed, summary "
        "rounds, prompt versions, sampling. Default: 'default'.",
    )
    # Output-knob overrides — resolve into the RunConfig; recorded in the run label.
    p.add_argument("--model", default=None, help="Override the config's model.")
    p.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    p.add_argument(
        "--summary-rounds", type=int, default=None, metavar="N",
        help="Override the config's number of summary rounds.",
    )
    p.add_argument(
        "--sampling", choices=available_sampling_presets(), default=None,
        help="Override the sampling preset (configs/sampling/<name>.yaml), e.g. qwen-no-thinking. "
        "Replaces the config's sampling and shows in the run label.",
    )
    # Transport (routing/auth, not part of the run identity).
    p.add_argument("--base-url", default=None, help="API base URL (e.g. a vLLM endpoint).")
    p.add_argument("--api-key", default=None, help="API key passed straight to the model call.")
    # Runtime.
    p.add_argument(
        "--inference", choices=["llm", "codeact", "both"], default="codeact",
        help="Which stage-4 inference to run (default: codeact). 'both' runs both "
        "over one extraction.",
    )
    p.add_argument(
        "--workers", type=int, default=10,
        help="Max tasks in parallel — one subprocess per task (default: 10).",
    )
    p.add_argument(
        "--doc-workers", type=int, default=None, metavar="N",
        help="Max documents processed concurrently WITHIN each task (the summaries + extraction "
        "fan-out). Sets R3CON_DOC_WORKERS for the spawned children; unset → the env var else 16. "
        "Peak endpoint load ≈ --workers × --doc-workers.",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Stream per-stage progress into each logs/<run-folder>/run_task.log "
        "(sets R3CON_LOG_LEVEL=INFO, inherited by the children). Quiet by default.",
    )
    return p


def resolve_config(args: argparse.Namespace) -> RunConfig:
    """Load the selected ``--config`` and apply the CLI overrides → the resolved RunConfig."""
    return load_config(
        args.config,
        model=getattr(args, "model", None),
        seed=getattr(args, "seed", None),
        summary_rounds=getattr(args, "summary_rounds", None),
        sampling=getattr(args, "sampling", None),
    )


def apply_verbose(args: argparse.Namespace) -> None:
    """Set ``R3CON_LOG_LEVEL=INFO`` iff ``--verbose`` (inherited by the children's
    ``configure_logging``)."""
    if getattr(args, "verbose", False):
        os.environ["R3CON_LOG_LEVEL"] = "INFO"


def apply_doc_workers(args: argparse.Namespace) -> None:
    """Set ``R3CON_DOC_WORKERS`` from ``--doc-workers`` iff given, so the spawned children (which
    do the actual per-task doc fan-out) inherit it. Unset → the ambient env var, else the default
    (:data:`evals.r3con.pipeline.settings.Settings.DOC_WORKERS`)."""
    n = getattr(args, "doc_workers", None)
    if n is not None:
        os.environ["R3CON_DOC_WORKERS"] = str(n)


def strategies_from_arg(inference: str) -> tuple[str, ...]:
    """Map ``--inference`` to the strategy tuple ``run``/``gr_answer`` expect."""
    return ("llm", "codeact") if inference == "both" else (inference,)


def print_run_header(
    task_ids: Sequence[str], config: RunConfig, strategies: tuple[str, ...], *,
    api_base: str | None = None, name: str = "Loong",
) -> None:
    print(f"Benchmark: {name} | tasks: {len(task_ids)} | run: {config.label()}")
    print(f"Model: {normalize_model_name(config.model)} | seed: {config.seed} | summary rounds: {config.summary_rounds}"
          f" | doc-workers: {active_doc_workers()}")
    if config.sampling_preset:
        print(f"Sampling preset: {config.sampling_preset}")
    if api_base:
        print(f"Endpoint: {api_base}")
    print(f"Inference: {', '.join(strategies)}  (one flat logs/<folder>/ per task)")
