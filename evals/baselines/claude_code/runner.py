"""Runner for the Claude Code baseline (v1, exploratory).

    python -m evals.baselines.claude_code --benchmark {loong,corpusqa} [flags]

Flat per-run-folder logging (the rlm/readagent layout) + resumption. Runs **one task at a time**
by default (``--max-workers 1`` — a deliberate choice for this baseline; bump it only if you know
your Max rate limits can take it). Each task shells out to ``claude -p`` (see ``run.py``); auth is
the logged-in Max subscription (the env's ``ANTHROPIC_API_KEY`` is popped for the CLI).

There is no ``--seed`` (the CLI takes none) and no ``--base-url`` / ``--api-key``
(the CLI owns auth). The behavioural knobs are ``--model`` + ``--effort``.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.claude_code.run import (
    SUPPORTED_BENCHMARKS,
    _DEFAULT_EFFORT,
    _DEFAULT_MODEL,
    _RUN_VERSION,
    run_one,
)
from evals.llm.usage import usage_scope

BASELINE = "claude-code"
MODULE = "evals.baselines.claude_code"
_EFFORT_CHOICES = ["low", "medium", "high", "xhigh", "max"]


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the Claude Code baseline (headless `claude -p` agent over the docs). "
                    "One task at a time; not a same-model comparison (runs on Claude via your Max login).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                   help=f"Claude model (alias like 'opus'/'sonnet' or full id). Default {_DEFAULT_MODEL}. "
                        "Pinned full ids are safest (the 'opus' alias may lag the newest).")
    p.add_argument("--effort", type=str, default=_DEFAULT_EFFORT,
                   help="Reasoning effort passed straight to `claude --effort` — free-form so you can "
                        f"test any level the CLI accepts (known: {', '.join(_EFFORT_CHOICES)}). "
                        f"Default {_DEFAULT_EFFORT}.")
    p.add_argument("--max-workers", "-w", type=int, default=1,
                   help="Concurrent tasks. Default 1 (one at a time — Max rate limits + cost).")
    p.add_argument("--limit", type=int, default=None, help="Run at most N PENDING tasks (stable order).")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run identity (model + effort are behavioural). No seed (the CLI
    has none)."""
    return {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "effort": args.effort,
        "run_version": _RUN_VERSION,
    }


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    # Not litellm — the CLI owns transport/auth. We pass the RAW --model id for the CLI's --model.
    return {"model": args.model}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    return [sys.executable, "-m", MODULE,
            "--benchmark", args.benchmark,
            "--model", args.model, "--effort", args.effort,
            "--task-id", task_id, "--run-tag", run_tag]


def _run_one_task(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> None:
    """CHILD MODE: run one task into ``base/{run_tag}/``; manifest.json on success, error.json on failure."""
    run_dir = base / run_tag
    benchmark = _common.load_benchmark_module(args.benchmark)
    try:
        with usage_scope():  # claude_code returns its own usage; the scope is just harness-uniform
            record_extra = run_one(
                benchmark=benchmark, task_id=task_id, run_config=run_config,
                run_dir=run_dir, litellm_kwargs=_litellm_kwargs(args),
            )
    except Exception as exc:  # noqa: BLE001 — record + exit cleanly; no retry.
        _common.write_error(run_dir, _common.build_error_record(task_id, run_config, exc, phase="run_one"))
        return
    usage = record_extra.pop("usage", {})
    manifest = {"task_id": str(task_id), "config": run_config, "usage": usage, **record_extra}
    _common.write_manifest(run_dir, manifest)


def _dispatch(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> str:
    """PARENT: launch one child ONCE into ``base/{run_tag}/``; return "ok"/"errored"."""
    r = _common.run_child(build_child_cmd(args, task_id, run_tag))
    run_dir = base / run_tag
    if (run_dir / _common.MANIFEST_FILE).exists():
        return "ok"
    if (run_dir / _common.ERROR_FILE).exists():
        rec = json.loads((run_dir / _common.ERROR_FILE).read_text())
        print(f"✗ {task_id} ({rec.get('error_type', 'error')})")
        return "errored"
    lines = (r.stderr or "").strip().splitlines()
    detail = "\n".join(lines[-20:]) if lines else f"exit {r.returncode}"
    last = lines[-1] if lines else f"exit {r.returncode}"
    _common.write_error(run_dir, _common.build_error_record(
        task_id, run_config, phase="child_crash", error_type="ChildCrash",
        message=f"child exited {r.returncode}:\n{detail}"))
    print(f"✗ {task_id} (child_crash: {last})")
    return "errored"


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    base = _common.base_dir(args.benchmark, BASELINE)

    # ---- CHILD MODE ----
    if args.task_id is not None:
        if args.run_tag is None:
            raise SystemExit("child mode (--task-id) requires --run-tag (the parent sets it).")
        _run_one_task(args, build_run_config(args), base, args.task_id, args.run_tag)
        return

    # ---- PARENT MODE: resume, then run the rest (one at a time by default). ----
    benchmark = _common.load_benchmark_module(args.benchmark)
    all_ids = benchmark.get_task_ids(**benchmark.STARTER_FILTER)
    if not all_ids:
        print(f"{args.benchmark}/{BASELINE}: nothing to run")
        return

    run_config = build_run_config(args)
    completed = _common.scan_completed_task_ids(base, run_config)
    pending = [tid for tid in all_ids if tid not in completed]
    n_pending = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]
    n_done = len(all_ids) - n_pending
    limit_note = f"; --limit {args.limit}" if args.limit is not None else ""
    print(f"{args.benchmark}/{BASELINE}: {len(all_ids)} total — {n_done} done, "
          f"{n_pending} left; running {len(pending)} now{limit_note}")
    if not pending:
        return

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_dispatch, args, run_config, base, tid, _new_run_tag()): tid
                   for tid in pending}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                status = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"✗ {tid} (dispatch {type(e).__name__}: {e})")
                n_err += 1
                continue
            if status == "ok":
                n_ok += 1
                print(f"✓ {tid}")
            else:
                n_err += 1
    print(f"{args.benchmark}/{BASELINE}: {n_ok} ok, {n_err} errored "
          f"(of {len(pending)} dispatched; {n_pending} were pending{limit_note})")
