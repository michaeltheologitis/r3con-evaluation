"""Runner for the RLM (Recursive Language Models) baseline.

    python -m evals.baselines.rlm --benchmark {loong,corpusqa,dracula} \
        --model Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 --api-key <key>

RLM is **token-heavy by design** (a code-REPL agent that iterates + spawns recursive sub-LM
calls), so it MUST run against a local vLLM endpoint — ``--base-url`` is **required** and an
OpenAI endpoint is rejected. There is no default OpenAI model here (unlike the other runners).

SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets its OWN folder
``logs/{benchmark}/rlm/{run_tag}/`` holding RLM's full trajectory (the native ``RLMLogger`` jsonl,
written live), ``manifest.json`` (TOTAL tokens) and ``calls.json``. Grading happens outside this
repo. No ``inferences/``, no shared index. It DOES resume — the parent scans run folders
and skips tasks already completed for this exact config (model / seed / max_iterations / max_depth /
run_version). ``--limit N`` runs the next N PENDING tasks.

Each task runs in a child subprocess ONCE (crash isolation; also isolates the in-process REPL ``exec``
to that child); a caught error → ``error.json``, a hard crash → the parent writes ``error.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.rlm.run import SUPPORTED_BENCHMARKS, _RUN_VERSION, run_one
from evals.llm.usage import usage_scope

BASELINE = "rlm"
MODULE = "evals.baselines.rlm"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the RLM (Recursive Language Models) baseline: the document bundle is loaded "
                    "into a Python REPL and a code-agent explores it + spawns recursive sub-LM calls. "
                    "Token-heavy — runs on local vLLM only (no OpenAI). Each task gets its own folder; resumes.",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    # No OpenAI default: --model is required and --base-url must be a non-OpenAI (vLLM) endpoint.
    p.add_argument("--model", type=str, required=True,
                   help="The vLLM-served model id (e.g. Qwen/Qwen3.5-35B-A3B). Used as the root "
                        "and sub-LM. A 'hosted_vllm/'/'openai/' prefix is stripped for the vLLM client.")
    p.add_argument("--base-url", type=str, required=True,
                   help="REQUIRED: the vLLM OpenAI-compatible endpoint (e.g. http://localhost:8555/v1). "
                        "An OpenAI endpoint is rejected — RLM is token-heavy and must run on local vLLM.")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-iterations", type=int, default=30,
                   help="Max root-agent REPL turns (RLM default 30).")
    p.add_argument("--max-depth", type=int, default=1,
                   help="Max recursion depth for rlm_query sub-calls (RLM default 1).")
    p.add_argument("--max-timeout", type=float, default=None,
                   help="Optional per-task wall-clock backstop in seconds (checked between iterations; "
                        "RLM returns its best partial answer if exceeded). Default: none (faithful).")
    p.add_argument("--max-workers", "-w", type=int, default=8,
                   help="Concurrent tasks. RLM is heavy (many calls + local exec per task); keep modest.")
    p.add_argument("--limit", type=int, default=None, help="Run at most N PENDING tasks (then stop).")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def _validate_endpoint(args: argparse.Namespace) -> None:
    """Enforce the no-OpenAI rule structurally (RLM must run on local vLLM)."""
    if not args.base_url or "api.openai.com" in args.base_url:
        raise SystemExit(
            "rlm requires --base-url to be a non-OpenAI (vLLM) endpoint, e.g. "
            "http://localhost:8555/v1. RLM is token-heavy and must run on local vLLM, never OpenAI."
        )


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run identity these logs are grouped by downstream (by whatever
    grades them later) + what resumption keys on.
    Endpoints/secrets (base_url, api_key) are NOT here — they ride in ``litellm_kwargs``."""
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "seed": args.seed,
        "max_iterations": args.max_iterations,
        "max_depth": args.max_depth,
        "max_timeout": args.max_timeout,
        "run_version": _RUN_VERSION,
    }
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    """Transport config (NOT hashed/saved): the vLLM endpoint. ``model`` is passed through and the
    provider prefix is stripped in run.py for RLM's OpenAI client."""
    return {"model": args.model, "api_base": args.base_url, "api_key": args.api_key}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--base-url", args.base_url,
           "--seed", str(args.seed),
           "--max-iterations", str(args.max_iterations),
           "--max-depth", str(args.max_depth),
           "--task-id", task_id, "--run-tag", run_tag]
    if args.max_timeout is not None:
        cmd += ["--max-timeout", str(args.max_timeout)]
    if args.api_key is not None:
        cmd += ["--api-key", args.api_key]
    return cmd


def _run_one_task(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> None:
    """CHILD MODE: run one task into ``base/{run_tag}/``; manifest.json (+ calls.json) on success,
    error.json on failure."""
    run_dir = base / run_tag
    benchmark = _common.load_benchmark_module(args.benchmark)
    try:
        with usage_scope() as task_usage:
            record_extra = run_one(
                benchmark=benchmark, task_id=task_id, run_config=run_config,
                run_dir=run_dir, litellm_kwargs=_litellm_kwargs(args),
            )
    except Exception as exc:  # noqa: BLE001 — record the failure + exit cleanly; no retry.
        _common.write_error(run_dir, _common.build_error_record(task_id, run_config, exc, phase="run_one"))
        return
    usage = record_extra.pop("usage") if "usage" in record_extra else dict(task_usage)
    calls_full = record_extra.pop("calls_full", None)
    manifest = {"task_id": str(task_id), "config": run_config, "usage": usage, **record_extra}
    _common.write_manifest(run_dir, manifest)
    if calls_full is not None:
        (run_dir / "calls.json").write_text(json.dumps(calls_full, indent=2, ensure_ascii=False))


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
    _validate_endpoint(args)
    base = _common.base_dir(args.benchmark, BASELINE)

    # ---- CHILD MODE: one task into its run folder, then exit. ----
    if args.task_id is not None:
        if args.run_tag is None:
            raise SystemExit("child mode (--task-id) requires --run-tag (the parent sets it).")
        _run_one_task(args, build_run_config(args), base, args.task_id, args.run_tag)
        return

    # ---- PARENT MODE: RESUME — skip tasks already done for this config, run the rest. ----
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
