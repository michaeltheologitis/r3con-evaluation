"""Runner for the MemAgent baseline.

    python -m evals.baselines.memagent --benchmark {loong,corpusqa,dracula} \
        --model BytedTsinghua-SIA/RL-MemoryAgent-14B \
        --base-url http://localhost:8000/v1 --api-key <key> [flags]

SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets its OWN folder
``logs/{benchmark}/memagent/{run_tag}/`` with EVERYTHING inside — the memory trajectory
(``memory_trajectory.json``), ``manifest.json`` (TOTAL cost), ``calls.json``, and a
live ``progress.json``. No ``inferences/``, no shared index, no content-addressing; a
pending task always rebuilds from scratch. Nothing here grades — the raw answer is saved
as-is for the external scoring repo that reads these logs.

It DOES resume: the parent scans existing run folders and **skips tasks already completed
for this exact config** (model / seed / chunk_size / max_new / max_context_len / --config
/ run_version) — so a re-run only does what's missing. ``--limit N`` runs the next N
PENDING. To force a full re-run, delete the baseline dir.

MemAgent's model IS the RL-trained checkpoint (default ``BytedTsinghua-SIA/RL-MemoryAgent-14B``,
served on vLLM) — pass ``--base-url``/``--api-key`` to the vLLM endpoint. Each task runs in
a child subprocess ONCE (crash isolation).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.memagent.run import SUPPORTED_BENCHMARKS, _RUN_VERSION, run_one
from evals.llm.usage import usage_scope
from evals.model_sampling import MODEL_SAMPLING_CONFIG

BASELINE = "memagent"
MODULE = "evals.baselines.memagent"

# MemAgent IS the RL-trained checkpoint (Qwen2.5-14B-Instruct-based); default to the 14B.
# NOT the harness's DEFAULT_COMPLETION_MODEL — a generic model isn't MemAgent.
DEFAULT_MODEL = "BytedTsinghua-SIA/RL-MemoryAgent-14B"

# Upstream canonical inference constants (quickstart.py): 5000-token chunks, 1024-token
# memory/answer cap. max_context_len defaults to 0 (no clip) — see chunker.py / PROVENANCE.
_DEFAULT_CHUNK_SIZE = 5000
_DEFAULT_MAX_NEW = 1024
_DEFAULT_MAX_CONTEXT_LEN = 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the MemAgent baseline (recurrent fixed-size memory over token "
                    "chunks). Each task gets its own folder; resumes (skips done tasks).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--model", type=str, default=DEFAULT_MODEL,
                   help=f"The RL-trained MemAgent model (default: {DEFAULT_MODEL}), served on "
                        "vLLM. Its tokenizer drives the token-window chunker.")
    p.add_argument("--config", type=str, default=None, choices=sorted(MODEL_SAMPLING_CONFIG),
                   help="Named sampling preset (applies to every completion). Recorded in the config.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", type=str, default=None,
                   help="vLLM endpoint for the RL model (REQUIRED for the default model).")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--chunk-size", type=int, default=_DEFAULT_CHUNK_SIZE,
                   help=f"Tokens per memory-update chunk (upstream default {_DEFAULT_CHUNK_SIZE}).")
    p.add_argument("--max-new", type=int, default=_DEFAULT_MAX_NEW,
                   help=f"max_tokens per call = the memory/answer size cap (upstream default "
                        f"{_DEFAULT_MAX_NEW}; load-bearing — bounds the fixed memory).")
    p.add_argument("--max-context-len", type=int, default=_DEFAULT_MAX_CONTEXT_LEN,
                   help="Clip the context to head+tail this many tokens BEFORE chunking (0 = no "
                        "clip; upstream demo used 120000). We default to 0 so the method processes "
                        "the whole leave-no-doc-behind bundle.")
    p.add_argument("--max-workers", "-w", type=int, default=32,
                   help="Number of tasks processed in parallel (each is a subprocess). MemAgent "
                        "fires many sequential calls per task, so total endpoint load ≈ max_workers.")
    p.add_argument("--limit", type=int, default=None, help="Run at most N PENDING tasks (stable order).")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run identity: resumption matches on it, and whatever
    analyses these logs later groups by it. The chunk knobs (chunk_size / max_new /
    max_context_len) change the run's output, so they are part of the identity (a different
    chunk size is a different run)."""
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "max_new": args.max_new,
        "max_context_len": args.max_context_len,
        "run_version": _RUN_VERSION,
    }
    if args.config is not None:
        config["config_name"] = args.config
        config["completion_params"] = MODEL_SAMPLING_CONFIG[args.config]
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    return {"model": _common.with_provider_prefix(args.model),
            "api_base": args.base_url, "api_key": args.api_key}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    """Argv for a child that runs exactly one task."""
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--seed", str(args.seed),
           "--chunk-size", str(args.chunk_size), "--max-new", str(args.max_new),
           "--max-context-len", str(args.max_context_len),
           "--task-id", task_id, "--run-tag", run_tag]
    if args.config is not None:
        cmd += ["--config", args.config]
    if args.base_url is not None:
        cmd += ["--base-url", args.base_url]
    if args.api_key is not None:
        cmd += ["--api-key", args.api_key]
    return cmd


def _run_one_task(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> None:
    """CHILD MODE: run one task into ``base/{run_tag}/``; write manifest.json (+ calls.json)
    on success, error.json on failure."""
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
