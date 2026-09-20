"""Runner for the structrag baseline.

    python -m evals.baselines.structrag --benchmark {loong,loogle,longbenchv2} [flags]

Self-contained: this runner owns its argparse, run-config, parent fan-out, and
child-mode (one task per subprocess, for crash isolation). It shares only the
small stateless mechanism in ``evals.baselines._common`` — there is no central
runner that dispatches every baseline. StructRAG restructures documents per task,
so (like direct-llm) there is no ``setup()`` and no index / search knobs.

This file is harness PLUMBING (not vendored StructRAG code); it mirrors
``direct_llm/runner.py``. The StructRAG-specific work lives in ``run.py`` (the
connector) and ``upstream/`` (the vendored pipeline).

Failure policy (no infinite retries): each task runs in a child subprocess ONCE.
On a caught ``run_one`` error the child writes ``error.json`` and exits cleanly; if
a child dies hard the parent writes ``error.json`` itself. Either file counts as
done — a deterministic failure is recorded, not retried. To retry, delete its
``error.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.structrag.run import SUPPORTED_BENCHMARKS, run_one
from evals.llm.usage import usage_scope
from evals.model_sampling import MODEL_SAMPLING_CONFIG
from evals.settings import DEFAULT_COMPLETION_MODEL

BASELINE = "structrag"
MODULE = "evals.baselines.structrag"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the StructRAG baseline on one benchmark (route → structurize → utilize).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS),
                   help="Which benchmark to run (StructRAG supports these).")
    p.add_argument("--model", type=str, default=DEFAULT_COMPLETION_MODEL)
    p.add_argument("--config", type=str, default=None, choices=sorted(MODEL_SAMPLING_CONFIG),
                   help="Named sampling preset from evals.model_sampling.MODEL_SAMPLING_CONFIG "
                        "(temperature / top_p / extra_body / …). Applied to EVERY StructRAG "
                        "completion call. Folded into the run identity (hashed + recorded in the "
                        "manifest). Omit for model/provider defaults.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--max-workers", "-w", type=int, default=64)
    p.add_argument("--limit", type=int, default=None,
                   help="Run at most N PENDING tasks (then stop). Re-run for the next N.")
    p.add_argument("--task-id", type=str, default=None,
                   help="Child mode: run this one task and exit.")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The inference identity (what gets hashed). No retrieval/index knobs for
    structrag — same shape as direct-llm.

    A ``--config`` preset folds BOTH its resolved ``completion_params`` (so the hash
    reflects exactly what was sent) AND its ``config_name`` (a readable label) into
    the identity. Without ``--config`` the shape is unchanged.
    """
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "seed": args.seed,
    }
    if args.config is not None:
        config["config_name"] = args.config
        config["completion_params"] = MODEL_SAMPLING_CONFIG[args.config]
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    return {"model": _common.with_provider_prefix(args.model),
            "api_base": args.base_url, "api_key": args.api_key}


def build_child_cmd(args: argparse.Namespace, task_id: str) -> list[str]:
    """Argv for a child that runs exactly one task."""
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--seed", str(args.seed),
           "--task-id", task_id]
    if args.config is not None:
        cmd += ["--config", args.config]
    if args.base_url is not None:
        cmd += ["--base-url", args.base_url]
    if args.api_key is not None:
        cmd += ["--api-key", args.api_key]
    return cmd


def _run_one_task(args: argparse.Namespace, run_config: dict, base, task_id: str) -> None:
    """CHILD MODE: run one task; write manifest.json on success, error.json on failure."""
    inf_dir = _common.inference_dir(base, _common.compute_inference_hash(run_config, task_id))
    benchmark = _common.load_benchmark_module(args.benchmark)
    try:
        with usage_scope() as task_usage:
            record_extra = run_one(
                benchmark=benchmark, task_id=task_id, run_config=run_config,
                base_dir=base, litellm_kwargs=_litellm_kwargs(args),
            )
    except Exception as exc:  # noqa: BLE001 — record the failure + exit cleanly; no retry.
        _common.write_error(inf_dir, _common.build_error_record(task_id, run_config, exc, phase="run_one"))
        return
    # Prefer usage the baseline captured straight from its responses (deterministic;
    # StructRAG fires many sequential SYNC completions, which the usage_scope callback
    # path would under-count). Fall back to the scope tally only if absent.
    usage = record_extra.pop("usage") if "usage" in record_extra else dict(task_usage)
    calls_full = record_extra.pop("calls_full", None)
    manifest = {"task_id": str(task_id), "config": run_config,
                "usage": usage, **record_extra}
    _common.write_manifest(inf_dir, manifest)
    # The full request/response of every internal LLM call, beside the manifest: the
    # manifest holds the model OUTPUT + token cost; calls.json holds the raw call trace
    # (router / per-doc structurize / decompose / extract / merge) — everything sent and
    # returned via litellm, with the generated content (and a thinking model's
    # reasoning_content) surfaced as explicit per-call fields. StructRAG fans out many
    # calls per task, so this is where the detail lives. Can be large (each structurize
    # call carries a whole document).
    if calls_full is not None:
        (inf_dir / "calls.json").write_text(json.dumps(calls_full, indent=2, ensure_ascii=False))


def _dispatch(args: argparse.Namespace, run_config: dict, base, task_id: str) -> str:
    """PARENT: launch one child ONCE (no retry). Ensure exactly one of
    manifest.json / error.json exists, returning "ok" or "errored"."""
    r = _common.run_child(build_child_cmd(args, task_id))
    inf_dir = _common.inference_dir(base, _common.compute_inference_hash(run_config, task_id))
    if (inf_dir / _common.MANIFEST_FILE).exists():
        return "ok"
    if (inf_dir / _common.ERROR_FILE).exists():
        rec = json.loads((inf_dir / _common.ERROR_FILE).read_text())
        print(f"✗ {task_id} ({rec.get('error_type', 'error')})")
        return "errored"
    # Child died before recording anything (hard crash / OOM-kill): record it here.
    lines = (r.stderr or "").strip().splitlines()
    detail = "\n".join(lines[-20:]) if lines else f"exit {r.returncode}"
    last = lines[-1] if lines else f"exit {r.returncode}"
    _common.write_error(inf_dir, _common.build_error_record(
        task_id, run_config, phase="child_crash", error_type="ChildCrash",
        message=f"child exited {r.returncode}:\n{detail}"))
    print(f"✗ {task_id} (child_crash: {last})")
    return "errored"


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_config = build_run_config(args)
    base = _common.base_dir(args.benchmark, BASELINE)

    # ---- CHILD MODE: one task, then exit. ----
    if args.task_id is not None:
        _run_one_task(args, run_config, base, args.task_id)
        return

    # ---- PARENT MODE: fan out (no setup for an inference-time baseline). ----
    benchmark = _common.load_benchmark_module(args.benchmark)
    all_ids = benchmark.get_task_ids(**benchmark.STARTER_FILTER)
    completed = _common.scan_completed_inference_hashes(base)
    pending = [tid for tid in all_ids
               if _common.compute_inference_hash(run_config, tid) not in completed]
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
        futures = {pool.submit(_dispatch, args, run_config, base, tid): tid for tid in pending}
        for fut in as_completed(futures):
            tid = futures[fut]
            try:
                status = fut.result()
            except Exception as e:  # noqa: BLE001 — a dispatch-level failure shouldn't kill the run.
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
