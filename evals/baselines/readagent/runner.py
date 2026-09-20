"""Runner for the ReadAgent baseline.

    python -m evals.baselines.readagent --benchmark {loong,corpusqa,dracula} [flags]

SIMPLE NO-INDEX-REUSE logging (the flat per-run-folder layout): each task run gets its OWN folder
``logs/{benchmark}/readagent/{run_tag}/`` with EVERYTHING inside it — the gist memory
(``gist_memory.json``), ``manifest.json`` (with the TOTAL cost), ``calls.json``, and (at
score time) ``score.json``. No ``inferences/`` folder, no shared ``_indices/`` store, and no
content-addressing; a pending task always builds its gist memory from scratch inside its own
folder (no gist memory is ever reused).

It DOES resume, though: the parent scans existing run folders and **skips tasks already
completed for this exact config** (model / seed / ``--config`` / ``run_version`` — matched
against each record's saved ``config``), so a re-run only does what's missing — it does NOT
re-run finished experiments. ``--limit N`` runs the next N PENDING tasks. To force a full
re-run, delete the baseline dir (or the specific run folders).

Only **ReadAgent-P** (one batched look-up call) is wired — it is the only look-up variant
upstream implements in code (ReadAgent-S is a prompt template upstream ships with no
implementation; see PROVENANCE.md D5). There is no variant flag.

Each task runs in a child subprocess ONCE (crash isolation); a caught error → ``error.json``,
a hard crash → the parent writes ``error.json``. Both count as done (a recorded failure isn't
retried — delete its folder to retry).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.readagent.run import SUPPORTED_BENCHMARKS, run_one, run_version_for
from evals.llm.usage import usage_scope
from evals.model_sampling import MODEL_SAMPLING_CONFIG
from evals.settings import DEFAULT_COMPLETION_MODEL

BASELINE = "readagent"
MODULE = "evals.baselines.readagent"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the ReadAgent baseline (ReadAgent-P: paginate → gist → look up → answer). "
                    "Each task gets its own folder; no gist memory is reused; resumes (skips done tasks).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--model", type=str, default=DEFAULT_COMPLETION_MODEL,
                   help="The LLM for every stage (pagination, gisting, look-up, answer). "
                        "ReadAgent uses no embeddings.")
    p.add_argument("--config", type=str, default=None, choices=sorted(MODEL_SAMPLING_CONFIG),
                   help="Named sampling preset (applies to every completion). Recorded in the manifest config.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", type=str, default=None)
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--max-workers", "-w", type=int, default=64)
    p.add_argument("--gist-workers", type=int, default=8,
                   help="INNER-task concurrency: how many pages gist (and how many docs paginate) "
                        "at once WITHIN a task. Default 8. A pure throughput knob — outputs are "
                        "order-preserved + identical, and it is NOT part of the run identity, so it "
                        "never splits resumption/analysis groups (existing sequential runs resume "
                        "under it). NOTE total load on the endpoint is max_workers × gist_workers.")
    p.add_argument("--limit", type=int, default=None, help="Run at most N tasks (stable order).")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run identity the analysis CLI groups by. Purely
    descriptive (no hashing — nothing is reused)."""
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "seed": args.seed,
        "run_version": run_version_for(args.benchmark),  # per-benchmark: loong/dracula "v2", corpusqa "v4"
    }
    if args.config is not None:
        config["config_name"] = args.config
        config["completion_params"] = MODEL_SAMPLING_CONFIG[args.config]
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    # gist_workers rides here (a runtime channel), NOT in run_config — it's a throughput knob,
    # not part of the run identity (so it never splits resumption/analysis). run_one reads it;
    # ReadAgentLLM ignores the extra key.
    return {"model": _common.with_provider_prefix(args.model),
            "api_base": args.base_url, "api_key": args.api_key,
            "gist_workers": args.gist_workers}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    """Argv for a child that runs exactly one task."""
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--seed", str(args.seed),
           "--gist-workers", str(args.gist_workers),
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
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base = _common.base_dir(args.benchmark, BASELINE)

    # ---- CHILD MODE: one task into its run folder, then exit. ----
    if args.task_id is not None:
        if args.run_tag is None:
            raise SystemExit("child mode (--task-id) requires --run-tag (the parent sets it).")
        _run_one_task(args, build_run_config(args), base, args.task_id, args.run_tag)
        return

    # ---- PARENT MODE: RESUME — skip tasks already done for this config, run the rest (each
    #      pending task → its own fresh folder; its gist memory is built from scratch —
    #      resumption skips re-RUNNING, not gist-memory reuse). ----
    benchmark = _common.load_benchmark_module(args.benchmark)
    # STARTER_FILTER is the benchmark's wired subset.
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
