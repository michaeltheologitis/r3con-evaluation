"""Runner for the HippoRAG 2 baseline.

    python -m evals.baselines.hipporag --benchmark {loong,corpusqa,dracula} \
        [--model openai/gpt-5.4-nano] [--base-url URL --api-key KEY] [flags]

SIMPLE NO-REUSE logging (the readagent/rlm/memagent layout): each task run gets its OWN
folder ``logs/{benchmark}/hipporag/{run_tag}/`` with EVERYTHING inside — the HippoRAG
index under ``index/`` (the OpenIE knowledge graph + embedding stores), ``manifest.json``
(TOTAL cost — every OpenIE / filter / reader call folded into one number) and
``calls.json``. No ``inferences/``, no shared index, no content-addressing; a pending
task always rebuilds its graph from scratch. Nothing here grades a run — these folders
are what an external scoring repo reads.

It DOES resume: the parent scans existing run folders and skips tasks already completed
for this exact config (model / seed / chunk_size / embedding_model / run_version).
``--limit N`` runs the next N PENDING. To force a full re-run, delete the baseline dir.

Runs on OpenAI OR a vLLM endpoint (``--base-url``) — HippoRAG emits JSON (no tool-calling
needed); embeddings ALWAYS go to OpenAI. **Expensive** (per-passage OpenIE at index time),
so run a stratified ``--limit`` subset, not full sets. Each task runs in a child
subprocess ONCE (crash isolation).
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from evals.baselines import _common
from evals.baselines.hipporag.chunker import CHUNK_SIZES
from evals.baselines.hipporag.run import EMBEDDING_MODEL, SUPPORTED_BENCHMARKS, _RUN_VERSION, run_one
from evals.llm.usage import usage_scope
from evals.settings import DEFAULT_COMPLETION_MODEL

BASELINE = "hipporag"
MODULE = "evals.baselines.hipporag"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the HippoRAG 2 baseline (OpenIE knowledge graph + Personalized "
                    "PageRank). Each task gets its own folder; resumes (skips done tasks).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--model", type=str, default=DEFAULT_COMPLETION_MODEL,
                   help="The completion model — drives OpenIE (NER + triples), the recognition-"
                        "memory triple filter, and the QA reader. OpenAI or vLLM (--base-url).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--base-url", type=str, default=None,
                   help="vLLM endpoint for the completion model (embeddings still go to OpenAI).")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--max-workers", "-w", type=int, default=8,
                   help="Tasks processed in parallel (each is a subprocess). Indexing fires many "
                        "OpenIE calls per task, so total endpoint load ≈ max_workers × passages.")
    p.add_argument("--limit", type=int, default=None, help="Run at most N PENDING tasks (stable order).")
    p.add_argument("--tasks", nargs="+", default=None, metavar="VARIANT",
                   help="Restrict to these task-id variants — the part after '@' in the id "
                        "(corpusqa: the context-length tier, e.g. --tasks 1m; corpusqa is "
                        "1m-only, so that selects everything there). A parent-mode "
                        "selection filter only: NOT part of a task's identity, so a task's "
                        "manifest is byte-identical whether or not you filter, and a later "
                        "full run resumes the ones already done. Loong and dracula ids carry "
                        "no '@', so the filter matches nothing there.")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run's identity: resumption skips a task already done
    under it, and whatever reads these logs groups by it. The per-benchmark ``chunk_size``
    changes the run's output (passage boundaries → the whole graph), so it is part of the
    identity."""
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "embedding_model": EMBEDDING_MODEL,
        "seed": args.seed,
        "chunk_size": CHUNK_SIZES[args.benchmark],
        "run_version": _RUN_VERSION,
    }
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    return {"model": _common.with_provider_prefix(args.model),
            "api_base": args.base_url, "api_key": args.api_key}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def _filter_task_variants(task_ids: list[str], variants) -> list[str]:
    """Keep only task ids whose ``@``-suffix variant is in ``variants`` (the ``--tasks`` filter).
    The variant is the part after the last ``@`` (corpusqa ``…@1m``); ids with no ``@``
    (loong, dracula) never match, so ``--tasks`` is a no-op-then-error there."""
    wanted = set(variants)
    return [t for t in task_ids if t.rsplit("@", 1)[-1] in wanted]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    """Argv for a child that runs exactly one task."""
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--seed", str(args.seed),
           "--task-id", task_id, "--run-tag", run_tag]
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
    if args.tasks:  # parent-mode subset filter (e.g. --tasks 1m); children still run one id each
        all_ids = _filter_task_variants(all_ids, args.tasks)
        if not all_ids:
            raise SystemExit(
                f"--tasks {sorted(set(args.tasks))} matched no {args.benchmark} task ids "
                f"(the variant is the part after '@' in the id, e.g. '1m' for CorpusQA).")
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
