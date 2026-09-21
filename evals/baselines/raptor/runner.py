"""Runner for the RAPTOR baseline.

    python -m evals.baselines.raptor --benchmark {loong,corpusqa,dracula} [flags]

SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets its OWN folder
``logs/{benchmark}/raptor/{run_tag}/`` with EVERYTHING inside it — the built ``tree.json``,
``manifest.json`` (with the TOTAL cost), ``calls.json``, and a live ``progress.json``. Grading
is not this repo's job — the separate scoring repo reads these folders. No ``inferences/``
folder, no shared ``_indices/`` store, no content-addressing; a pending task always rebuilds
its tree from scratch inside its own folder (no tree is ever reused).

**Offline-embed / online-build split** (``--phase {all,embed,build}``): ``embed`` runs phase 1 — the
OpenAI leaf embeddings (the heavy ~82% burst), NO vLLM — and writes a checkpoint (``embed.json`` +
``leaves.pkl``); ``build`` runs phase 2 (cluster + LLM-summarize + QA) against the served LLM,
reusing the checkpoint's run folder. ``all`` (default) does both in one pass. ``--embed-workers``
optionally caps the leaf-embed concurrency so the OpenAI burst doesn't rate-limit.

It DOES resume (config-scoped): ``embed``/``all`` skip tasks already done; ``build`` finds the
existing ``embed.json`` checkpoints not yet built and reuses their folders. ``--limit N`` runs the
next N PENDING. To force a full re-run, delete the baseline dir.

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
from evals.baselines.raptor.run import (
    SUPPORTED_BENCHMARKS, _EMBED_FILE, _RUN_VERSION,
    build_one, embed_one, load_embed, run_one, save_embed,
)
from evals.llm.usage import usage_scope
from evals.settings import DEFAULT_COMPLETION_MODEL, DEFAULT_EMBEDDING_MODEL

BASELINE = "raptor"
MODULE = "evals.baselines.raptor"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=f"python -m {MODULE}",
        description="Run the RAPTOR baseline (recursive-summary tree → collapse-tree retrieval). "
                    "Each task gets its own folder; no tree is reused; resumes (skips done tasks).",
    )
    p.add_argument("--benchmark", required=True, choices=sorted(SUPPORTED_BENCHMARKS))
    p.add_argument("--model", type=str, default=DEFAULT_COMPLETION_MODEL,
                   help="The LLM for BOTH the cluster summaries (tree build) and the QA answer.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paper-hparams", action="store_true",
                   help="Use RAPTOR's PUBLISHED tree hyperparameters (chunk 100 / recluster 3500 / "
                        "summary 100 / retrieval top-10 @3500 tok) instead of the D12 large-corpus "
                        "re-scale. For docs in RAPTOR's validated regime (<=~78K tok); "
                        "do NOT use on the 1M-token corpora (chunk=100 explodes the leaf/summary count). "
                        "Folded into the run config (hashed), so paper vs D12 runs never mix.")
    p.add_argument("--base-url", type=str, default=None,
                   help="Route the completion LLM through a vLLM/OpenAI-compatible endpoint "
                        "(embeddings always go to OpenAI).")
    p.add_argument("--api-key", type=str, default=None)
    p.add_argument("--embedding-model", type=str, default=DEFAULT_EMBEDDING_MODEL,
                   help="OpenAI embedding model for the tree nodes + query (embeddings-always-openai).")
    p.add_argument("--phase", choices=("all", "embed", "build"), default="all",
                   help="all (default): embed + build in one pass (backward compatible). "
                        "embed: the OFFLINE leaf embeddings only → writes embed.json + leaves.pkl "
                        "(NO vLLM; run it before/without the GPU, rate-limited). build: cluster + "
                        "LLM-summarize + QA over the saved leaves (needs the vLLM). Split so the "
                        "rate-limit-prone OpenAI embedding burst doesn't tie up / contend with the LLM phase.")
    p.add_argument("--embed-workers", type=int, default=None,
                   help="Cap the per-task leaf-embed concurrency (rate-limit the OpenAI burst). "
                        "Default: RAPTOR's unbounded ThreadPool. Applies to --phase embed and all.")
    p.add_argument("--summary-workers", type=int, default=None,
                   help="Cap/RAISE the per-task cluster-SUMMARY concurrency (the inner tree-build "
                        "ThreadPool). Default (None): RAPTOR's min(32, cpu+4) — UNCHANGED. Lets you "
                        "raise inner LLM concurrency without more --max-workers subprocesses. "
                        "Output-identical; applies to --phase build and all.")
    p.add_argument("--max-workers", "-w", type=int, default=8,
                   help="Tasks in flight at once (each task builds its own tree). NOTE RAPTOR also "
                        "multithreads WITHIN a task, so endpoint load is higher than this number.")
    p.add_argument("--limit", type=int, default=None, help="Run at most N tasks (stable order).")
    p.add_argument("--tasks", nargs="+", default=None, metavar="VARIANT",
                   help="Restrict to these task-id variants — the part after '@' in the id "
                        "(corpusqa: the context-length tier, e.g. --tasks 1m). A parent-mode "
                        "selection filter only: NOT part of a task's identity, so a task's "
                        "manifest is byte-identical whether or not you filter, and a later "
                        "full run resumes the ones already done. Only corpusqa ids carry a "
                        "variant, and it has a single wired tier (1m), so this can only select "
                        "all of them or nothing.")
    p.add_argument("--task-id", type=str, default=None, help="Child mode: run this one task and exit.")
    p.add_argument("--run-tag", type=str, default=None,
                   help="Child mode: the unique folder name for this task run (set by the parent).")
    return p


def build_run_config(args: argparse.Namespace) -> dict:
    """The manifest ``config`` — the run identity resumption keys on, and what an external
    consumer of these logs groups runs by.
    ``--phase`` / ``--embed-workers`` / ``--summary-workers`` are NOT here (they don't change a run's
    output — same embeddings, same tree — so a split run is byte-compatible with ``--phase all``)."""
    config = {
        "benchmark": args.benchmark,
        "baseline": BASELINE,
        "model": _common.canonical_model_id(args.model),
        "seed": args.seed,
        "embedding_model": args.embedding_model,
        "run_version": _RUN_VERSION,
    }
    # Only add the key when set → config stays byte-identical (same hash) for existing D12 runs.
    if args.paper_hparams:
        config["paper_hparams"] = True
    return config


def _litellm_kwargs(args: argparse.Namespace) -> dict:
    return {"model": _common.with_provider_prefix(args.model),
            "api_base": args.base_url, "api_key": args.api_key}


def _new_run_tag() -> str:
    return uuid.uuid4().hex[:12]


def build_child_cmd(args: argparse.Namespace, task_id: str, run_tag: str) -> list[str]:
    """Argv for a child that runs exactly one task (for this phase)."""
    cmd = [sys.executable, "-m", MODULE,
           "--benchmark", args.benchmark,
           "--model", args.model, "--seed", str(args.seed),
           "--embedding-model", args.embedding_model,
           "--phase", args.phase,
           "--task-id", task_id, "--run-tag", run_tag]
    if args.embed_workers is not None:
        cmd += ["--embed-workers", str(args.embed_workers)]
    if args.summary_workers is not None:
        cmd += ["--summary-workers", str(args.summary_workers)]
    if args.paper_hparams:
        cmd += ["--paper-hparams"]
    if args.base_url is not None:
        cmd += ["--base-url", args.base_url]
    if args.api_key is not None:
        cmd += ["--api-key", args.api_key]
    return cmd


def _run_one_task(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> None:
    """CHILD MODE (phase-aware): run one task into ``base/{run_tag}/``.
      - embed: leaf embeddings (no vLLM) → ``embed.json`` + ``leaves.pkl``; on failure ``error.json``.
      - build: load the checkpoint → cluster+summarize+QA → ``manifest.json`` (+ ``calls.json``).
      - all:   embed + build in one shot → ``manifest.json`` (+ ``calls.json``).
    A recorded failure is not retried (delete its folder to retry)."""
    run_dir = base / run_tag
    benchmark = _common.load_benchmark_module(args.benchmark)
    try:
        if args.phase == "embed":
            checkpoint = embed_one(benchmark, task_id, run_config, run_dir,
                                   embed_workers=args.embed_workers)
            save_embed(run_dir, task_id, run_config, checkpoint)
            return  # phase 1 produces embed.json + leaves.pkl, NOT a manifest
        if args.phase == "build":
            checkpoint = load_embed(run_dir)
            with usage_scope() as task_usage:
                record_extra = build_one(
                    benchmark, task_id, run_config, run_dir, _litellm_kwargs(args), checkpoint,
                    summary_workers=args.summary_workers)
        else:  # all
            with usage_scope() as task_usage:
                record_extra = run_one(
                    benchmark=benchmark, task_id=task_id, run_config=run_config,
                    run_dir=run_dir, litellm_kwargs=_litellm_kwargs(args),
                    embed_workers=args.embed_workers, summary_workers=args.summary_workers)
    except Exception as exc:  # noqa: BLE001 — record the failure + exit cleanly; no retry.
        _common.write_error(run_dir, _common.build_error_record(task_id, run_config, exc, phase=args.phase))
        return
    usage = record_extra.pop("usage") if "usage" in record_extra else dict(task_usage)
    calls_full = record_extra.pop("calls_full", None)
    manifest = {"task_id": str(task_id), "config": run_config, "usage": usage, **record_extra}
    _common.write_manifest(run_dir, manifest)
    if calls_full is not None:
        (run_dir / "calls.json").write_text(json.dumps(calls_full, indent=2, ensure_ascii=False))


def _dispatch(args: argparse.Namespace, run_config: dict, base, task_id: str, run_tag: str) -> str:
    """PARENT: launch one child ONCE into ``base/{run_tag}/``; return "ok"/"errored". Success is an
    ``embed.json`` for ``--phase embed``, else a ``manifest.json``."""
    r = _common.run_child(build_child_cmd(args, task_id, run_tag))
    run_dir = base / run_tag
    success_file = _EMBED_FILE if args.phase == "embed" else _common.MANIFEST_FILE
    if (run_dir / success_file).exists():
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


def _scan_phase_state(base, run_config: dict) -> tuple[dict[str, str], set[str]]:
    """Map this config's existing run folders to their phase state:
      - ``embedded``: ``{task_id -> run_tag}`` for folders with a phase-1 ``embed.json`` (awaiting build)
      - ``completed``: ``{task_id}`` for folders with a ``manifest.json`` / ``error.json`` (fully done)
    Config-scoped (canonical-JSON equality), exactly like ``_common.scan_completed_task_ids``. A task
    can be in both; ``completed`` wins. (``embed.json`` records its ``task_id`` + ``config`` for the
    match, like manifest/error.)"""
    embedded: dict[str, str] = {}
    completed: set[str] = set()
    if not base.exists():
        return embedded, completed
    target = json.dumps(run_config, sort_keys=True)
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        for filename in (_common.MANIFEST_FILE, _common.ERROR_FILE):
            path = folder / filename
            if not path.exists():
                continue
            try:
                rec = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                break  # half-written → treat as pending
            if rec.get("task_id") and json.dumps(rec.get("config", {}), sort_keys=True) == target:
                completed.add(str(rec["task_id"]))
            break  # exactly one of manifest/error per folder
        epath = folder / _EMBED_FILE
        if epath.exists():
            try:
                rec = json.loads(epath.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if rec.get("task_id") and json.dumps(rec.get("config", {}), sort_keys=True) == target:
                embedded[str(rec["task_id"])] = folder.name
    return embedded, completed


def _pending_for_phase(args, all_ids, embedded: dict, completed: set) -> list[tuple[str, str]]:
    """The (task_id, run_tag) units to dispatch for this phase, in stable ``all_ids`` order:
      - build: EXISTING checkpoints (``embed.json``) not yet built → reuse their run_tag.
      - embed: tasks with no checkpoint and not done → a fresh run_tag.
      - all:   tasks not done → a fresh run_tag (today's behavior)."""
    if args.phase == "build":
        return [(tid, embedded[tid]) for tid in all_ids
                if tid in embedded and tid not in completed]
    if args.phase == "embed":
        return [(tid, _new_run_tag()) for tid in all_ids
                if tid not in embedded and tid not in completed]
    return [(tid, _new_run_tag()) for tid in all_ids if tid not in completed]


def _filter_task_variants(task_ids: list[str], variants) -> list[str]:
    """Keep only task ids whose ``@``-suffix variant is in ``variants`` (the ``--tasks`` filter).
    The variant is the part after the last ``@`` (corpusqa ``…@1m``); ids with no ``@`` (Loong,
    Dracula) never match, so ``--tasks`` is a no-op-then-error there."""
    wanted = set(variants)
    return [t for t in task_ids if t.rsplit("@", 1)[-1] in wanted]


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_config = build_run_config(args)
    base = _common.base_dir(args.benchmark, BASELINE)

    # ---- CHILD MODE: one task into its run folder, then exit. ----
    if args.task_id is not None:
        if args.run_tag is None:
            raise SystemExit("child mode (--task-id) requires --run-tag (the parent sets it).")
        _run_one_task(args, run_config, base, args.task_id, args.run_tag)
        return

    # ---- PARENT MODE: RESUME (config-scoped) per phase — see _pending_for_phase. `build` reuses the
    #      phase-1 checkpoint folders; `embed`/`all` mint fresh folders for not-yet-done tasks. ----
    benchmark = _common.load_benchmark_module(args.benchmark)
    all_ids = benchmark.get_task_ids(**benchmark.STARTER_FILTER)
    if args.tasks:  # parent-mode subset filter (e.g. --tasks 1m); children still run one id each
        all_ids = _filter_task_variants(all_ids, args.tasks)
        if not all_ids:
            raise SystemExit(
                f"--tasks {sorted(set(args.tasks))} matched no {args.benchmark} task ids "
                f"(the variant is the part after '@' in the id, e.g. '1m' for corpusqa).")
    if not all_ids:
        print(f"{args.benchmark}/{BASELINE}: nothing to run")
        return

    embedded, completed = _scan_phase_state(base, run_config)
    pending = _pending_for_phase(args, all_ids, embedded, completed)
    n_pending = len(pending)
    if args.limit is not None:
        pending = pending[:args.limit]
    limit_note = f"; --limit {args.limit}" if args.limit is not None else ""
    print(f"{args.benchmark}/{BASELINE} [--phase {args.phase}]: {len(all_ids)} tasks, "
          f"{n_pending} pending, running {len(pending)} now{limit_note}")
    if args.phase == "build" and n_pending == 0 and (set(all_ids) - completed - set(embedded)):
        print("  (no embed.json checkpoints to build — run `--phase embed` first)")
    if not pending:
        return

    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_dispatch, args, run_config, base, tid, tag): tid
                   for tid, tag in pending}
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
    done_word = "embedded" if args.phase == "embed" else "answered"
    print(f"{args.benchmark}/{BASELINE} [--phase {args.phase}]: {n_ok} {done_word}, {n_err} errored "
          f"(of {len(pending)} dispatched; {n_pending} were pending{limit_note})")


if __name__ == "__main__":  # pragma: no cover
    main()
