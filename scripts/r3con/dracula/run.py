"""Run R3Con on the Dracula showcase mini-benchmark.

Spawns one ``scripts/r3con/dracula/run_task.py`` subprocess per task (true concurrency +
crash isolation). Each task writes its answer and its per-stage artifacts into its own
folder under ``logs/r3con/``.

Dracula has no filters (one compositional question so far, ``death_toll``, over the whole
corpus ≈ 210k tokens). ``--limit N`` truncates to the first N questions; omit it to run all.

    uv run python scripts/r3con/dracula/run.py --inference both            # all Dracula tasks
    uv run python scripts/r3con/dracula/run.py --limit 1 --inference both  # cheap smoke
"""

from __future__ import annotations

# The repo root on sys.path, so `evals` imports whether this file is run directly
# (`python scripts/r3con/.../x.py`) or the launcher spawns it as a child. The repo is
# used from a checkout and is not pip-installed, matching `python -m evals.baselines.*`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

from evals.r3con.harness import dracula, runner
from evals.r3con.pipeline.logging_setup import configure_logging
from evals.r3con.pipeline.settings import settings

def main(argv: list[str] | None = None) -> int:
    p = runner.build_common_argparser(__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--limit", type=int, default=None,
                   help="Run the first N questions (questions.json order). Omit to run ALL.")
    p.add_argument("--force", action="store_true",
                   help="Re-run tasks already completed for this run (default: resume — skip tasks "
                        "already done for this exact config/prompts/model under every requested strategy).")
    args = p.parse_args(argv)

    configure_logging()
    runner.apply_verbose(args)  # sets R3CON_LOG_LEVEL=INFO for the spawned children
    runner.apply_doc_workers(args)  # sets R3CON_DOC_WORKERS (within-task fan-out) for the children
    config = runner.resolve_config(args)  # configs/<name>.yaml + CLI overrides

    task_ids = dracula.list_task_ids(limit=args.limit)
    if not task_ids:
        raise SystemExit("No Dracula tasks.")

    strategies = runner.strategies_from_arg(args.inference)

    # Resume by default — skip tasks already done for this exact run identity, scoped by
    # `benchmark=` to folders whose manifest records Dracula (the run label carries no
    # benchmark and all three benchmarks share one logs/r3con/).
    n_total = len(task_ids)
    done = runner.completed_task_ids(settings.LOGS_DIR, config.label(), strategies,
                                     benchmark=dracula.NAME)
    n_done = sum(1 for t in task_ids if t in done)
    if args.force:
        print(f"Tasks: {n_total} requested · {n_done} already done · re-running ALL {n_total} (--force).")
    else:
        task_ids = [t for t in task_ids if t not in done]
        print(f"Tasks: {n_total} requested · {n_done} already done · {len(task_ids)} to run "
              f"(--force to re-run the done ones).")
        if not task_ids:
            print("Nothing to run — all requested tasks already done for this run.")
            return 0

    runner.print_run_header(task_ids, config, strategies, api_base=args.base_url, name=dracula.NAME)
    summary = runner.run(
        task_ids, config, strategies=strategies, workers=args.workers,
        api_base=args.base_url, api_key=args.api_key,
        run_task_script="scripts/r3con/dracula/run_task.py",
    )
    runner.print_launch_summary(summary)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
