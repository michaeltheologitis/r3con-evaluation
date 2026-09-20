"""Run ONE Dracula task end-to-end through the R3Con pipeline; write its artifacts.

Spawned once per task by the launcher (``scripts/r3con/dracula/run.py`` via
:func:`evals.r3con.harness.runner.run`), which passes ``--run-folder``; also runnable by hand. Writes
``logs/r3con/<run-folder>/…`` and exits non-zero iff R3Con hit a top-level failure.

    uv run python scripts/r3con/dracula/run_task.py --task-id death_toll --inference both --config default
"""

from __future__ import annotations

# The repo root on sys.path, so `evals` imports whether this file is run directly
# (`python scripts/r3con/.../x.py`) or the launcher spawns it as a child. The repo is
# used from a checkout and is not pip-installed, matching `python -m evals.baselines.*`.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse

from evals.r3con.pipeline.config import available_configs, available_sampling_presets, load_config
from evals.r3con.harness import runner
from evals.r3con.harness.dracula.run_task import run_task
from evals.r3con.pipeline.logging_setup import configure_logging
from evals.r3con.pipeline.runs import new_run_folder

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--task-id", required=True)
    p.add_argument("--config", choices=available_configs(), default="default")
    p.add_argument("--model", default=None, help="Override the config's model.")
    p.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    p.add_argument("--summary-rounds", type=int, default=None, metavar="N")
    p.add_argument("--sampling", choices=available_sampling_presets(), default=None,
                   help="Override the sampling preset (configs/sampling/<name>.yaml).")
    p.add_argument("--inference", choices=["llm", "codeact", "both"], default="codeact")
    p.add_argument(
        "--run-folder", default=None,
        help="Write into logs/<run-folder>/ (the launcher passes this; a hand-run gets a fresh one).",
    )
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)

    configure_logging("INFO" if args.verbose else None)
    config = load_config(args.config, model=args.model, seed=args.seed,
                         summary_rounds=args.summary_rounds, sampling=args.sampling)
    run_folder = args.run_folder or new_run_folder()
    strategies = runner.strategies_from_arg(args.inference)

    outcome = run_task(
        args.task_id, config=config, run_folder=run_folder, strategies=strategies,
        api_base=args.base_url, api_key=args.api_key,
    )

    bits = [f"{strategy}=FAIL({err[:60]})" if err else f"{strategy}=ok"
            for strategy, (ans, err) in outcome.results.items()]
    print(f"{args.task_id} {'FAILED' if outcome.failed else 'OK'} {' '.join(bits)}")
    return 1 if outcome.failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
