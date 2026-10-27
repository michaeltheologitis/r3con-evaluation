"""Run the "w/o structuring" ablation: CodeAct over each finished run's summaries, with no parse.

Over every ``logs/r3con/<run-folder>/`` that has computed summaries and does not yet have the
ablation output, runs :mod:`evals.r3con.harness.ablation_nostruct` — recomputing nothing upstream
and changing nothing in the pipeline core. The result lands in each folder as a third strategy,
``inference/codeact_ablation_nostruct/``, beside ``llm`` and ``codeact``.

    uv run python scripts/r3con/ablation_nostruct.py \
        --model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 \
        --api-key <key> --workers 20 \
        --label 'default[model=Qwen3.5-35B-A3B,seed=42,sr=2,prompts=(sum=v3,prop=v4,ext=v2,llm=v4,cod=v4)]'

``--label`` restricts to folders whose recomputed run label matches exactly (the paper ran it over
the default configuration above); ``--benchmark`` restricts to one benchmark's folders. One
subprocess per folder, as in the method's runner; each child's output goes to the folder's
``ablation_nostruct.log``. Nothing here grades anything.
"""

from __future__ import annotations

# The repo root on sys.path, so `evals` imports whether this file is run directly
# (`python scripts/r3con/ablation_nostruct.py`) or spawned as its own worker child.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import subprocess
import sys
import time

from evals.r3con.harness import ablation_nostruct
from evals.r3con.pipeline.logging_setup import configure_logging
from evals.r3con.pipeline.settings import settings

_TERMINATE_GRACE_S = 10.0


def _worker(args: argparse.Namespace) -> int:
    """Worker mode: ablate ONE folder (``--folder``)."""
    ans, err = ablation_nostruct.ablate_one(
        args.folder, model=args.model, api_base=args.base_url, api_key=args.api_key
    )
    print(f"{args.folder} {'FAIL(' + err[:50] + ')' if err else 'ok'}")
    return 1 if err else 0


def _child_argv(folder: str, args: argparse.Namespace) -> list[str]:
    argv = [
        sys.executable, str(settings.ROOT / "scripts" / "r3con" / "ablation_nostruct.py"),
        "--folder", folder, "--model", args.model,
    ]
    if args.base_url:
        argv += ["--base-url", args.base_url]
    if args.api_key:
        argv += ["--api-key", args.api_key]
    if args.verbose:
        argv += ["--verbose"]
    return argv


def _launch(folders: list[str], args: argparse.Namespace) -> int:
    n = len(folders)
    workers = max(1, args.workers)
    print(f"=== ablating {n} folder(s) [codeact, summaries-only] · ≤{workers} concurrent subprocess(es) ===")
    queue = list(folders)
    inflight: dict = {}  # proc -> (folder, open logfile)
    ok = failed = 0

    def start(folder: str) -> None:
        # A file, not a pipe, for the child's output — a pipe could fill and deadlock.
        logf = open(settings.LOGS_DIR / folder / "ablation_nostruct.log", "w")
        try:
            proc = subprocess.Popen(
                _child_argv(folder, args), stdout=logf, stderr=subprocess.STDOUT,
                cwd=str(settings.ROOT), start_new_session=True,
            )
        except BaseException:
            logf.close()
            raise
        inflight[proc] = (folder, logf)

    try:
        while queue or inflight:
            while queue and len(inflight) < workers:
                start(queue.pop(0))
            finished = [p for p in inflight if p.poll() is not None]
            for p in finished:
                folder, logf = inflight.pop(p)
                logf.close()
                if int(p.returncode) == 0:
                    ok += 1
                    print(f"  ✓ {folder}")
                else:
                    failed += 1
                    print(f"  ✗ {folder} (exit {p.returncode})")
            if not finished:
                time.sleep(0.05)
    except KeyboardInterrupt:
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
            except (subprocess.TimeoutExpired, OSError, ValueError):
                try:
                    p.kill()
                except OSError:
                    pass
        print(f"\n⚠  interrupted — stopped {len(live)} running worker(s). Re-running picks the folders back up.")
        raise
    finally:
        for _folder, logf in inflight.values():
            try:
                logf.close()
            except OSError:
                pass

    print(f"\n=== done: {ok} ok · {failed} failed ===")
    print(f"Answers: {settings.LOGS_DIR}/<run-folder>/inference/{ablation_nostruct.STRATEGY}/result.json")
    return 0 if failed == 0 else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    p.add_argument("--model", required=True,
                   help="Routable model string for the call, e.g. hosted_vllm/Qwen/Qwen3.5-35B-A3B. "
                        "(The manifest's bare model name is identity, not routable.)")
    p.add_argument("--base-url", default=None, help="API base URL (e.g. a vLLM endpoint).")
    p.add_argument("--api-key", default=None, help="API key passed to the model call.")
    p.add_argument("--workers", type=int, default=10, help="Max folders in parallel (default: 10).")
    p.add_argument("--label", default=None,
                   help="Restrict to folders whose recomputed run label matches this exactly.")
    p.add_argument("--benchmark", default=None, choices=["Loong", "CorpusQA", "Dracula"],
                   help="Restrict to one benchmark's folders (the run label spans benchmarks).")
    p.add_argument("--limit", type=int, default=None, help="Ablate at most N folders (a safe subset).")
    p.add_argument("--folder", default=None, help="(worker mode) ablate this one folder and exit.")
    p.add_argument("--verbose", action="store_true", help="Stream per-stage progress at INFO.")
    args = p.parse_args(argv)
    configure_logging("INFO" if args.verbose else None)

    if args.folder:
        return _worker(args)

    folders = ablation_nostruct.folders_to_ablate(
        settings.LOGS_DIR, label=args.label, benchmark=args.benchmark
    )
    if args.limit is not None:
        folders = folders[: args.limit]
    if not folders:
        print("Nothing to ablate (no folders with summaries missing the ablation output).")
        return 0
    return _launch(folders, args)


if __name__ == "__main__":
    raise SystemExit(main())
