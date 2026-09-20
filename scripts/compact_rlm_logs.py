"""Compact existing RLM trajectories: drop the REPL-variable snapshots, keep the trajectory.

RLM's native trajectory (``logs/{benchmark}/rlm/{run_tag}/rlm_*.jsonl``) re-serialized the whole
REPL namespace — the full document bundle twice, plus every slice derived from it — after every
code block, which was 94-99.6% of each file. New runs no longer write it
(``evals/baselines/rlm/logger.py``); this script applies the SAME rule (``drop_repl_locals``) to
trajectories written before that: every ``locals`` is removed, everything else — metadata line,
prompts, responses, code, stdout/stderr, sub-calls, final answers — is kept.

Each file is rewritten line by line into ``<file>.compacting`` and swapped in with ``os.replace``
only after the line count matches, so an interruption never leaves a half-written trajectory (a
stale ``.compacting`` is simply overwritten next time). Lines without ``locals`` are copied
byte-for-byte, and a file with no ``locals`` left is not rewritten, so re-running is a no-op.
Don't run it while an rlm batch is still writing into these folders.

    python scripts/compact_rlm_logs.py                     # every rlm benchmark
    python scripts/compact_rlm_logs.py --benchmark loong
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Allow direct invocation as a path script (`python scripts/compact_rlm_logs.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.baselines.rlm.logger import drop_repl_locals  # noqa: E402
from evals.settings import _slug, settings  # noqa: E402

BASELINE = "rlm"


def compact_file(path: Path) -> tuple[int, int, bool]:
    """Drop ``locals`` from one trajectory file in place. Returns ``(bytes_before, bytes_after,
    rewritten)``; a file with no ``locals`` key is not rewritten. (Inside logged text a quote is
    JSON-escaped, so the byte check ``"locals"`` only matches the key itself.)"""
    path = Path(path)
    before = path.stat().st_size
    tmp = path.with_name(path.name + ".compacting")
    changed = False
    n_in = n_out = 0
    with open(path, "rb") as src, open(tmp, "wb") as dst:
        for line in src:
            n_in += 1
            if b'"locals"' in line:
                # Same serializer RLMLogger uses (json.dump defaults), so the kept fields are
                # written exactly as before.
                line = (json.dumps(drop_repl_locals(json.loads(line))) + "\n").encode("utf-8")
                changed = True
            dst.write(line)
            n_out += 1
    if not changed:
        tmp.unlink()
        return before, before, False
    if n_in != n_out:  # defensive: never swap in a file that lost lines
        tmp.unlink()
        raise RuntimeError(f"{path}: line count changed ({n_in} -> {n_out}); left untouched")
    os.replace(tmp, path)
    return before, path.stat().st_size, True


def trajectory_files(logs_dir: Path, benchmark: str | None = None) -> list[Path]:
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    return sorted(logs_dir.glob(f"{benchmark_glob}/{BASELINE}/*/rlm_*.jsonl"))


def compact_rlm_logs(logs_dir: Path, *, benchmark: str | None = None, workers: int = 4) -> dict:
    """Compact every rlm trajectory under ``logs_dir``. Returns per-benchmark
    ``{files, rewritten, bytes_before, bytes_after}``."""
    files = trajectory_files(logs_dir, benchmark)
    summary: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "rewritten": 0, "bytes_before": 0, "bytes_after": 0})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, (path, (before, after, rewritten)) in enumerate(
                zip(files, pool.map(compact_file, files)), start=1):
            s = summary[path.parents[2].name]
            s["files"] += 1
            s["rewritten"] += rewritten
            s["bytes_before"] += before
            s["bytes_after"] += after
            if i % 100 == 0 or i == len(files):
                print(f"  {i}/{len(files)} files", file=sys.stderr, flush=True)
    return dict(summary)


def _format_summary(summary: dict) -> str:
    if not summary:
        return "no rlm trajectories found"
    lines = []
    for bench in sorted(summary):
        s = summary[bench]
        lines.append(f"{bench}: rewrote {s['rewritten']}/{s['files']} trajectories, "
                     f"{s['bytes_before'] / 1e9:.2f} GB -> {s['bytes_after'] / 1e6:.1f} MB")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python scripts/compact_rlm_logs.py",
        description="Drop the REPL-variable snapshots (`locals`) from RLM trajectory files, keeping "
                    "the trajectory itself (prompts, responses, code, stdout, sub-calls).",
    )
    parser.add_argument("--benchmark", default=None, help="Only this benchmark (default: all).")
    parser.add_argument("--workers", type=int, default=4,
                        help="Files compacted in parallel (each holds one iteration line in memory).")
    parser.add_argument("--logs-dir", default=None,
                        help="Override the logs root (default: the project's logs/ dir).")
    args = parser.parse_args(argv)
    logs_dir = Path(args.logs_dir) if args.logs_dir else settings.LOGS_DIR
    print(_format_summary(compact_rlm_logs(logs_dir, benchmark=args.benchmark, workers=args.workers)))


if __name__ == "__main__":
    main()
