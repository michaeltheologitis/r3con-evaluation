"""Claude Code baseline — pose each task to a headless ``claude -p`` agent over the docs.

EXPLORATORY / v1 (we're still finding the right shape). Per task: write the document bundle to
files in a fresh temp dir OUTSIDE this repo (so the agent doesn't auto-discover our ``CLAUDE.md``),
then run the **Claude Code CLI headless** (`claude -p`) in that dir — constrained to read/search/
compute tools with web DISABLED — and parse its structured output.

This is **NOT a published method** — it's a frontier general-agent reference point. It runs on
**Claude models via the `claude` CLI** (your Max login, or an API key), so it does NOT use the
served vLLM and is not a same-model comparison with the other baselines. Token + cost come straight
from the CLI's result JSON (`usage` + `total_cost_usd`). ``--output-format stream-json`` gives the
full trajectory (every turn: thinking, tool calls [Read/Grep/Bash], tool results) → ``trajectory.jsonl``.

Flat per-run-folder logging (the rlm/readagent layout). Runs **one task at a time** (no inner/outer
parallelism — the maintainer's call for this baseline). SUPPORTED_BENCHMARKS = {loong, corpusqa}.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
