"""Claude Code baseline — pose each task to a headless ``claude -p`` agent over the docs.

EXPLORATORY / v1 (we're still finding the right shape). Per task: write the document bundle to
files in a fresh temp dir OUTSIDE this repo (so the agent doesn't auto-discover our ``CLAUDE.md``),
then run the **Claude Code CLI headless** (`claude -p`) in that dir — Claude Code's default toolset
minus web and ``AskUserQuestion`` — and parse its structured output.

This is **NOT a published method** — it's a frontier general-agent reference point. It runs on
**Claude models via the `claude` CLI** (your Max login — the env's ``ANTHROPIC_API_KEY`` is popped
for the CLI), so it does NOT use the served vLLM and is not a same-model comparison with the other
baselines. Token COUNTS come straight from the CLI's result JSON (its cumulative per-model
``modelUsage``); the CLI's cache-discounted ``total_cost_usd`` rides along raw in the trace for
provenance, but is NOT the cost metric. ``--output-format stream-json`` gives the full trajectory
(every turn: thinking, tool calls [Read/Grep/Bash], tool results) → ``trajectory.jsonl``.

Flat per-run-folder logging (the rlm/readagent layout). Runs **one task at a time** (no inner/outer
parallelism — a deliberate choice for this baseline). SUPPORTED_BENCHMARKS = {loong, corpusqa}.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
