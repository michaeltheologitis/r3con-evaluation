"""Claude Code baseline pipeline (v1, exploratory).

Per task: materialize the document bundle as files in a temp dir OUTSIDE this repo, run
``claude -p`` headless in that dir (Claude Code's default toolset minus web + ``AskUserQuestion``),
and parse the ``stream-json`` output — the full trajectory (saved to ``trajectory.jsonl``) plus a
final ``result`` message carrying the answer + TOTAL token usage + ``total_cost_usd``.

Auth: we POP ``ANTHROPIC_API_KEY`` from the subprocess env on purpose, so the CLI falls back to the
logged-in (Max subscription) auth — a deliberate choice. Set the key back (or remove the pop)
to bill against the pay-per-token API instead.

Env hygiene: ``_subprocess_env`` POPs ``ANTHROPIC_API_KEY`` (→ Max auth) AND scrubs the Claude-Code
session vars (``CLAUDE*`` / ``AI_AGENT`` / ``BAGGAGE``), so a child launched from inside a Claude Code
session behaves identically to one from a clean terminal (otherwise the inherited env shifts the
toolset and ``CLAUDE_EFFORT`` overrides ``--effort``). The isolation flags drop user-global
settings / hooks / plugins / MCP; the operator's ``~/.claude/CLAUDE.md`` is empty, so no user memory
leaks. (Project-level pollution is moot too — the agent runs in a temp dir outside any repo.)

Cost note: we cost from TOKEN COUNTS at the full rate (NO cache discount), consistent with every
other baseline — ``usage`` carries token counts only, no USD. Anthropic's cache-discounted
``total_cost_usd`` rides along raw in ``trace.cli_result`` for provenance, but is NOT the metric.

SUPPORTED_BENCHMARKS = {loong, corpusqa}.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa"})

_RUN_VERSION = "v1"
_TRAJECTORY_FILE = "trajectory.jsonl"

_DEFAULT_MODEL = "claude-opus-4-8"
_DEFAULT_EFFORT = "high"

# Toolset = Claude Code's DEFAULT set (we do NOT pass ``--tools`` — that allowlist wrongly dropped
# Glob/Grep), with web + ``AskUserQuestion`` removed via ``--disallowedTools`` (a deliberate choice:
# let Workflow exist; just take away web — whatever else Claude Code ships by default is fine):
#   - **web** (WebSearch/WebFetch): the Loong/CorpusQA sources are real public docs, so web access
#     would be RETRIEVING the answer, not grounded reasoning.
#   - **AskUserQuestion**: headless (`-p`) — there is no human to answer, so a call just wastes a turn.
# Everything else Claude Code ships by default is KEPT (Read/Write/Edit/Bash/Glob/Grep/Task/Skill/
# ToolSearch/Workflow/…). Their token cost IS fully captured: the CLI's final ``result.modelUsage``
# is the CUMULATIVE per-model session aggregate — it includes the tokens of EVERY sub-activity
# (Task subagents, Workflow sub-runs, the aux Haiku helper), not just the visible main-thread turns,
# and ``_to_harness_usage`` sums every model key in it. Verified live: in a run whose visible
# main-thread emitted 9 output tokens, ``modelUsage`` reported 811, and ``total_cost_usd`` ==
# Σ ``modelUsage.costUSD`` (the authoritative whole-session bill is computed from exactly what we
# capture). The EXACT granted toolset is recorded per run in ``trace.cli_init.tools`` (its
# ``claude_code_version`` pins which CLI build produced it — the default set can shift across CLI
# versions; e.g. 2.1.181 added Workflow/DesignSync and defers Glob/Grep/Skill behind ToolSearch).
# NOTE: under ``--permission-mode bypassPermissions`` the default set now includes side-effecting
# tools (Cron*/RemoteTrigger/PushNotification/Worktree/ScheduleWakeup); they are very unlikely to
# fire on a local-file QA task, but add them back to this list if you want them denied for safety.
_DISALLOWED_TOOLS = ["WebSearch", "WebFetch", "AskUserQuestion"]

# Isolation so the agent is a CLEAN agent, not the operator's personal Claude Code (no hooks /
# plugins / skills / MCP / memory) — while KEEPING the Max (OAuth) login (``apiKeySource`` stays
# "none"). ``--bare`` would do all this but forces ANTHROPIC_API_KEY auth (drops Max), so we strip
# each source by hand:
#   --setting-sources project     : drop USER-global settings → no personal hooks/plugins/skills
#   --strict-mcp-config + empty    : no MCP servers (no Gmail/Drive/Calendar &c.)
#   --disable-slash-commands       : no slash-command skills
# Verified: this keeps Max auth and removes plugins/MCP/hooks. (A few built-in managed skills still
# LIST in ``init.skills``; ``Skill`` is kept in the toolset, so the agent CAN invoke them — a
# deliberate choice.)
_ISOLATION_FLAGS = [
    "--setting-sources", "project",
    "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}',
    "--disable-slash-commands",
]


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _build_task(benchmark, task_id: str) -> str:
    """The question posed to the agent — the SAME components every other baseline poses (the
    documents are files in the cwd, dropped from the question)."""
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    raise ValueError(f"claude_code has no task assembly for benchmark {name!r}")


def _write_docs(workdir: Path, documents: list[str]) -> list[str]:
    """Write each document as ``doc_NNN.md`` in the working dir; return the filenames."""
    names: list[str] = []
    for i, doc in enumerate(documents):
        name = f"doc_{i + 1:03d}.md"
        (workdir / name).write_text(doc, encoding="utf-8")
        names.append(name)
    return names


def _build_prompt(task: str, doc_names: list[str]) -> str:
    """Pose the question + tell the agent the docs are the files in its cwd (the offload — the
    documents are NOT in the prompt, the agent reads the files)."""
    listing = ", ".join(doc_names) if len(doc_names) <= 8 else f"{', '.join(doc_names[:8])}, …"
    return (
        f"{task}\n\n"
        f"The source documents are the {len(doc_names)} file(s) in your current working directory "
        f"({listing}). Read and analyze them to answer; rely ONLY on these files, not on prior "
        f"knowledge or the web. Provide your final answer directly."
    )


def _build_cmd(prompt: str, model: str, effort: str) -> list[str]:
    return [
        "claude", "-p", prompt,
        "--model", model,
        "--effort", effort,
        "--output-format", "stream-json", "--verbose",
        *_ISOLATION_FLAGS,
        # No --tools allowlist — keep Claude Code's DEFAULT toolset, only DENY web + out-of-band ops.
        "--disallowedTools", *_DISALLOWED_TOOLS,
        "--permission-mode", "bypassPermissions",
        "--no-session-persistence",
        "--exclude-dynamic-system-prompt-sections",
    ]


def _subprocess_env() -> dict[str, str]:
    """A clean env for the CLI so a run is IDENTICAL no matter where it's launched from.

    - **POP ``ANTHROPIC_API_KEY``** → the CLI falls back to the logged-in (Max) auth, not a
      pay-per-token key the harness may have loaded from ``.env``.
    - **STRIP the Claude-Code session vars** (anything starting with ``CLAUDE``, plus ``AI_AGENT`` /
      ``BAGGAGE``). When ``claude -p`` is spawned from INSIDE a Claude Code session, the child
      inherits these and behaves differently — the toolset shifts (Glob/Grep get hidden behind
      tool-search), ``CLAUDE_EFFORT`` silently overrides ``--effort``, and it runs in child-session
      mode. A fresh terminal / SLURM job has none of them, so scrubbing makes every launch context
      match the fresh-terminal baseline. (``ANTHROPIC_BASE_URL`` is kept — it doesn't start with
      ``CLAUDE`` — so a custom endpoint still works; default is api.anthropic.com.)
    """
    return {
        k: v for k, v in os.environ.items()
        if not (k.startswith("CLAUDE") or k in ("AI_AGENT", "BAGGAGE", "ANTHROPIC_API_KEY"))
    }


def _sandbox_prefix(workdir: Path) -> list[str]:
    """macOS: a Seatbelt (``sandbox-exec``) prefix that CONFINES the CLI — and its Bash + every child
    process (the sandbox is inherited) — so it cannot read/write the operator's ``$HOME`` (where
    solutions/other files live), EXCEPT claude's own config/binary. System paths + network stay
    allowed (so auth + the API + the doc workdir still work). The temp workdir lives in ``$TMPDIR``
    (outside ``$HOME``) so it's readable by default; we also allow it explicitly.

    Returns ``[]`` off macOS or if ``sandbox-exec`` is missing — and ``run_one`` then REFUSES to run
    (a Linux/bubblewrap sandbox isn't wired yet), so the agent is never accidentally let loose on the
    whole machine. Verified: ``cat ~/secret`` → "Operation not permitted"; the workdir reads fine."""
    if platform.system() != "Darwin" or not shutil.which("sandbox-exec"):
        return []
    home = str(Path.home())
    profile = (
        "(version 1)\n"
        "(allow default)\n"
        # Block the operator's whole $HOME (solutions, repos, ~/Documents, ~/Library/Messages, …) …
        f'(deny file-read* file-write* (subpath "{home}"))\n'
        # … except the narrow set the CLI genuinely needs (verified empirically):
        f'(allow file-read* file-write* (subpath "{home}/.claude"))\n'          # config/state dir
        f'(allow file-read* file-write* (regex #"^{home}/\\.claude\\.json"))\n'  # config FILE (+ .backup/.tmp)
        f'(allow file-read* (subpath "{home}/.local"))\n'                        # the binary + runtime
        f'(allow file-read* file-write* (subpath "{home}/Library/Keychains"))\n' # Max OAuth token (auth!)
        f'(allow file-read* file-write* (subpath "{home}/Library/Caches"))\n'    # node/claude caches
        # The doc workdir (resolved; in $TMPDIR, outside $HOME — explicit for robustness).
        f'(allow file-read* file-write* (subpath "{workdir}"))\n'
    )
    return ["sandbox-exec", "-p", profile]


def _to_harness_usage(result_msg: dict[str, Any], primary_model: str) -> dict[str, Any]:
    """Map the CLI's ``modelUsage`` into the harness ``{total, calls}`` shape — **token counts only,
    no USD**. ``modelUsage`` is the CLI's CUMULATIVE per-model session total — it already folds in
    EVERY sub-activity (Task subagents, Workflow sub-runs, the aux Haiku helper), so summing every
    model key here captures the WHOLE session, not just the visible main-thread turns (verified live:
    ``total_cost_usd`` == Σ ``modelUsage.costUSD``). We deliberately do NOT save the CLI's per-model ``costUSD``: it bakes in Anthropic's
    cache-discounted dollar pricing, and we want to price from the raw tokens later (at whatever
    per-rate convention we choose). The fields needed for that are the DISAGGREGATED, disjoint input
    buckets the CLI reports — ``input_tokens`` (fresh/uncached), ``cache_read_input_tokens`` (cache
    hits, ~10% rate), ``cache_creation_input_tokens`` (cache writes, ~125% rate) — plus
    ``output_tokens``. ``prompt_tokens`` is their input SUM (fresh + cache-read + cache-creation) and
    ``total_tokens`` adds output — convenience lumps; for accurate cost use the disaggregated fields.

    ``num_calls``: the CLI's ``modelUsage`` exposes per-model TOKENS but no per-model CALL count. The
    agent loop runs on ``primary_model`` (the requested ``--model``) at ~one completion per turn, so
    it gets ``num_turns``. Any OTHER model is Claude Code's internal helper (e.g. a single background
    Haiku call for the session title); its count isn't exposed and is ~1 per session, so we record 1
    rather than ``num_turns`` (which would massively OVERcount it). ``num_turns`` is also kept in
    ``trace``."""
    num_turns = int(result_msg.get("num_turns", 1) or 1)
    total: dict[str, Any] = {}
    for model, mu in (result_msg.get("modelUsage") or {}).items():
        inp = int(mu.get("inputTokens", 0) or 0)
        out = int(mu.get("outputTokens", 0) or 0)
        cache_read = int(mu.get("cacheReadInputTokens", 0) or 0)
        cache_create = int(mu.get("cacheCreationInputTokens", 0) or 0)
        is_primary = model == primary_model or model.startswith(primary_model)
        total[model] = {
            "num_calls": num_turns if is_primary else 1,  # aux models (Haiku helper): count not exposed → 1
            "prompt_tokens": inp + cache_read + cache_create,
            "completion_tokens": out,
            "total_tokens": inp + cache_read + cache_create + out,
            "input_tokens": inp,
            "output_tokens": out,
            "cache_read_input_tokens": cache_read,
            "cache_creation_input_tokens": cache_create,
        }
    return {"total": total, "calls": [{"model": m, "usage": dict(b)} for m, b in total.items()]}


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task Claude Code pipeline. Writes ``trajectory.jsonl`` (the full stream) and returns the
    harness record — ``raw_answer`` + token-count ``usage`` + a ``trace`` that carries the CLI's
    ``result`` and ``init`` messages verbatim (``cli_result`` / ``cli_init``)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = benchmark.get_documents(task_id)
    task = _build_task(benchmark, task_id)
    model = litellm_kwargs.get("model") or _DEFAULT_MODEL
    effort = run_config.get("effort", _DEFAULT_EFFORT)

    # Temp dir OUTSIDE the repo → the agent doesn't auto-discover this repo's CLAUDE.md.
    with tempfile.TemporaryDirectory(prefix="claude_code_") as tmp:
        workdir = Path(tmp).resolve()
        doc_names = _write_docs(workdir, documents)
        prompt = _build_prompt(task, doc_names)
        sandbox = _sandbox_prefix(workdir)
        if not sandbox and os.environ.get("CLAUDE_CODE_ALLOW_UNSANDBOXED") != "1":
            raise RuntimeError(
                "Refusing to run claude_code UNSANDBOXED — the agent's Bash could read the whole "
                "machine. macOS confines it via sandbox-exec automatically; a Linux/bubblewrap "
                "sandbox is not wired yet. Set CLAUDE_CODE_ALLOW_UNSANDBOXED=1 to override (DANGEROUS)."
            )
        proc = subprocess.run(
            [*sandbox, *_build_cmd(prompt, model, effort)],
            cwd=workdir, env=_subprocess_env(),
            stdin=subprocess.DEVNULL,  # -p reads stdin; close it so it doesn't wait ~3s for input
            capture_output=True, text=True,
        )

    # Persist the full trajectory verbatim (every stream-json line) — the rich CodeAct-style log.
    (run_dir / _TRAJECTORY_FILE).write_text(proc.stdout, encoding="utf-8")

    messages: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            messages.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    result_msg = next((m for m in reversed(messages) if m.get("type") == "result"), None)
    if result_msg is None:
        raise RuntimeError(
            f"claude -p produced no result message (exit {proc.returncode}). "
            f"stderr tail: {(proc.stderr or '')[-1000:]}"
        )
    if result_msg.get("is_error"):
        raise RuntimeError(
            f"claude -p returned an error result ({result_msg.get('subtype')}; "
            f"api_error={result_msg.get('api_error_status')}). stderr tail: {(proc.stderr or '')[-500:]}"
        )

    raw_answer = result_msg.get("result") or ""
    init_msg = next(
        (m for m in messages if m.get("type") == "system" and m.get("subtype") == "init"), {}
    )
    return {
        "raw_answer": raw_answer,
        # Harness-standard usage `{total, calls}` (the shape every baseline here emits, so whatever
        # grades/aggregates these logs later — outside this repo — can put claude-code alongside the
        # other baselines): token counts only, no USD,
        # `prompt_tokens` = the FULL input (fresh + cache-read + cache-creation) so cost is priced at
        # the full rate with NO cache discount — consistent with every other baseline. The CLI's RAW
        # per-model/session usage (and its cache-discounted dollar figure) is kept verbatim below.
        "usage": _to_harness_usage(result_msg, model),
        "trace": {
            # What WE constructed (not emitted by the CLI):
            "task": task,
            "n_docs": len(documents),
            "doc_files": doc_names,
            "model": model,       # requested
            "effort": effort,     # requested
            "n_messages": len(messages),
            # The CLI's own messages, saved VERBATIM (native shape, no extraction — the more we keep
            # the better). `cli_result` carries the raw per-model `modelUsage` + session `usage` (the
            # disaggregated input / cache_read / cache_creation / output token counts, server_tool_use,
            # iterations, timings, stop_reason) + `total_cost_usd` (Anthropic's CACHE-DISCOUNTED total —
            # kept for provenance, NOT the cost metric). `cli_init` carries the granted tools / skills /
            # plugins / mcp_servers / apiKeySource / memory_paths / version. (Both also appear in
            # trajectory.jsonl; duplicated here for one-file access when these logs are read later.)
            "cli_result": result_msg,
            "cli_init": init_msg,
        },
    }
