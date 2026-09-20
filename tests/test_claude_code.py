"""Tests for the Claude Code baseline connector (``evals.baselines.claude_code``).

claude-code shells out to the ``claude -p`` CLI; there is nothing to vendor and no Python LLM seam.
The harness-side pieces are tested here with fakes — the ``claude`` subprocess is replaced by a
canned ``stream-json`` stdout, so NO real CLI / Max login is needed and the whole file runs by
default (no marker, like linearrag):

- ``run``    — _build_task / _build_prompt / _write_docs / _subprocess_env / _build_cmd /
               _sandbox_prefix / _to_harness_usage, and a run_one that drives the REAL parsing of
               the stream with only ``subprocess.run`` faked (so trajectory persistence + the
               result/init extraction + the usage mapping all run for real).
- ``runner`` — the run-config identity, the child command, child-mode manifest/error writes, and
               resumption (skip tasks already done for the config).
"""
from __future__ import annotations

import json
import os
import platform
from types import SimpleNamespace

import pytest

from evals.baselines.claude_code import run as cc_run


# ============================================================
# run — pure helpers
# ============================================================


def test_build_task_loong_and_corpusqa() -> None:
    loong = SimpleNamespace(__name__="evals.benchmarks.loong", get_task=lambda t: ("INSTR", "Q", ["d"]))
    loong_empty = SimpleNamespace(__name__="evals.benchmarks.loong",
                                  get_task=lambda t: ("INSTR-ONLY", "  ", ["d"]))
    corpusqa = SimpleNamespace(__name__="evals.benchmarks.corpusqa",
                               get_task=lambda t: ("OUTPUT-REQS", "Q2", ["d"]))
    assert cc_run._build_task(loong, "t") == "INSTR\n\nQ"
    assert cc_run._build_task(loong_empty, "t") == "INSTR-ONLY"        # empty question → instruction only
    assert cc_run._build_task(corpusqa, "t") == "Q2\n\nOUTPUT-REQS"    # output-reqs after the question


def test_build_task_rejects_unsupported_benchmark() -> None:
    other = SimpleNamespace(__name__="evals.benchmarks.nosuchbench", get_task=lambda t: ("a", "b", "c"))
    with pytest.raises(ValueError, match="no task assembly"):
        cc_run._build_task(other, "t")


def test_write_docs_names_and_contents(tmp_path) -> None:
    names = cc_run._write_docs(tmp_path, ["alpha", "beta", "gamma"])
    assert names == ["doc_001.md", "doc_002.md", "doc_003.md"]
    assert (tmp_path / "doc_002.md").read_text() == "beta"


def test_build_prompt_points_at_files_and_truncates_long_listings() -> None:
    short = cc_run._build_prompt("TASK", ["doc_001.md", "doc_002.md"])
    assert short.startswith("TASK")
    assert "doc_001.md, doc_002.md" in short
    assert "rely ONLY on these files" in short and "not on prior" in short
    # >8 docs → the listing is truncated with an ellipsis (but the count is exact).
    many = cc_run._build_prompt("T", [f"doc_{i:03d}.md" for i in range(1, 13)])
    assert "12 file(s)" in many and "…" in many
    assert "doc_012.md" not in many                      # truncated past the 8th


def test_subprocess_env_pops_key_and_scrubs_session_vars(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-be-popped")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://example.test/v1")  # kept (no CLAUDE prefix)
    monkeypatch.setenv("CLAUDE_EFFORT", "low")                            # scrubbed (would override --effort)
    monkeypatch.setenv("CLAUDECODE", "1")                                # scrubbed (CLAUDE* prefix)
    monkeypatch.setenv("AI_AGENT", "claude")                             # scrubbed
    monkeypatch.setenv("BAGGAGE", "x=y")                                 # scrubbed
    monkeypatch.setenv("PATH_SENTINEL_KEEP", "yes")                      # unrelated → kept

    env = cc_run._subprocess_env()

    assert "ANTHROPIC_API_KEY" not in env                # → falls back to the Max login
    assert "CLAUDE_EFFORT" not in env and "CLAUDECODE" not in env
    assert "AI_AGENT" not in env and "BAGGAGE" not in env
    assert env["ANTHROPIC_BASE_URL"] == "https://example.test/v1"
    assert env["PATH_SENTINEL_KEEP"] == "yes"


def test_build_cmd_keeps_default_toolset_and_denies_web_and_askuser() -> None:
    cmd = cc_run._build_cmd("PROMPT", "claude-opus-4-8", "high")
    assert cmd[:3] == ["claude", "-p", "PROMPT"]
    assert "--model" in cmd and "claude-opus-4-8" in cmd
    assert cmd[cmd.index("--effort") + 1] == "high"
    # We pass NO --tools allowlist (it wrongly dropped Glob/Grep); the default set is kept and we
    # DENY only web via --disallowedTools.
    assert "--tools" not in cmd
    assert "--disallowedTools" in cmd
    for denied in ("WebSearch", "WebFetch", "AskUserQuestion"):  # web + the headless no-op tool
        assert denied in cmd
    # Everything else Claude Code ships by default is KEPT (not denied) — incl. Workflow/Task/Skill/
    # ToolSearch. (Their token cost is still fully captured via result.modelUsage.)
    for kept in ("Workflow", "Task", "Skill", "ToolSearch", "Glob", "Grep"):
        assert kept not in cmd
    # headless + isolation flags present.
    assert cmd[cmd.index("--permission-mode") + 1] == "bypassPermissions"
    assert "--setting-sources" in cmd and "--disable-slash-commands" in cmd
    assert "--strict-mcp-config" in cmd and '{"mcpServers":{}}' in cmd


def test_sandbox_prefix_empty_off_darwin(monkeypatch) -> None:
    monkeypatch.setattr(cc_run.platform, "system", lambda: "Linux")
    assert cc_run._sandbox_prefix("/tmp/wd") == []        # no Seatbelt off macOS → run_one will refuse


def test_sandbox_prefix_confines_home_on_darwin(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(cc_run.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cc_run.shutil, "which", lambda name: "/usr/bin/sandbox-exec")
    monkeypatch.setattr(cc_run.Path, "home", staticmethod(lambda: tmp_path / "home"))
    prefix = cc_run._sandbox_prefix(tmp_path / "workdir")
    assert prefix[0] == "sandbox-exec" and prefix[1] == "-p"
    profile = prefix[2]
    assert "(allow default)" in profile
    assert f'(deny file-read* file-write* (subpath "{tmp_path / "home"}"))' in profile
    assert str(tmp_path / "workdir") in profile           # the doc workdir is re-allowed
    assert "Library/Keychains" in profile                 # the Max OAuth token stays readable


def test_to_harness_usage_maps_modelusage_and_attributes_calls() -> None:
    result_msg = {
        "num_turns": 4,
        "modelUsage": {
            "claude-opus-4-8": {
                "inputTokens": 8, "outputTokens": 1291,
                "cacheReadInputTokens": 112084, "cacheCreationInputTokens": 6030,
            },
            "claude-haiku-4-5-20251001": {
                "inputTokens": 738, "outputTokens": 17,
                "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0,
            },
        },
    }
    usage = cc_run._to_harness_usage(result_msg, "claude-opus-4-8")
    opus = usage["total"]["claude-opus-4-8"]
    haiku = usage["total"]["claude-haiku-4-5-20251001"]
    # primary model gets num_turns; the aux Haiku helper's call count isn't exposed → 1 (not over-counted).
    assert opus["num_calls"] == 4 and haiku["num_calls"] == 1
    # prompt_tokens = the FULL input (fresh + cache-read + cache-creation) → priced at the full rate.
    assert opus["prompt_tokens"] == 8 + 112084 + 6030 == 118122
    assert opus["completion_tokens"] == 1291 and opus["total_tokens"] == 119413
    # disaggregated buckets preserved so cost can be recomputed at any rate convention later.
    assert opus["input_tokens"] == 8 and opus["cache_read_input_tokens"] == 112084
    assert opus["cache_creation_input_tokens"] == 6030
    # the {total, calls} shape the analysis/cost layer consumes.
    assert {c["model"] for c in usage["calls"]} == {"claude-opus-4-8", "claude-haiku-4-5-20251001"}


# ============================================================
# run — run_one over a canned stream-json (only subprocess.run faked)
# ============================================================


def _canned_stream(answer: str = "THE ANSWER", num_turns: int = 4) -> str:
    """A minimal but faithful ``claude -p --output-format stream-json`` stdout: an init message,
    an assistant turn, and the final result message carrying the answer + modelUsage."""
    messages = [
        {"type": "system", "subtype": "init", "apiKeySource": "none", "mcp_servers": [],
         "plugins": [], "tools": ["Read", "Grep", "Bash", "Glob", "Skill"],
         "claude_code_version": "2.1.150"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": "reading docs"}]}},
        {"type": "result", "subtype": "success", "is_error": False, "num_turns": num_turns,
         "result": answer, "total_cost_usd": 0.0764,
         "modelUsage": {
             "claude-opus-4-8": {"inputTokens": 8, "outputTokens": 1291,
                                 "cacheReadInputTokens": 112084, "cacheCreationInputTokens": 6030},
             "claude-haiku-4-5-20251001": {"inputTokens": 738, "outputTokens": 17,
                                           "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
         }},
    ]
    return "\n".join(json.dumps(m) for m in messages) + "\n"


def _loong_benchmark():
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: ["Paris is the capital of France.", "Berlin is in Germany."],
        get_task=lambda tid: ("Answer from the documents.", "What is the capital of France?", []),
    )


def _patch_unsandboxed(monkeypatch) -> None:
    """Make run_one portable: pretend there's no Seatbelt and explicitly allow unsandboxed, so the
    test runs identically on macOS and Linux CI (the real sandbox is exercised by
    ``test_sandbox_prefix_confines_home_on_darwin``)."""
    monkeypatch.setattr(cc_run, "_sandbox_prefix", lambda wd: [])
    monkeypatch.setenv("CLAUDE_CODE_ALLOW_UNSANDBOXED", "1")


def test_run_one_persists_trajectory_and_returns_record(tmp_path, monkeypatch) -> None:
    _patch_unsandboxed(monkeypatch)
    captured: dict = {}

    def fake_run(cmd, **kwargs):  # noqa: ANN003
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return SimpleNamespace(stdout=_canned_stream(), stderr="", returncode=0)

    monkeypatch.setattr(cc_run.subprocess, "run", fake_run)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-popped")

    run_config = {"benchmark": "loong", "baseline": "claude-code", "model": "claude-opus-4-8",
                  "effort": "high", "run_version": cc_run._RUN_VERSION}
    rec = cc_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path,
                         {"model": "claude-opus-4-8"})

    # raw_answer comes straight from the CLI result message.
    assert rec["raw_answer"] == "THE ANSWER"
    # usage is the {total, calls} shape; the FULL input is in prompt_tokens (no cache discount).
    assert rec["usage"]["total"]["claude-opus-4-8"]["prompt_tokens"] == 118122
    assert rec["usage"]["total"]["claude-opus-4-8"]["num_calls"] == 4
    # the CLI's result + init messages are kept VERBATIM in the trace (proof of Max + clean + version).
    assert rec["trace"]["cli_result"]["total_cost_usd"] == 0.0764
    assert rec["trace"]["cli_init"]["apiKeySource"] == "none"
    assert rec["trace"]["n_docs"] == 2 and rec["trace"]["doc_files"][0] == "doc_001.md"
    assert "capital of France" in rec["trace"]["task"]
    # the full stream-json is persisted verbatim inside the run folder.
    traj = (tmp_path / cc_run._TRAJECTORY_FILE).read_text()
    assert traj == _canned_stream()
    # the CLI was invoked in a temp workdir OUTSIDE this repo, with the API key popped from its env.
    assert "ANTHROPIC_API_KEY" not in captured["env"]
    assert str(tmp_path) not in str(captured["cwd"])     # docs live in $TMPDIR, not the run folder


def test_run_one_refuses_unsandboxed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cc_run, "_sandbox_prefix", lambda wd: [])     # no Seatbelt available
    monkeypatch.delenv("CLAUDE_CODE_ALLOW_UNSANDBOXED", raising=False)
    monkeypatch.setattr(cc_run.subprocess, "run", lambda *a, **k: pytest.fail("must not shell out unsandboxed"))
    with pytest.raises(RuntimeError, match="UNSANDBOXED"):
        cc_run.run_one(_loong_benchmark(), "t1",
                       {"benchmark": "loong", "model": "claude-opus-4-8", "effort": "high"},
                       tmp_path, {"model": "claude-opus-4-8"})


def test_run_one_raises_on_no_result_message(tmp_path, monkeypatch) -> None:
    _patch_unsandboxed(monkeypatch)
    monkeypatch.setattr(cc_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout="", stderr="boom", returncode=1))
    with pytest.raises(RuntimeError, match="no result message"):
        cc_run.run_one(_loong_benchmark(), "t1",
                       {"benchmark": "loong", "model": "claude-opus-4-8", "effort": "high"},
                       tmp_path, {"model": "claude-opus-4-8"})


def test_run_one_raises_on_error_result(tmp_path, monkeypatch) -> None:
    _patch_unsandboxed(monkeypatch)
    err = json.dumps({"type": "result", "is_error": True, "subtype": "error_max_turns",
                      "api_error_status": None}) + "\n"
    monkeypatch.setattr(cc_run.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(stdout=err, stderr="", returncode=0))
    with pytest.raises(RuntimeError, match="error result"):
        cc_run.run_one(_loong_benchmark(), "t1",
                       {"benchmark": "loong", "model": "claude-opus-4-8", "effort": "high"},
                       tmp_path, {"model": "claude-opus-4-8"})


# ============================================================
# runner — config identity, child command, child-mode writes, resumption
# ============================================================

from evals.baselines.claude_code import runner as cc_runner  # noqa: E402
from evals.baselines import _common  # noqa: E402


def test_build_run_config_shape() -> None:
    p = cc_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "corpusqa", "--model", "claude-opus-4-8", "--effort", "max"])
    cfg = cc_runner.build_run_config(a)
    assert cfg == {"benchmark": "corpusqa", "baseline": "claude-code",
                   "model": _common.canonical_model_id("claude-opus-4-8"),
                   "effort": "max", "run_version": cc_run._RUN_VERSION}
    assert "seed" not in cfg and "completion_params" not in cfg     # the CLI takes neither


def test_effort_is_free_form_passthrough() -> None:
    p = cc_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "loong", "--effort", "xhigh"])
    assert a.effort == "xhigh" and cc_runner.build_run_config(a)["effort"] == "xhigh"
    # an UNLISTED value is accepted (no `choices` restriction) — the CLI itself validates it, and the
    # effort is part of the run identity so a different effort is its own config (own resumption set).
    a2 = p.parse_args(["--benchmark", "loong", "--effort", "experimental-99"])
    assert cc_runner.build_run_config(a2)["effort"] == "experimental-99"


def test_arg_parser_restricts_benchmarks_and_defaults() -> None:
    p = cc_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "loong"])
    assert a.model == cc_run._DEFAULT_MODEL and a.effort == cc_run._DEFAULT_EFFORT
    assert a.max_workers == 1                                       # one task at a time by default
    with pytest.raises(SystemExit):
        p.parse_args(["--benchmark", "nosuchbench"])               # unsupported → argparse rejects


def test_build_child_cmd_carries_task_and_run_tag() -> None:
    p = cc_runner.build_arg_parser()
    a = p.parse_args(["--benchmark", "loong", "--model", "claude-opus-4-8", "--effort", "high"])
    cmd = cc_runner.build_child_cmd(a, "task-xyz", "rtag123")
    assert "--task-id" in cmd and cmd[cmd.index("--task-id") + 1] == "task-xyz"
    assert "--run-tag" in cmd and cmd[cmd.index("--run-tag") + 1] == "rtag123"
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_child_mode_writes_manifest_with_usage(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cc_runner, "run_one",
                        lambda **k: {"raw_answer": "A", "usage": {"total": {"m": {"num_calls": 1}}},
                                     "trace": {"n_docs": 1}})
    args = cc_runner.build_arg_parser().parse_args(["--benchmark", "loong"])
    cfg = cc_runner.build_run_config(args)
    cc_runner._run_one_task(args, cfg, tmp_path, "t1", "rtag")
    manifest = json.loads((tmp_path / "rtag" / _common.MANIFEST_FILE).read_text())
    assert manifest["task_id"] == "t1" and manifest["config"] == cfg
    assert manifest["raw_answer"] == "A" and manifest["usage"]["total"]["m"]["num_calls"] == 1
    assert manifest["trace"] == {"n_docs": 1}


def test_child_mode_writes_error_on_failure(tmp_path, monkeypatch) -> None:
    def boom(**k):
        raise RuntimeError("claude blew up")
    monkeypatch.setattr(cc_runner, "run_one", boom)
    args = cc_runner.build_arg_parser().parse_args(["--benchmark", "loong"])
    cfg = cc_runner.build_run_config(args)
    cc_runner._run_one_task(args, cfg, tmp_path, "t1", "rtag")
    err = json.loads((tmp_path / "rtag" / _common.ERROR_FILE).read_text())
    assert err["task_id"] == "t1" and err["error_type"] == "RuntimeError"
    assert err["phase"] == "run_one" and "blew up" in err["message"]
    assert not (tmp_path / "rtag" / _common.MANIFEST_FILE).exists()


def test_runner_resumes_and_does_not_rerun_completed(tmp_path, monkeypatch) -> None:
    fake = SimpleNamespace(get_task_ids=lambda **k: ["t1", "t2", "t3"], STARTER_FILTER={})
    monkeypatch.setattr(_common, "load_benchmark_module", lambda n: fake)
    monkeypatch.setattr(_common, "base_dir", lambda b, bl: tmp_path)
    dispatched: list[str] = []

    def fake_dispatch(args, run_config, base, tid, run_tag):
        dispatched.append(tid)
        d = base / run_tag
        d.mkdir(parents=True, exist_ok=True)
        _common.write_manifest(d, {"task_id": tid, "config": run_config})
        return "ok"

    monkeypatch.setattr(cc_runner, "_dispatch", fake_dispatch)
    argv = ["--benchmark", "loong", "--model", "claude-opus-4-8"]
    cc_runner.main(argv)
    assert sorted(dispatched) == ["t1", "t2", "t3"]   # first run: all 3
    dispatched.clear()
    cc_runner.main(argv)
    assert dispatched == []                            # re-run: all done → nothing re-run
