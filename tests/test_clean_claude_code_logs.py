"""Tests for ``scripts/clean_claude_code_logs.py``.

Synthetic log trees only (no real ``claude -p`` runs). claude-code uses the same flat layout as
ReadAgent / RLM / CodeAct: each task run is ONE folder
``{benchmark}/claude-code/{run_tag}/`` directly under the baseline dir (no ``inferences/`` level, no
shared ``_indices/`` store), with the verbatim ``trajectory.jsonl`` stream INSIDE it. So these pin:
run folders are classified directly; a removed junk folder takes its ``trajectory.jsonl`` with it;
successful runs + genuine model failures are kept; the HTTP-429 rate-limit ``RuntimeError`` is
transient (cleared so the task re-runs); other baselines' layouts are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_claude_code_logs import clean_claude_code_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_traj: bool = False) -> Path:
    """Create ``{base}/{name}/`` (a run folder directly under the baseline dir) holding ``files``,
    optionally with a ``trajectory.jsonl`` (claude-code keeps the verbatim stream inside the run folder)."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_traj:
        (d / "trajectory.jsonl").write_text(json.dumps({"type": "result", "result": "x"}) + "\n")
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_junk_keeps_real_and_drops_their_trajectory(tmp_path) -> None:
    base = tmp_path / "loong" / "claude-code"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_traj=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_traj=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_traj=True)
    incomplete = base / "r_incomplete"          # killed mid-run: an empty folder, no manifest/error
    incomplete.mkdir(parents=True)

    summary = clean_claude_code_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert (manifest / "trajectory.jsonl").exists() and (cwe / "trajectory.jsonl").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2                  # manifest + context-window


def test_rate_limit_runtimeerror_is_transient_and_cleared(tmp_path) -> None:
    """The HTTP-429 rate-limit error claude-code hits under the Max plan is a ``RuntimeError``
    (NOT in FAILURE_ERROR_TYPES), so it classifies as transient and is cleared so the task re-runs."""
    base = tmp_path / "loong" / "claude-code"
    rate_limited = _mk(base, "r_429", _err("RuntimeError"), with_traj=True)
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_claude_code_logs(tmp_path)
    assert not rate_limited.exists() and cwe.exists()
    assert summary["by_category"].get("transient") == 1
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "corpusqa" / "claude-code"
    crash = _mk(base, "r_c", _err("ChildCrash"), with_traj=True)
    summary = clean_claude_code_logs(tmp_path, dry_run=True)
    assert crash.exists() and (crash / "trajectory.jsonl").exists()
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "claude-code"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_claude_code_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()
    assert summary["by_category"].get("real_error") == 1


def test_only_touches_claude_code_not_other_baselines(tmp_path) -> None:
    # arag (its inferences/ layout) must be untouched by the claude-code cleaner.
    arag = tmp_path / "loong" / "arag" / "inferences" / "h1"
    arag.mkdir(parents=True)
    (arag / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    # a sibling flat baseline (codeact) must also be untouched.
    codeact = _mk(tmp_path / "loong" / "codeact", "r_c", _err("ChildCrash"))
    clean_claude_code_logs(tmp_path)
    assert arag.exists() and codeact.exists()


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "claude-code", "r1", _err("ChildCrash"))
    cq = _mk(tmp_path / "corpusqa" / "claude-code", "r2", _err("ChildCrash"))
    clean_claude_code_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and cq.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "claude-code"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out
