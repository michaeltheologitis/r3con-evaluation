"""Tests for ``scripts/clean_memagent_logs.py``.

Synthetic log trees only (no real runs). MemAgent uses the same flat layout as
ReadAgent/RLM: each run is ONE folder ``{benchmark}/memagent/{run_tag}/`` directly under the
baseline dir (no ``inferences/`` level, no shared index store), with the memory trajectory
INSIDE it (``memory_trajectory.json``). So these pin: run folders classified directly; a
removed junk folder takes its ``memory_trajectory.json`` with it; successful runs + genuine
model failures kept; other baselines' layouts untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_memagent_logs import clean_memagent_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_trajectory: bool = False) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_trajectory:
        (d / "memory_trajectory.json").write_text(json.dumps({"n_chunks": 1, "memories": ["m"]}))
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_junk_keeps_real_and_drops_their_trajectory(tmp_path) -> None:
    base = tmp_path / "loong" / "memagent"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_trajectory=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_trajectory=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_trajectory=True)
    incomplete = base / "r_incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "memory_trajectory.json").write_text("partial")

    summary = clean_memagent_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert (manifest / "memory_trajectory.json").exists() and (cwe / "memory_trajectory.json").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2


def test_transient_removed_keeps_genuine_failure(tmp_path) -> None:
    base = tmp_path / "loong" / "memagent"
    timeout = _mk(base, "r_t", _err("Timeout"))
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_memagent_logs(tmp_path)
    assert not timeout.exists() and cwe.exists()
    assert summary["by_category"].get("transient") == 1
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "corpusqa" / "memagent"
    crash = _mk(base, "r_c", _err("ChildCrash"), with_trajectory=True)
    summary = clean_memagent_logs(tmp_path, dry_run=True)
    assert crash.exists() and (crash / "memory_trajectory.json").exists()
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "memagent"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_memagent_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()
    assert summary["by_category"].get("real_error") == 1


def test_only_touches_memagent_not_other_baselines(tmp_path) -> None:
    # arag (its inferences/ layout) must be untouched by the memagent cleaner.
    arag = tmp_path / "loong" / "arag" / "inferences" / "h1"
    arag.mkdir(parents=True)
    (arag / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    clean_memagent_logs(tmp_path)
    assert arag.exists()


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "memagent", "r1", _err("ChildCrash"))
    cq = _mk(tmp_path / "corpusqa" / "memagent", "r2", _err("ChildCrash"))
    clean_memagent_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and cq.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "memagent"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out
