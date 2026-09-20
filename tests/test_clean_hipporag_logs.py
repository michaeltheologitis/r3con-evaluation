"""Tests for ``scripts/clean_hipporag_logs.py``.

Synthetic log trees only (no real runs). HippoRAG uses the same flat layout as
ReadAgent/RLM/MemAgent: each run is ONE folder ``{benchmark}/hipporag/{run_tag}/`` directly
under the baseline dir (no ``inferences/`` level, no shared index store), with the OpenIE
graph index INSIDE it (``index/``). So these pin: run folders classified directly; a removed
junk folder takes its ``index/`` with it; successful runs + genuine model failures kept;
other baselines' layouts untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_hipporag_logs import clean_hipporag_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_index: bool = False) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_index:
        (d / "index").mkdir(exist_ok=True)
        (d / "index" / "graph.pickle").write_text("<graph bytes>")
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_junk_keeps_real_and_drops_their_index(tmp_path) -> None:
    base = tmp_path / "loong" / "hipporag"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_index=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_index=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_index=True)
    incomplete = base / "r_incomplete"
    (incomplete / "index").mkdir(parents=True)
    (incomplete / "index" / "graph.pickle").write_text("partial")

    summary = clean_hipporag_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert (manifest / "index").exists() and (cwe / "index").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2


def test_transient_removed_keeps_genuine_failure(tmp_path) -> None:
    base = tmp_path / "corpusqa" / "hipporag"
    timeout = _mk(base, "r_t", _err("Timeout"))
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_hipporag_logs(tmp_path)
    assert not timeout.exists() and cwe.exists()
    assert summary["by_category"].get("transient") == 1
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "dracula" / "hipporag"
    crash = _mk(base, "r_c", _err("ChildCrash"), with_index=True)
    summary = clean_hipporag_logs(tmp_path, dry_run=True)
    assert crash.exists() and (crash / "index").exists()
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "hipporag"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_hipporag_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()
    assert summary["by_category"].get("real_error") == 1


def test_only_touches_hipporag_not_other_baselines(tmp_path) -> None:
    # arag (its inferences/ layout) must be untouched by the hipporag cleaner.
    arag = tmp_path / "loong" / "arag" / "inferences" / "h1"
    arag.mkdir(parents=True)
    (arag / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    clean_hipporag_logs(tmp_path)
    assert arag.exists()


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "hipporag", "r1", _err("ChildCrash"))
    cq = _mk(tmp_path / "corpusqa" / "hipporag", "r2", _err("ChildCrash"))
    clean_hipporag_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and cq.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "hipporag"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out
