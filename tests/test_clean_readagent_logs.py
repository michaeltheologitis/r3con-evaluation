"""Tests for ``scripts/clean_readagent_logs.py``.

Synthetic log trees only (no real runs). ReadAgent uses the same flat layout as RLM/RAPTOR: each
run is ONE folder ``{benchmark}/readagent/{run_tag}/`` directly under the baseline dir (no
``inferences/`` level, no shared ``_indices/`` store), with the gist memory INSIDE it
(``gist_memory.json``). So these pin: run folders are classified directly; a removed junk folder
takes its ``gist_memory.json`` with it; successful runs + genuine model failures are kept; other
baselines' layouts are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_readagent_logs import clean_readagent_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_gist: bool = False) -> Path:
    """Create ``{base}/{name}/`` (a run folder, directly under the baseline dir) holding ``files``,
    optionally with a ``gist_memory.json`` (ReadAgent keeps the gist memory inside the run folder)."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_gist:
        (d / "gist_memory.json").write_text(json.dumps({"pages": [["p"]], "gists": ["g"], "n_docs": 1}))
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_junk_keeps_real_and_drops_their_gist(tmp_path) -> None:
    base = tmp_path / "loong" / "readagent"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_gist=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_gist=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_gist=True)
    # incomplete = a run killed mid-pipeline: a gist_memory.json but no manifest / error
    incomplete = base / "r_incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "gist_memory.json").write_text("partial")

    summary = clean_readagent_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    # the gist memory inside a kept folder stays; the removed ones are gone with their folder
    assert (manifest / "gist_memory.json").exists() and (cwe / "gist_memory.json").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2  # manifest + context-window


def test_transient_removed_keeps_genuine_failure(tmp_path) -> None:
    base = tmp_path / "loong" / "readagent"
    timeout = _mk(base, "r_t", _err("Timeout"))
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_readagent_logs(tmp_path)
    assert not timeout.exists() and cwe.exists()
    assert summary["by_category"].get("transient") == 1
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "corpusqa" / "readagent"
    crash = _mk(base, "r_c", _err("ChildCrash"), with_gist=True)
    summary = clean_readagent_logs(tmp_path, dry_run=True)
    assert crash.exists() and (crash / "gist_memory.json").exists()  # untouched
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "readagent"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_readagent_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()  # real error cleared; manifest always kept
    assert summary["by_category"].get("real_error") == 1


def test_unparseable_error_is_broken(tmp_path) -> None:
    base = tmp_path / "loong" / "readagent"
    bad = _mk(base, "r_b", {"error.json": "{not valid json"})
    summary = clean_readagent_logs(tmp_path)
    assert not bad.exists()
    assert summary["by_category"].get("broken") == 1


def test_only_touches_readagent_not_other_baselines(tmp_path) -> None:
    # arag (its inferences/ layout) must be untouched by the readagent cleaner.
    arag = tmp_path / "loong" / "arag" / "inferences" / "h1"
    arag.mkdir(parents=True)
    (arag / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    clean_readagent_logs(tmp_path)
    assert arag.exists()


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "readagent", "r1", _err("ChildCrash"))
    cq = _mk(tmp_path / "corpusqa" / "readagent", "r2", _err("ChildCrash"))
    clean_readagent_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and cq.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "readagent"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out


def test_main_cli_dry_run_previews(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "readagent"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    main(["--logs-dir", str(tmp_path), "--dry-run"])
    assert crash.exists()
    assert "would delete" in capsys.readouterr().out.lower()
