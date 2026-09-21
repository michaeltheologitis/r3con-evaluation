"""Tests for ``scripts/clean_structrag_logs.py``.

Synthetic log trees only (no real runs). structrag restructures documents per task,
so it has no index store, and this is the simple inference-only cleaner (no
``_indices/`` layer). These pin exactly which inference dirs are removed vs kept:

- interruption artifacts removed: ChildCrash, empty/unparseable error.json, a dir
  with neither manifest nor error (incomplete);
- always kept: a dir with a manifest.json, and a REAL run_one error
  (ContextWindowExceededError) — unless ``--all-errors``;
- ``--dry-run`` deletes nothing; OTHER baselines' dirs are never touched;
  ``--benchmark`` narrows the scope.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_structrag_logs import clean_structrag_logs, main


def _mk(base: Path, name: str, files: dict[str, str]) -> Path:
    d = base / "inferences" / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_interruption_artifacts_keeps_real(tmp_path) -> None:
    base = tmp_path / "loong" / "structrag"
    manifest = _mk(base, "m1", {"manifest.json": json.dumps({"task_id": "a"})})
    crash = _mk(base, "c1", _err("ChildCrash"))
    empty = _mk(base, "e1", {"error.json": ""})
    cwe = _mk(base, "w1", _err("ContextWindowExceededError"))
    incomplete = base / "inferences" / "i1"
    incomplete.mkdir(parents=True)

    summary = clean_structrag_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2  # manifest + context-window
    assert summary["dry_run"] is False


def test_transient_errors_removed_by_default_keeps_genuine_failures(tmp_path) -> None:
    """Non-whitelisted error types (Timeout &c.) are transient noise — removed so
    the task re-runs; the genuine model failures (FAILURE_ERROR_TYPES) survive."""
    base = tmp_path / "loong" / "structrag"
    timeout = _mk(base, "t1", _err("Timeout"))
    unknown = _mk(base, "u1", _err("RuntimeError"))
    cwe = _mk(base, "w1", _err("ContextWindowExceededError"))

    summary = clean_structrag_logs(tmp_path)

    assert not timeout.exists() and not unknown.exists()
    assert cwe.exists()
    assert summary["by_category"].get("transient") == 2
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "loong" / "structrag"
    crash = _mk(base, "c1", _err("ChildCrash"))
    summary = clean_structrag_logs(tmp_path, dry_run=True)
    assert crash.exists()
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_also_removes_real_errors_but_keeps_manifests(tmp_path) -> None:
    base = tmp_path / "loong" / "structrag"
    manifest = _mk(base, "m1", {"manifest.json": "{}"})
    cwe = _mk(base, "w1", _err("ContextWindowExceededError"))
    summary = clean_structrag_logs(tmp_path, all_errors=True)
    assert not cwe.exists()
    assert manifest.exists()
    assert summary["by_category"].get("real_error") == 1


def test_unparseable_error_is_broken_and_removed(tmp_path) -> None:
    base = tmp_path / "loong" / "structrag"
    bad = _mk(base, "b1", {"error.json": "{not valid json"})
    summary = clean_structrag_logs(tmp_path)
    assert not bad.exists()
    assert summary["by_category"].get("broken") == 1


def test_only_touches_structrag_not_other_baselines(tmp_path) -> None:
    arag = _mk(tmp_path / "loong" / "arag", "c1", _err("ChildCrash"))
    other = _mk(tmp_path / "loong" / "other-baseline", "c2", _err("ChildCrash"))
    clean_structrag_logs(tmp_path)
    assert arag.exists() and other.exists()  # other baselines are out of scope


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "structrag", "c1", _err("ChildCrash"))
    lg = _mk(tmp_path / "dracula" / "structrag", "c2", _err("ChildCrash"))
    clean_structrag_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and lg.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "structrag"
    crash = _mk(base, "c1", _err("ChildCrash"))
    _mk(base, "m1", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out


def test_main_cli_dry_run_previews(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "structrag"
    crash = _mk(base, "c1", _err("ChildCrash"))
    main(["--logs-dir", str(tmp_path), "--dry-run"])
    assert crash.exists()
    assert "would delete" in capsys.readouterr().out.lower()
