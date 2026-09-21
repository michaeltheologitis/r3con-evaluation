"""Tests for ``scripts/clean_rlm_logs.py``.

Synthetic log trees only (no real runs). RLM uses the same flat layout as ReadAgent/RAPTOR:
each run is ONE folder ``{benchmark}/rlm/{run_tag}/`` directly under the baseline dir (no
``inferences/`` level, no shared ``_indices/`` store), with RLM's trajectory inside it. So these
pin: run folders are classified directly; a removed junk folder takes its trajectory with it;
successful runs + genuine model failures are kept; other baselines' layouts are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_rlm_logs import clean_rlm_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_traj: bool = False) -> Path:
    """Create ``{base}/{name}/`` (a run folder, directly under the baseline dir) holding ``files``,
    optionally with an RLM trajectory (RLM keeps it inside the run folder)."""
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_traj:
        # The native RLMLogger trajectory (a `rlm_*.jsonl`, written live: 1 metadata line + 1 per
        # iteration). This is now the ONLY on-disk trajectory — we no longer dump a rlm_trajectory.json.
        (d / "rlm_log.jsonl").write_text('{"type":"metadata"}\n{"type":"iteration","response":"x"}\n')
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def test_default_removes_junk_keeps_real_and_drops_their_trajectory(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_traj=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_traj=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_traj=True)
    # incomplete = a run killed mid-trajectory (e.g. hung REPL cell SLURM-killed): traj but no record
    incomplete = base / "r_incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "rlm_log.jsonl").write_text("partial")

    summary = clean_rlm_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert (manifest / "rlm_log.jsonl").exists() and (cwe / "rlm_log.jsonl").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2  # manifest + context-window


def test_transient_removed_keeps_genuine_failure(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    timeout = _mk(base, "r_t", _err("Timeout"))
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_rlm_logs(tmp_path)
    assert not timeout.exists() and cwe.exists()
    assert summary["by_category"].get("transient") == 1
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "corpusqa" / "rlm"
    crash = _mk(base, "r_c", _err("ChildCrash"), with_traj=True)
    summary = clean_rlm_logs(tmp_path, dry_run=True)
    assert crash.exists() and (crash / "rlm_log.jsonl").exists()  # untouched
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_w", _err("ContextWindowExceededError"))
    summary = clean_rlm_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()
    assert summary["by_category"].get("real_error") == 1


def test_only_touches_rlm_not_other_baselines(tmp_path) -> None:
    # arag (its inferences/ layout) must be untouched by the rlm cleaner.
    arag = tmp_path / "loong" / "arag" / "inferences" / "h1"
    arag.mkdir(parents=True)
    (arag / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    clean_rlm_logs(tmp_path)
    assert arag.exists()


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk(tmp_path / "loong" / "rlm", "r1", _err("ChildCrash"))
    cq = _mk(tmp_path / "corpusqa" / "rlm", "r2", _err("ChildCrash"))
    clean_rlm_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and cq.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "rlm"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out


# --- --max-iter: also clear successful runs that hit the iteration cap ---


def _capped(n_iter: int = 30, max_iter: int = 30) -> dict[str, str]:
    """A SUCCESSFUL run (manifest.json) whose trace says it exhausted its budget."""
    return {"manifest.json": json.dumps(
        {"task_id": "t", "trace": {"n_iterations": n_iter, "max_iterations": max_iter}})}


def test_max_iter_removes_capped_keeps_finished_and_failures(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    capped = _mk(base, "r_capped", _capped(30, 30), with_traj=True)       # n_iter == max → capped
    finished = _mk(base, "r_finished", _capped(5, 30))                    # n_iter < max → NOT capped
    plain = _mk(base, "r_plain", {"manifest.json": json.dumps({"task_id": "p"})})  # no budget trace
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"))          # genuine failure

    summary = clean_rlm_logs(tmp_path, max_iter=True)

    assert not capped.exists()                                           # capped success + its traj removed
    assert (capped / "rlm_log.jsonl").exists() is False
    assert finished.exists() and plain.exists() and cwe.exists()         # everything else kept
    assert summary["by_category"] == {"max_iter": 1}
    assert summary["kept"] == 3                                          # finished + plain + cwe


def test_max_iter_off_by_default_keeps_capped(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    capped = _mk(base, "r_capped", _capped(30, 30))
    summary = clean_rlm_logs(tmp_path)                                   # no max_iter
    assert capped.exists()
    assert summary["deleted"] == [] and summary["kept"] == 1


def test_max_iter_dry_run_previews_without_deleting(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    capped = _mk(base, "r_capped", _capped(31, 30), with_traj=True)      # n_iter > max also counts
    summary = clean_rlm_logs(tmp_path, max_iter=True, dry_run=True)
    assert capped.exists()                                              # nothing deleted
    assert [p for p, _ in summary["deleted"]] == [capped]
    assert summary["by_category"] == {"max_iter": 1}
    assert summary["dry_run"] is True


def test_main_cli_max_iter_flag(tmp_path) -> None:
    base = tmp_path / "loong" / "rlm"
    capped = _mk(base, "r_capped", _capped(30, 30))
    main(["--max-iter", "--logs-dir", str(tmp_path)])                    # flag name maps to args.max_iter
    assert not capped.exists()
