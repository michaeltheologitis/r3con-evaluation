"""Tests for ``scripts/clean_raptor_logs.py``.

Synthetic log trees only (no real runs). RAPTOR uses the same flat layout as
ReadAgent/RLM/CodeAct: each run is ONE folder ``{benchmark}/raptor/{run_tag}/`` directly under
the baseline dir (no ``inferences/`` level, no shared ``_indices/`` store), with the built
``tree.json`` inside it. So these pin: run folders are classified directly; a removed junk folder
takes its tree with it; successful runs + genuine model failures are kept; other baselines' layouts
are untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_raptor_logs import clean_raptor_logs, main


def _mk(base: Path, name: str, files: dict[str, str], with_tree: bool = False) -> Path:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    if with_tree:
        (d / "tree.json").write_text(json.dumps({"n_nodes": 3}))
        (d / "calls.json").write_text("[]")
    return d


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def _embed_ckpt(error_type: str | None = None) -> dict[str, str]:
    """A phase-1 embed checkpoint folder's files (embed.json marker + leaves.pkl data), optionally
    with a build-phase error.json beside them."""
    files = {"embed.json": json.dumps({"task_id": "t", "config": {}}), "leaves.pkl": "PICKLEBYTES"}
    if error_type is not None:
        files["error.json"] = json.dumps({"task_id": "t", "error_type": error_type})
    return files


def test_default_removes_junk_keeps_real_and_drops_their_tree(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    manifest = _mk(base, "r_manifest", {"manifest.json": json.dumps({"task_id": "a"})}, with_tree=True)
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_tree=True)
    empty = _mk(base, "r_empty", {"error.json": ""})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"), with_tree=True)
    incomplete = base / "r_incomplete"
    incomplete.mkdir(parents=True)
    (incomplete / "tree.json").write_text("partial")  # tree but no record → killed mid-build

    summary = clean_raptor_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert (manifest / "tree.json").exists() and (cwe / "tree.json").exists()
    assert summary["by_category"] == {"crash": 1, "broken": 1, "incomplete": 1}
    assert summary["kept"] == 2  # manifest + context-window


def test_transient_removed_keeps_genuine_failure(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    timeout = _mk(base, "r_t", _err("Timeout"))
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"))
    summary = clean_raptor_logs(tmp_path)
    assert not timeout.exists() and cwe.exists()
    assert summary["kept"] == 1


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    crash = _mk(base, "r_crash", _err("ChildCrash"), with_tree=True)
    summary = clean_raptor_logs(tmp_path, dry_run=True)
    assert crash.exists()
    assert [p for p, _ in summary["deleted"]] == [crash]
    assert summary["dry_run"] is True


def test_all_errors_removes_real_keeps_manifest(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    manifest = _mk(base, "r_m", {"manifest.json": "{}"})
    cwe = _mk(base, "r_cwe", _err("ContextWindowExceededError"))
    summary = clean_raptor_logs(tmp_path, all_errors=True)
    assert not cwe.exists() and manifest.exists()
    assert summary["kept"] == 1


def test_only_touches_raptor_not_other_baselines(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    _mk(base, "r_crash", _err("ChildCrash"))
    other = tmp_path / "loong" / "other-baseline"
    other.mkdir(parents=True)
    (other / "error.json").write_text(json.dumps({"task_id": "t", "error_type": "ChildCrash"}))
    clean_raptor_logs(tmp_path)
    assert (other / "error.json").exists()  # untouched


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "raptor"
    crash = _mk(base, "r_c", _err("ChildCrash"))
    _mk(base, "r_m", {"manifest.json": "{}"})
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert not crash.exists()
    assert "deleted" in out and "loong" in out


# ---- phase split (embed/build) safety: keep checkpoints, revert build failures ----


def test_keeps_phase1_embed_checkpoint(tmp_path) -> None:
    """A phase-1 embed checkpoint (embed.json + leaves.pkl, no manifest/error) is KEPT — cleaning
    errored phase-1 stuff beside it must NOT wipe the (heavy) offline leaf-embedding work."""
    base = tmp_path / "loong" / "raptor"
    ckpt = _mk(base, "r_ckpt", _embed_ckpt())            # awaiting --phase build → rescued (kept)
    errored = _mk(base, "r_err", _err("Timeout"))        # a failed phase-1 embed → removed
    summary = clean_raptor_logs(tmp_path)
    assert ckpt.exists() and (ckpt / "embed.json").exists() and (ckpt / "leaves.pkl").exists()
    assert not errored.exists()
    assert summary["kept"] == 1 and summary["by_category"] == {"transient": 1}
    assert summary["reverted"] == []


def test_reverts_build_failure_keeps_checkpoint(tmp_path) -> None:
    """A --phase build failure drops error.json beside the embed.json + leaves.pkl. The folder is
    REVERTED (only error.json removed), preserving the embed work so --phase build retries it."""
    base = tmp_path / "loong" / "raptor"
    folder = _mk(base, "r_buildfail", _embed_ckpt(error_type="ChildCrash"))
    summary = clean_raptor_logs(tmp_path)
    assert folder.exists() and (folder / "embed.json").exists() and (folder / "leaves.pkl").exists()
    assert not (folder / "error.json").exists()          # reverted: only the error is gone
    assert summary["reverted"] == [folder]
    assert summary["deleted"] == [] and summary["kept"] == 0   # neither deleted nor (re)counted kept


def test_build_failure_revert_is_dry_run_safe(tmp_path) -> None:
    base = tmp_path / "loong" / "raptor"
    folder = _mk(base, "r_buildfail", _embed_ckpt(error_type="Timeout"))
    summary = clean_raptor_logs(tmp_path, dry_run=True)
    assert (folder / "error.json").exists()              # nothing removed under --dry-run
    assert summary["reverted"] == [folder] and summary["deleted"] == []
