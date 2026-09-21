"""Tests for ``scripts/clean_arag_logs.py``.

Synthetic log trees only (no real runs). A-RAG is the one baseline with a content-
addressed ``_indices/`` store, so its cleaner needs the index-store layer on top of
the shared inference-folder cleanup. These pin:

- **Inference folders** behave exactly like the other cleaners: ChildCrash / empty /
  incomplete / transient removed by default; manifest.json + real errors kept (real
  errors removed only with ``--all-errors``).
- **Index store**: a *half-built* index (missing the ``index_meta.json`` /
  ``index_usage.json`` receipts) is removed by default; a *complete but orphaned*
  index is KEPT by default and removed only with ``--prune-orphan-indices``; a
  *referenced* index is NEVER removed — even with the flag.
- A-RAG's index is **LLM-independent** (``index_hash`` excludes the completion
  model/seed), so an index referenced by manifests from DIFFERENT completion models
  is kept once — the cleaner uses ``_clean_common``'s ``clean_index_store``.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.clean_arag_logs import clean_arag_logs, main


def _mk_inf(base: Path, name: str, files: dict[str, str]) -> Path:
    d = base / "inferences" / name
    d.mkdir(parents=True, exist_ok=True)
    for filename, content in files.items():
        (d / filename).write_text(content)
    return d


def _manifest(index_ref: str | None = None, **config: str) -> dict[str, str]:
    record: dict[str, object] = {"task_id": "t", "config": config, "raw_answer": "a"}
    if index_ref is not None:
        record["index_ref"] = index_ref
    return {"manifest.json": json.dumps(record)}


def _err(error_type: str) -> dict[str, str]:
    return {"error.json": json.dumps({"task_id": "t", "error_type": error_type})}


def _mk_index(base: Path, index_hash: str, *, complete: bool = True) -> Path:
    """Create ``{base}/_indices/{index_hash}/``. A *complete* index carries the
    build's completion receipt (``index_meta.json`` + ``index_usage.json``, written
    last); a half-built one has only partial output."""
    d = base / "_indices" / index_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / "chunks.json").write_text("[]")  # always present
    if complete:
        (d / "index_meta.json").write_text(json.dumps({"index_hash": index_hash, "doc_key": "k"}))
        (d / "index_usage.json").write_text(json.dumps({"total": {}}))
    return d


# ---------------------------------------------------------------------------
# Inference folders — same contract as the other cleaners
# ---------------------------------------------------------------------------


def test_default_removes_interruption_inferences_keeps_real(tmp_path) -> None:
    base = tmp_path / "dracula" / "arag"
    manifest = _mk_inf(base, "m1", _manifest(index_ref="idx", model="x"))
    crash = _mk_inf(base, "c1", _err("ChildCrash"))
    empty = _mk_inf(base, "e1", {"error.json": ""})
    cwe = _mk_inf(base, "w1", _err("ContextWindowExceededError"))
    incomplete = base / "inferences" / "i1"
    incomplete.mkdir(parents=True)
    _mk_index(base, "idx", complete=True)  # keep the referenced index alive

    summary = clean_arag_logs(tmp_path)

    assert not crash.exists() and not empty.exists() and not incomplete.exists()
    assert manifest.exists() and cwe.exists()
    assert summary["by_category"]["crash"] == 1
    assert summary["by_category"]["broken"] == 1
    assert summary["by_category"]["incomplete"] == 1
    assert summary["dry_run"] is False


def test_transient_errors_removed_by_default_keeps_genuine_failures(tmp_path) -> None:
    """Non-whitelisted error types (Timeout, the live NotFoundError &c.) are transient
    noise — removed so the task re-runs; genuine model failures survive."""
    base = tmp_path / "loong" / "arag"
    timeout = _mk_inf(base, "t1", _err("Timeout"))
    notfound = _mk_inf(base, "n1", _err("NotFoundError"))
    cwe = _mk_inf(base, "w1", _err("ContextWindowExceededError"))

    summary = clean_arag_logs(tmp_path)

    assert not timeout.exists() and not notfound.exists()
    assert cwe.exists()
    assert summary["by_category"].get("transient") == 2


def test_all_errors_removes_real_error_inferences_keeps_manifests(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    manifest = _mk_inf(base, "m1", _manifest(index_ref="idx"))
    cwe = _mk_inf(base, "w1", _err("ContextWindowExceededError"))
    _mk_index(base, "idx", complete=True)

    summary = clean_arag_logs(tmp_path, all_errors=True)

    assert not cwe.exists()
    assert manifest.exists()
    assert summary["by_category"].get("real_error") == 1


# ---------------------------------------------------------------------------
# Index store
# ---------------------------------------------------------------------------


def test_half_built_index_removed_by_default(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    half = _mk_index(base, "h1", complete=False)  # interrupted build, unreferenced
    summary = clean_arag_logs(tmp_path)
    assert not half.exists()
    assert summary["by_category"].get("half_built") == 1


def test_complete_index_referenced_by_manifest_is_kept(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    _mk_inf(base, "m1", _manifest(index_ref="ref1"))
    referenced = _mk_index(base, "ref1", complete=True)
    summary = clean_arag_logs(tmp_path)
    assert referenced.exists()
    assert summary["n_referenced_indices"] == 1
    assert "orphan" not in summary["by_category"]


def test_orphan_complete_index_kept_by_default_pruned_with_flag(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    orphan = _mk_index(base, "o1", complete=True)  # complete, but unreferenced

    kept = clean_arag_logs(tmp_path)
    assert orphan.exists()
    assert "orphan" not in kept["by_category"]

    pruned = clean_arag_logs(tmp_path, prune_orphan_indices=True)
    assert not orphan.exists()
    assert pruned["by_category"].get("orphan") == 1


def test_referenced_index_never_pruned_even_with_flag(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    _mk_inf(base, "m1", _manifest(index_ref="shared"))
    referenced = _mk_index(base, "shared", complete=True)
    orphan = _mk_index(base, "lonely", complete=True)

    summary = clean_arag_logs(tmp_path, prune_orphan_indices=True)

    assert referenced.exists()
    assert not orphan.exists()
    assert summary["by_category"].get("orphan") == 1


def test_index_shared_across_completion_models_is_kept(tmp_path) -> None:
    """A-RAG's index is LLM-independent, so manifests from two DIFFERENT completion
    models reference the SAME index — it's kept once; an index referenced by neither
    is an orphan."""
    base = tmp_path / "dracula" / "arag"
    _mk_inf(base, "a1", _manifest(index_ref="shared", model="gpt-5-4-nano"))
    _mk_inf(base, "b1", _manifest(index_ref="shared", model="qwen3-5-35b-a3b"))
    shared = _mk_index(base, "shared", complete=True)
    orphan = _mk_index(base, "other", complete=True)

    summary = clean_arag_logs(tmp_path, prune_orphan_indices=True)

    assert shared.exists()
    assert not orphan.exists()
    assert summary["n_referenced_indices"] == 1
    assert summary["by_category"].get("orphan") == 1


def test_half_built_but_referenced_index_is_kept(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    _mk_inf(base, "m1", _manifest(index_ref="ref"))
    weird = _mk_index(base, "ref", complete=False)  # referenced yet missing receipts
    summary = clean_arag_logs(tmp_path)
    assert weird.exists()
    assert "half_built" not in summary["by_category"]


# ---------------------------------------------------------------------------
# Scope + safety
# ---------------------------------------------------------------------------


def test_dry_run_deletes_nothing(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    crash = _mk_inf(base, "c1", _err("ChildCrash"))
    half = _mk_index(base, "h1", complete=False)
    summary = clean_arag_logs(tmp_path, prune_orphan_indices=True, dry_run=True)
    assert crash.exists() and half.exists()
    deleted_paths = {p for p, _ in summary["deleted"]}
    assert crash in deleted_paths and half in deleted_paths
    assert summary["dry_run"] is True


def test_only_touches_arag_not_other_baselines(tmp_path) -> None:
    structrag = _mk_inf(tmp_path / "loong" / "structrag", "c1", _err("ChildCrash"))
    other = _mk_inf(tmp_path / "loong" / "other-baseline", "c2", _err("ChildCrash"))
    clean_arag_logs(tmp_path)
    assert structrag.exists() and other.exists()  # other baselines out of scope


def test_benchmark_filter_limits_scope(tmp_path) -> None:
    lo = _mk_index(tmp_path / "loong" / "arag", "h1", complete=False)
    lb = _mk_index(tmp_path / "dracula" / "arag", "h2", complete=False)
    clean_arag_logs(tmp_path, benchmark="loong")
    assert not lo.exists() and lb.exists()


def test_main_cli_deletes_and_prints_summary(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "arag"
    _mk_inf(base, "c1", _err("ChildCrash"))
    _mk_index(base, "h1", complete=False)
    main(["--logs-dir", str(tmp_path)])
    out = capsys.readouterr().out.lower()
    assert "deleted" in out and "loong" in out


def test_main_cli_dry_run_previews(tmp_path, capsys) -> None:
    base = tmp_path / "loong" / "arag"
    half = _mk_index(base, "h1", complete=False)
    main(["--logs-dir", str(tmp_path), "--dry-run"])
    assert half.exists()
    assert "would delete" in capsys.readouterr().out.lower()
