"""Tests for `evals.baselines._common` — the stateless mechanism the per-baseline
runners share (hashing, paths, model-id, resumption, manifest/error serialization,
the subprocess primitive). No benchmark/baseline branching lives here.
"""
from __future__ import annotations

import json
import sys

import pytest

from evals.baselines import _common
from evals.baselines._common import (
    base_dir,
    build_error_record,
    canonical_model_id,
    compute_config_hash,
    compute_inference_hash,
    load_benchmark_module,
    run_child,
    scan_completed_inference_hashes,
    with_provider_prefix,
    write_error,
    write_manifest,
)
from evals.settings import settings


# ============================================================
# with_provider_prefix
# ============================================================


@pytest.mark.parametrize("model", [
    "hosted_vllm/Qwen/Qwen3-8B", "ollama_chat/qwen3:8b", "ollama/qwen3:8b", "openai/gpt-5.4-nano",
])
def test_with_provider_prefix_passthrough_when_already_prefixed(model: str) -> None:
    assert with_provider_prefix(model) == model


def test_with_provider_prefix_adds_default_when_no_prefix() -> None:
    assert with_provider_prefix("Qwen/Qwen3-8B") == "hosted_vllm/Qwen/Qwen3-8B"


def test_default_prefix_is_hosted_vllm() -> None:
    assert _common.DEFAULT_LITELLM_PREFIX == "hosted_vllm/"


# ============================================================
# canonical_model_id
# ============================================================


@pytest.mark.parametrize("model, expected", [
    ("Qwen/Qwen3.5-9B", "qwen3-5-9b"),
    ("hosted_vllm/Qwen/Qwen3.5-9B", "qwen3-5-9b"),
    ("ollama_chat/qwen3.5:9b", "qwen3-5-9b"),
    ("openai/gpt-5-nano", "gpt-5-nano"),
])
def test_canonical_model_id_examples(model: str, expected: str) -> None:
    assert canonical_model_id(model) == expected


def test_canonical_model_id_collapses_across_providers() -> None:
    assert canonical_model_id("hosted_vllm/Qwen/Qwen3.5-9B") == canonical_model_id("ollama_chat/qwen3.5:9b")


# ============================================================
# compute_config_hash / compute_inference_hash / paths
# ============================================================


def test_config_hash_deterministic_and_key_order_independent() -> None:
    a = {"benchmark": "dracula", "model": "qwen3-8b", "seed": 42}
    b = {"seed": 42, "model": "qwen3-8b", "benchmark": "dracula"}
    assert compute_config_hash(a) == compute_config_hash(b) == compute_config_hash(a)


def test_config_hash_sensitive_and_truncated() -> None:
    assert compute_config_hash({"seed": 42}) != compute_config_hash({"seed": 43})
    assert len(compute_config_hash({"a": 1})) == 12
    int(compute_config_hash({"a": 1}), 16)  # hex


def test_compute_inference_hash_distinguishes_task_and_config() -> None:
    cfg = {"benchmark": "loong", "baseline": "arag", "model": "m", "seed": 1,
           "search_method": "global", "embedding_model": "e"}
    assert compute_inference_hash(cfg, "t1") != compute_inference_hash(cfg, "t2")
    assert compute_inference_hash(cfg, "t1") != compute_inference_hash({**cfg, "search_method": "basic"}, "t1")


def test_base_dir_assembles_and_slugifies(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    assert base_dir("dracula", "codeact") == tmp_path / "dracula" / "codeact"
    assert base_dir("My.Bench", "Some/Baseline") == tmp_path / "my-bench" / "some-baseline"


# ============================================================
# Resumption: a folder is "done" with manifest.json OR error.json
# ============================================================


def test_scan_returns_empty_when_no_inferences_dir(tmp_path) -> None:
    assert scan_completed_inference_hashes(tmp_path) == set()


def test_scan_counts_manifest_or_error_as_done(tmp_path) -> None:
    inf = tmp_path / "inferences"
    (inf / "ok").mkdir(parents=True)
    (inf / "ok" / "manifest.json").write_text("{}")
    (inf / "failed").mkdir(parents=True)
    (inf / "failed" / "error.json").write_text("{}")
    (inf / "partial").mkdir(parents=True)  # neither file → still pending
    assert scan_completed_inference_hashes(tmp_path) == {"ok", "failed"}


# ============================================================
# scan_completed_task_ids — resumption for the FLAT per-run-folder layout
# (linearrag / readagent): config-scoped, reads task_id from the record itself
# ============================================================


def _flat_run(base, run_tag: str, *, task_id: str, config: dict, fname: str = "manifest.json") -> None:
    """A flat per-run folder ``base/{run_tag}/`` holding one manifest.json / error.json record."""
    d = base / run_tag
    d.mkdir(parents=True, exist_ok=True)
    (d / fname).write_text(json.dumps({"task_id": task_id, "config": config}))


def test_scan_task_ids_empty_when_base_missing(tmp_path) -> None:
    assert _common.scan_completed_task_ids(tmp_path / "nope", {"a": 1}) == set()


def test_scan_task_ids_manifest_and_error_both_count_for_matching_config(tmp_path) -> None:
    cfg = {"benchmark": "loong", "baseline": "readagent", "model": "m", "lookup_method": "parallel"}
    _flat_run(tmp_path, "r1", task_id="t1", config=cfg)                       # success
    _flat_run(tmp_path, "r2", task_id="t2", config=cfg, fname="error.json")   # recorded failure → done
    (tmp_path / "r3").mkdir()                                                 # neither → pending
    assert _common.scan_completed_task_ids(tmp_path, cfg) == {"t1", "t2"}


def test_scan_task_ids_is_config_scoped(tmp_path) -> None:
    # The SAME task under a DIFFERENT config (here: a different lookup_method) is NOT "done"
    # for this config — so running the other variant still re-runs it. Key-order-independent.
    par = {"baseline": "readagent", "model": "m", "lookup_method": "parallel"}
    seq = {"model": "m", "baseline": "readagent", "lookup_method": "sequential"}
    _flat_run(tmp_path, "r1", task_id="t1", config=par)
    _flat_run(tmp_path, "r2", task_id="t2", config=seq)
    assert _common.scan_completed_task_ids(tmp_path, par) == {"t1"}   # only the parallel one
    assert _common.scan_completed_task_ids(tmp_path, seq) == {"t2"}   # only the sequential one


def test_scan_task_ids_skips_half_written_record(tmp_path) -> None:
    cfg = {"baseline": "readagent"}
    _flat_run(tmp_path, "good", task_id="t1", config=cfg)
    (tmp_path / "torn").mkdir()
    (tmp_path / "torn" / "manifest.json").write_text("{not json")  # interrupted write → pending
    assert _common.scan_completed_task_ids(tmp_path, cfg) == {"t1"}


# ============================================================
# manifest / error serialization
# ============================================================


def test_write_manifest_and_error_roundtrip(tmp_path) -> None:
    write_manifest(tmp_path / "h1", {"task_id": "a", "raw_answer": "x"})
    assert json.loads((tmp_path / "h1" / "manifest.json").read_text())["raw_answer"] == "x"
    write_error(tmp_path / "h2", {"task_id": "b", "error_type": "Boom"})
    assert json.loads((tmp_path / "h2" / "error.json").read_text())["error_type"] == "Boom"


def test_build_error_record_from_exception_captures_type_message_traceback() -> None:
    try:
        raise ValueError("context window exceeded")
    except ValueError as exc:
        rec = build_error_record("t1", {"benchmark": "loong"}, exc, phase="run_one")
    assert rec["task_id"] == "t1"
    assert rec["config"] == {"benchmark": "loong"}
    assert rec["phase"] == "run_one"
    assert rec["error_type"] == "ValueError"
    assert "context window exceeded" in rec["message"]
    assert rec["traceback"] and "ValueError" in rec["traceback"]


def test_build_error_record_explicit_fields_no_exc() -> None:
    rec = build_error_record("t2", {}, phase="child_crash", error_type="ChildCrash", message="exit 137")
    assert rec["error_type"] == "ChildCrash" and rec["phase"] == "child_crash"
    assert rec["traceback"] is None


def test_build_error_record_truncates_long_message() -> None:
    rec = build_error_record("t3", {}, phase="run_one", error_type="X", message="z" * 5000)
    assert len(rec["message"]) == 2000


# ============================================================
# load_benchmark_module + run_child
# ============================================================


def test_load_benchmark_module_known_and_unknown() -> None:
    assert hasattr(load_benchmark_module("dracula"), "get_task_ids")
    with pytest.raises(ValueError, match="Unknown benchmark"):
        load_benchmark_module("not-a-benchmark")


def test_run_child_runs_once_and_returns_completed_process() -> None:
    ok = run_child([sys.executable, "-c", "import sys; sys.exit(0)"])
    assert ok.returncode == 0
    bad = run_child([sys.executable, "-c", "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"])
    assert bad.returncode == 3
    assert "boom" in bad.stderr
