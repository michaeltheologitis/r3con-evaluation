"""Shared, stateless helpers for the per-baseline runners.

Each baseline owns its OWN runner (``evals.baselines.<baseline>.runner``, invoked
as ``python -m evals.baselines.<baseline> --benchmark <X>``) — there is no central
runner that dispatches every baseline. This module holds only the small, pure
mechanism those runners share: content hashing, the log-store paths, model-id
canonicalization, the resumption scan, and the manifest / error serialization.
Nothing here branches on a benchmark or baseline name.

Log layout (per (benchmark, baseline)) — index store and inference store are
DECOUPLED, both content-addressed; no "run" folder:

    logs/{benchmark}/{baseline}/
      _indices/{index_hash}/                # graphrag only; built once, shared
          <parquets>, lancedb/, index_usage.json, index_meta.json
      inferences/{inference_hash}/
          manifest.json                     # ONE successful inference (model output)
          error.json                        # OR: this task failed (recorded, not retried)

A dispatched task ends with EXACTLY ONE of ``manifest.json`` / ``error.json``, so
both count as "done" for resumption (a recorded failure is not re-attempted).
"""
from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import traceback as _traceback
from pathlib import Path
from typing import Any

from evals.settings import (
    LITELLM_PROVIDER_PREFIXES,
    _slug,
    canonical_model_id,
    settings,
)

# Load API keys from .env into os.environ so LiteLLM and graphrag both pick them
# up. Importing evals.llm would also do this, but that pulls litellm into the
# import graph even for baselines that don't use it; doing it explicitly here
# keeps the helper self-contained.
try:
    from dotenv import load_dotenv
    load_dotenv(settings.ENV_FILE)
except ImportError:
    pass  # python-dotenv is a baseline dep; if it's missing we just skip silently.


# LITELLM_PROVIDER_PREFIXES and canonical_model_id now live in evals.settings (the
# model registry) and are re-exported here (imported above) so existing
# `from evals.baselines._common import canonical_model_id` callers keep working.
DEFAULT_LITELLM_PREFIX = "hosted_vllm/"

# How to resolve a ``--benchmark`` name to its module. The per-baseline runner
# restricts the valid choices to its own ``SUPPORTED_BENCHMARKS``.
BENCHMARK_IMPORT_MAP = {
    "loong": "evals.benchmarks.loong",
    "corpusqa": "evals.benchmarks.corpusqa",
    "dracula": "evals.benchmarks.dracula",
}
BENCHMARK_NAMES = tuple(BENCHMARK_IMPORT_MAP)


# ============================================================
# Provider-routing helpers
# ============================================================


def with_provider_prefix(model: str) -> str:
    """Prepend the default LiteLLM provider prefix if none is present."""
    if model.startswith(LITELLM_PROVIDER_PREFIXES):
        return model
    return DEFAULT_LITELLM_PREFIX + model


# ============================================================
# Content hashing + store layout
# ============================================================


def compute_config_hash(config: dict[str, Any], length: int = 12) -> str:
    """Stable hex hash of a config dict.

    Canonical JSON (sorted keys, no whitespace, no NaN, ASCII-escaped non-ASCII)
    so the hash is reproducible across processes and Python versions. Length 12
    is short enough to fit in a directory name, long enough to avoid collisions
    across the configs this project will ever explore.
    """
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def base_dir(benchmark_name: str, baseline_name: str) -> Path:
    """Root for a (benchmark, baseline): holds ``inferences/`` and, for
    index-building baselines, ``_indices/``. Does not create the directory."""
    return settings.LOGS_DIR / _slug(benchmark_name) / _slug(baseline_name)


def compute_inference_hash(run_config: dict[str, Any], task_id: str) -> str:
    """Content hash identifying EXACTLY ONE inference: the run config + task id."""
    return compute_config_hash({**run_config, "task_id": str(task_id)})


def inference_dir(base: Path, inference_hash: str) -> Path:
    """Folder for one inference's ``manifest.json`` / ``error.json``. Does not create it."""
    return base / "inferences" / inference_hash


# ============================================================
# Resumption scan
# ============================================================


def scan_completed_inference_hashes(base: Path) -> set[str]:
    """Inference hashes already finished under ``base/inferences/``.

    "Finished" = the inference folder holds a ``manifest.json`` (succeeded) OR an
    ``error.json`` (failed, recorded). Either way the task is not re-dispatched —
    a deterministic failure (e.g. context-window-exceeded) won't be retried every
    run. A half-written folder with neither file is treated as pending. (To retry
    a recorded failure, delete its ``error.json``.)
    """
    inferences = base / "inferences"
    if not inferences.exists():
        return set()
    return {
        d.name for d in inferences.iterdir()
        if d.is_dir() and ((d / "manifest.json").exists() or (d / "error.json").exists())
    }


def scan_completed_task_ids(base: Path, run_config: dict[str, Any]) -> set[str]:
    """Task ids already finished for ``run_config`` under a FLAT per-run-folder baseline.

    The content-addressed baselines name each folder by ``inference_hash`` (config +
    task), so resumption there is a folder-name scan. readagent / rlm instead use
    one random-hex folder per RUN with the index inside it — NO content-addressing, NO
    folder-name signal. But every ``manifest.json`` / ``error.json`` records its
    ``task_id`` AND its full ``config``, so resumption needs neither: scan the run
    folders (direct children of ``base``) and collect the task ids whose record's
    ``config`` matches ``run_config`` exactly.

    "Finished" = a ``manifest.json`` (succeeded) OR ``error.json`` (failed, recorded) —
    a recorded failure is NOT retried (delete its folder to retry), same policy as the
    content-addressed scan. Config-scoping (canonical-JSON equality) is what keeps a
    different experiment — another model / seed / ``--config`` / ``lookup_method`` /
    ``run_version`` — from counting as "done": those rebuild from scratch. This restores
    resumption WITHOUT restoring index reuse (a pending task still builds its own fresh
    index inside its own folder).
    """
    if not base.exists():
        return set()
    target = json.dumps(run_config, sort_keys=True)
    done: set[str] = set()
    for folder in base.iterdir():
        if not folder.is_dir():
            continue
        for filename in (MANIFEST_FILE, ERROR_FILE):
            path = folder / filename
            if not path.exists():
                continue
            try:
                record = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                break  # half-written record → treat the folder as pending
            if record.get("task_id") and json.dumps(record.get("config", {}), sort_keys=True) == target:
                done.add(str(record["task_id"]))
            break  # exactly one of manifest/error per folder
    return done


# ============================================================
# Manifest / error serialization
# ============================================================


MANIFEST_FILE = "manifest.json"
ERROR_FILE = "error.json"

# error_type values (recorded in error.json) that represent a genuine failed
# PREDICTION — the model could not produce an answer for this input — as opposed
# to infra noise (a transient timeout, an OOM child-crash). The single source of
# truth for that judgment, defined beside ``build_error_record`` (which writes
# ``error_type``) and consumed by BOTH downstream readers so they can't diverge:
# the analysis CLI folds these into its "all" (⁺) view as worst-score predictions,
# and the log cleaners KEEP exactly these by default (every other recorded error
# is retryable noise, cleared so the task re-runs). This is a property of the
# ERROR, not the benchmark. Context-window is the canonical case.
FAILURE_ERROR_TYPES = frozenset({"ContextWindowExceededError"})


def hit_step_cap(manifest: dict[str, Any]) -> bool:
    """True if an agentic baseline's task EXHAUSTED its step/iteration budget without
    finishing — so it produced no genuine answer, only a forced/truncated one.

    Two signals, both written into the manifest ``trace`` by the baselines that have a
    budget: codeact (smolagents) sets ``state == "max_steps_error"``; rlm sets
    ``n_iterations`` + ``max_iterations`` (capped ⇔ used them all). Every other baseline
    writes neither, so this is ``False`` for them.

    Like ``FAILURE_ERROR_TYPES``, this is a SINGLE source of truth shared so the two
    readers can't diverge: the analysis CLI counts a capped run as a failed prediction
    in its "all" (⁺) view, and the rlm/codeact log cleaners can opt to remove capped
    run folders (``--max-iter``) so the task re-runs — both keying on this one judgment.
    """
    trace = manifest.get("trace") or {}
    if trace.get("state") == "max_steps_error":
        return True
    n_iter, max_iter = trace.get("n_iterations"), trace.get("max_iterations")
    return n_iter is not None and max_iter is not None and n_iter >= max_iter


def write_manifest(inf_dir: Path, manifest: dict[str, Any]) -> None:
    """Persist one successful inference's ``manifest.json`` (the model's output)."""
    inf_dir.mkdir(parents=True, exist_ok=True)
    (inf_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def build_error_record(
    task_id: str,
    config: dict[str, Any],
    exc: BaseException | None = None,
    *,
    phase: str,
    error_type: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """A principled, self-contained failure record (mirrors the manifest shape:
    ``task_id`` + ``config`` + why it failed).

    Pass an ``exc`` to capture its class / message / traceback automatically, or
    pass ``error_type`` / ``message`` directly (e.g. for a child that crashed
    before it could record anything itself).
    """
    if exc is not None:
        error_type = type(exc).__name__
        message = str(exc)
        tb = "".join(_traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        tb = None
    return {
        "task_id": str(task_id),
        "config": config,
        "phase": phase,                 # "run_one" (caught in-child) | "child_crash" (parent-detected)
        "error_type": error_type or "Unknown",
        "message": (message or "")[:2000],
        "traceback": (tb[-4000:] if tb else None),
    }


def write_error(inf_dir: Path, error_record: dict[str, Any]) -> None:
    """Persist one failed inference's ``error.json``. Counts as done for resumption."""
    inf_dir.mkdir(parents=True, exist_ok=True)
    (inf_dir / ERROR_FILE).write_text(json.dumps(error_record, indent=2, ensure_ascii=False))


# ============================================================
# Module loading + subprocess
# ============================================================


def load_benchmark_module(name: str):
    """Import a benchmark module by its public name (e.g. ``loong``)."""
    if name not in BENCHMARK_IMPORT_MAP:
        raise ValueError(f"Unknown benchmark: {name!r}. Known: {sorted(BENCHMARK_IMPORT_MAP)}")
    return importlib.import_module(BENCHMARK_IMPORT_MAP[name])


def run_child(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run one child subprocess ONCE (no retry) and return the completed process.

    stdout is discarded; stderr is captured so the parent can record the tail in
    an ``error.json`` if the child crashed before recording one itself.
    """
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
