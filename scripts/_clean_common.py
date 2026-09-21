"""Shared inference-folder cleanup logic for the per-baseline log cleaners.

The cleaners remove every inference folder whose task should RE-RUN — a folder
the resumption scan counts as "done" even though no genuine answer or genuine
failure was recorded:

- a folder with **neither** ``manifest.json`` nor ``error.json`` (incomplete);
- an **empty / unparseable** ``error.json`` (a half-written file an interrupt
  caught mid-flush);
- a ``ChildCrash`` ``error.json`` (the child died on a signal mid-request);
- any other ``error.json`` whose ``error_type`` is NOT a genuine failed
  prediction (``FAILURE_ERROR_TYPES`` — the same whitelist whatever grades these
  logs later keys on): a ``Timeout`` &c. is transient infra noise, so keeping it
  would silently park the task forever.

while KEEPING real results (any ``manifest.json``) and the genuine model
failures (``ContextWindowExceededError`` — removed only with ``--all-errors``).
This module owns that per-folder classification + deletion so every per-baseline
cleaner shares ONE copy (arag layers its ``_indices/`` store handling on top).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable

# The whitelist of error_types that are genuine failed predictions — defined once
# beside build_error_record (which writes error_type) and shared with whatever
# grades these logs later, so "what the cleaner keeps" and "what a report counts
# as a failed prediction" can never diverge.
from evals.baselines._common import FAILURE_ERROR_TYPES

# Categories removed by default; "real_error" is a genuine failed prediction
# (FAILURE_ERROR_TYPES), removed only with --all-errors. Shared so the cleaners
# report the same vocabulary.
INTERRUPTION_CATEGORIES = ("crash", "broken", "incomplete", "transient")


def classify_inference(inf_dir: Path, all_errors: bool) -> str | None:
    """Return the deletion category for an inference folder, or None to KEEP it.

    - ``manifest.json`` present                  -> None  (a real answer; always kept)
    - ``error.json`` is ``ChildCrash``           -> "crash"
    - ``error.json`` empty / unparseable         -> "broken"
    - error_type NOT in ``FAILURE_ERROR_TYPES``  -> "transient" (Timeout &c. — retryable)
    - a genuine failed prediction                -> "real_error" if all_errors else None
    - neither manifest nor error                 -> "incomplete"
    """
    if (inf_dir / "manifest.json").exists():
        return None  # real result — never touched
    error = inf_dir / "error.json"
    if not error.exists():
        return "incomplete"  # half-written: no manifest, no error
    raw = error.read_text(errors="replace")
    if not raw.strip():
        return "broken"  # 0-byte / whitespace — interrupted write
    try:
        record = json.loads(raw)
    except json.JSONDecodeError:
        return "broken"  # truncated JSON — interrupted write
    if record.get("error_type") == "ChildCrash":
        return "crash"  # child died on a signal (SIGINT/hard crash)
    if record.get("error_type") not in FAILURE_ERROR_TYPES:
        return "transient"  # recorded but retryable (Timeout &c.) — clear so it re-runs
    return "real_error" if all_errors else None  # ContextWindowExceededError


def clean_inference_dirs(
    inferences: Path,
    *,
    all_errors: bool,
    dry_run: bool,
    extra_remove: Callable[[Path], str | None] | None = None,
    keep_if: Callable[[Path], bool] | None = None,
) -> tuple[list[tuple[Path, str]], int]:
    """Classify + remove interrupted inference folders under one ``inferences/`` dir.

    Returns ``(deleted, kept)`` where ``deleted`` is a list of ``(folder,
    category)`` and ``kept`` is the count of folders left intact. In ``dry_run``
    nothing is removed but the same would-be-deleted list is returned.

    ``extra_remove`` is an optional ``(inf_dir) -> str | None`` predicate consulted ONLY
    for folders ``classify_inference`` would KEEP (a real ``manifest.json``): a non-None
    return is used as the deletion category, so a caller can additionally clear
    *successful* runs that meet a custom condition (e.g. rlm/codeact's ``--max-iter`` —
    runs that exhausted their iteration budget). Default ``None`` = no extra removal, so
    every existing caller is unchanged.

    ``keep_if`` is an optional ``(inf_dir) -> bool`` predicate consulted ONLY for folders that
    would otherwise be classified ``"incomplete"`` (neither manifest nor error): a truthy return
    RESCUES the folder (keeps it). This is for a valid mid-pipeline state that looks "incomplete"
    by file presence — e.g. raptor's phase-1 ``embed.json`` checkpoint awaiting ``--phase
    build``. Default ``None`` = no rescue, so every existing caller is unchanged.
    """
    deleted: list[tuple[Path, str]] = []
    kept = 0
    for inf_dir in sorted(p for p in inferences.iterdir() if p.is_dir()):
        category = classify_inference(inf_dir, all_errors)
        if category == "incomplete" and keep_if is not None and keep_if(inf_dir):
            category = None  # a valid mid-pipeline checkpoint (e.g. retrieval.json) — keep it
        if category is None and extra_remove is not None:
            category = extra_remove(inf_dir)  # may promote a kept (manifest) folder to deleted
        if category is None:
            kept += 1
            continue
        deleted.append((inf_dir, category))
        if not dry_run:
            shutil.rmtree(inf_dir)
    return deleted, kept


# ---- index-store cleaning (used by the one INDEXED baseline here: arag) ----

# An index build writes these two receipt files LAST (after a clean build), so their
# joint presence means "build completed"; a dir missing either was interrupted mid-build.
# Same sentinel arag's `_index_is_built` probes, so the cleaner and the runtime agree.
INDEX_RECEIPTS = ("index_meta.json", "index_usage.json")

# The index-dir deletion categories (vs the inference INTERRUPTION_CATEGORIES above), so
# the cleaners' summaries split inference lines from index lines using one vocabulary.
INDEX_CATEGORIES = ("half_built", "orphan")


def _index_is_complete(index_dir: Path) -> bool:
    """Whether ``index_dir`` carries the build's completion receipt (both
    ``INDEX_RECEIPTS`` files)."""
    return all((index_dir / name).exists() for name in INDEX_RECEIPTS)


def classify_index(index_dir: Path, referenced: set[str], prune_orphans: bool) -> str | None:
    """Deletion category for a content-addressed index dir, or None to KEEP it.

    - referenced by any surviving manifest -> None  (in use — NEVER deleted; the safety
      invariant: a content-addressed index is shared, e.g. across arag's completion
      models — the index is LLM-independent)
    - unreferenced + missing receipts      -> "half_built"  (interrupted build)
    - unreferenced + complete              -> "orphan" if prune_orphans else None
    """
    if index_dir.name in referenced:
        return None
    if not _index_is_complete(index_dir):
        return "half_built"
    return "orphan" if prune_orphans else None


def clean_index_store(
    indices: Path, referenced: set[str], *, prune_orphans: bool, dry_run: bool
) -> tuple[list[tuple[Path, str]], int, int]:
    """Classify + remove dangling index dirs under one ``_indices/`` store.

    Returns ``(deleted, kept, n_referenced_present)`` where ``deleted`` is a list of
    ``(index_dir, category)``. Never removes a referenced index (the safety invariant);
    in ``dry_run`` nothing is removed but the same would-be-deleted list is returned.
    Used by the indexed baseline's cleaner (arag).
    """
    deleted: list[tuple[Path, str]] = []
    kept = 0
    if not indices.is_dir():
        return deleted, kept, 0
    present = {p.name for p in indices.iterdir() if p.is_dir()}
    n_referenced = len(referenced & present)
    for index_dir in sorted(p for p in indices.iterdir() if p.is_dir()):
        category = classify_index(index_dir, referenced, prune_orphans)
        if category is None:
            kept += 1
            continue
        deleted.append((index_dir, category))
        if not dry_run:
            shutil.rmtree(index_dir)
    return deleted, kept, n_referenced


def referenced_index_hashes(inferences: Path) -> set[str]:
    """The set of ``index_hash`` strings any surviving ``manifest.json`` references.

    An indexing baseline's manifest records its index via ``index_ref`` (the
    ``index_hash`` string). Reads every ``manifest.json`` under ``inferences/`` and
    collects the non-null ``index_ref`` values — the indices that MUST be kept (one
    LLM-independent index serves many completion models, so an index is "in use" if
    ANY manifest points at it). Unreadable manifests are skipped (don't drop a real
    reference on a transient read error — but a half-written manifest has no answer to
    keep anyway)."""
    refs: set[str] = set()
    if not inferences.is_dir():
        return refs
    for path in inferences.glob("*/manifest.json"):
        try:
            record = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        ref = record.get("index_ref")
        if ref:
            refs.add(ref)
    return refs
