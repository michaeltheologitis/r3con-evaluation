"""Compute + cache a self-contained ``score.json`` per inference.

The manifest is purely the model's output (``raw_answer``). Grading lives here:
for each inference, read the manifest's raw answer, grade it via the benchmark's
``score_details`` (parse-then-match for the label benchmarks, an LLM judge for the
free-form ones), and write a self-contained ``score.json`` next to the manifest.

``score.json`` duplicates the inference ``config`` ON PURPOSE so a scoreboard reads
ONE file per inference and never re-opens the manifest::

    {
      "task_id": "...",
      "config": { ...full inference config... },
      "score": 17,                   # the metric (0/1 or 1–100)
      "scorer": "loong-judge",       # benchmark-specific mechanism id
      "scored_with": "gpt-5-4-nano", # grader model → cache-invalidation key
      "parsed": "D." | "Supported" | null,   # label benchmarks: what parse extracted
      "rationale": "..." | null              # judge benchmarks: the grader's reasoning
    }

Cached on **(``scored_with``, ``scorer``)** — the grader model AND the grading
mechanism id: an existing score.json graded by the same model and the same
mechanism is reused, so re-running the scoreboard never re-pays the parse / judge
LLM (important now that *label* scoring also hits an LLM — parse moved to score
time). A grader-MODEL change flips ``scored_with``; a grading-MECHANISM change (the
parse / judge prompt, schema, or vocabulary) is signalled by bumping the
benchmark's ``SCORER`` id (e.g. ``loong-judge`` → ``loong-judge-v2``), which flips
``scorer`` — both auto-invalidate the cache (same discipline as graphrag's
``_INDEX_VERSION``). ``force=True`` (the CLI's ``--rescore``) recomputes everything
regardless.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from evals.baselines._common import canonical_model_id

SCORE_FILENAME = "score.json"


def grader_slug(benchmark_module: Any) -> str:
    """Canonical slug of the benchmark's grader model — score.json's ``scored_with``
    and the cache key. Same canonicalization as the model slug in an inference
    config, so a vLLM-vs-OpenAI grader of the same model collapses to one key."""
    return canonical_model_id(benchmark_module.GRADER_MODEL)


def read_score(inference_dir: Path) -> dict | None:
    """The ``score.json`` under ``inference_dir``, or None if absent / unreadable."""
    path = inference_dir / SCORE_FILENAME
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _is_fresh(inference_dir: Path, grader: str, scorer: str, force: bool) -> bool:
    """Whether a usable score.json already exists for this grader + mechanism.

    A cache hit requires the file to exist, carry a usable ``score``, AND match
    BOTH the grader model (``scored_with``) and the grading mechanism id
    (``scorer``). Requiring ``"score" in existing`` keeps freshness symmetric with
    the readback in ``aggregate.attach_scores`` (which needs a ``score`` key) — so a
    same-grader file missing its score recomputes instead of being trusted-then-
    silently-dropped from the metric.
    """
    if force:
        return False
    existing = read_score(inference_dir)
    return (
        existing is not None
        and "score" in existing
        and existing.get("scored_with") == grader
        and existing.get("scorer") == scorer
    )


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (temp file in the same dir + os.replace).

    score.json is a cached grading artifact a later run may reuse without
    recomputing, so a reader must never see a half-written file: a process killed
    mid-write leaves the prior good file intact, not a truncated one (same write
    discipline as the manifest / index artifacts)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".score-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def ensure_scores(
    inferences: list[tuple[Path, dict]],
    benchmark_module: Any,
    *,
    force: bool = False,
) -> dict[str, int]:
    """Ensure every inference has a fresh ``score.json``; compute the missing/stale.

    ``inferences`` is a list of ``(inference_dir, manifest)``. Inferences whose
    score.json is missing, unreadable, incomplete, graded by a different model or a
    different grading mechanism, or ``force``-flagged are (re)graded in ONE batched
    ``score_details`` call (parse / judge fans out internally); fresh ones are left
    untouched (cache hit). Returns ``{"scored": n_computed, "cached": n_reused,
    "total": n}``.
    """
    grader = grader_slug(benchmark_module)
    scorer = benchmark_module.SCORER

    todo = [(d, m) for d, m in inferences if not _is_fresh(d, grader, scorer, force)]
    cached = len(inferences) - len(todo)

    if todo:
        task_ids = [m["task_id"] for _, m in todo]
        # `or ""` (not just a default) so an explicitly-null raw_answer collapses to
        # the empty-answer short-circuit (score 0) instead of crashing on .strip().
        answers = [(m.get("raw_answer") or "") for _, m in todo]
        results = benchmark_module.score_details(task_ids, answers)
        for (inference_dir, manifest), result in zip(todo, results):
            payload = {
                "task_id": manifest["task_id"],
                "config": manifest.get("config", {}),
                "score": result["score"],
                "scorer": scorer,
                "scored_with": grader,
                # parsed (label benchmarks) / rationale (judge benchmarks) are
                # mutually-exclusive optional provenance — absent → null.
                "parsed": result.get("parsed"),
                "rationale": result.get("rationale"),
            }
            _write_atomic(
                inference_dir / SCORE_FILENAME,
                json.dumps(payload, indent=2, ensure_ascii=False),
            )

    return {"scored": len(todo), "cached": cached, "total": len(inferences)}
