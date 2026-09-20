"""Tests for `evals.analysis.score` — the self-contained score.json compute + cache.

A fake benchmark module (``GRADER_MODEL`` / ``SCORER`` / ``score_details``) stands
in for a real one so nothing hits an LLM; ``ScoreResult`` is the real shape.

Coverage:
- `grader_slug`: canonicalizes the benchmark's grader model.
- `ensure_scores`: writes a self-contained score.json (config + score + provenance);
  batches one score_details call; caches on `scored_with`; recomputes on a model
  mismatch or `force`; carries label `parsed` and judge `rationale`.
- `read_score`: returns the dict, or None for missing / unreadable.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from evals.analysis.score import (
    SCORE_FILENAME,
    ensure_scores,
    grader_slug,
    read_score,
)
from evals.benchmarks._scoring import ScoreResult


# ============================================================
# Helpers
# ============================================================


def _bench(score_details, *, grader="openai/gpt-5.4-nano", scorer="fake-judge"):
    return SimpleNamespace(GRADER_MODEL=grader, SCORER=scorer, score_details=score_details)


def _inference(tmp_path, name, manifest):
    """Create an inference folder with a manifest.json; return (dir, manifest)."""
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest))
    return d, manifest


# ============================================================
# grader_slug
# ============================================================


def test_grader_slug_canonicalizes_the_grader_model() -> None:
    assert grader_slug(_bench(None)) == "gpt-5-4-nano"
    assert grader_slug(_bench(None, grader="hosted_vllm/Qwen/Qwen3-32B")) == "qwen3-32b"


# ============================================================
# ensure_scores — write + self-containment
# ============================================================


def test_ensure_scores_writes_self_contained_score_json(tmp_path) -> None:
    calls: list = []

    def fake_details(task_ids, answers):
        calls.append((list(task_ids), list(answers)))
        return [ScoreResult(score=100, rationale="meets every criterion") for _ in task_ids]

    cfg = {"benchmark": "loong", "baseline": "arag", "model": "qwen3-32b", "seed": 42}
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": cfg, "raw_answer": "the answer"})

    stats = ensure_scores([(d, m)], _bench(fake_details, scorer="loong-judge"))

    assert stats == {"scored": 1, "cached": 0, "total": 1}
    assert calls == [(["t1"], ["the answer"])]  # one batched grader call
    score_json = json.loads((d / SCORE_FILENAME).read_text())
    assert score_json == {
        "task_id": "t1",
        "config": cfg,                 # duplicated from the manifest ON PURPOSE
        "score": 100,
        "scorer": "loong-judge",
        "scored_with": "gpt-5-4-nano",
        "parsed": None,                # judge benchmark → no parsed label
        "rationale": "meets every criterion",
    }


def test_ensure_scores_carries_label_parsed_and_no_rationale(tmp_path) -> None:
    """Label benchmarks: score.json carries `parsed` (the extracted verdict) and a
    null `rationale`."""
    def fake_details(task_ids, answers):
        return [ScoreResult(score=1, parsed="Supported") for _ in task_ids]

    d, m = _inference(tmp_path, "h1", {"task_id": "c1", "config": {"benchmark": "dracula"},
                                       "raw_answer": "verdict: supported"})
    ensure_scores([(d, m)], _bench(fake_details, scorer="dracula-parse"))
    sj = json.loads((d / SCORE_FILENAME).read_text())
    assert sj["score"] == 1 and sj["parsed"] == "Supported" and sj["rationale"] is None


# ============================================================
# ensure_scores — caching on scored_with
# ============================================================


def test_ensure_scores_caches_on_matching_grader(tmp_path) -> None:
    calls: list = []

    def fake_details(task_ids, answers):
        calls.append(list(task_ids))
        return [ScoreResult(score=1) for _ in task_ids]

    bench = _bench(fake_details)
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})

    first = ensure_scores([(d, m)], bench)
    second = ensure_scores([(d, m)], bench)  # score.json now fresh → cache hit

    assert first["scored"] == 1
    assert second == {"scored": 0, "cached": 1, "total": 1}
    assert calls == [["t1"]]  # grader called only once


def test_ensure_scores_recomputes_on_grader_mismatch(tmp_path) -> None:
    """A score.json graded by a different MODEL is stale → recompute (scorer matches
    here, so the grader-model axis is isolated)."""
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})
    (d / SCORE_FILENAME).write_text(json.dumps({
        "task_id": "t1", "config": {}, "score": 0, "scorer": "fake-judge",
        "scored_with": "some-old-model", "parsed": None, "rationale": None,
    }))

    stats = ensure_scores([(d, m)], _bench(lambda t, a: [ScoreResult(score=1) for _ in t]))
    assert stats["scored"] == 1
    assert json.loads((d / SCORE_FILENAME).read_text())["score"] == 1  # overwritten


def test_ensure_scores_recomputes_on_scorer_mismatch(tmp_path) -> None:
    """A score.json graded by a different MECHANISM (scorer id) is stale even when
    the grader model matches — so bumping SCORER after a prompt change auto-
    invalidates the cache."""
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})
    (d / SCORE_FILENAME).write_text(json.dumps({
        "task_id": "t1", "config": {}, "score": 0, "scorer": "fake-judge-v1",
        "scored_with": "gpt-5-4-nano", "parsed": None, "rationale": None,
    }))
    bench = _bench(lambda t, a: [ScoreResult(score=1) for _ in t], scorer="fake-judge-v2")
    stats = ensure_scores([(d, m)], bench)
    assert stats["scored"] == 1
    assert json.loads((d / SCORE_FILENAME).read_text())["score"] == 1  # overwritten


def test_ensure_scores_recomputes_when_score_key_missing(tmp_path) -> None:
    """A same-grader/same-scorer file with no usable `score` is NOT a cache hit —
    it recomputes (keeps freshness symmetric with the aggregate readback, which
    needs a `score` key; otherwise the inference would be trusted-then-dropped)."""
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})
    (d / SCORE_FILENAME).write_text(json.dumps({
        "task_id": "t1", "config": {}, "scorer": "fake-judge",
        "scored_with": "gpt-5-4-nano",  # matches, but no "score"
    }))
    stats = ensure_scores([(d, m)], _bench(lambda t, a: [ScoreResult(score=1) for _ in t]))
    assert stats["scored"] == 1
    assert json.loads((d / SCORE_FILENAME).read_text())["score"] == 1


def test_ensure_scores_handles_null_raw_answer(tmp_path) -> None:
    """An explicitly-null raw_answer collapses to the empty string (the empty-answer
    short-circuit), not a crash on .strip()."""
    seen: list = []

    def fake_details(task_ids, answers):
        seen.extend(answers)
        return [ScoreResult(score=0) for _ in task_ids]

    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": None})
    ensure_scores([(d, m)], _bench(fake_details))
    assert seen == [""]  # None became "" before reaching the grader


def test_ensure_scores_writes_atomically_with_no_temp_leftovers(tmp_path) -> None:
    """The atomic write leaves only score.json behind — no .score-*.tmp staging file."""
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})
    ensure_scores([(d, m)], _bench(lambda t, a: [ScoreResult(score=1) for _ in t]))
    leftovers = [p.name for p in d.iterdir()]
    assert SCORE_FILENAME in leftovers
    assert not any(n.startswith(".score-") for n in leftovers), leftovers


def test_ensure_scores_force_recomputes_even_when_fresh(tmp_path) -> None:
    calls: list = []

    def fake_details(task_ids, answers):
        calls.append(list(task_ids))
        return [ScoreResult(score=1) for _ in task_ids]

    bench = _bench(fake_details)
    d, m = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})

    ensure_scores([(d, m)], bench)
    ensure_scores([(d, m)], bench, force=True)  # force → recompute despite fresh cache
    assert calls == [["t1"], ["t1"]]


def test_ensure_scores_only_grades_the_stale_subset(tmp_path) -> None:
    """A mixed batch: one fresh, one missing → only the missing one is graded."""
    graded: list = []

    def fake_details(task_ids, answers):
        graded.extend(task_ids)
        return [ScoreResult(score=1) for _ in task_ids]

    bench = _bench(fake_details)
    d1, m1 = _inference(tmp_path, "h1", {"task_id": "t1", "config": {}, "raw_answer": "a"})
    d2, m2 = _inference(tmp_path, "h2", {"task_id": "t2", "config": {}, "raw_answer": "b"})
    # h1 already scored with the current grader → fresh.
    (d1 / SCORE_FILENAME).write_text(json.dumps({
        "task_id": "t1", "config": {}, "score": 1, "scorer": "fake-judge",
        "scored_with": "gpt-5-4-nano", "parsed": None, "rationale": None,
    }))

    stats = ensure_scores([(d1, m1), (d2, m2)], bench)
    assert stats == {"scored": 1, "cached": 1, "total": 2}
    assert graded == ["t2"]  # only the un-scored one hit the grader


# ============================================================
# read_score
# ============================================================


def test_read_score_missing_returns_none(tmp_path) -> None:
    assert read_score(tmp_path / "nope") is None


def test_read_score_unparseable_returns_none(tmp_path) -> None:
    d = tmp_path / "h1"
    d.mkdir()
    (d / SCORE_FILENAME).write_text("{not json")
    assert read_score(d) is None


def test_read_score_returns_the_dict(tmp_path) -> None:
    d = tmp_path / "h1"
    d.mkdir()
    payload = {"task_id": "t1", "score": 42}
    (d / SCORE_FILENAME).write_text(json.dumps(payload))
    assert read_score(d) == payload
