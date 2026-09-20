"""Live end-to-end wiring tests for the scoring stack (`pytest -m live`).

These exercise the FULL grading wiring against the real grader LLM
(`openai/gpt-5.4-nano`): a benchmark's gold answer fed back through
`evals.analysis.score.ensure_scores` must land a **correct** grade in `score.json`,
and a known-wrong answer must land an **incorrect** one — for all four benchmarks,
label (parse-then-match) and free-form (LLM judge) alike. Re-running must hit the
cache (no second grader call). This is the "does the plumbing actually carry a
real grade end to end" check the unit tests (mocked LLM) can't give.

De-selected by default (real LLM, costs a few cents); run with `pytest -m live`.
Needs `OPENAI_API_KEY` (loaded from `.env` when `evals.llm` imports).

One small task per benchmark, two inferences each (gold + wrong), so the whole
module is ~8 grader calls.
"""
from __future__ import annotations

import json

import pytest

from evals.analysis.score import SCORE_FILENAME, ensure_scores, read_score


# ============================================================
# Helper: drive the real score.json scoreboard for a benchmark
# ============================================================


def _run_scoreboard(tmp_path, benchmark_module, items):
    """Write a manifest per item, run the REAL ensure_scores, return everything.

    ``items`` = list of ``(name, task_id, raw_answer)``. Returns
    ``(stats, scores_by_name, inferences)`` where ``scores_by_name[name]`` is the
    parsed ``score.json`` dict and ``inferences`` is the ``(dir, manifest)`` list
    (so a caller can re-run ensure_scores to check caching).
    """
    inferences = []
    for name, task_id, raw_answer in items:
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        manifest = {
            "task_id": task_id,
            "config": {"benchmark": benchmark_module.__name__.rsplit(".", 1)[-1],
                       "baseline": "live-test", "model": "gold-vs-wrong", "seed": 0},
            "raw_answer": raw_answer,
        }
        (d / "manifest.json").write_text(json.dumps(manifest))
        inferences.append((d, manifest))

    stats = ensure_scores(inferences, benchmark_module)
    scores = {name: read_score(d) for (name, _, _), (d, _) in zip(items, inferences)}
    return stats, scores, inferences


def _assert_cache_hit_on_rerun(benchmark_module, inferences) -> None:
    """A second ensure_scores over the same inferences must grade nothing."""
    re_stats = ensure_scores(inferences, benchmark_module)
    assert re_stats == {"scored": 0, "cached": len(inferences), "total": len(inferences)}


# ============================================================
# Label benchmarks — parse-then-match (gold letter/verdict in, wrong in)
# ============================================================


@pytest.mark.live
def test_live_wiring_corpusqa_scoreboard(tmp_path) -> None:
    from evals.benchmarks import corpusqa as bench

    tid = bench.get_task_ids(domains=["Single-Document QA"], lengths=["short"], limit=1)[0]
    gold = bench.get_task_answer(tid)  # e.g. "C."
    choices = bench.get_task_choices(tid)
    gold_choice = next(c for c in choices if c.startswith(gold))
    wrong_choice = next(c for c in choices if not c.startswith(gold))

    stats, scores, inferences = _run_scoreboard(tmp_path, bench, [
        ("gold", tid, f"After weighing the options, the answer is {gold_choice}."),
        ("wrong", tid, f"I'm confident the answer is {wrong_choice}."),
    ])

    assert stats["scored"] == 2
    # Gold raw answer → parsed to the gold letter → score 1, provenance recorded.
    assert scores["gold"]["score"] == 1
    assert scores["gold"]["parsed"] == gold
    assert scores["gold"]["scorer"] == "corpusqa-parse"
    assert scores["gold"]["scored_with"] == "gpt-5-4-nano"
    assert scores["gold"]["rationale"] is None  # label benchmark
    # Wrong raw answer → a different letter → score 0.
    assert scores["wrong"]["score"] == 0
    assert scores["wrong"]["parsed"] != gold

    _assert_cache_hit_on_rerun(bench, inferences)


@pytest.mark.live
def test_live_wiring_dracula_scoreboard(tmp_path) -> None:
    from evals.benchmarks import dracula as bench

    tid = bench.get_task_ids(limit=1)[0]
    gold = bench.get_task_answer(tid)  # Supported / Refuted / Overruled
    wrong = next(v for v in ("Supported", "Refuted", "Overruled") if v != gold)

    stats, scores, inferences = _run_scoreboard(tmp_path, bench, [
        ("gold", tid, f"Having reviewed the precedent, my verdict is that the claim is {gold}."),
        ("wrong", tid, f"Having reviewed the precedent, my verdict is that the claim is {wrong}."),
    ])

    assert stats["scored"] == 2
    assert scores["gold"]["score"] == 1
    assert scores["gold"]["parsed"] == gold
    assert scores["gold"]["scorer"] == "dracula-parse"
    assert scores["wrong"]["score"] == 0
    assert scores["wrong"]["parsed"] == wrong

    _assert_cache_hit_on_rerun(bench, inferences)


# ============================================================
# Free-form benchmarks — LLM judge (gold answer in → high/correct; nonsense → low)
# ============================================================

_NONSENSE = "Bananas are an igneous rock formed deep underground over millennia."


@pytest.mark.live
def test_live_wiring_loong_scoreboard(tmp_path) -> None:
    from evals.benchmarks import loong as bench

    tid = bench.get_task_ids(subsets=["shortdep_qa"], limit=1)[0]
    gold = bench.get_task_answer(tid)

    stats, scores, inferences = _run_scoreboard(tmp_path, bench, [
        ("gold", tid, gold),          # gold answer fed back → judged correct
        ("wrong", tid, _NONSENSE),    # unrelated → judged incorrect
    ])

    assert stats["scored"] == 2
    assert scores["gold"]["score"] == 1
    assert scores["gold"]["scorer"] == "loong-judge"
    assert scores["gold"]["parsed"] is None             # judge benchmark
    assert scores["gold"]["rationale"]                  # judge recorded a rationale
    assert scores["wrong"]["score"] == 0

    _assert_cache_hit_on_rerun(bench, inferences)


@pytest.mark.live
def test_live_wiring_loong_scoreboard(tmp_path) -> None:
    from evals.benchmarks import loong as bench
    from evals.benchmarks.loong import judge

    # A task whose gold is a plain string (some Loong golds are JSON objects).
    tid = next(t for t in bench.get_task_ids() if isinstance(bench.get_task_answer(t), str))
    gold = judge._format_gold(bench.get_task_answer(tid))

    stats, scores, inferences = _run_scoreboard(tmp_path, bench, [
        ("gold", tid, gold),          # gold fed back → rated high
        ("wrong", tid, _NONSENSE),    # unrelated → rated low
    ])

    assert stats["scored"] == 2
    assert scores["gold"]["scorer"] == "loong-judge"
    assert scores["gold"]["parsed"] is None
    assert scores["gold"]["rationale"]
    # 1–100 rating: gold near the top, nonsense near the floor, gold clearly above.
    assert scores["gold"]["score"] >= 80
    assert scores["wrong"]["score"] <= 40
    assert scores["gold"]["score"] > scores["wrong"]["score"]

    _assert_cache_hit_on_rerun(bench, inferences)
