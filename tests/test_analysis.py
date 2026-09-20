"""Tests for `evals.analysis.aggregate`.

Synthetic logs only — no real benchmark.score calls. The aggregation logic is
pure-Python over file contents, so the tests don't round-trip a real benchmark.

Coverage:
- `find_groups`: groups manifests by config; filters by benchmark/baseline;
  ignores base dirs without an `inferences/` subdir; attaches `_inference_dir`.
- `attach_scores`: reads each inference's score.json onto `_score`; computes the
  missing ones via ensure_scores when asked; reads-only when `compute=False`.
- `construction_usage`: sums DISTINCT indices' build cost (deduped by index_ref).
- `aggregate_group`: accuracy + inference/construction/total cost split.
- `sum_usage` / `cost_per_model_usd`: per-model arithmetic.
- CLI: filters propagate; default computes+caches score.json; --no-score skips
  the compute; empty → "no inferences found".
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evals.analysis import aggregate as aggregate_module
from evals.analysis.aggregate import (
    FAILURE_ERROR_TYPES,
    aggregate_group,
    attach_scores,
    construction_usage,
    cost_per_model_usd,
    failure_manifests,
    find_groups,
    main,
    sum_usage,
)
from evals.analysis.score import SCORE_FILENAME
from evals.benchmarks._scoring import ScoreResult


# ============================================================
# Helpers
# ============================================================


def _config(benchmark="loong", baseline="arag", **extra) -> dict:
    cfg = {"benchmark": benchmark, "baseline": baseline, "model": "gpt-5-4-nano",
           "seed": 42}
    cfg.update(extra)
    return cfg


def _write_manifest(base: Path, inference_hash: str, manifest: dict) -> None:
    d = base / "inferences" / inference_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(manifest))


def _write_error(base: Path, inference_hash: str, error_record: dict) -> None:
    d = base / "inferences" / inference_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / "error.json").write_text(json.dumps(error_record))


def _write_index_usage(base: Path, index_hash: str, rollup: dict) -> None:
    """Write an index's build-cost file (the new {total, calls} envelope) at the
    flat content-addressed path ``_indices/{index_hash}/index_usage.json``."""
    d = base / "_indices" / index_hash
    d.mkdir(parents=True, exist_ok=True)
    (d / "index_usage.json").write_text(json.dumps({"total": rollup, "calls": []}))


def _rollup(model: str, total: int, calls: int = 1) -> dict:
    """A per-model ``total`` rollup."""
    return {model: {"prompt_tokens": total, "completion_tokens": 0,
                    "total_tokens": total, "num_calls": calls}}


def _usage(model: str, total: int, calls: int = 1) -> dict:
    """A manifest ``usage`` field: {total: rollup, calls: []}."""
    return {"total": _rollup(model, total, calls), "calls": []}


# ============================================================
# find_groups
# ============================================================


def test_find_groups_groups_manifests_by_config(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    # Two inferences with search_method=global, one with =basic.
    _write_manifest(base, "h1", {"task_id": "a", "config": _config(search_method="global")})
    _write_manifest(base, "h2", {"task_id": "b", "config": _config(search_method="global")})
    _write_manifest(base, "h3", {"task_id": "a", "config": _config(search_method="basic")})

    groups = find_groups(logs_dir=tmp_path)
    by_method = {g["config"]["search_method"]: g for g in groups}
    assert set(by_method) == {"global", "basic"}
    assert len(by_method["global"]["manifests"]) == 2
    assert len(by_method["basic"]["manifests"]) == 1


def test_find_groups_filters_by_benchmark_and_baseline(tmp_path) -> None:
    _write_manifest(tmp_path / "loong" / "arag", "h1",
                    {"task_id": "a", "config": _config()})
    _write_manifest(tmp_path / "loong" / "codeact", "h2",
                    {"task_id": "a", "config": _config(baseline="codeact")})
    _write_manifest(tmp_path / "dracula" / "arag", "h3",
                    {"task_id": "a", "config": _config(benchmark="dracula")})

    assert len(find_groups(benchmark="loong", logs_dir=tmp_path)) == 2
    assert len(find_groups(baseline="arag", logs_dir=tmp_path)) == 2
    assert len(find_groups(benchmark="loong", baseline="codeact", logs_dir=tmp_path)) == 1


def test_find_groups_ignores_base_dir_without_inferences(tmp_path) -> None:
    (tmp_path / "loong" / "arag" / "_indices").mkdir(parents=True)  # index store only
    assert find_groups(logs_dir=tmp_path) == []


def test_find_groups_empty_when_logs_dir_missing(tmp_path) -> None:
    assert find_groups(logs_dir=tmp_path / "nope") == []


def test_find_groups_skips_unparseable_manifest(tmp_path, capsys: pytest.CaptureFixture) -> None:
    base = tmp_path / "loong" / "arag"
    (base / "inferences" / "broken").mkdir(parents=True)
    (base / "inferences" / "broken" / "manifest.json").write_text("{nope")
    _write_manifest(base, "good", {"task_id": "a", "config": _config()})
    groups = find_groups(logs_dir=tmp_path)
    assert len(groups) == 1 and len(groups[0]["manifests"]) == 1
    assert "skipping unparseable" in capsys.readouterr().out


# ============================================================
# attach_scores — read score.json onto _score (+ optional compute)
# ============================================================


def _fake_scoring_benchmark(score_details, *, grader="openai/gpt-5.4-nano", scorer="fake-judge"):
    return SimpleNamespace(GRADER_MODEL=grader, SCORER=scorer, score_details=score_details)


def _group_with_inferences(tmp_path, *manifests):
    """A group whose manifests carry `_inference_dir` (as find_groups would set)."""
    out = []
    for i, m in enumerate(manifests):
        d = tmp_path / f"inf{i}"
        d.mkdir(parents=True, exist_ok=True)
        m = {**m, "_inference_dir": d}
        out.append(m)
    return {"base_dir": tmp_path, "config": _config(), "manifests": out}


def test_attach_scores_reads_existing_score_json_without_computing(tmp_path) -> None:
    """compute=False: read whatever score.json is on disk onto `_score`, never call
    the grader."""
    group = _group_with_inferences(
        tmp_path,
        {"task_id": "1", "raw_answer": "a"},
        {"task_id": "2", "raw_answer": "b"},
    )
    # Pre-write a score.json for the first inference only.
    d0 = group["manifests"][0]["_inference_dir"]
    (d0 / SCORE_FILENAME).write_text(json.dumps({"task_id": "1", "score": 1}))

    called: list = []
    bench = _fake_scoring_benchmark(lambda t, a: called.append(1) or [])
    stats = attach_scores(group, bench, compute=False)

    assert called == []  # compute=False → grader never called
    assert stats == {"scored": 0, "cached": 0, "total": 2}
    assert group["manifests"][0]["_score"] == 1
    assert "_score" not in group["manifests"][1]  # no score.json → unscored


def test_attach_scores_computes_missing_then_reads_back(tmp_path) -> None:
    group = _group_with_inferences(
        tmp_path,
        {"task_id": "1", "raw_answer": "won the game"},
        {"task_id": "2", "raw_answer": "no idea"},
    )

    def fake_details(task_ids, answers):
        return [ScoreResult(score=1 if "won" in a else 0) for a in answers]

    bench = _fake_scoring_benchmark(fake_details)
    stats = attach_scores(group, bench, compute=True)

    assert stats == {"scored": 2, "cached": 0, "total": 2}
    assert [m["_score"] for m in group["manifests"]] == [1, 0]
    # score.json now persisted for both.
    assert all((m["_inference_dir"] / SCORE_FILENAME).exists() for m in group["manifests"])


# ============================================================
# sum_usage + cost_per_model_usd
# ============================================================


def test_sum_usage_combines_per_model_rollups() -> None:
    # sum_usage operates on per-model rollups (the `total` shape), not manifests.
    out = sum_usage(_rollup("m", 100, 1), _rollup("m", 50, 2))
    assert out["m"]["total_tokens"] == 150
    assert out["m"]["num_calls"] == 3


def test_sum_usage_recurses_into_nested_token_details() -> None:
    a = {"m": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 4}}}
    b = {"m": {"prompt_tokens": 20, "prompt_tokens_details": {"cached_tokens": 6}}}
    out = sum_usage(a, b)
    assert out["m"]["prompt_tokens"] == 30
    assert out["m"]["prompt_tokens_details"]["cached_tokens"] == 10


def test_sum_usage_empty_inputs() -> None:
    assert sum_usage() == {}
    assert sum_usage(None, {}) == {}


def test_cost_per_model_usd_known_model_priced() -> None:
    costs = cost_per_model_usd(_rollup("openai/gpt-5.4-nano", 1500))
    assert costs["openai/gpt-5.4-nano"] is not None and costs["openai/gpt-5.4-nano"] >= 0


def test_cost_per_model_usd_unknown_model_returns_none(tmp_path) -> None:
    costs = cost_per_model_usd(_rollup("some-vllm-model-xyz", 1000))
    assert costs["some-vllm-model-xyz"] is None


def test_cost_per_model_usd_manual_pricing_fallback() -> None:
    # claude-sonnet-5 isn't in LiteLLM yet → settings.MANUAL_MODEL_PRICING fills it in
    # ($2/1M input, $10/1M output) → 1M of each = $2 + $10 = $12.
    costs = cost_per_model_usd({"claude-sonnet-5": {"prompt_tokens": 1_000_000,
                                                    "completion_tokens": 1_000_000}})
    assert costs["claude-sonnet-5"] == 12.0
    # the provider-prefixed form canonicalizes to the same manual-pricing key
    costs2 = cost_per_model_usd({"anthropic/claude-sonnet-5": {"prompt_tokens": 500_000,
                                                               "completion_tokens": 0}})
    assert costs2["anthropic/claude-sonnet-5"] == 1.0


# ============================================================
# construction_usage — dedup across shared indices
# ============================================================


def test_construction_usage_counts_distinct_indices_once(tmp_path) -> None:
    """Two inferences (e.g. global + basic) referencing the SAME index must
    count its build cost ONCE."""
    base = tmp_path / "loong" / "arag"
    _write_index_usage(base, "idxA", _rollup("m", 1000, 50))
    manifests = [
        {"task_id": "doc-1", "index_ref": "idxA"},
        {"task_id": "doc-1", "index_ref": "idxA"},
    ]
    usage, n_indices = construction_usage(base, manifests)
    assert n_indices == 1
    assert usage["m"]["total_tokens"] == 1000  # counted once, not 2000


def test_construction_usage_sums_distinct_per_example_indices(tmp_path) -> None:
    """loong: each task has its OWN index (distinct doc → distinct hash) → sum."""
    base = tmp_path / "loong" / "arag"
    _write_index_usage(base, "hA", _rollup("m", 1000))
    _write_index_usage(base, "hB", _rollup("m", 1500))
    manifests = [
        {"task_id": "doc-1", "index_ref": "hA"},
        {"task_id": "doc-2", "index_ref": "hB"},
    ]
    usage, n_indices = construction_usage(base, manifests)
    assert n_indices == 2
    assert usage["m"]["total_tokens"] == 2500


def test_construction_usage_zero_for_inferences_without_index_ref(tmp_path) -> None:
    """a no-index baseline (codeact) has no index → no construction cost."""
    usage, n_indices = construction_usage(tmp_path, [{"task_id": "1"}])
    assert usage == {} and n_indices == 0


# ============================================================
# aggregate_group
# ============================================================


def test_aggregate_group_splits_inference_and_construction_cost(tmp_path) -> None:
    base = tmp_path / "loong" / "arag"
    _write_index_usage(base, "hA", _rollup("m", 1000, 50))
    _write_index_usage(base, "hB", _rollup("m", 1000, 50))
    group = {"base_dir": base, "config": _config(search_method="basic"), "manifests": [
        {"task_id": "doc-1", "raw_answer": "a", "usage": _usage("m", 20, 1), "index_ref": "hA"},
        {"task_id": "doc-2", "raw_answer": "b", "usage": _usage("m", 30, 1), "index_ref": "hB"},
    ]}
    summary = aggregate_group(group)
    # inference = 20 + 30 = 50; construction = 1000 + 1000 = 2000; total = 2050.
    assert summary["usage"]["inference"]["m"]["total_tokens"] == 50
    assert summary["usage"]["construction"]["m"]["total_tokens"] == 2000
    assert summary["usage"]["total"]["m"]["total_tokens"] == 2050
    assert summary["n_indices"] == 2


def test_aggregate_group_headline_accuracy(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(), "manifests": [
        {"task_id": "1", "raw_answer": "a", "_score": 1},
        {"task_id": "2", "raw_answer": "b", "_score": 0},
        {"task_id": "3", "raw_answer": "c", "_score": 1},
    ]}
    summary = aggregate_group(group)
    assert summary["n"] == 3 and summary["n_correct"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)


def test_aggregate_group_no_scored_returns_none_accuracy(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(), "manifests": [
        {"task_id": "1", "raw_answer": "a"},
    ]}
    assert aggregate_group(group)["accuracy"] is None


def test_aggregate_group_rating_metric_for_perfect_score_benchmark(tmp_path) -> None:
    """A benchmark exposing PERFECT_SCORE (Loong) → Avg Score (mean rating) +
    Perfect Rate (fraction == PERFECT_SCORE), not accuracy."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"), "manifests": [
        {"task_id": "1", "raw_answer": "a", "_score": 100},
        {"task_id": "2", "raw_answer": "b", "_score": 50},
        {"task_id": "3", "raw_answer": "c", "_score": 100},
    ]}
    summary = aggregate_group(group, SimpleNamespace(PERFECT_SCORE=100))
    m = summary["metric"]
    assert m["kind"] == "rating"
    assert m["avg_score"] == pytest.approx((100 + 50 + 100) / 3)
    assert m["n_perfect"] == 2
    assert m["perfect_rate"] == pytest.approx(2 / 3)
    # Top-level accuracy/n_correct are meaningless for a 1–100 rating benchmark
    # (a sum of ratings isn't "n correct") → None; readers use `metric` instead.
    assert summary["accuracy"] is None
    assert summary["n_correct"] is None


def test_aggregate_group_accuracy_metric_without_perfect_score(tmp_path) -> None:
    """The 0/1 label benchmarks (no PERFECT_SCORE) keep the accuracy metric."""
    group = {"base_dir": tmp_path, "config": _config(), "manifests": [
        {"task_id": "1", "_score": 1}, {"task_id": "2", "_score": 0},
    ]}
    summary = aggregate_group(group)  # no benchmark module → accuracy kind
    assert summary["metric"]["kind"] == "accuracy"
    assert summary["metric"]["accuracy"] == pytest.approx(0.5)


def test_aggregate_group_dual_view_answered_and_all(tmp_path) -> None:
    """Context-window errors are folded into the 'all'/total view (scored at the worst
    value); the summary carries BOTH the answered (`metric`) and total (`metric_all`)
    numbers in its data, even though the report renders only the total. The same
    numerator, a bigger denominator."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="corpusqa"),
             "manifests": [{"task_id": "a", "_score": 1}],
             "errors": [{"task_id": "e", "error_type": "ContextWindowExceededError"}]}
    meta = {"a": {"length": "short"}, "e": {"length": "long"}}
    bench = _fake_scoring_benchmark(lambda t, ans: [ScoreResult(score=0) for _ in t])
    bench.get_task_metadata = lambda t: meta[t]

    summary = aggregate_group(group, bench)
    assert summary["metric"]["accuracy"] == 1.0       # answered: 1/1
    assert summary["metric_all"]["accuracy"] == 0.5   # all: 1/2 (ctx err scored 0)
    assert summary["n_failures"] == 1
    short = summary["by_axis"]["length"]["short"]
    long = summary["by_axis"]["length"]["long"]
    assert short["accuracy"] == 1.0 and short["all"]["accuracy"] == 1.0  # no err here
    assert long["accuracy"] is None and long["all"]["accuracy"] == 0.0   # all-errored bucket


def test_hit_step_cap_predicate() -> None:
    """Detect a budget-exhausted run from the manifest trace: codeact's
    `state == max_steps_error`, or rlm's `n_iterations >= max_iterations`. Anything
    else (other baselines, no trace) is not capped."""
    from evals.analysis.aggregate import _hit_step_cap
    assert _hit_step_cap({"trace": {"state": "max_steps_error"}}) is True
    assert _hit_step_cap({"trace": {"state": "success"}}) is False
    assert _hit_step_cap({"trace": {"n_iterations": 30, "max_iterations": 30}}) is True
    assert _hit_step_cap({"trace": {"n_iterations": 31, "max_iterations": 30}}) is True
    assert _hit_step_cap({"trace": {"n_iterations": 5, "max_iterations": 30}}) is False
    assert _hit_step_cap({"trace": {}}) is False          # no signal (e.g. codeact/structrag)
    assert _hit_step_cap({}) is False                      # no trace at all


def test_aggregate_group_folds_step_capped_into_all_view(tmp_path) -> None:
    """A step/iteration-capped run (codeact `max_steps_error`, rlm `n_iterations>=max`)
    keeps its REAL judged score in the answered view but is counted WRONG in the ⁺ all
    view — re-scored at the benchmark's empty-answer worst, like a context-window error.
    So the headline `answered` credits a lucky cap-survivor; `⁺ all` doesn't."""
    group = {"base_dir": tmp_path,
             "config": _config(benchmark="corpusqa", baseline="codeact"),
             "manifests": [
                 {"task_id": "ok",      "_score": 1, "trace": {"state": "success"}},
                 {"task_id": "cap_hit", "_score": 1, "trace": {"state": "max_steps_error"}},  # lucky → wrong in ⁺
                 {"task_id": "cap_miss","_score": 0, "trace": {"state": "max_steps_error"}},
                 {"task_id": "rlm_cap", "_score": 1, "trace": {"n_iterations": 30, "max_iterations": 30}},
             ]}
    meta = {t: {"domain": "education"} for t in ("ok", "cap_hit", "cap_miss", "rlm_cap")}
    bench = _fake_scoring_benchmark(lambda t, ans: [ScoreResult(score=0) for _ in t])  # empty → 0
    bench.get_task_metadata = lambda t: meta[t]

    summary = aggregate_group(group, bench)
    assert summary["n_capped"] == 3
    assert summary["metric"]["accuracy"] == pytest.approx(3 / 4)       # answered credits the 3 "correct"
    assert summary["metric_all"]["accuracy"] == pytest.approx(1 / 4)   # ⁺ all: only `ok` survives → 1/4
    edu = summary["by_axis"]["domain"]["education"]
    assert edu["accuracy"] == pytest.approx(3 / 4)                     # per-axis answered matches
    assert edu["all"]["accuracy"] == pytest.approx(1 / 4)             # per-axis ⁺ folds caps too


def test_aggregate_group_no_caps_leaves_all_view_unchanged(tmp_path) -> None:
    """A baseline that records no step/iteration budget (no `state`/`n_iterations` in the
    trace) is untouched — `⁺ all` equals the answered view (no spurious cap-folding)."""
    group = {"base_dir": tmp_path,
             "config": _config(benchmark="corpusqa", baseline="codeact"),
             "manifests": [{"task_id": "a", "_score": 1, "trace": {"response": "..."}},
                           {"task_id": "b", "_score": 0, "trace": {}}]}
    bench = _fake_scoring_benchmark(lambda t, ans: [ScoreResult(score=0) for _ in t])
    bench.get_task_metadata = lambda t: {"domain": "education"}
    summary = aggregate_group(group, bench)
    assert summary["n_capped"] == 0
    assert summary["metric"]["accuracy"] == summary["metric_all"]["accuracy"] == pytest.approx(0.5)


def test_aggregate_group_rating_per_axis_breakdown(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"), "manifests": [
        {"task_id": "1", "_score": 100}, {"task_id": "2", "_score": 60},
    ]}
    meta = {"1": {"task_name": "Clustering"}, "2": {"task_name": "Clustering"}}
    bench = SimpleNamespace(PERFECT_SCORE=100, get_task_metadata=lambda t: meta[t])
    cell = aggregate_group(group, bench)["by_axis"]["task_name"]["Clustering"]
    assert cell["kind"] == "rating"
    assert cell["avg_score"] == pytest.approx(80.0)
    assert cell["n_perfect"] == 1 and cell["n"] == 2


def _render_to_text(summary, width: int = 140) -> str:
    """Render a group summary to plain text (color stripped) for assertions."""
    import io

    from rich.console import Console

    from evals.analysis.aggregate import render_report

    buf = io.StringIO()
    render_report(summary, Console(file=buf, force_terminal=False, no_color=True, width=width))
    return buf.getvalue()


def test_render_report_accuracy_has_tables_and_values(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(benchmark="corpusqa"),
             "manifests": [{"task_id": "1", "_score": 1}, {"task_id": "2", "_score": 0}],
             "errors": []}
    meta = {"1": {"length": "short"}, "2": {"length": "long"}}
    summary = aggregate_group(group, SimpleNamespace(get_task_metadata=lambda t: meta[t]))
    text = _render_to_text(summary)
    assert "accuracy" in text and "0.500" in text  # 1/2 correct
    assert "by length" in text and "short" in text and "long" in text


def test_render_report_rating_shows_avg_not_accuracy(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"),
             "manifests": [{"task_id": "1", "_score": 100}, {"task_id": "2", "_score": 50}],
             "errors": []}
    summary = aggregate_group(group, SimpleNamespace(PERFECT_SCORE=100))
    text = _render_to_text(summary)
    assert "avg score" in text and "perfect rate" in text
    assert "accuracy" not in text


def test_render_report_shows_error_coverage(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(benchmark="corpusqa"),
             "manifests": [{"task_id": "1", "_score": 1}],
             "errors": [{"task_id": "e", "error_type": "ContextWindowExceededError"}]}
    text = _render_to_text(aggregate_group(group))
    assert "ContextWindowExceededError" in text and "error" in text.lower()


def test_render_report_shows_config_name_hides_param_dump(tmp_path) -> None:
    """When a run used a --config preset, the report labels the group by the config
    NAME and does NOT dump the verbose completion_params dict into the header."""
    cfg = {"benchmark": "loong", "baseline": "codeact", "model": "qwen3-30b", "seed": 42,
           "config_name": "Qwen3.5-MoE-Instruct",
           "completion_params": {"temperature": 0.7, "extra_body": {"top_k": 20}}}
    group = {"base_dir": tmp_path, "config": cfg,
             "manifests": [{"task_id": "1", "_score": 1}], "errors": []}
    text = _render_to_text(aggregate_group(group))
    assert "Qwen3.5-MoE-Instruct" in text       # the name labels the group
    assert "completion_params" not in text       # the verbose dict is hidden
    assert "top_k" not in text


def test_render_report_skips_high_cardinality_axis(tmp_path) -> None:
    """A continuous / high-cardinality axis (e.g. loong's raw token `length`) must
    NOT render as one row per task — it's skipped with a note, while real grouping
    axes still render."""
    manifests = [{"task_id": str(i), "_score": i % 2} for i in range(30)]
    meta = {str(i): {"length": 1000 + i, "domain": "paper"} for i in range(30)}
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"),
             "manifests": manifests, "errors": []}
    bench = SimpleNamespace(get_task_metadata=lambda t: meta[t])

    text = _render_to_text(aggregate_group(group, bench))
    assert "by domain" in text       # 1 distinct value → kept
    assert "30 distinct" in text     # the length axis is noted as skipped
    assert "1000" not in text and "1029" not in text  # NOT rendered as 30 rows


def test_render_report_shows_single_total_view(tmp_path) -> None:
    """The report shows ONE total accuracy (context-window errors + caps counted wrong) —
    no separate answered/⁺ columns. A bucket entirely context-window-errored renders its
    total (0.0 over its failures), not an answered N/A."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="corpusqa"),
             "manifests": [{"task_id": "a", "_score": 1}],
             "errors": [{"task_id": "e", "error_type": "ContextWindowExceededError"}]}
    meta = {"a": {"length": "short"}, "e": {"length": "long"}}
    bench = _fake_scoring_benchmark(lambda t, ans: [ScoreResult(score=0) for _ in t])
    bench.get_task_metadata = lambda t: meta[t]

    text = _render_to_text(aggregate_group(group, bench))
    assert "acc⁺" not in text                    # no dual ⁺ column anymore
    assert "0.500" in text                        # the single TOTAL accuracy: 1/2 (ctx err counted wrong)
    assert "by length" in text and "short" in text and "long" in text
    assert "1.000" in text and "0.000" in text    # per-axis totals: short 1.0, long 0.0 (the all-errored bucket)


def test_aggregate_group_shared_index_counted_once_in_total(tmp_path) -> None:
    """The headline cost win: 4 modes sharing one index pay construction once."""
    base = tmp_path / "loong" / "arag"
    _write_index_usage(base, "h", _rollup("m", 1000, 50))
    # global + basic inferences of the same task, same index.
    group = {"base_dir": base, "config": _config(), "manifests": [
        {"task_id": "doc-1", "raw_answer": "a", "usage": _usage("m", 20), "index_ref": "h"},
        {"task_id": "doc-1", "raw_answer": "b", "usage": _usage("m", 25), "index_ref": "h"},
    ]}
    summary = aggregate_group(group)
    assert summary["n_indices"] == 1
    assert summary["usage"]["construction"]["m"]["total_tokens"] == 1000  # once
    assert summary["usage"]["total"]["m"]["total_tokens"] == 1045  # 1000 + 20 + 25


# ============================================================
# CLI
# ============================================================


def test_main_prints_no_inferences_when_empty(tmp_path, monkeypatch, capsys) -> None:
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    main(["--benchmark", "loong"])
    assert "no inferences found" in capsys.readouterr().out


def test_main_no_score_skips_scoring(tmp_path, monkeypatch, capsys) -> None:
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    _write_manifest(tmp_path / "loong" / "arag", "h1",
                    {"task_id": "a", "raw_answer": "x", "config": _config()})
    main(["--benchmark", "loong", "--no-score"])
    out = capsys.readouterr().out
    assert "0 scored" in out
    # --no-score must not write a score.json.
    assert not (tmp_path / "loong" / "arag" / "inferences" / "h1" / SCORE_FILENAME).exists()


def _patch_benchmark_import(monkeypatch, fake) -> None:
    """Make the CLI's benchmark import resolve to ``fake`` (no real LLM / dataset)."""
    monkeypatch.setattr(aggregate_module.importlib, "import_module", lambda path: fake)


def test_main_default_computes_and_caches_score_json(tmp_path, monkeypatch, capsys) -> None:
    """Default run grades each inference, writes a self-contained score.json, and
    reports the metric; a second run reads the cache and never re-grades."""
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    _write_manifest(tmp_path / "loong" / "arag", "h1",
                    {"task_id": "a", "raw_answer": "the home team won", "config": _config()})

    calls: list = []

    def fake_details(task_ids, answers):
        calls.append(list(task_ids))
        return [ScoreResult(score=1) for _ in task_ids]

    fake = SimpleNamespace(score_details=fake_details, SCORER="loong-judge",
                           GRADER_MODEL="openai/gpt-5.4-nano")
    _patch_benchmark_import(monkeypatch, fake)

    main(["--benchmark", "loong"])
    out = capsys.readouterr().out
    assert calls == [["a"]]
    assert "accuracy" in out and "1.000" in out

    score_path = tmp_path / "loong" / "arag" / "inferences" / "h1" / SCORE_FILENAME
    sj = json.loads(score_path.read_text())
    assert sj["score"] == 1
    assert sj["scorer"] == "loong-judge"
    assert sj["scored_with"] == "gpt-5-4-nano"
    assert sj["config"]["benchmark"] == "loong"  # self-contained config

    # Second run → cache hit: the grader is NOT called again, metric still shown.
    main(["--benchmark", "loong"])
    assert calls == [["a"]]  # unchanged
    out2 = capsys.readouterr().out
    assert "accuracy" in out2 and "1.000" in out2


def test_main_rescore_forces_recompute(tmp_path, monkeypatch, capsys) -> None:
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    _write_manifest(tmp_path / "loong" / "arag", "h1",
                    {"task_id": "a", "raw_answer": "x", "config": _config()})

    calls: list = []

    def fake_details(task_ids, answers):
        calls.append(list(task_ids))
        return [ScoreResult(score=0) for _ in task_ids]

    fake = SimpleNamespace(score_details=fake_details, SCORER="loong-judge",
                           GRADER_MODEL="openai/gpt-5.4-nano")
    _patch_benchmark_import(monkeypatch, fake)

    main(["--benchmark", "loong"])             # grades once
    main(["--benchmark", "loong", "--rescore"])  # forces a re-grade
    capsys.readouterr()
    assert calls == [["a"], ["a"]]


def test_main_scoring_failure_does_not_abort_the_cli(tmp_path, monkeypatch, capsys) -> None:
    """A scoring error (e.g. a stale task_id) is bounded to its group: the CLI warns
    and falls back to cached scores instead of crashing the whole run."""
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    _write_manifest(tmp_path / "loong" / "arag", "h1",
                    {"task_id": "ghost", "raw_answer": "x", "config": _config()})

    def boom(task_ids, answers):
        raise KeyError(f"Unknown task id(s): {task_ids}")

    fake = SimpleNamespace(score_details=boom, SCORER="loong-judge",
                           GRADER_MODEL="openai/gpt-5.4-nano")
    _patch_benchmark_import(monkeypatch, fake)

    main(["--benchmark", "loong"])  # must NOT raise
    out = capsys.readouterr().out
    assert "scoring failed" in out
    assert "0 scored" in out  # no cached score → unscored, but the CLI completed


# ============================================================
# Error-as-failure scoring (find_groups errors + failure_manifests + dual answered/all view)
# ============================================================


def test_find_groups_collects_error_records_by_config(tmp_path) -> None:
    """A group carries the parsed error.json records for its config alongside
    its manifests (same config key)."""
    base = tmp_path / "corpusqa" / "codeact"
    cfg = _config(benchmark="corpusqa", baseline="codeact")
    _write_manifest(base, "h1", {"task_id": "ok", "config": cfg})
    _write_error(base, "e1", {"task_id": "cwe", "config": cfg,
                              "error_type": "ContextWindowExceededError"})
    [group] = find_groups(logs_dir=tmp_path)
    assert len(group["manifests"]) == 1
    assert len(group["errors"]) == 1
    assert group["errors"][0]["task_id"] == "cwe"
    assert group["errors"][0]["error_type"] == "ContextWindowExceededError"
    assert group["errors"][0]["_inference_dir"].name == "e1"


def test_find_groups_skips_unparseable_error_json(tmp_path, capsys) -> None:
    base = tmp_path / "corpusqa" / "codeact"
    (base / "inferences" / "broken").mkdir(parents=True)
    (base / "inferences" / "broken" / "error.json").write_text("{nope")  # half-written
    _write_error(base, "e1", {"task_id": "cwe", "config": _config(),
                              "error_type": "ContextWindowExceededError"})
    [group] = find_groups(logs_dir=tmp_path)
    assert len(group["errors"]) == 1  # the broken one is skipped
    assert "skipping unparseable" in capsys.readouterr().out


def test_find_groups_config_seen_only_in_errors_forms_group(tmp_path) -> None:
    """A config that errored on every task still forms a group (no manifests)."""
    base = tmp_path / "corpusqa" / "codeact"
    _write_error(base, "e1", {"task_id": "cwe", "config": _config(),
                              "error_type": "ContextWindowExceededError"})
    [group] = find_groups(logs_dir=tmp_path)
    assert group["manifests"] == []
    assert len(group["errors"]) == 1


def test_FAILURE_ERROR_TYPES_is_context_window_only() -> None:
    """The policy: only a context-window error is a failed PREDICTION; infra noise
    (crash, timeout) is not."""
    assert "ContextWindowExceededError" in FAILURE_ERROR_TYPES
    assert "ChildCrash" not in FAILURE_ERROR_TYPES
    assert "Timeout" not in FAILURE_ERROR_TYPES


def test_FAILURE_ERROR_TYPES_is_shared_with_the_cleaners() -> None:
    """One source of truth: the analysis layer's failed-prediction whitelist IS the
    set of errors the log cleaners keep — the two judgments can't diverge."""
    from evals.baselines import _common
    from scripts import _clean_common

    assert FAILURE_ERROR_TYPES is _common.FAILURE_ERROR_TYPES
    assert FAILURE_ERROR_TYPES is _clean_common.FAILURE_ERROR_TYPES


def test_failure_manifests_scores_context_window_as_empty_excludes_crash(tmp_path) -> None:
    """A context-window error becomes a pseudo-manifest scored as an EMPTY answer
    (worst value); a child-crash is excluded (infra noise, not a prediction)."""
    cfg = _config()
    group = {
        "base_dir": tmp_path, "config": cfg, "manifests": [],
        "errors": [
            {"task_id": "cwe", "config": cfg, "error_type": "ContextWindowExceededError",
             "_inference_dir": tmp_path / "e1"},
            {"task_id": "crash", "config": cfg, "error_type": "ChildCrash",
             "_inference_dir": tmp_path / "e2"},
        ],
    }
    seen: list = []

    def fake_details(task_ids, answers):
        seen.append((list(task_ids), list(answers)))
        return [ScoreResult(score=0) for _ in task_ids]  # label benchmark: empty → 0

    fakes = failure_manifests(group, _fake_scoring_benchmark(fake_details))
    # Only the context-window error was scored, and as an EMPTY answer.
    assert seen == [(["cwe"], [""])]
    assert len(fakes) == 1
    assert fakes[0]["task_id"] == "cwe"
    assert fakes[0]["_score"] == 0
    assert fakes[0]["_from_error"] == "ContextWindowExceededError"


def test_failure_manifests_uses_benchmark_worst_not_hardcoded_zero(tmp_path) -> None:
    """failure_manifests must use the BENCHMARK's own empty-answer score, not a
    hardcoded 0 — a 1–100 rating benchmark floors an empty answer at 1."""
    cfg = _config(benchmark="loong")
    group = {"base_dir": tmp_path, "config": cfg, "manifests": [],
             "errors": [{"task_id": "x", "config": cfg,
                         "error_type": "ContextWindowExceededError",
                         "_inference_dir": tmp_path / "e"}]}
    # rating benchmark: empty answer → floor (1), NOT 0.
    bench = _fake_scoring_benchmark(lambda t, a: [ScoreResult(score=1) for _ in t],
                                    scorer="loong-judge")
    [fake] = failure_manifests(group, bench)
    assert fake["_score"] == 1


def test_failure_manifests_empty_without_benchmark_or_errors(tmp_path) -> None:
    group = {"base_dir": tmp_path, "config": _config(), "manifests": [], "errors": []}
    assert failure_manifests(group, None) == []  # no benchmark module
    assert failure_manifests(group, _fake_scoring_benchmark(lambda t, a: [])) == []  # no errors


def test_main_shows_total_view_counting_errors(
    tmp_path, monkeypatch, capsys
) -> None:
    """One run shows a SINGLE total accuracy in which context-window errors count as
    wrong; the answered-only view is not shown. Child-crash is never counted."""
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    base = tmp_path / "corpusqa" / "codeact"
    cfg = _config(benchmark="corpusqa", baseline="codeact")
    _write_manifest(base, "h1", {"task_id": "ok", "raw_answer": "D", "config": cfg})
    _write_error(base, "e1", {"task_id": "cwe", "config": cfg,
                              "error_type": "ContextWindowExceededError"})
    _write_error(base, "e2", {"task_id": "crash", "config": cfg,
                              "error_type": "ChildCrash"})

    def fake_details(task_ids, answers):
        # the real answer scores 1; an empty (folded failure) scores 0
        return [ScoreResult(score=0 if a == "" else 1) for a in answers]

    fake = SimpleNamespace(score_details=fake_details, SCORER="corpusqa-parse",
                           GRADER_MODEL="openai/gpt-5.4-nano")
    _patch_benchmark_import(monkeypatch, fake)

    main(["--benchmark", "corpusqa", "--baseline", "codeact"])  # NO flag
    out = capsys.readouterr().out
    assert "0.500" in out      # the total: 1 correct / (1 answered + 1 ctx-error) = 1/2; crash excluded
    assert "1.000" not in out  # the answered-only view is no longer shown
    assert "ContextWindowExceededError" in out


# ============================================================
# Model filter + per-benchmark display config (hidden axes, value labels)
# ============================================================


def test_find_groups_filters_by_model(tmp_path) -> None:
    """`--model` accepts a human-readable id (e.g. 'Qwen3.5-35B-A3B') and matches the
    CANONICAL slug stored in each config — so the user needn't know the slug form."""
    base = tmp_path / "corpusqa" / "codeact"
    _write_manifest(base, "h1", {"task_id": "a", "config": _config(
        benchmark="corpusqa", baseline="codeact", model="qwen3-5-35b-a3b")})
    _write_manifest(base, "h2", {"task_id": "b", "config": _config(
        benchmark="corpusqa", baseline="codeact", model="gpt-5-4-nano")})

    groups = find_groups(model="Qwen3.5-35B-A3B", logs_dir=tmp_path)
    assert len(groups) == 1
    assert groups[0]["config"]["model"] == "qwen3-5-35b-a3b"
    # An already-canonical slug works too.
    assert len(find_groups(model="qwen3-5-35b-a3b", logs_dir=tmp_path)) == 1
    # No --model → both groups.
    assert len(find_groups(logs_dir=tmp_path)) == 2


def test_aggregate_group_hides_axes_declared_by_benchmark(tmp_path) -> None:
    """A benchmark may declare ANALYSIS_HIDE_AXES; those axes are dropped from the
    per-axis breakdown (loong hides task_name + length; corpusqa hides sub_domain)."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"),
             "manifests": [{"task_id": "1", "_score": 100}, {"task_id": "2", "_score": 60}],
             "errors": []}
    meta = {"1": {"set": 1, "task": 3, "task_name": "Clustering"},
            "2": {"set": 1, "task": 3, "task_name": "Clustering"}}
    bench = SimpleNamespace(PERFECT_SCORE=100,
                            ANALYSIS_HIDE_AXES=frozenset({"task_name"}),
                            get_task_metadata=lambda t: meta[t])
    by_axis = aggregate_group(group, bench)["by_axis"]
    assert "task_name" not in by_axis          # hidden
    assert "set" in by_axis and "task" in by_axis  # kept


def test_render_report_labels_axis_values_from_benchmark(tmp_path) -> None:
    """ANALYSIS_VALUE_LABELS relabels the value column (loong set 1 → '1 · 10–50K tok',
    task level 3 → '3 · Clustering') so a reader needn't know the integer encoding."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"),
             "manifests": [{"task_id": "1", "_score": 100}, {"task_id": "2", "_score": 60}],
             "errors": []}
    meta = {"1": {"set": 1, "task": 3}, "2": {"set": 1, "task": 3}}
    bench = SimpleNamespace(PERFECT_SCORE=100, get_task_metadata=lambda t: meta[t],
                            ANALYSIS_VALUE_LABELS={"set": {"1": "1 · 10–50K tok"},
                                                   "task": {"3": "3 · Clustering"}})
    text = _render_to_text(aggregate_group(group, bench))
    assert "50K tok" in text         # set value labeled with its token range
    assert "Clustering" in text      # task value labeled with its type name


def test_render_report_hidden_axis_absent_with_no_note(tmp_path) -> None:
    """A hidden axis produces NO table and NO 'skipped' note (unlike the
    high-cardinality guard, which prints a note)."""
    group = {"base_dir": tmp_path, "config": _config(benchmark="loong"),
             "manifests": [{"task_id": str(i), "_score": 50} for i in range(5)],
             "errors": []}
    meta = {str(i): {"set": 1, "length": 1000 + i} for i in range(5)}
    bench = SimpleNamespace(PERFECT_SCORE=100,
                            ANALYSIS_HIDE_AXES=frozenset({"length"}),
                            get_task_metadata=lambda t: meta[t])
    text = _render_to_text(aggregate_group(group, bench))
    assert "by set" in text
    assert "by length" not in text   # gone entirely
    assert "distinct" not in text    # and no high-cardinality skip note


def test_main_model_filter_limits_to_one_model(tmp_path, monkeypatch, capsys) -> None:
    """`--model Qwen3.5-35B-A3B` reports only the matching config group."""
    from evals.settings import settings
    monkeypatch.setattr(settings, "LOGS_DIR", tmp_path)
    base = tmp_path / "corpusqa" / "codeact"
    _write_manifest(base, "h1", {"task_id": "a", "raw_answer": "x", "config": _config(
        benchmark="corpusqa", baseline="codeact", model="qwen3-5-35b-a3b")})
    _write_manifest(base, "h2", {"task_id": "b", "raw_answer": "y", "config": _config(
        benchmark="corpusqa", baseline="codeact", model="gpt-5-4-nano")})
    fake = SimpleNamespace(score_details=lambda t, a: [], SCORER="corpusqa-parse",
                           GRADER_MODEL="openai/gpt-5.4-nano")
    _patch_benchmark_import(monkeypatch, fake)

    main(["--benchmark", "corpusqa", "--baseline", "codeact",
          "--model", "Qwen3.5-35B-A3B", "--no-score"])
    out = capsys.readouterr().out
    assert "qwen3-5-35b-a3b" in out      # the matching group's header
    assert "gpt-5-4-nano" not in out     # the other model is filtered out
