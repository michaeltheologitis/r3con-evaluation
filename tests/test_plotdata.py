"""Tests for `evals.analysis.plotdata` — the DataFrame-friendly seam.

`plotdata` turns the rich summary dicts that `evals.analysis.aggregate.aggregate_group`
returns into flat, plot-ready rows (one dict per group, or per metadata-axis value),
so the results-visualization notebook stays a *thin* layer over `evals.analysis`:
it calls `find_groups` -> `aggregate_group` -> these helpers -> pandas/matplotlib,
with no flattening logic buried in notebook cells.

These functions are PURE over the summary shape (no LLM, no disk), so the tests build
synthetic summaries directly. One integration test feeds a REAL `aggregate_group`
output through the helpers to pin the contract against the actual producer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from evals.analysis.aggregate import aggregate_group
from evals.analysis.plotdata import (
    axis_rows,
    headline_rows,
    plottable_axes,
)


# ============================================================
# Synthetic summaries (the aggregate_group output shape)
# ============================================================


def _accuracy_summary(**cfg_extra) -> dict:
    """A label-benchmark (accuracy) summary, as aggregate_group returns one."""
    config = {"benchmark": "corpusqa", "baseline": "codeact",
              "model": "qwen3-5-35b-a3b", "seed": 42, **cfg_extra}
    return {
        "config": config,
        "base_dir": None,
        "n": 10, "n_scored": 8, "n_correct": 6, "accuracy": 0.75,
        "metric": {"kind": "accuracy", "n": 8, "n_correct": 6, "accuracy": 0.75},
        "metric_all": {"kind": "accuracy", "n": 10, "n_correct": 6, "accuracy": 0.6},
        "n_indices": 0, "n_errors": 2, "errors_by_type": {"ContextWindowExceededError": 2},
        "n_failures": 2,
        "by_axis": {
            "difficulty": {
                "easy": {"kind": "accuracy", "n": 4, "n_correct": 4, "accuracy": 1.0,
                         "all": {"kind": "accuracy", "n": 4, "n_correct": 4, "accuracy": 1.0}},
                "hard": {"kind": "accuracy", "n": 4, "n_correct": 2, "accuracy": 0.5,
                         "all": {"kind": "accuracy", "n": 6, "n_correct": 2, "accuracy": 1 / 3}},
            }
        },
        "value_labels": {},
        "usage": {"total": {"qwen3-5-35b-a3b": {"total_tokens": 1200, "num_calls": 10}}},
        "cost_usd": {"total": {"qwen3-5-35b-a3b": 0.0123}},
    }


def _rating_summary(**cfg_extra) -> dict:
    """A judge-rating benchmark (Loong, PERFECT_SCORE=100) summary."""
    config = {"benchmark": "loong", "baseline": "arag", "model": "gpt-5-4-nano",
              "seed": 7, **cfg_extra}
    return {
        "config": config,
        "base_dir": None,
        "n": 4, "n_scored": 4, "n_correct": None, "accuracy": None,
        "metric": {"kind": "rating", "perfect_score": 100, "n": 4,
                   "avg_score": 80.0, "n_perfect": 2, "perfect_rate": 0.5},
        "metric_all": {"kind": "rating", "perfect_score": 100, "n": 5,
                       "avg_score": 64.0, "n_perfect": 2, "perfect_rate": 0.4},
        "n_indices": 1, "n_errors": 1, "errors_by_type": {"ContextWindowExceededError": 1},
        "n_failures": 1,
        "by_axis": {
            "set": {
                "1": {"kind": "rating", "perfect_score": 100, "n": 2, "avg_score": 90.0,
                      "n_perfect": 1, "perfect_rate": 0.5,
                      "all": {"kind": "rating", "perfect_score": 100, "n": 2,
                              "avg_score": 90.0, "n_perfect": 1, "perfect_rate": 0.5}},
                "2": {"kind": "rating", "perfect_score": 100, "n": 2, "avg_score": 70.0,
                      "n_perfect": 1, "perfect_rate": 0.5,
                      "all": {"kind": "rating", "perfect_score": 100, "n": 3,
                              "avg_score": 46.67, "n_perfect": 1, "perfect_rate": 1 / 3}},
            }
        },
        "value_labels": {"set": {"1": "1 · 10–50K tok", "2": "2 · 50–100K tok"}},
        "usage": {"total": {"gpt-5-4-nano": {"total_tokens": 5000, "num_calls": 20}}},
        "cost_usd": {"total": {"gpt-5-4-nano": 0.05}},
    }


# ============================================================
# headline_rows
# ============================================================


def test_headline_rows_flattens_accuracy_metric() -> None:
    [row] = headline_rows([_accuracy_summary()])
    assert row["benchmark"] == "corpusqa"
    assert row["baseline"] == "codeact"
    assert row["kind"] == "accuracy"
    assert row["accuracy"] == 0.75
    assert row["accuracy_all"] == 0.6
    # The unified cross-benchmark fraction is the accuracy itself.
    assert row["metric_frac"] == 0.75
    assert row["metric_frac_all"] == 0.6
    # Rating-only columns are present but None for an accuracy benchmark.
    assert row["avg_score"] is None and row["perfect_rate"] is None
    # Counts: answered (8), all incl. ctx-errors (10), logged (10).
    assert row["n_scored"] == 8 and row["n_all"] == 10 and row["n_logged"] == 10
    assert row["n_failures"] == 2


def test_headline_rows_flattens_rating_metric() -> None:
    [row] = headline_rows([_rating_summary()])
    assert row["kind"] == "rating"
    assert row["avg_score"] == 80.0 and row["avg_score_all"] == 64.0
    assert row["perfect_rate"] == 0.5 and row["perfect_rate_all"] == 0.4
    # The unified fraction normalizes the 1–100 rating by PERFECT_SCORE.
    assert row["metric_frac"] == pytest.approx(0.80)
    assert row["metric_frac_all"] == pytest.approx(0.64)
    # Accuracy columns present-but-None for a rating benchmark.
    assert row["accuracy"] is None and row["accuracy_all"] is None


def test_headline_rows_label_carries_distinguishing_config() -> None:
    [row] = headline_rows([_accuracy_summary(
        search_method="global", config_name="Qwen3.5-MoE-Instruct")])
    # The label is a compact descriptor of the config knobs that separate runs —
    # benchmark/baseline/model live in their own columns, so they're excluded.
    assert "global" in row["label"]
    assert "Qwen3.5-MoE-Instruct" in row["label"]
    assert "seed=42" in row["label"]
    assert "corpusqa" not in row["label"] and "codeact" not in row["label"]


def test_headline_rows_one_row_per_summary_order_preserved() -> None:
    rows = headline_rows([_accuracy_summary(), _rating_summary()])
    assert [r["benchmark"] for r in rows] == ["corpusqa", "loong"]


def test_headline_rows_cost_and_tokens_summed_across_models() -> None:
    summary = _accuracy_summary()
    summary["usage"]["total"]["second-model"] = {"total_tokens": 300, "num_calls": 5}
    summary["cost_usd"]["total"]["second-model"] = 0.002
    [row] = headline_rows([summary])
    assert row["total_tokens"] == 1500 and row["num_calls"] == 15
    assert row["cost_total_usd"] == pytest.approx(0.0143)


def test_headline_rows_cost_none_when_all_models_unpriced() -> None:
    """A vLLM model litellm can't price → cost_usd None; the row reports None, not 0."""
    summary = _accuracy_summary()
    summary["cost_usd"]["total"] = {"qwen3-5-35b-a3b": None}
    [row] = headline_rows([summary])
    assert row["cost_total_usd"] is None


# ============================================================
# axis_rows
# ============================================================


def test_axis_rows_accuracy_axis_sorted_with_both_views() -> None:
    rows = axis_rows(_accuracy_summary(), "difficulty")
    assert [r["value"] for r in rows] == ["easy", "hard"]  # sorted by raw value
    hard = next(r for r in rows if r["value"] == "hard")
    assert hard["accuracy"] == 0.5
    assert hard["accuracy_all"] == pytest.approx(1 / 3)
    assert hard["metric_frac"] == 0.5
    assert hard["n"] == 4 and hard["n_all"] == 6


def test_axis_rows_rating_axis_uses_relabeled_value() -> None:
    rows = axis_rows(_rating_summary(), "set")
    first = rows[0]
    assert first["value"] == "1"            # raw value preserved (for sorting)
    assert first["label"] == "1 · 10–50K tok"  # ANALYSIS_VALUE_LABELS applied
    assert first["avg_score"] == 90.0
    assert first["metric_frac"] == pytest.approx(0.90)


def test_axis_rows_missing_axis_returns_empty() -> None:
    assert axis_rows(_accuracy_summary(), "nonexistent") == []


# ============================================================
# plottable_axes
# ============================================================


def test_plottable_axes_lists_low_cardinality_axes() -> None:
    assert plottable_axes(_accuracy_summary()) == ["difficulty"]


def test_plottable_axes_skips_high_cardinality_axis() -> None:
    summary = _accuracy_summary()
    # A continuous axis (e.g. loong's raw token `length`) has one bucket per task —
    # too granular to plot, mirroring the CLI's MAX_BREAKDOWN_VALUES guard.
    summary["by_axis"]["length"] = {
        str(i): {"kind": "accuracy", "n": 1, "n_correct": 1, "accuracy": 1.0,
                 "all": {"kind": "accuracy", "n": 1, "n_correct": 1, "accuracy": 1.0}}
        for i in range(40)
    }
    assert plottable_axes(summary) == ["difficulty"]  # length omitted


# ============================================================
# Integration — feed a REAL aggregate_group output through the helpers
# ============================================================


def test_helpers_consume_real_aggregate_group_output() -> None:
    """Pins the plotdata contract against the actual producer: build a group, run the
    real `aggregate_group`, and flatten its summary. Catches shape drift between the
    two modules."""
    group = {
        "base_dir": None,
        "config": {"benchmark": "loong", "baseline": "codeact",
                   "model": "gpt-5-4-nano", "seed": 1},
        "manifests": [
            {"task_id": "1", "_score": 100}, {"task_id": "2", "_score": 60},
        ],
        "errors": [],
    }
    meta = {"1": {"set": "1"}, "2": {"set": "2"}}
    bench = SimpleNamespace(PERFECT_SCORE=100, get_task_metadata=lambda t: meta[t])
    summary = aggregate_group(group, bench)

    [row] = headline_rows([summary])
    assert row["kind"] == "rating"
    assert row["avg_score"] == pytest.approx(80.0)
    assert row["metric_frac"] == pytest.approx(0.80)

    rows = axis_rows(summary, "set")
    assert {r["value"] for r in rows} == {"1", "2"}
    assert "set" in plottable_axes(summary)
