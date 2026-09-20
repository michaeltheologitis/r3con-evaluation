"""Flatten ``aggregate_group`` summaries into plot-ready rows.

The analysis CLI renders summaries with ``rich`` (terminal tables). The
results-visualization notebook wants the same numbers as flat rows it can drop
into a pandas DataFrame and plot. This module is that seam — pure functions over
the summary dict shape that :func:`evals.analysis.aggregate.aggregate_group`
returns (no LLM, no disk), so the notebook stays a *thin* layer:

    groups   = find_groups(benchmark=..., baseline=...)
    summaries = [aggregate_group(attach_scores(g, mod) or g, mod) for g, mod in ...]
    df = pandas.DataFrame(headline_rows(summaries))         # one row per run
    df = pandas.DataFrame(axis_rows(summary, "difficulty")) # one row per axis value

Both metric kinds (accuracy for the 0/1 label benchmarks; the 1–100 judge rating
for Loong) are flattened into the SAME columns, with the kind-irrelevant ones left
``None``. ``metric_frac`` is a unified 0..1 headline (accuracy as-is; a rating
normalized by ``PERFECT_SCORE``) so runs across different benchmarks can share one
bar chart. The ``⁺`` "all" view (context-window errors counted as worst-score
predictions) travels alongside the answered-only view as ``*_all`` columns.
"""
from __future__ import annotations

from typing import Any, Iterable

from evals.analysis.aggregate import MAX_BREAKDOWN_VALUES

# Config keys that get their own dedicated columns (or are too verbose to label
# with), so they're excluded from the compact run `label`.
_LABEL_SKIP = frozenset({"benchmark", "baseline", "model", "completion_params"})


def _frac(value: float | None, perfect: float | None) -> float | None:
    """Normalize a metric to 0..1: accuracy is already a fraction (``perfect`` None);
    a 1–100 rating is divided by ``PERFECT_SCORE``."""
    if value is None:
        return None
    if perfect is None:  # accuracy benchmark — already a fraction, no normalization
        return value
    return value / perfect


def _config_label(config: dict[str, Any]) -> str:
    """A compact descriptor of the config knobs that distinguish runs (search_method,
    config_name, seed, …) — benchmark/baseline/model live in their own columns."""
    return ", ".join(
        f"{k}={v}" for k, v in sorted(config.items()) if k not in _LABEL_SKIP
    )


def _sum_costs(costs: dict[str, float | None]) -> float | None:
    """Total USD across models. ``None`` (not ``0``) when every model is unpriced
    (e.g. a vLLM model litellm can't price); ``0.0`` when there's no cost at all."""
    if not costs:
        return 0.0
    known = [c for c in costs.values() if c is not None]
    return sum(known) if known else None


def _apply_metric(row: dict[str, Any], metric: dict[str, Any],
                  metric_all: dict[str, Any]) -> None:
    """Fill ``row``'s metric columns (answered + ``_all``) for either kind."""
    if metric["kind"] == "rating":
        perfect = metric["perfect_score"]
        row["avg_score"] = metric["avg_score"]
        row["avg_score_all"] = metric_all["avg_score"]
        row["perfect_rate"] = metric["perfect_rate"]
        row["perfect_rate_all"] = metric_all["perfect_rate"]
        row["metric_frac"] = _frac(metric["avg_score"], perfect)
        row["metric_frac_all"] = _frac(metric_all["avg_score"], perfect)
    else:
        row["accuracy"] = metric["accuracy"]
        row["accuracy_all"] = metric_all["accuracy"]
        row["metric_frac"] = metric["accuracy"]
        row["metric_frac_all"] = metric_all["accuracy"]


def _blank_metric_cols() -> dict[str, Any]:
    """Every metric column, defaulted to ``None`` — the kind-irrelevant ones stay
    ``None`` so accuracy and rating rows share one schema (one DataFrame)."""
    return {
        "accuracy": None, "accuracy_all": None,
        "avg_score": None, "avg_score_all": None,
        "perfect_rate": None, "perfect_rate_all": None,
        "metric_frac": None, "metric_frac_all": None,
    }


def headline_rows(summaries: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """One flat row per group summary — the master table for a cross-run comparison.

    Columns: ``benchmark`` / ``baseline`` / ``model`` / ``label`` (config identity);
    ``kind``; the count trio ``n_logged`` / ``n_scored`` / ``n_all`` plus
    ``n_errors`` / ``n_failures``; the metric columns (see :func:`_apply_metric`);
    and cost (``cost_total_usd``, ``total_tokens``, ``num_calls``). Order is
    preserved.
    """
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        config = summary["config"]
        metric = summary["metric"]
        metric_all = summary.get("metric_all") or metric
        total_usage = (summary.get("usage") or {}).get("total") or {}
        row: dict[str, Any] = {
            "benchmark": config.get("benchmark"),
            "baseline": config.get("baseline"),
            "model": config.get("model"),
            "label": _config_label(config),
            "kind": metric["kind"],
            "n_logged": summary["n"],
            "n_scored": metric["n"],
            "n_all": metric_all["n"],
            "n_errors": summary.get("n_errors", 0),
            "n_failures": summary.get("n_failures", 0),
            **_blank_metric_cols(),
            "cost_total_usd": _sum_costs((summary.get("cost_usd") or {}).get("total") or {}),
            "total_tokens": sum(b.get("total_tokens", 0) for b in total_usage.values()),
            "num_calls": sum(b.get("num_calls", 0) for b in total_usage.values()),
        }
        _apply_metric(row, metric, metric_all)
        rows.append(row)
    return rows


def axis_rows(summary: dict[str, Any], axis: str) -> list[dict[str, Any]]:
    """Flatten one metadata-axis breakdown into rows, sorted by raw value.

    Each row carries the answered metric AND the ``⁺`` "all" view (``*_all``), plus
    ``value`` (the raw value, for sorting) and ``label`` (the benchmark's
    ``ANALYSIS_VALUE_LABELS`` mapping, e.g. loong ``set`` ``"1"`` → ``"1 · 10–50K
    tok"``). An axis the summary doesn't carry → ``[]`` (notebook-friendly).
    """
    breakdown = (summary.get("by_axis") or {}).get(axis)
    if not breakdown:
        return []
    labels = (summary.get("value_labels") or {}).get(axis, {})
    rows: list[dict[str, Any]] = []
    for value, cell in sorted(breakdown.items()):
        all_cell = cell.get("all", cell)
        row: dict[str, Any] = {
            "axis": axis,
            "value": value,
            "label": labels.get(value, value),
            "kind": cell["kind"],
            "n": cell["n"],
            "n_all": all_cell["n"],
            **_blank_metric_cols(),
        }
        _apply_metric(row, cell, all_cell)
        rows.append(row)
    return rows


def plottable_axes(summary: dict[str, Any],
                   max_values: int = MAX_BREAKDOWN_VALUES) -> list[str]:
    """The metadata axes worth plotting: those with ``<= max_values`` distinct values.

    Mirrors the CLI's high-cardinality guard — a continuous axis (e.g. loong's raw
    token ``length``, ~unique per task) has one bucket per task and is noise as a bar
    chart, so it's omitted. Sorted for stable iteration.
    """
    by_axis = summary.get("by_axis") or {}
    return sorted(a for a, bd in by_axis.items() if len(bd) <= max_values)
