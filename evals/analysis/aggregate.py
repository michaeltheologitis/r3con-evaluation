"""Aggregate inference manifests into headline accuracy + cost, grouped by config.

There is no "run" on disk — inferences live under
``logs/{benchmark}/{baseline}/inferences/{inference_hash}/manifest.json``, each
self-describing (its ``config``). Grouping is a query-time job: this module
globs the manifests, groups them by ``config`` (the inference identity), and
reports per-group accuracy + cost.

Cost has two parts, kept separate:
- **inference** — sum of each manifest's ``usage`` (the query cost).
- **construction** — the build cost of the DISTINCT indices the group's
  manifests reference (via ``index_ref``), read from the index store's
  ``index_usage.json`` and **counted once per index** (the four search modes
  share indices, so construction is not multiplied).

The **metric** reads each inference's cached ``score.json`` (written by
``evals.analysis.score`` — parse-then-match for the label benchmarks, an LLM judge
for the free-form ones), NOT a re-run of ``benchmark.score``. So re-aggregating
never re-pays the parse / judge LLM. The CLI computes any missing/stale
``score.json`` first (``--no-score`` skips that compute; ``--rescore`` forces it),
then reads the scores back. Designed cheap so it can run after every measurement.
"""
from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import litellm
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from evals.analysis.score import ensure_scores, read_score

# FAILURE_ERROR_TYPES (which error_types are genuine failed PREDICTIONS vs infra
# noise) lives in _common beside build_error_record — shared with the log
# cleaners, which KEEP exactly that set. Here it drives the "all" (⁺) view:
# those errors are folded in as worst-score predictions beside the answered-only
# view. Re-exported under its old name for callers/tests.
from evals.baselines._common import (
    BENCHMARK_IMPORT_MAP,
    FAILURE_ERROR_TYPES,
    canonical_model_id,
    hit_step_cap,
)
from evals.llm.usage import _merge_numeric  # recursive numeric-dict sum (single source)
from evals.settings import MANUAL_MODEL_PRICING, _slug, settings

# A metadata axis with more distinct values than this is treated as continuous /
# high-cardinality (e.g. loong's raw token `length`, ~unique per task) and is NOT
# tabulated — one row per task is noise, not a grouping. The real per-benchmark
# grouping axes are all small (longbenchv2 sub_domain = 16, loong set/task = 4).
MAX_BREAKDOWN_VALUES = 25


# ============================================================
# Discovery + reading
# ============================================================


def _read_records(inferences: Path, filename: str, kind: str) -> dict[str, list[dict]]:
    """Parse every ``{hash}/{filename}`` under ``inferences/``, grouped by config key.

    Each record is tagged with ``_inference_dir`` (its folder). An unparseable file
    — a half-written one a run was killed mid-write — is skipped with a note.
    """
    by_config: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(inferences.glob(f"*/{filename}")):
        try:
            record = json.loads(path.read_text())
        except json.JSONDecodeError:
            print(f"skipping unparseable {kind}: {path}")
            continue
        record["_inference_dir"] = path.parent
        config_key = json.dumps(record.get("config", {}), sort_keys=True)
        by_config[config_key].append(record)
    return by_config


def find_groups(
    *,
    benchmark: str | None = None,
    baseline: str | None = None,
    model: str | None = None,
    logs_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Return inference GROUPS matching the filters.

    A group = all inference records under one ``{benchmark}/{baseline}/`` base dir
    that share the same ``config`` (the analysis unit — what a "run" used to be).
    Each group is ``{base_dir, config, manifests, errors}``: ``manifests`` are the
    succeeded inferences (``manifest.json``), ``errors`` the recorded failures
    (``error.json``). A config seen ONLY in errors still forms a group (a model that
    context-window-failed every task still reports).

    ``benchmark`` / ``baseline`` filter via the directory glob; ``model`` filters at
    the config level — it's **canonicalized** (``canonical_model_id``) before
    matching, so a human-readable id like ``"Qwen3.5-35B-A3B"`` (or the full
    ``hosted_vllm/Qwen/Qwen3.5-35B-A3B``) matches the ``qwen3-5-35b-a3b`` slug stored
    in each config.
    """
    base = logs_dir or settings.LOGS_DIR
    if not base.exists():
        return []
    benchmark_glob = _slug(benchmark) if benchmark else "*"
    baseline_glob = _slug(baseline) if baseline else "*"
    model_slug = canonical_model_id(model) if model else None

    groups: list[dict[str, Any]] = []
    for base_dir in sorted(base.glob(f"{benchmark_glob}/{baseline_glob}")):
        # Most baselines nest inferences under `inferences/{hash}/` (one folder per content-hash,
        # with a shared `_indices/` store). readagent/rlm use a flat, no-reuse layout — one folder per
        # RUN, `{run_tag}/`, directly under the baseline dir, with the index inside it. Scan
        # whichever exists; `_read_records` globs `*/manifest.json` either way.
        inferences = base_dir / "inferences"
        scan_dir = inferences if inferences.is_dir() else base_dir
        if not scan_dir.is_dir():
            continue
        manifests_by_config = _read_records(scan_dir, "manifest.json", "manifest")
        errors_by_config = _read_records(scan_dir, "error.json", "error")
        for config_key in sorted(set(manifests_by_config) | set(errors_by_config)):
            config = json.loads(config_key)
            if model_slug is not None and config.get("model") != model_slug:
                continue
            groups.append({
                "base_dir": base_dir,
                "config": config,
                "manifests": manifests_by_config.get(config_key, []),
                "errors": errors_by_config.get(config_key, []),
            })
    return groups


# ============================================================
# Scoring
# ============================================================


def attach_scores(
    group: dict[str, Any],
    benchmark_module: Any,
    *,
    compute: bool = True,
    force: bool = False,
) -> dict[str, int]:
    """Attach each inference's cached score onto its manifest as ``_score``.

    When ``compute`` (the default), first ensures every inference has a fresh
    ``score.json`` via :func:`evals.analysis.score.ensure_scores` (computing the
    missing/stale ones in one batched grader call; ``force`` recomputes all).
    Then reads each inference's ``score.json`` and, if present, sets the manifest's
    ``_score`` — the metric is read from disk, NEVER recomputed here, so
    re-aggregating doesn't re-pay the grader. Returns ``ensure_scores``' counts
    (all-zero when ``compute`` is False or the benchmark module is unknown).
    """
    inferences = [(m["_inference_dir"], m) for m in group["manifests"]]
    stats = {"scored": 0, "cached": 0, "total": len(inferences)}
    if compute and benchmark_module is not None:
        stats = ensure_scores(inferences, benchmark_module, force=force)
    for inference_dir, manifest in inferences:
        score_json = read_score(inference_dir)
        if score_json is not None and "score" in score_json:
            manifest["_score"] = score_json["score"]
    return stats


def failure_manifests(
    group: dict[str, Any],
    benchmark_module: Any | None,
    error_types: frozenset[str] = FAILURE_ERROR_TYPES,
) -> list[dict[str, Any]]:
    """Pseudo-manifests for the group's errors that are genuine failed PREDICTIONS.

    A context-window error means the model produced no answer — semantically an
    empty ``raw_answer``, which every benchmark already scores at its worst value
    with NO grader call (its empty-answer short-circuit). So we score these failures
    through the benchmark's own ``score_details(tids, [""])`` — free, and
    per-benchmark by construction (``0`` for the label / 0-1 benchmarks, the 1–100
    floor for Loong). Returns manifest-shaped dicts with ``_score`` set, ready to be
    merged into ``group["manifests"]`` for aggregation; they carry no ``usage`` (no
    cost) and a ``_from_error`` tag. Empty when there's no benchmark module or no
    matching error.
    """
    if benchmark_module is None:
        return []
    failures = [e for e in group.get("errors", []) if e.get("error_type") in error_types]
    if not failures:
        return []
    task_ids = [e["task_id"] for e in failures]
    results = benchmark_module.score_details(task_ids, [""] * len(task_ids))
    return [
        {
            "task_id": e["task_id"],
            "config": e.get("config", group["config"]),
            "_score": r["score"],
            "_from_error": e.get("error_type"),
            "_inference_dir": e.get("_inference_dir"),
        }
        for e, r in zip(failures, results)
    ]


# Whether a manifest's task exhausted its step/iteration budget (codeact
# `max_steps_error` / rlm `n_iterations >= max_iterations`). The SINGLE source of truth
# lives in `_common` (shared with the log cleaners' `--max-iter`, so the ⁺ view and the
# cleaner can't disagree on what "capped" means). Aliased under the old private name so
# the existing call site + test keep working.
_hit_step_cap = hit_step_cap


def _compute_metric(scores: list, perfect_score: int | None) -> dict[str, Any]:
    """Reduce a list of per-inference scores to a headline metric.

    Two metric kinds, picked by whether the benchmark declares a ``PERFECT_SCORE``:

    - **accuracy** (the 0/1 label benchmarks): ``accuracy`` = mean (= fraction
      correct), plus ``n_correct``.
    - **rating** (a 1–100 judge benchmark like Loong, ``perfect_score`` set):
      ``avg_score`` = mean rating, ``perfect_rate`` = fraction == ``perfect_score``
      (the two upstream Loong metrics), plus ``n_perfect``.

    ``n`` is the number of scored items; ``None`` for the means when ``n == 0``.
    Used for both the headline number and each per-metadata-axis cell.
    """
    n = len(scores)
    if perfect_score is not None:
        n_perfect = sum(1 for s in scores if s == perfect_score)
        return {
            "kind": "rating",
            "perfect_score": perfect_score,
            "n": n,
            "avg_score": (sum(scores) / n) if n else None,
            "n_perfect": n_perfect,
            "perfect_rate": (n_perfect / n) if n else None,
        }
    n_correct = sum(scores)
    return {
        "kind": "accuracy",
        "n": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else None,
    }


# ============================================================
# Cost
# ============================================================


def sum_usage(*usages: dict[str, dict]) -> dict[str, dict]:
    """Sum N per-model usage ROLLUPS (the ``total`` shape: ``{model_id:
    {prompt_tokens, completion_tokens, total_tokens, num_calls, <nested
    details>}}``). Recurses, so nested detail counters survive the sum.
    """
    out: dict[str, dict] = {}
    for usage in usages:
        if not usage:
            continue
        for model, bucket in usage.items():
            _merge_numeric(out.setdefault(model, {}), bucket)
    return out


def _index_total(index_usage: dict) -> dict:
    """The per-model rollup from an ``index_usage.json`` ({total, calls})."""
    return (index_usage or {}).get("total") or {}


def construction_usage(base_dir: Path, manifests: list[dict]) -> tuple[dict, int]:
    """Build cost of the DISTINCT indices these manifests reference.

    The four search modes share indices, so an index's build cost is counted
    ONCE — keyed by ``index_ref`` (the ``index_hash`` string). Reads each index's
    ``index_usage.json`` rollup from the store. Returns
    ``(summed_rollup, n_indices)``.
    """
    seen: set[str] = set()
    rollups: list[dict] = []
    for manifest in manifests:
        ref = manifest.get("index_ref")
        if not ref or ref in seen:
            continue
        seen.add(ref)
        usage_path = base_dir / "_indices" / ref / "index_usage.json"
        if usage_path.exists():
            try:
                rollups.append(_index_total(json.loads(usage_path.read_text())))
            except json.JSONDecodeError:
                print(f"skipping unparseable index usage: {usage_path}")
    return sum_usage(*rollups), len(seen)


def cost_per_model_usd(usage: dict[str, dict[str, int]]) -> dict[str, float]:
    """USD cost per model via ``litellm.cost_per_token``, with a manual-pricing fallback
    (``settings.MANUAL_MODEL_PRICING``, keyed by canonical id) for models LiteLLM doesn't
    price yet — e.g. a just-released model. Still unknown → None (the CLI shows "?")."""
    out: dict[str, float] = {}
    for model, bucket in usage.items():
        prompt = int(bucket.get("prompt_tokens", 0))
        completion = int(bucket.get("completion_tokens", 0))
        try:
            prompt_cost, completion_cost = litellm.cost_per_token(
                model=model, prompt_tokens=prompt, completion_tokens=completion,
            )
            out[model] = float(prompt_cost) + float(completion_cost)
        except Exception:  # noqa: BLE001 — litellm raises a bare Exception for unknown models
            price = MANUAL_MODEL_PRICING.get(canonical_model_id(model))  # our fallback table
            out[model] = (prompt * price[0] + completion * price[1]) if price else None  # type: ignore[assignment]
    return out


# ============================================================
# Aggregation
# ============================================================


def aggregate_group(
    group: dict[str, Any],
    benchmark_module: Any | None = None,
) -> dict[str, Any]:
    """Build a group summary: accuracy + per-metadata breakdown + cost.

    Cost ``usage`` is split into ``inference`` (sum of manifest ``usage``),
    ``construction`` (distinct indices' build cost, counted once), and
    ``total``.
    """
    manifests = group["manifests"]
    n = len(manifests)
    scored = [m for m in manifests if "_score" in m]
    n_scored = len(scored)

    errors = group.get("errors", [])
    errors_by_type = dict(Counter(e.get("error_type", "Unknown") for e in errors))

    # Rating benchmarks (Loong) declare PERFECT_SCORE → Avg Score + Perfect Rate;
    # the 0/1 label benchmarks don't → accuracy. `_compute_metric` carries the
    # kind-aware headline — read it (`summary["metric"]`), NOT the top-level fields.
    perfect_score = (
        getattr(benchmark_module, "PERFECT_SCORE", None)
        if benchmark_module is not None else None
    )
    # Per-benchmark report display config (read generically via getattr, so the
    # analysis layer never branches on a benchmark name): axes to omit from the
    # breakdown, and value labels for the small integer axes.
    hide_axes = (
        getattr(benchmark_module, "ANALYSIS_HIDE_AXES", frozenset())
        if benchmark_module is not None else frozenset()
    )
    value_labels = (
        getattr(benchmark_module, "ANALYSIS_VALUE_LABELS", {})
        if benchmark_module is not None else {}
    )

    # Context-window errors are a genuine failed prediction (the model produced no
    # answer). Score them at the benchmark's worst value (free — its empty-answer
    # short-circuit) and ALWAYS fold them into a parallel "all" view alongside the
    # answered-only view; the renderer shows both. Guarded — a benchmark without
    # score_details, or a stale task_id, just yields no failures.
    try:
        failures = failure_manifests(group, benchmark_module)
    except Exception:  # noqa: BLE001 — failure-scoring must not break aggregation
        failures = []
    n_failures = len(failures)

    # Capped runs: an agentic baseline that exhausted its step/iteration budget
    # (codeact `max_steps_error`, rlm `n_iterations>=max_iterations`) produced no genuine
    # answer — only a forced/truncated one. Its REAL judged score stays in the answered
    # view; in the ⁺ ALL view it's re-scored as EMPTY (the benchmark's worst, free via the
    # empty-answer short-circuit), exactly like a context-window failure. So ⁺ all = "tasks
    # the model didn't actually complete (no answer OR out of budget) count as wrong".
    for m in scored:
        m["_all_score"] = m["_score"]
    capped = [m for m in scored if _hit_step_cap(m)]
    n_capped = len(capped)
    if capped and benchmark_module is not None and hasattr(benchmark_module, "score_details"):
        try:
            worst = benchmark_module.score_details([m["task_id"] for m in capped], [""] * len(capped))
            for m, r in zip(capped, worst):
                m["_all_score"] = r["score"]
        except Exception:  # noqa: BLE001 — cap-folding must never break aggregation
            pass

    answered_scores = [m["_score"] for m in scored]
    all_scores = [m["_all_score"] for m in scored]          # capped → worst; everyone else → real
    failure_scores = [f["_score"] for f in failures]
    metric = _compute_metric(answered_scores, perfect_score)                  # answered only
    metric_all = _compute_metric(all_scores + failure_scores, perfect_score)  # + ctx errors + capped

    # Top-level n_correct / accuracy are the ANSWERED view, and meaningful ONLY for
    # the 0/1 label benchmarks (a "mean rating" mislabeled as accuracy is nonsense),
    # so leave them None for rating benchmarks — everything routes through `metric`.
    is_rating = perfect_score is not None
    n_correct = None if is_rating else sum(answered_scores)
    accuracy = None if is_rating else ((n_correct / n_scored) if n_scored else None)

    by_axis: dict[str, dict[str, dict[str, Any]]] = {}
    if benchmark_module is not None and hasattr(benchmark_module, "get_task_metadata"):
        answered_by: dict[tuple[str, str], list] = defaultdict(list)
        all_by: dict[tuple[str, str], list] = defaultdict(list)

        def _bucket(record: dict, into_all_only: bool) -> None:
            try:
                meta = benchmark_module.get_task_metadata(record["task_id"])
            except (KeyError, ValueError):
                return  # task id no longer in the dataset — skip from breakdown
            # The ⁺ all view uses `_all_score` (capped manifests → worst); the answered
            # view uses the real `_score`. Failures carry only `_score` (already worst).
            all_score = record.get("_all_score", record["_score"])
            for axis, value in meta.items():
                if axis in hide_axes:
                    continue  # benchmark hides this axis from the report breakdown
                if not into_all_only:
                    answered_by[(axis, str(value))].append(record["_score"])
                all_by[(axis, str(value))].append(all_score)

        for manifest in scored:
            _bucket(manifest, into_all_only=False)
        for failure in failures:
            _bucket(failure, into_all_only=True)  # context-window errors → "all" only

        for axis, value in set(answered_by) | set(all_by):
            cell = _compute_metric(answered_by.get((axis, value), []), perfect_score)
            cell["all"] = _compute_metric(all_by.get((axis, value), []), perfect_score)
            by_axis.setdefault(axis, {})[value] = cell

    # Each manifest's `usage` is {total: {model: rollup}, calls: [...]}; the
    # inference cost is the per-model `total` rollup.
    inference_usage = sum_usage(*((m.get("usage") or {}).get("total") or {} for m in manifests))
    construction, n_indices = construction_usage(group["base_dir"], manifests)
    total_usage = sum_usage(inference_usage, construction)

    return {
        "config": group["config"],
        "base_dir": group["base_dir"],
        "n": n,
        "n_scored": n_scored,
        "n_correct": n_correct,
        "accuracy": accuracy,
        "metric": metric,
        "metric_all": metric_all,
        "n_indices": n_indices,
        "n_errors": len(errors),
        "errors_by_type": errors_by_type,
        "n_failures": n_failures,
        "n_capped": n_capped,
        "by_axis": by_axis,
        "value_labels": value_labels,
        "usage": {
            "inference": inference_usage,
            "construction": construction,
            "total": total_usage,
        },
        "cost_usd": {
            "inference": cost_per_model_usd(inference_usage),
            "construction": cost_per_model_usd(construction),
            "total": cost_per_model_usd(total_usage),
        },
    }


# ============================================================
# CLI
# ============================================================


def _fmt(value: float | None, places: int) -> str:
    return "N/A" if value is None else f"{value:.{places}f}"


def _metric_style(fraction: float | None) -> str:
    """Color band for a 0..1 metric fraction (accuracy, or avg/perfect_score):
    red below 0.40, yellow to 0.65, green above; dim when undefined."""
    if fraction is None:
        return "dim"
    if fraction < 0.40:
        return "red"
    if fraction < 0.65:
        return "yellow"
    return "green"


def _band(value: float | None, places: int, fraction: float | None = None) -> str:
    """A metric value, color-banded by its (optionally normalized) magnitude."""
    style = _metric_style(value if fraction is None else fraction)
    return f"[{style}]{_fmt(value, places)}[/{style}]"


# The single per-row view: the TOTAL — context-window errors AND step/iteration-capped
# runs counted as the benchmark's worst score (the answered-only view is not shown).
_METRIC_COLS = {"accuracy": ("acc", "n"), "rating": ("avg", "perfect", "n")}


def _metric_cells(stats: dict[str, Any]) -> tuple[str, ...]:
    """The metric cell(s) for ONE view of an axis row, value color-banded. An empty
    bucket (nothing in this view) renders N/A counts instead of a bare 0/0."""
    n = stats["n"]
    if stats["kind"] == "rating":
        avg = stats["avg_score"]
        frac = (avg / stats["perfect_score"]) if avg is not None else None
        return (_band(avg, 2, frac), _fmt(stats["perfect_rate"], 3),
                f"{stats['n_perfect']}/{n}" if n else "N/A")
    return (_band(stats["accuracy"], 3),
            f"{stats['n_correct']}/{n}" if n else "N/A")


def _headline_markup(metric_all: dict[str, Any], n_logged: int) -> str:
    """The colored headline — a SINGLE total metric: the view in which context-window
    errors AND step/iteration-capped runs count as the benchmark's worst score. (The
    answered-only view isn't shown; the errors/cap lines below say what's folded in.)"""
    n_all = metric_all["n"]
    if metric_all["kind"] == "rating":
        avg = metric_all["avg_score"]
        frac = (avg / metric_all["perfect_score"]) if avg is not None else None
        return (f"[b]avg score[/b] {_band(avg, 2, frac)}   "
                f"[b]perfect rate[/b] {_fmt(metric_all['perfect_rate'], 3)}   "
                f"[dim]({n_all} scored incl. errors + capped · {n_logged} logged)[/dim]")
    return (f"[b]accuracy[/b] {_band(metric_all['accuracy'], 3)} "
            f"[dim]({metric_all['n_correct']}/{n_all})[/dim]")


def _axis_table(axis: str, breakdown: dict[str, dict[str, Any]],
                value_labels: dict[str, str] | None = None) -> Table:
    """A table for one metadata axis: each row shows the TOTAL metric (context-window
    errors + step/iteration-capped runs counted as wrong), value color-banded.

    ``value_labels`` (from the benchmark's ``ANALYSIS_VALUE_LABELS``) relabels the
    ``value`` column — e.g. loong's ``set`` ``"1"`` → ``"1 · 10–50K tok"`` — while
    rows still SORT by the raw value, so the encoding order is preserved.
    """
    labels = value_labels or {}
    rating = next(iter(breakdown.values()))["kind"] == "rating"
    headers = _METRIC_COLS["rating" if rating else "accuracy"]
    table = Table(title=f"by {axis}", title_justify="left", title_style="bold",
                  box=box.HEAVY_HEAD, padding=(0, 1), expand=False)
    table.add_column("value", style="cyan", no_wrap=True)
    for name in headers:
        table.add_column(name, justify="right", style="dim" if name == "n" else None)
    for value, stats in sorted(breakdown.items()):
        table.add_row(labels.get(value, value), *_metric_cells(stats["all"]))
    return table


def _cost_table(summary: dict[str, Any]) -> Table | None:
    """The per-model cost table (inference vs construction split), or None if no usage."""
    total_usage = summary["usage"]["total"]
    if not total_usage:
        return None
    total_cost = summary["cost_usd"]["total"]
    infer_cost = summary["cost_usd"]["inference"]
    build_cost = summary["cost_usd"]["construction"]
    table = Table(title=f"cost · {summary['n_indices']} index/indices built",
                  title_justify="left", title_style="bold", box=box.HEAVY_HEAD,
                  padding=(0, 1), expand=False)
    table.add_column("model", style="cyan", no_wrap=True)
    table.add_column("USD", justify="right")
    table.add_column("tokens", justify="right", style="dim")
    table.add_column("calls", justify="right", style="dim")
    table.add_column("split", style="dim")
    for model in sorted(total_usage):
        bucket = total_usage[model]
        cost = total_cost.get(model)
        parts = []
        bc, ic = build_cost.get(model), infer_cost.get(model)
        if bc:
            parts.append(f"build ${bc:.4f}")
        if ic:
            parts.append(f"infer ${ic:.4f}")
        table.add_row(model, "?" if cost is None else f"${cost:.4f}",
                      f"{bucket['total_tokens']:,}", f"{bucket['num_calls']:,}",
                      " / ".join(parts))
    grand = sum((c for c in total_cost.values() if c is not None), 0.0)
    denom = summary["n_scored"] or summary["n"]
    per = grand / denom if denom else 0.0
    table.add_row("[b]TOTAL[/b]", f"[b]${grand:.4f}[/b]", "", "", f"${per:.4f}/inference")
    return table


def render_report(summary: dict[str, Any], console: Console) -> None:
    """Render one group's summary: a rounded panel (config + headline metric + the
    errors-on-disk coverage line) followed by a table per metadata axis and a cost
    table. Metric values are color-banded (red→yellow→green); color degrades to plain
    text when the console isn't a terminal (e.g. piped to a file)."""
    cfg = summary["config"]
    bench = cfg.get("benchmark", "?")
    baseline = cfg.get("baseline", "?")
    # `completion_params` (the resolved sampling dict) is verbose — the run is
    # labeled by `config_name` instead; the full params stay in the manifest.
    extra = ", ".join(f"{k}={v}" for k, v in sorted(cfg.items())
                      if k not in {"benchmark", "baseline", "completion_params"})
    title = f"[b]{bench}[/b] / {baseline}" + (f"  [dim]{extra}[/dim]" if extra else "")

    n, n_scored = summary["n"], summary["n_scored"]
    metric_all = summary.get("metric_all") or summary["metric"]
    body: list[str] = []
    if n_scored == 0 and not summary.get("n_failures"):
        body.append(f"[dim]{n} logged · 0 scored — run with scoring to get the metric[/dim]")
    else:
        body.append(_headline_markup(metric_all, n))

    # Recorded failures on disk — always shown so the denominator is legible. The
    # context-window ones are the ones counted in the ⁺ ("all") columns.
    if summary.get("n_errors"):
        by_type = ", ".join(
            f"{cnt} {etype}"
            for etype, cnt in sorted(summary["errors_by_type"].items(), key=lambda kv: (-kv[1], kv[0]))
        )
        line = f"[dim]errors[/dim] {summary['n_errors']} on disk · {by_type}"
        if summary.get("n_failures"):
            line += (f"  [magenta]({summary['n_failures']} context-window counted "
                     f"wrong)[/magenta]")
        body.append(line)

    # Step/iteration-capped runs (codeact max_steps / rlm max_iterations) produced no
    # genuine answer — only a forced/truncated one — so they're counted WRONG in the total.
    if summary.get("n_capped"):
        body.append(f"[magenta]{summary['n_capped']}/{n_scored} hit the step/iteration cap "
                    f"(counted wrong)[/magenta]")

    console.print(Panel("\n".join(body), title=title, title_align="left",
                        box=box.ROUNDED, expand=False, padding=(0, 1)))

    value_labels = summary.get("value_labels", {})
    for axis, breakdown in sorted(summary["by_axis"].items()):
        if len(breakdown) > MAX_BREAKDOWN_VALUES:
            console.print(f"[dim]by {axis}: {len(breakdown)} distinct values — "
                          f"too granular to tabulate (skipped)[/dim]")
            continue
        console.print(_axis_table(axis, breakdown, value_labels.get(axis)))

    cost = _cost_table(summary)
    if cost is not None:
        console.print(cost)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate inference manifests into accuracy + cost summaries, "
                    "grouped by config (per CLAUDE.md's cheap per-iteration step)."
    )
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=sorted(BENCHMARK_IMPORT_MAP),
                        help="Filter to this benchmark. Omit for all.")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Filter to this baseline (e.g. structrag, arag, rlm, codeact).")
    parser.add_argument("--model", type=str, default=None,
                        help="Filter to one model (e.g. 'Qwen3.5-35B-A3B'); canonicalized "
                             "to match the slug stored in each config. Omit for all models.")
    parser.add_argument("--no-score", action="store_true",
                        help="Skip computing score.json (cheap inspection; the parse / judge "
                             "is an LLM call). Any already-cached scores are still read + shown.")
    parser.add_argument("--rescore", action="store_true",
                        help="Force-recompute score.json even when a cached one exists. The cache "
                             "auto-invalidates on a grader-MODEL change (scored_with) or a SCORER-id "
                             "bump; use this after any OTHER grader change (a parse/judge prompt edit "
                             "without bumping SCORER). Re-pays the parse / judge LLM.")
    args = parser.parse_args(argv)

    groups = find_groups(benchmark=args.benchmark, baseline=args.baseline, model=args.model)
    if not groups:
        print("no inferences found")
        return

    console = Console()  # auto-detects the terminal: color when interactive, plain when piped
    for group in groups:
        benchmark_name = group["config"].get("benchmark")
        module_path = BENCHMARK_IMPORT_MAP.get(benchmark_name) if benchmark_name else None
        benchmark_module = importlib.import_module(module_path) if module_path else None
        # Compute+cache missing/stale score.json (unless --no-score), then read the
        # scores back onto the manifests. The metric is read from score.json, never
        # recomputed here — so re-aggregating doesn't re-pay the parse / judge LLM.
        # A scoring error (e.g. a logged task_id that no longer resolves after a
        # dataset shift) must not abort the whole CLI: bound it to this group and
        # fall back to whatever scores are already cached.
        try:
            stats = attach_scores(
                group, benchmark_module, compute=not args.no_score, force=args.rescore
            )
        except Exception as e:  # noqa: BLE001 — one group's scoring must not kill the rest
            print(f"[score] {benchmark_name}: scoring failed "
                  f"({type(e).__name__}: {e}); reporting cached scores only")
            stats = attach_scores(group, benchmark_module, compute=False)
        if stats["scored"]:
            print(f"[score] {benchmark_name}: graded {stats['scored']} "
                  f"({stats['cached']} cached) of {stats['total']}")
        # aggregate_group computes BOTH views (answered-only and context-window-
        # counted); render_report shows them side by side — no flag.
        summary = aggregate_group(group, benchmark_module)
        render_report(summary, console)
        console.print()
