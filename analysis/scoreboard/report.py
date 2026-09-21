"""Shape the per-inference table into scoreboards — pure functions returning DataFrames.

NO printing or plotting here; that is the notebook's job (`analysis/results.ipynb`).

- :func:`headline` — one row per run (full identity): Avg Score / Perfect Rate (loong) or
  Accuracy (corpusqa), answered-only + ⁺all (context-window errors counted as 0), tokens.
- :func:`crosstab` — the banded scoreboard: an **S** (avg score, ctx-inclusive) and **EM**
  (perfect rate) cell per task-type (or domain), plus an Overall block; one row per run.

In both, **baselines come first and our method (`grounded-*`) last** ("baseline" sorts
before "method"), each block by score descending. The method's configs (summary_rounds
ablation, thinking vs no-thinking, both strategies) each get their own row.
"""
from __future__ import annotations

import pandas as pd

from scoreboard import load, meta, style

HEADLINE_COLUMNS = [
    "source", "run", "method", "benchmark", "model", "n", "n_scored", "n_err", "n_capped",
    "avg_score", "perfect_rate", "avg_score_all", "perfect_rate_all",
    "mean_tokens", "total_tokens", "config",
]

# A run's full identity. `config` deliberately omits method/benchmark (they're columns),
# so it is NOT unique on its own — direct-llm and structrag both have `model=…|seed=42`.
# Always group by the whole key, never `config` alone (this has bitten the crosstab).
RUN_KEY = ["source", "method", "benchmark", "config"]

# The model the comparison centers on (served vLLM); other models get a [model] tag in the
# run label so rows stay unique (claude-code's Opus, gpt wire-tests, other Qwen sizes).
PRIMARY_MODEL = "qwen3-5-35b-a3b"

# Models shown WITHOUT a `[model]` tag in run labels: each is the single canonical model of its
# method, so the tag is redundant (the served model; MemAgent's own trained model). Claude Code's
# Opus/Sonnet stay tagged (it runs several), so they're deliberately NOT here.
UNTAGGED_MODELS = frozenset({PRIMARY_MODEL, "rl-memoryagent-14b"})

# R3Con's DEFAULT configuration, in one place so every pin and every ablation agrees:
# the coding agent downstream, summary_rounds=2.
DEFAULT_METHOD = "grounded-codeact"
DEFAULT_SR = 2

# ── Error and cap counting ────────────────────────────────────────────────────────────
# The ⁺all view re-scores genuine PREDICTION FAILURES at the benchmark's WORST value, Two kinds fold in:
#   1. capped agentic runs — codeact `max_steps_error` / rlm `max_iterations` (the `capped`
#      column, from the manifest trace) re-scored within the scored set, BOTH benchmarks. Only
#      the baselines write a trace, so R3Con's runs are never cap-folded.
#   2. error rows — folded into the denominator at worst:
#        loong    → only ContextWindowExceededError (FAILURE_ERROR_TYPES); a Timeout or
#                   RuntimeError is infrastructure noise — reported, but NOT folded.
#        corpusqa → EVERY error counts wrong (0/1: any failure is unambiguously a miss).
# WORST = the benchmark floor an empty answer earns (loong judge `_WORST_SCORE` = 1; the 0/1
# label judges = 0). PERFECT = the top of the scale (loong 100; the 0/1 benchmarks 1).
FAILURE_ERROR_TYPES = frozenset({"ContextWindowExceededError"})
# A 0/1 accuracy benchmark folds EVERY error as wrong (any failure is a miss); Loong (1-100)
# folds only context-window errors.
EVERY_ERROR_BENCHMARKS = frozenset({"corpusqa"})
WORST = {"loong": 1.0, "corpusqa": 0.0}
PERFECT = {"loong": 100.0, "corpusqa": 1.0}


def _scores(sub: pd.DataFrame) -> tuple[pd.Series, pd.Series, int, int]:
    """(answered, all, n_folded_errors, n_capped) score arrays for one slice, per the fold
    rule above. ``answered`` = real judged scores; ``all`` = those with prediction-failures
    re-scored to the benchmark's worst (capped runs re-scored in place + folded error rows
    appended). Empty slice → empty series."""
    if sub.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0, 0
    bench = sub["benchmark"].iloc[0]
    worst = WORST.get(bench, 0.0)
    scored = sub[sub["scored"]]
    real = scored["score"].astype(float)
    # capped agentic runs -> worst (both benchmarks). `capped` comes from the manifest trace,
    # which only the baselines write (codeact max_steps / rlm max_iter), so R3Con is never
    # cap-folded.
    capped = scored["capped"].fillna(False).astype(bool)
    all_scored = real.mask(capped, worst)
    err = sub[sub["status"] == "error"]
    n_fold = len(err) if bench in EVERY_ERROR_BENCHMARKS else int(err["error_type"].isin(FAILURE_ERROR_TYPES).sum())
    # dtype is explicit: an untyped empty Series makes pandas warn (and one day change
    # behaviour) on concat, and the warning would print an absolute path into the notebook.
    folded = pd.Series([worst] * n_fold, dtype=float)
    all_view = pd.concat([all_scored, folded], ignore_index=True)
    return real.reset_index(drop=True), all_view, n_fold, int(capped.sum())


# The default config + the two single-PHASE ablations off it (all else equal) — (label, method,
# summary_rounds). Each removes one phase of the just-in-time pipeline (the paper's Fig. r3con):
#   w/o extracting relevance → summary_rounds=0 (no relevance state 𝓡; same CodeAct downstream)
#   w/o structuring          → `grounded-codeact_ablation_nostruct` (CodeAct over 𝓡, no parse)
# Swapping the DOWNSTREAM reasoner is deliberately NOT one of these (owner, 2026-09-21): it
# leaves the representation intact, so it's a separate comparison — see :data:`DOWNSTREAM`.
ABLATIONS = [
    ("Default",                  DEFAULT_METHOD,                       DEFAULT_SR),
    ("w/o extracting relevance", DEFAULT_METHOD,                       0),
    ("w/o structuring",          "grounded-codeact_ablation_nostruct", DEFAULT_SR),
]
# The SAME representation (both phases intact) handed to a different downstream reasoner:
# the direct LLM vs the default coding agent. Winner-last, as everywhere else.
DOWNSTREAM = [
    ("LLM",                    "grounded-llm", DEFAULT_SR),
    ("Coding agent (default)", DEFAULT_METHOD, DEFAULT_SR),
]
ABLATION_COLUMNS = ["ablation", "em", "n"]
DOWNSTREAM_COLUMNS = ["downstream", "em", "n"]


def _em_rows(df: pd.DataFrame, spec: list, label_col: str) -> pd.DataFrame:
    """Each ``(label, method, summary_rounds)`` in ``spec`` reduced to its TOTAL Exact-Match rate
    in the ⁺all view (loong = perfect rate, fraction == 100; corpusqa = accuracy, fraction
    correct). ``df`` = the per-inference table for ONE benchmark; only the served model's
    thinking (``variant == "default"``) method runs are used. ``em`` is 0–1."""
    cols = [label_col, "em", "n"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    loong = df["benchmark"].iloc[0] == "loong"
    rows = []
    for label, method, sr in spec:
        g = df[(df["source"] == "method") & (df["method"] == method) & (df["variant"] == "default")
               & (df["summary_rounds"] == sr) & (df["model"] == PRIMARY_MODEL)]
        _real, all_view, _, _ = _scores(g)
        em = (float("nan") if len(all_view) == 0
              else ((all_view == PERFECT["loong"]).mean() if loong else all_view.mean()))
        rows.append({label_col: label, "em": em, "n": len(all_view)})
    return pd.DataFrame(rows, columns=cols)


def ablations(df: pd.DataFrame) -> pd.DataFrame:
    """The default config + the two single-phase ablations (:data:`ABLATIONS`)."""
    return _em_rows(df, ABLATIONS, "ablation")


def downstream(df: pd.DataFrame) -> pd.DataFrame:
    """The default representation consumed by each downstream reasoner (:data:`DOWNSTREAM`)."""
    return _em_rows(df, DOWNSTREAM, "downstream")


# Names and row order come from the central registry (scoreboard.style): names via style.name,
# the display sequence via style.ORDER (method rows always sort last).


def _mode(variant: str) -> str:
    """Readable run-knob tag: thinking/no-thinking, claude-code effort, else the raw value."""
    if variant in ("Qwen3.5-MoE-Instruct", "qwen-no-thinking"):
        return "no-think"
    if variant.startswith("effort="):
        return variant.split("=", 1)[1]
    return variant


# Readable strategy names for the run label (the raw `inference/<strategy>/` folder name).
_STRATEGY_LABEL = {"codeact_ablation_nostruct": "codeact no-parse"}


def run_label(row: pd.Series) -> str:
    """A short, readable label per run, using the central display name (:mod:`scoreboard.style`).
    Method rows surface the strategy + summary_rounds ablation; a non-primary model gets a
    ``[model]`` tag so labels stay unique across models."""
    name = style.name(row["method"])
    tag = "" if row["model"] in UNTAGGED_MODELS else f" [{row['model']}]"
    # Keyed off the METHOD, not the source: the branch is about the label's SHAPE, so any
    # method-shaped row gets the strategy + sr suffix (source "method" <=> grounded-*).
    if str(row["method"]).startswith("grounded-"):
        strat = row["method"].split("-", 1)[-1]    # strategy: grounded-codeact -> codeact
        bits = [_STRATEGY_LABEL.get(strat, strat)]
        bits.append("think" if row["variant"] == "default" else _mode(row["variant"]))
        if pd.notna(row.get("summary_rounds")):
            bits.append(f"sr{int(row['summary_rounds'])}")
        return f"{name} ({', '.join(bits)}){tag}"
    if row["variant"] == "default":
        return f"{name}{tag}"
    return f"{name} ({_mode(row['variant'])}){tag}"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["run"] = df.apply(run_label, axis=1)
    return df


def _sort_for_display(out: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Order rows by the registry sequence (:data:`scoreboard.style.ORDER`), with the method
    (grounded-*) rows always last; ties within a group by ``score_col`` descending."""
    def rank(r):
        base = style.ORDER.index(r["method"]) if r["method"] in style.ORDER else len(style.ORDER)
        return (1, base) if r["source"] == "method" else (0, base)
    keys = out.apply(rank, axis=1)
    out = out.assign(_g=[k[0] for k in keys], _i=[k[1] for k in keys])
    out = out.sort_values(["_g", "_i", score_col], ascending=[True, True, False], na_position="last")
    return out.drop(columns=["_g", "_i"]).reset_index(drop=True)


def headline(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run (source × method × benchmark × config), baselines first, method last."""
    if df.empty:
        return pd.DataFrame(columns=HEADLINE_COLUMNS)
    df = add_labels(df)
    rows = []
    for _, g in df.groupby(RUN_KEY, sort=False):
        bench = g["benchmark"].iloc[0]
        rating = bench == "loong"                           # 1–100 rating vs 0/1 accuracy
        perfect = PERFECT.get(bench, 100.0)
        places = 1 if rating else 3                          # loong score 1dp; corpusqa acc 3dp
        real, all_view, _, n_capped = _scores(g)            # ⁺all folds failures at worst
        n_err = int((g["status"] == "error").sum())         # all errors on disk (folded subset varies)
        ok_tokens = g.loc[g["status"] == "ok", "total_tokens"]
        rows.append({
            "source": g["source"].iloc[0], "run": g["run"].iloc[0], "method": g["method"].iloc[0],
            "benchmark": bench, "model": g["model"].iloc[0],
            "n": len(g), "n_scored": len(real), "n_err": n_err, "n_capped": n_capped,
            "avg_score": round(real.mean(), places) if len(real) else None,
            "perfect_rate": round((real == perfect).mean(), 3) if (rating and len(real)) else None,
            "avg_score_all": round(all_view.mean(), places) if len(all_view) else None,
            "perfect_rate_all": round((all_view == perfect).mean(), 3) if (rating and len(all_view)) else None,
            "mean_tokens": int(ok_tokens.mean()) if len(ok_tokens) else None,
            "total_tokens": int(ok_tokens.sum()), "config": g["config"].iloc[0],
        })
    out = pd.DataFrame(rows, columns=HEADLINE_COLUMNS)
    return _sort_for_display(out, "avg_score_all")


COST_COLUMNS = ["source", "run", "method", "model", "n", "S", "EM",
                "in_tok", "out_tok", "tok", "usd_per_task", "usd_total"]


def cost_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per run: the overall score (S = avg, EM = perfect rate — the boards' ctx-
    inclusive "Overall"), mean tokens/task (input / output / total), and USD (per task +
    total). Baselines first then method last. Cost is the per-inference `usd` column (priced
    at load, per model — see `scoreboard.load.PRICE`); a run on an unpriced model shows `–` for USD."""
    if df.empty:
        return pd.DataFrame(columns=COST_COLUMNS)
    df = add_labels(df)
    rows = []
    for _, g in df.groupby(RUN_KEY, sort=False):
        ok = g[g["status"] == "ok"]
        usd = ok["usd"]
        priced = usd.notna().any()
        s, em = _se(g)                                  # same overall S / EM as the boards
        rows.append({
            "source": g["source"].iloc[0], "run": g["run"].iloc[0], "method": g["method"].iloc[0],
            "model": g["model"].iloc[0], "n": len(ok), "S": s, "EM": em,
            "in_tok": int(ok["input_tokens"].mean()) if len(ok) else None,
            "out_tok": int(ok["output_tokens"].mean()) if len(ok) else None,
            "tok": int(ok["total_tokens"].mean()) if len(ok) else None,
            "usd_per_task": usd.mean() if priced else None,
            "usd_total": usd.sum() if priced else None,
        })
    out = pd.DataFrame(rows, columns=COST_COLUMNS)
    return _sort_for_display(out, "S")


def _se(sub: pd.DataFrame) -> tuple[float, float]:
    """(S, EM) for a slice in the ⁺all view (failures folded at worst — see :func:`_scores`).
    S = mean score (loong rating / corpusqa accuracy), EM = fraction == the benchmark's
    perfect score (degenerate == S for 0/1 corpusqa). NaN when the slice has nothing to score."""
    _real, all_view, _, _ = _scores(sub)
    if len(all_view) == 0:
        return float("nan"), float("nan")
    perfect = PERFECT.get(sub["benchmark"].iloc[0], 100.0)
    return all_view.mean(), (all_view == perfect).mean()


def _uniquify(labels: list[str]) -> list[str]:
    """Append ``#2``, ``#3``… to repeated labels so the row index stays unique."""
    seen: dict[str, int] = {}
    out = []
    for lbl in labels:
        seen[lbl] = seen.get(lbl, 0) + 1
        out.append(lbl if seen[lbl] == 1 else f"{lbl} #{seen[lbl]}")
    return out


def _value_order(d: pd.DataFrame, col: str) -> list:
    """The distinct values of ``col``, in display order: the known ordering first (anything
    unexpected appended), else sorted."""
    present = list(pd.unique(d[col].dropna()))
    known = meta.TASK_ORDER if col == "task_name" else None
    if known is None:
        return sorted(present, key=str)
    return [v for v in known if v in present] + [v for v in present if v not in known]


def crosstab(df: pd.DataFrame, col: str = "task_name", set_n=None, *,
             benchmark: str = "loong") -> pd.DataFrame:
    """Banded scoreboard: rows = runs (method last), columns = ``(value, {S, EM})`` over
    ``col`` plus an Overall block. Keyed by the full run identity so runs never merge.
    ``benchmark`` selects the rows and the S/EM scale (Loong rating / CorpusQA accuracy) and
    ``set_n`` restricts to one context-length tier. ``df`` must carry the matching metadata
    columns (see :func:`scoreboard.meta.attach_meta`)."""
    d = add_labels(df[df["benchmark"] == benchmark].copy())
    if set_n is not None:
        d = d[d["set"] == set_n]
    values = _value_order(d, col)
    cols = pd.MultiIndex.from_tuples([(v, m) for v in [*values, "Overall"] for m in ("S", "EM")])
    groups = {k: g for k, g in d.groupby(RUN_KEY, sort=False)}   # key by the FULL identity
    data, labels = [], []
    for _, r in headline(d).iterrows():                          # headline order: method last
        g = groups[tuple(r[k] for k in RUN_KEY)]
        row = {}
        for v in values:
            row[(v, "S")], row[(v, "EM")] = _se(g[g[col] == v])
        row[("Overall", "S")], row[("Overall", "EM")] = _se(g)
        data.append(row)
        labels.append(r["run"])
    out = pd.DataFrame(data, index=_uniquify(labels), columns=cols, dtype=float)
    out.index.name = "run"
    return out                          # full precision; the notebook formats (S 1dp, EM 2dp)


def build(baselines_root=None, method_root=None) -> pd.DataFrame:
    """Read ``logs/`` and attach Loong metadata -> the table the Loong boards are built from."""
    return meta.attach_meta(load.load_all(baselines_root, method_root))


def build_corpusqa(baselines_root=None, method_root=None) -> pd.DataFrame:
    """The CorpusQA twin of :func:`build`."""
    return meta.attach_corpusqa_meta(load.load_all(baselines_root, method_root))

