# analysis

Rebuilds every table and figure in the paper from this repo's `logs/`.

```bash
uv sync --extra analysis
uv run jupyter lab analysis/results.ipynb        # then Restart & Run All
```

Or re-execute it without opening Jupyter:

```bash
uv run jupytext --sync --execute analysis/results.py
```

## What's here

```
results.ipynb    the notebook, with its outputs — open this
results.py       the same notebook as a script (jupytext pair); the source we edit
scoreboard/      the small library it imports
  paths.py       where the logs are
  load.py        logs -> one tidy row per inference (tokens, USD, score, errors, caps)
  meta.py        per-task benchmark metadata, from vendored snapshots
  report.py      rows -> scoreboards (DataFrames only)
  style.py       one display name / colour / marker per method
figures/         written by the notebook: PDF for the paper, PNG as a check render
```

`results.py` and `results.ipynb` are a [jupytext](https://jupytext.readthedocs.io) pair —
edit either one and `jupytext --sync` updates the other.

## Two things worth knowing

**Nothing is re-judged.** Each run's `score.json` was written by the judge when the run
happened. The notebook reads those verdicts back, so it regenerates in seconds and gives the
same numbers every time. No model is called anywhere in here.

**The logs are stripped.** `logs/` keeps only the fields this analysis reads — the run
config, per-model token totals, the judge's score, the step/iteration-cap signal, and each
R3Con stage's token totals. Questions, gold answers, agent traces and per-call dumps are not
included, which is what makes the tree small enough to ship. R3Con's just-in-time artifacts
(the relevance state, the structured parse, the proposed schema) *are* included for the
default configuration, so the intermediate-representation sizes in **Method internals** are
counted from the real text rather than taken on trust.

## Conventions the numbers depend on

- **Scoring views.** Loong reports an average 1–100 judge score and an exact-match rate
  (fraction scoring exactly 100); CorpusQA reports 0/1 accuracy. Every headline number is the
  **⁺all** view: a genuine prediction failure is re-scored at the benchmark's worst value
  rather than dropped. That covers agentic runs that exhausted their step budget, plus
  context-window errors on Loong and *every* error on CorpusQA (0/1 — any failure is a miss).
- **Runs are grouped by their full identity** — source, method, benchmark and the whole
  config string. Two runs differing by any knob stay separate rows.
- **Tokens are the cost**, and USD comes from a small per-model price table
  (`scoreboard.load.PRICE`, OpenRouter rates). Cache counts as normal input. Every size of one
  model family is priced on one provider's endpoint, so a size comparison isn't a pricing
  artefact. An embedder's tokens price into USD but are not counted as language-model tokens.
- **Scope.** Loong uses context tiers 2–4 (the 10–50K tier is dropped at the source);
  CorpusQA uses the `1m` tier. Every method is compared on the same served model,
  `Qwen3.5-35B-A3B` — the two exceptions are labelled where they appear (Claude Code as a
  frontier reference, and the 9B/4B size study).

The **Method internals** section counts tokens with the served model's own tokenizer, fetched
from the Hugging Face hub once and cached locally — the only step here that needs network
access.
