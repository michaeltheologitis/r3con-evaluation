"""Turn this repo's ``logs/`` into the paper's tables and figures.

Read-only over the logs: no model is called and no answer is re-judged. The judge verdicts
already sit in each run's ``score.json`` and are reproduced, not recomputed, so the whole
notebook regenerates in seconds and gives the same numbers every time.

- :mod:`scoreboard.paths`  — where the logs are
- :mod:`scoreboard.load`   — logs -> one tidy row per inference
- :mod:`scoreboard.meta`   — per-task benchmark metadata (vendored snapshots)
- :mod:`scoreboard.report` — rows -> scoreboards (DataFrames only; no printing or plotting)
- :mod:`scoreboard.style`  — one display name / colour / marker per method

Rendering lives in ``analysis/results.ipynb``, never in here.
"""
