# analysis

Rebuilds every table and figure in the paper from this repo's `logs/`.

```bash
tar -xf logs.tar.zst                             # from the repo root, if you haven't yet
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

The notebook itself explains each table and figure as it builds it. The one step that needs
network access is **Method internals**, which fetches the served model's tokenizer from the
Hugging Face hub once and caches it.
