"""Where the analysis reads from — this repo's own ``logs/``.

Both sides of the comparison live in one tree, at the paths the runners write:

- ``logs/{benchmark}/{baseline}/…`` — the baselines (``evals/baselines/_common.py``)
- ``logs/r3con/<run-folder>/``      — R3Con (``evals/r3con/pipeline/settings.py``)

So a baseline you re-run lands beside the shipped run rather than somewhere new, and the
analysis picks it up with no configuration.
"""
from __future__ import annotations

from pathlib import Path

# Figures are written here as PDF (vector, for the paper) with a PNG alongside as a check
# render. Edit this one line to send them somewhere else.
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


def repo_root() -> Path:
    """The repo root — the nearest ancestor holding ``pyproject.toml``."""
    here = Path(__file__).resolve()
    for d in here.parents:
        if (d / "pyproject.toml").is_file():
            return d
    return here.parent.parent.parent


def logs_dir() -> Path:
    """``<repo>/logs/`` — the root of both log trees."""
    return repo_root() / "logs"


def baselines_logs() -> Path:
    """The baselines root: one directory per benchmark, each holding one per baseline.
    ``logs/r3con/`` sits beside them and is skipped by name — see
    :data:`scoreboard.load.NON_BENCHMARK_DIRS`."""
    return logs_dir()


def method_logs() -> Path:
    """R3Con's root: one opaque run-folder per task."""
    return logs_dir() / "r3con"
