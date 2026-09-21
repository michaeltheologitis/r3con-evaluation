"""Standard `logging` for the pipeline.

Library modules log through ``get_logger(...)``; the CLI entry points
(``scripts/r3con/<benchmark>/{run,run_task}.py``) call ``configure_logging()`` once at
startup. Enabling progress output is one knob:

- ``R3CON_LOG_LEVEL`` env (``DEBUG`` / ``INFO`` / ``WARNING``; default
  ``WARNING`` = quiet), or
- ``--verbose`` on the runner, which sets that env to ``INFO`` for the spawned
  per-task children (whose stdout the launcher redirects to
  ``logs/r3con/<run-folder>/run_task.log``).

No handler is attached until ``configure_logging`` runs, so importing the library
in a notebook/test stays silent.
"""

from __future__ import annotations

import logging
import os
import sys

_NAME = "r3con"


def get_logger(name: str | None = None) -> logging.Logger:
    """The pipeline logger (``r3con`` or ``r3con.<name>``)."""
    return logging.getLogger(_NAME if name is None else f"{_NAME}.{name}")


def configure_logging(level: str | int | None = None) -> None:
    """Attach a single stdout handler to the ``r3con`` logger (idempotent).

    Level comes from ``level``, else ``R3CON_LOG_LEVEL``, else ``WARNING``.
    """
    logger = logging.getLogger(_NAME)
    if level is None:
        level = os.environ.get("R3CON_LOG_LEVEL", "WARNING")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
