"""Deterministic task-id ordering shared by every benchmark's ``get_task_ids``.

Every benchmark shuffles its task ids with a SINGLE hardcoded seed
(``settings.TASK_ID_SHUFFLE_SEED``) before returning, so callers always get the
same order — and an optional ``limit`` then yields a stable, *representative*
sample (a spread across the dataset, not the first N in file order). The seed is
deliberately NOT exposed (no parameter, no env var): the order must be
reproducible everywhere.
"""
from __future__ import annotations

import random

from evals.settings import TASK_ID_SHUFFLE_SEED


def order_task_ids(task_ids: list[str], limit: int | None = None) -> list[str]:
    """Deterministically shuffle ``task_ids`` (fixed seed) then optionally take the
    first ``limit``.

    - The shuffle is seeded with ``settings.TASK_ID_SHUFFLE_SEED`` and uses a
      fresh ``random.Random`` instance, so the result depends only on the input
      list + that seed — same input, same output, every process/run. (Python's
      Mersenne-Twister + ``Random.shuffle`` are stable across versions.)
    - ``limit=None`` (default) → all; ``limit`` ≥ ``len`` → all; ``limit=0`` → ``[]``.
    - Does not mutate the input (operates on a copy).
    """
    ids = list(task_ids)
    random.Random(TASK_ID_SHUFFLE_SEED).shuffle(ids)
    if limit is not None:
        ids = ids[:limit]
    return ids
