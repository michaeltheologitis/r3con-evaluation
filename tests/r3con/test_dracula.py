"""Unit tests for `evals.r3con.harness.dracula` — the pure / no-I/O bits (offline).

No touching of the upstream loader: the adapter is (deliberately) pure delegation — ``load`` passes
the question through, ``score_*`` coerce/serialize — so the only offline-testable bits are the
constants and the empty-input short-circuits. The real verification (loading the real 45-doc corpus
+ the correctness judge, against real vendored data) is in ``tests/live/test_dracula.py``.

Run with:  uv run python tests/unit/test_dracula.py
"""

from __future__ import annotations

from evals.r3con.harness import dracula


def test_constants() -> None:
    assert dracula.NAME == "Dracula"
