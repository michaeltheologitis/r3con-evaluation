"""Unit tests for `evals.r3con.harness.dracula` — the pure / no-I/O bits (offline).

No touching of the upstream loader: the adapter is (deliberately) pure delegation — ``load`` passes
the question through, ``gold`` / ``metadata`` hand straight off to the benchmark loader — so the
only offline-testable bit is the adapter's own constant surface. The real 46-document corpus and
the correctness judge are exercised against the real vendored data by the benchmark's own tests in
``tests/test_dracula.py``.

Run with:  uv run pytest tests/r3con/test_dracula.py
"""

from __future__ import annotations

from evals.r3con.harness import dracula


def test_constants() -> None:
    assert dracula.NAME == "Dracula"
