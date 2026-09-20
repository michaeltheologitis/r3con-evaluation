"""Tests for `evals.settings`.

The legacy ``RunMetadata`` class has been removed (its responsibilities moved
into ``evals.baselines._common``'s config-hash-based log dir layout). This
file now pins ``_slug`` — the one helper that survived and is shared between
``settings`` and ``_common`` for filesystem-safe directory segments.
"""
from __future__ import annotations

from evals.settings import _slug


def test_slug_lowercases_and_collapses_spaces() -> None:
    assert _slug("Hello World") == "hello-world"


def test_slug_collapses_runs_of_hyphens_and_spaces() -> None:
    assert _slug("a   b---c") == "a-b-c"


def test_slug_strips_leading_and_trailing_separators() -> None:
    assert _slug("--Hello World--") == "hello-world"
    assert _slug("  spaced  ") == "spaced"


def test_slug_replaces_non_word_chars_with_hyphen() -> None:
    # Dots, slashes, colons all become hyphens; runs collapse.
    assert _slug("Qwen/Qwen3.5-9B") == "qwen-qwen3-5-9b"
    assert _slug("qwen3.5:9b") == "qwen3-5-9b"


def test_slug_preserves_internal_digits_and_underscores() -> None:
    # `\w` includes underscore; only colons/dots/slashes become hyphens.
    assert _slug("foo_bar_42") == "foo_bar_42"


def test_slug_empty_string_returns_empty() -> None:
    assert _slug("") == ""
