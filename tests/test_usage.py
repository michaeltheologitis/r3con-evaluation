"""Tests for `evals.llm.usage`.

The module registers a LiteLLM `CustomLogger` at import time. These tests
exercise the callback path directly by calling `_on_litellm_success` with
synthetic LiteLLM-shaped arguments — no real LLM calls.

A scope yields ``{"total": {model: rollup}, "calls": [per-call records]}``. We
pin both: the per-model rollup (recursive numeric sum incl. nested token
details) AND the full per-call capture (messages, params, response, usage,
latency) — with credentials scrubbed.
"""
from __future__ import annotations

from types import SimpleNamespace

import litellm

from evals.llm.usage import (
    _on_litellm_success,
    total_tokens,
    usage_envelope,
    usage_scope,
)


def _resp(prompt=0, completion=0, total=0, **extra):
    """A fake LiteLLM response: `.usage` is a pydantic-like object (has
    `model_dump`, returning the full dict incl. nested details), and the
    response itself has `model_dump` — mirroring litellm's real shapes."""
    usage_dict = {"prompt_tokens": prompt, "completion_tokens": completion,
                  "total_tokens": total, **extra}
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            total_tokens=total, model_dump=lambda: usage_dict, **extra)
    body = {"choices": [{"message": {"content": "hi"}}], "usage": usage_dict}
    return SimpleNamespace(usage=usage, model_dump=lambda: body)


# ============================================================
# Registration
# ============================================================


def test_usage_logger_registered_in_litellm_callbacks() -> None:
    from evals.llm.usage import UsageLogger
    assert any(isinstance(cb, UsageLogger) for cb in litellm.callbacks)


def test_registration_is_idempotent() -> None:
    from evals.llm import usage as usage_mod
    usage_mod._register()
    usage_mod._register()
    count = sum(1 for cb in litellm.callbacks if isinstance(cb, usage_mod.UsageLogger))
    assert count == 1


# ============================================================
# Outside a scope, callback no-ops
# ============================================================


def test_callback_outside_any_scope_does_nothing() -> None:
    _on_litellm_success({"model": "openai/x"}, _resp(100, 50, 150), 0, 1)
    with usage_scope() as usage:
        pass
    assert usage == {"total": {}, "calls": []}


# ============================================================
# Per-model rollup (`total`)
# ============================================================


def test_callback_tallies_single_call_into_total() -> None:
    with usage_scope() as usage:
        _on_litellm_success({"model": "openai/gpt-5.4-nano"}, _resp(100, 50, 150), 0, 1)
    assert usage["total"] == {
        "openai/gpt-5.4-nano": {
            "num_calls": 1, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        }
    }


def test_callback_sums_multiple_calls_to_same_model() -> None:
    """Critical for graphrag: one build = many calls to the same model."""
    with usage_scope() as usage:
        for _ in range(3):
            _on_litellm_success({"model": "openai/gpt-5.4-nano"}, _resp(100, 50, 150), 0, 1)
    bucket = usage["total"]["openai/gpt-5.4-nano"]
    assert bucket["prompt_tokens"] == 300
    assert bucket["completion_tokens"] == 150
    assert bucket["total_tokens"] == 450
    assert bucket["num_calls"] == 3


def test_callback_keys_by_model_so_completion_and_embedding_are_separate() -> None:
    with usage_scope() as usage:
        _on_litellm_success({"model": "openai/gpt-5.4-nano"}, _resp(500, 200, 700), 0, 1)
        _on_litellm_success({"model": "openai/text-embedding-3-small"}, _resp(2000, 0, 2000), 0, 1)
    assert set(usage["total"]) == {"openai/gpt-5.4-nano", "openai/text-embedding-3-small"}
    assert usage["total"]["openai/gpt-5.4-nano"]["total_tokens"] == 700
    assert usage["total"]["openai/text-embedding-3-small"]["total_tokens"] == 2000


def test_rollup_sums_nested_token_details_recursively() -> None:
    """Provider usage carries nested detail counters (cached_tokens,
    reasoning_tokens); the rollup must sum them too, not drop them."""
    with usage_scope() as usage:
        for _ in range(2):
            _on_litellm_success(
                {"model": "openai/gpt-5.4-nano"},
                _resp(100, 50, 150,
                      prompt_tokens_details={"cached_tokens": 40},
                      completion_tokens_details={"reasoning_tokens": 10}),
                0, 1,
            )
    bucket = usage["total"]["openai/gpt-5.4-nano"]
    assert bucket["prompt_tokens_details"]["cached_tokens"] == 80
    assert bucket["completion_tokens_details"]["reasoning_tokens"] == 20


def test_callback_handles_response_with_no_usage() -> None:
    """A response without `.usage` still records the CALL (we save everything)
    and counts it in num_calls, but contributes no tokens to the rollup."""
    response = SimpleNamespace(model_dump=lambda: {"choices": []})  # no `.usage`
    with usage_scope() as usage:
        _on_litellm_success({"model": "openai/x"}, response, 0, 1)
    assert usage["total"] == {"openai/x": {"num_calls": 1}}  # counted, zero tokens
    assert len(usage["calls"]) == 1
    assert usage["calls"][0]["usage"] is None


def test_callback_handles_usage_as_dict() -> None:
    response = SimpleNamespace(usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                               model_dump=lambda: {"usage": {"total_tokens": 15}})
    with usage_scope() as usage:
        _on_litellm_success({"model": "openai/x"}, response, 0, 1)
    assert usage["total"]["openai/x"]["total_tokens"] == 15


def test_callback_handles_missing_model_kwarg() -> None:
    with usage_scope() as usage:
        _on_litellm_success({}, _resp(10, 5, 15), 0, 1)
    assert "<unknown>" in usage["total"]


# ============================================================
# Per-call cost capture
# ============================================================


def test_callback_saves_per_call_usage_only() -> None:
    """Each call saves just {model, usage} — the per-call token cost, with the
    usage object's full detail. No prompts / responses / params / latency."""
    with usage_scope() as usage:
        _on_litellm_success(
            {"model": "openai/gpt-5.4-nano", "messages": [{"role": "user", "content": "…"}],
             "optional_params": {"temperature": 0.7, "api_key": "sk-SECRET"}},
            _resp(100, 50, 150, prompt_tokens_details={"cached_tokens": 20}),
            0, 1,
        )
    assert len(usage["calls"]) == 1
    call = usage["calls"][0]
    assert set(call) == {"model", "usage"}                   # nothing else captured
    assert call["model"] == "openai/gpt-5.4-nano"
    assert call["usage"]["total_tokens"] == 150
    assert call["usage"]["prompt_tokens_details"]["cached_tokens"] == 20  # full token detail kept
    # No prompt / response / params anywhere in the record.
    assert "messages" not in call and "response" not in call and "params" not in call


# ============================================================
# Scope isolation
# ============================================================


def test_nested_scopes_are_isolated() -> None:
    with usage_scope() as outer:
        _on_litellm_success({"model": "openai/setup"}, _resp(100, 50, 150), 0, 1)
        with usage_scope() as inner:
            _on_litellm_success({"model": "openai/task"}, _resp(1, 1, 2), 0, 1)
        assert "openai/task" in inner["total"] and "openai/setup" not in inner["total"]
        assert "openai/setup" in outer["total"] and "openai/task" not in outer["total"]
        assert len(inner["calls"]) == 1 and len(outer["calls"]) == 1

    _on_litellm_success({"model": "openai/leaked"}, _resp(999, 999, 999), 0, 1)
    assert "openai/leaked" not in inner["total"]
    assert "openai/leaked" not in outer["total"]


def test_scope_reset_on_exception() -> None:
    try:
        with usage_scope():
            raise ValueError("boom")
    except ValueError:
        pass
    with usage_scope() as usage:
        pass
    assert usage == {"total": {}, "calls": []}


# ============================================================
# Concurrent asyncio calls (graphrag's production pattern)
# ============================================================


def test_callback_tallies_concurrent_asyncio_calls() -> None:
    import asyncio

    async def fake_llm_call(model: str) -> None:
        _on_litellm_success({"model": model}, _resp(1, 1, 2), 0, 1)

    async def driver() -> None:
        await asyncio.gather(*[fake_llm_call("openai/x") for _ in range(100)])

    with usage_scope() as usage:
        asyncio.run(driver())

    assert usage["total"]["openai/x"]["num_calls"] == 100
    assert usage["total"]["openai/x"]["total_tokens"] == 200
    assert len(usage["calls"]) == 100


# ============================================================
# total_tokens helper
# ============================================================


def test_total_tokens_sums_across_models() -> None:
    usage = {"total": {
        "openai/gpt-5.4-nano": {"total_tokens": 700, "num_calls": 1},
        "openai/text-embedding-3-small": {"total_tokens": 2000, "num_calls": 1},
    }, "calls": []}
    assert total_tokens(usage) == 2700


def test_total_tokens_empty_usage() -> None:
    assert total_tokens({"total": {}, "calls": []}) == 0
    assert total_tokens({}) == 0


# ============================================================
# usage_envelope — deterministic capture from one response (the bug fix)
# ============================================================


def test_usage_envelope_builds_total_and_calls_from_response() -> None:
    r = _resp(100, 50, 150, prompt_tokens_details={"cached_tokens": 20})
    r.model = "Qwen/Qwen3.6-35B-A3B"
    env = usage_envelope(r)
    assert env["total"] == {"Qwen/Qwen3.6-35B-A3B": {
        "num_calls": 1, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20}}}
    assert env["calls"] == [{"model": "Qwen/Qwen3.6-35B-A3B", "usage": {
        "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
        "prompt_tokens_details": {"cached_tokens": 20}}}]


def test_usage_envelope_is_identical_to_a_one_call_scope() -> None:
    """The whole point: the deterministic envelope == what usage_scope records if
    the callback DID fire once. So switching to it changes reliability, not values."""
    r = _resp(123, 45, 168); r.model = "M"
    env = usage_envelope(r)
    with usage_scope() as via_callback:
        _on_litellm_success({"model": "M"}, r, 0, 1)
    assert env == via_callback


def test_usage_envelope_empty_when_response_has_no_usage() -> None:
    r = SimpleNamespace(model="M", model_dump=lambda: {})  # no `.usage`
    assert usage_envelope(r) == {"total": {}, "calls": []}


def test_usage_envelope_unknown_model_when_missing() -> None:
    r = _resp(1, 1, 2)  # no `.model` set
    assert "<unknown>" in usage_envelope(r)["total"]
