"""Tests for `evals.llm.chat`.

`litellm.completion` is mocked end-to-end — these tests are about the *request shape*
the wrapper builds and the *return path* it takes, not about LiteLLM itself.

Coverage:
- `_enforce_strict_objects` recursion: nested dicts and lists, defaults set without
  overwriting an explicit value.
- `litellm_chat_completion_full` request shape: messages, optional api_base/api_key/seed,
  schema → response_format with json_schema/strict + recursive additionalProperties:false.
- `litellm_chat_completion` return-path: text when schema is None, parsed model when
  schema is given, ValueError when schema is given but the model returned empty text.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import BaseModel

from evals.llm.chat import (
    _enforce_strict_objects,
    litellm_chat_completion,
    litellm_chat_completion_full,
)


# ============================================================
# Test fixtures / helpers
# ============================================================


class _Answer(BaseModel):
    """Trivial schema used to exercise the structured-output path."""

    answer: str


def _fake_completion(text: str = "hello"):
    """Build a mock LiteLLM response with a single choice and a `choices[0].message.content`."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


# ============================================================
# _enforce_strict_objects
# ============================================================


def test_enforce_strict_objects_sets_default_on_object_node() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    _enforce_strict_objects(schema)
    assert schema["additionalProperties"] is False


def test_enforce_strict_objects_does_not_overwrite_explicit_setting() -> None:
    """If a caller already declared `additionalProperties: True`, leave it alone — `setdefault`
    is deliberate, otherwise a caller couldn't opt back in to permissive validation anywhere
    in their schema tree."""
    schema = {"type": "object", "additionalProperties": True}
    _enforce_strict_objects(schema)
    assert schema["additionalProperties"] is True


def test_enforce_strict_objects_recurses_into_nested_objects() -> None:
    schema = {
        "type": "object",
        "properties": {
            "inner": {"type": "object", "properties": {"deep": {"type": "string"}}},
        },
    }
    _enforce_strict_objects(schema)
    assert schema["additionalProperties"] is False
    assert schema["properties"]["inner"]["additionalProperties"] is False


def test_enforce_strict_objects_recurses_through_lists() -> None:
    """`anyOf` / `oneOf` / `prefixItems` use lists — recursion must descend into those."""
    schema = {
        "anyOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}},
            {"type": "object", "properties": {"b": {"type": "number"}}},
        ]
    }
    _enforce_strict_objects(schema)
    for branch in schema["anyOf"]:
        assert branch["additionalProperties"] is False


def test_enforce_strict_objects_ignores_non_object_leaves() -> None:
    """Strings, numbers, bools don't get touched."""
    schema = {"type": "string"}
    _enforce_strict_objects(schema)
    assert "additionalProperties" not in schema


def test_enforce_strict_objects_returns_same_object() -> None:
    """Caller relies on the return value to use the result inline."""
    schema = {"type": "object"}
    assert _enforce_strict_objects(schema) is schema


# ============================================================
# litellm_chat_completion_full — request shape
# ============================================================


def test_full_minimum_request_has_messages_and_model_only() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion()) as mock:
        litellm_chat_completion_full(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
        )

    request = mock.call_args.kwargs
    assert request["model"] == "openai/gpt-5.4-nano"
    assert request["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    # No optional knobs were touched.
    assert "api_base" not in request
    assert "api_key" not in request
    assert "seed" not in request
    assert "response_format" not in request
    # ...but transport-level retries are always set (default 3).
    assert request["num_retries"] == 3


def test_full_empty_system_prompt_omits_system_message() -> None:
    """An empty system_prompt means 'no system message' — the request carries ONLY
    the user message, not a `{role: system, content: ""}` placeholder. A general
    convenience for any caller with no genuine instruction (current baselines all
    pass a non-empty system prompt, so this is the defensive path)."""
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion()) as mock:
        litellm_chat_completion_full(
            system_prompt="",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
        )
    request = mock.call_args.kwargs
    assert request["messages"] == [{"role": "user", "content": "usr"}]


def test_full_forwards_optional_kwargs() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion()) as mock:
        litellm_chat_completion_full(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
            api_base="http://localhost:8555/v1",
            api_key="secret",
            seed=42,
            temperature=0.7,  # extra kwarg passes straight through
        )

    request = mock.call_args.kwargs
    assert request["api_base"] == "http://localhost:8555/v1"
    assert request["api_key"] == "secret"
    assert request["seed"] == 42
    assert request["temperature"] == 0.7


def test_full_caller_can_override_num_retries() -> None:
    """num_retries defaults to 3 but a caller may override it via kwargs."""
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion()) as mock:
        litellm_chat_completion_full(
            system_prompt="sys", user_prompt="usr", model="openai/gpt-5.4-nano",
            num_retries=7,
        )
    assert mock.call_args.kwargs["num_retries"] == 7


def test_full_schema_sets_strict_json_response_format() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion('{"answer": "x"}')) as mock:
        litellm_chat_completion_full(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
            schema=_Answer,
        )

    rf = mock.call_args.kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "_Answer"
    assert rf["json_schema"]["strict"] is True
    # _enforce_strict_objects ran on the schema body.
    assert rf["json_schema"]["schema"]["additionalProperties"] is False


# ============================================================
# litellm_chat_completion — text vs schema return path
# ============================================================


def test_completion_returns_text_when_no_schema() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion("hello world")):
        out = litellm_chat_completion(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
        )

    assert out == "hello world"


def test_completion_returns_parsed_model_when_schema_given() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion('{"answer": "42"}')):
        out = litellm_chat_completion(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
            schema=_Answer,
        )

    assert isinstance(out, _Answer)
    assert out.answer == "42"


def test_completion_raises_on_empty_text_with_schema() -> None:
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion("")):
        with pytest.raises(ValueError, match="empty text"):
            litellm_chat_completion(
                system_prompt="sys",
                user_prompt="usr",
                model="openai/gpt-5.4-nano",
                schema=_Answer,
            )


def test_completion_returns_empty_string_when_no_schema_and_empty_response() -> None:
    """Without a schema, empty content is *not* an error — it's just an empty string.
    The `or ""` guard turns a None content into ""."""
    with patch("evals.llm.chat.litellm.completion", return_value=_fake_completion("")):
        out = litellm_chat_completion(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
        )
    assert out == ""


def test_completion_treats_none_content_as_empty_string() -> None:
    """LiteLLM can return `message.content=None` (tool-call responses, refusals).
    The `text = ... or ""` guard must convert that to `""`, not propagate None."""
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))])
    with patch("evals.llm.chat.litellm.completion", return_value=response):
        out = litellm_chat_completion(
            system_prompt="sys",
            user_prompt="usr",
            model="openai/gpt-5.4-nano",
        )
    assert out == ""
