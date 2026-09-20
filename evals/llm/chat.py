from __future__ import annotations

from typing import Any

import litellm
from pydantic import BaseModel

# Transport-level retries for every LLM call (litellm retries with exponential
# backoff, via tenacity, ONLY on transient errors — connection refused/reset,
# timeout, 5xx; deterministic errors like a 400 context-window-exceeded are NOT
# retried). One blip then no longer fails the whole task. Hardcoded; a caller can
# still override by passing `num_retries=` in kwargs.
_NUM_RETRIES = 3


def _enforce_strict_objects(schema: Any) -> Any:
    """Recursively set ``additionalProperties: false`` on every object node — required by OpenAI strict mode."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for v in schema.values():
            _enforce_strict_objects(v)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_strict_objects(item)
    return schema


def litellm_chat_completion_full(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    schema: type[BaseModel] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> Any:
    """Call an LLM via LiteLLM and return the full response object.

    ``model`` uses LiteLLM's provider-prefixed form, e.g.
    ``"hosted_vllm/Qwen/Qwen3-8B"`` against a vLLM endpoint,
    or ``"anthropic/claude-sonnet-4"`` against Anthropic.

    An **empty** ``system_prompt`` means "no system message" — for tasks with no
    genuine instruction (e.g. a long-context QA where the grounding document
    belongs in the user turn, not the system role), the system message is omitted
    entirely rather than sent as an empty string.

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.

    Every call gets ``num_retries`` (default ``_NUM_RETRIES`` = 3) so litellm
    retries transient failures (connection/timeout/5xx) with backoff; pass
    ``num_retries=`` in kwargs to override.
    """

    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    request: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **kwargs,
    }
    if api_base is not None:
        request["api_base"] = api_base
    if api_key is not None:
        request["api_key"] = api_key
    if seed is not None:
        request["seed"] = seed
    if schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": _enforce_strict_objects(schema.model_json_schema()),
                "strict": True,
            },
        }

    # setdefault: kwargs is spread into `request` above, so a caller-supplied
    # num_retries already won. Otherwise default to _NUM_RETRIES.
    request.setdefault("num_retries", _NUM_RETRIES)

    return litellm.completion(**request)


def litellm_chat_completion(
    *,
    system_prompt: str,
    user_prompt: str,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    schema: type[BaseModel] | None = None,
    seed: int | None = None,
    **kwargs: Any,
) -> str | BaseModel:
    """Call an LLM via LiteLLM and optionally parse structured output.

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.
    """

    response = litellm_chat_completion_full(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        api_base=api_base,
        api_key=api_key,
        schema=schema,
        seed=seed,
        **kwargs,
    )

    text = response.choices[0].message.content or ""

    if schema is None:
        return text

    if not text:
        raise ValueError("Structured output was requested, but model returned empty text.")

    return schema.model_validate_json(text)
