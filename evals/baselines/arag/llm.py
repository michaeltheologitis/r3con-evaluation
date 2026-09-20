"""A-RAG LLM seam — re-host of upstream ``core/llm.LLMClient`` over the harness's
LiteLLM transport (PROVENANCE D2).

REPLACES upstream's ``LLMClient`` (raw ``requests.post`` to ``{base_url}/chat/completions``
+ a hardcoded ``PRICING`` table) with the SAME interface the vendored ``BaseAgent``
consumes — ``chat(messages, tools=…, temperature=…, max_tokens=…) -> {"message": <OpenAI
dict>, "cost", "input_tokens", "output_tokens", "raw_response"}`` — so the vendored ReAct
loop / tools / prompts run UNMODIFIED, but calls route through ``litellm.completion``
(vLLM / OpenAI / Ollama, with provider routing, ``--seed``, ``--config`` sampling, and
``num_retries`` transport retries).

The maintainer green-lit this re-seam: it changes the TRANSPORT, not A-RAG's method
(the agent still decides every tool call and when to answer). ``cost`` is best-effort
from litellm and unused by the harness (which prices via litellm on the captured token
counts); A-RAG's own ``PRICING`` table is dropped. Usage is accumulated DETERMINISTICALLY
(``usage_envelope`` per call → ``{total, calls}``, like ``structrag.StructRAGLLM``) because
the agent fires many SYNC completions, which the litellm-callback ``usage_scope`` path
under-counts. Every call's full request + response is captured to ``full_calls`` →
written to ``calls.json`` by the runner.

Full deviation ledger: evals/baselines/arag/PROVENANCE.md
"""
from __future__ import annotations

from typing import Any

import litellm

from evals.llm.usage import _merge_numeric, usage_envelope

# Transport-level retries (litellm backs off on transient connection/5xx; a 400
# context-window error is NOT retried). Matches evals.llm.chat's default.
_NUM_RETRIES = 3

# Per-call generation budget. A-RAG ships 16384; we raise it to 32768 because we run
# REASONING models — a thinking model can spend the whole 16384 on reasoning_content and
# emit an EMPTY answer (finish_reason=length, 0 content). Mirrors StructRAG's 32768 bump.
# Kept in lockstep with model_budget._GENERATION_RESERVE so the answer never re-clamps.
_MAX_TOKENS = 32768


def _clean_message(message: Any) -> dict[str, Any]:
    """The assistant message as a minimal OpenAI dict the agent appends and re-sends.

    ``model_dump()`` emits ``function_call: None`` / ``tool_calls: None`` scaffolding
    that some providers reject when echoed back in the next request, so we keep only
    ``role`` / ``content`` and ``tool_calls`` when actually present."""
    dumped = message.model_dump()
    cleaned: dict[str, Any] = {
        "role": dumped.get("role", "assistant"),
        "content": dumped.get("content"),
    }
    if dumped.get("tool_calls"):
        cleaned["tool_calls"] = dumped["tool_calls"]
    return cleaned


def _response_cost(response: Any) -> float:
    """Best-effort USD cost litellm attached to the response (the harness ignores it
    for accounting — it prices via litellm on the captured tokens — but the agent
    surfaces it in its trajectory)."""
    try:
        return float(getattr(response, "_hidden_params", {}).get("response_cost") or 0.0)
    except Exception:  # noqa: BLE001
        return 0.0


class AragLLM:
    """Tool-calling LLM client with the interface A-RAG's ``BaseAgent`` expects,
    over ``litellm.completion``. Accumulates ``{total, calls}`` usage + per-call
    full request/response for ``calls.json``."""

    def __init__(self, litellm_kwargs: dict[str, Any], *, seed: int | None = None,
                 completion_params: dict[str, Any] | None = None, max_tokens: int = _MAX_TOKENS):
        self._litellm_kwargs = litellm_kwargs
        self._seed = seed
        self._completion_params = completion_params or {}
        self._max_tokens = max_tokens
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None,
             temperature: float | None = None, max_tokens: int | None = None) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._litellm_kwargs["model"],
            "messages": messages,
            "max_tokens": max_tokens or self._max_tokens,
            "num_retries": _NUM_RETRIES,
            # --config sampling preset (temperature / top_p / extra_body / …). A
            # caller-supplied `temperature` (the force-answer call passes 0.0) wins below.
            **self._completion_params,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"
        if self._litellm_kwargs.get("api_base") is not None:
            request["api_base"] = self._litellm_kwargs["api_base"]
        if self._litellm_kwargs.get("api_key") is not None:
            request["api_key"] = self._litellm_kwargs["api_key"]
        if self._seed is not None:
            request["seed"] = self._seed

        response = litellm.completion(**request)
        message = response.choices[0].message
        self._record(messages, tools, request, response, message)
        usage = getattr(response, "usage", None)
        return {
            "message": _clean_message(message),
            "cost": _response_cost(response),
            "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "raw_response": response,
        }

    def _record(self, messages, tools, request, response, message) -> None:
        env = usage_envelope(response)
        for model, bucket in env["total"].items():
            _merge_numeric(self._total.setdefault(model, {}), bucket)
        self._calls.extend(env["calls"])
        entry: dict[str, Any] = {
            "request": {
                "model": self._litellm_kwargs["model"],
                "seed": self._seed,
                "max_tokens": request.get("max_tokens"),
                "completion_params": self._completion_params,
                "tools": [t.get("function", {}).get("name") for t in (tools or [])],
                "messages": messages,
            },
            "content": getattr(message, "content", None),
        }
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            entry["reasoning_content"] = reasoning
        tool_calls = _clean_message(message).get("tool_calls")
        if tool_calls:
            entry["tool_calls"] = tool_calls
        entry["response"] = _safe_dump(response)
        self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` completion-cost record for the task."""
        return {"total": self._total, "calls": self._calls}

    @property
    def full_calls(self) -> list[dict[str, Any]]:
        """Per-call full request + response, written verbatim to ``calls.json``."""
        return self._full_calls


def _safe_dump(response: Any) -> Any:
    """The full litellm response as a dict (best-effort)."""
    try:
        return response.model_dump()
    except Exception:  # noqa: BLE001
        message = getattr(response.choices[0], "message", None)
        return {"content": getattr(message, "content", None)}
