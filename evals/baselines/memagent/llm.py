"""MemAgent LLM seam — the recurrent-memory calls over the harness transport.

REPLACES upstream ``quickstart.py``'s raw ``aiohttp`` POST to an OpenAI-compatible
endpoint with the SAME shape — one single-user-message completion returning text — but
routed through the harness's LiteLLM wrapper (vLLM / OpenAI, with provider routing,
``--seed`` and ``num_retries`` transport retries). Both call
types (every per-chunk memory update AND the final answer) go through ``complete``.

Deviations from upstream (also in PROVENANCE.md):
  • ``max_tokens`` = ``RECURRENT_MAX_NEW`` (upstream default **1024**) is KEPT and sent on
    every call — unlike readagent/structrag/arag (which drop the cap for reasoning
    models), MemAgent's cap is **load-bearing**: it bounds the memory, which is the whole
    point (fixed-window, linear processing). MemAgent's model is Qwen2.5-Instruct
    (non-thinking), so 1024 is not eaten by reasoning tokens.
  • Temperature is NOT pinned to upstream's 0.7 — it follows the served model/provider
    default (the harness convention shared by readagent / structrag / arag, so the
    method is measured "as served").

Usage is captured DETERMINISTICALLY (``usage_envelope`` per call → ``{total, calls}``)
since MemAgent fires many SYNC completions (the litellm-callback ``usage_scope`` path
under-counts those). Every call's full request + response is captured to ``full_calls``
→ written to ``calls.json`` by the runner. The loop is inherently sequential (each memory
update depends on the previous), so no locking is needed.
"""
from __future__ import annotations

from typing import Any

from evals.llm.chat import litellm_chat_completion_full
from evals.llm.usage import _merge_numeric, usage_envelope

# Per-call generation budget = the memory / answer size cap. Upstream RECURRENT_MAX_NEW.
_MAX_NEW = 1024


class MemAgentLLM:
    """``.complete(prompt) -> str`` seam over the harness LiteLLM wrapper. Accumulates
    per-model ``{total, calls}`` usage AND the full request/response of every call
    (``full_calls`` → ``calls.json``)."""

    def __init__(
        self,
        litellm_kwargs: dict[str, Any],
        *,
        seed: int | None = None,
        completion_params: dict[str, Any] | None = None,
        max_new: int = _MAX_NEW,
    ):
        self._litellm_kwargs = litellm_kwargs
        self._seed = seed
        self._completion_params = completion_params or {}
        self._max_new = max_new
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []

    def complete(self, prompt: str) -> str:
        """One single-user-message completion (upstream sends no system message),
        returning the model's text. ``max_tokens`` caps the output (the memory size)."""
        response = litellm_chat_completion_full(
            system_prompt="",                 # upstream sends one user message, no system role
            user_prompt=prompt,
            model=self._litellm_kwargs["model"],
            api_base=self._litellm_kwargs.get("api_base"),
            api_key=self._litellm_kwargs.get("api_key"),
            seed=self._seed,
            max_tokens=self._max_new,          # the memory / answer cap — load-bearing (fixed window)
            **self._completion_params,
        )
        self._record(prompt, response)
        return response.choices[0].message.content or ""

    def _record(self, prompt: str, response: Any) -> None:
        env = usage_envelope(response)
        message = getattr(response.choices[0], "message", None)
        entry: dict[str, Any] = {
            "request": {
                "model": self._litellm_kwargs["model"],
                "max_tokens": self._max_new,
                "seed": self._seed,
                "completion_params": self._completion_params,
                "prompt": prompt,
            },
            "content": getattr(message, "content", None),
        }
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            entry["reasoning_content"] = reasoning
        entry["response"] = _safe_dump(response)
        for model, bucket in env["total"].items():
            _merge_numeric(self._total.setdefault(model, {}), bucket)
        self._calls.extend(env["calls"])
        self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` cost record — the TOTAL for the task
        (every memory update + the final answer)."""
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
