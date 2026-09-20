"""ReadAgent LLM seam — re-host of upstream's ``query_model`` over the harness transport.

REPLACES upstream's ``query_gpt_model`` / ``query_gemini_model`` (raw ``openai`` /
``google.generativeai`` clients, one user message, ``temperature=0.0``,
``max_tokens=512``) with the SAME shape — one single-user-message completion returning
text — but routed through the harness's LiteLLM wrapper (vLLM / OpenAI / Ollama, with
provider routing, ``--seed``, ``--config`` sampling, and ``num_retries`` transport
retries). Every ReadAgent stage (pagination, gisting, look-up, answer) calls
``complete(prompt)``.

Deviations from upstream (also in PROVENANCE.md):
  • ``max_tokens`` = **32768**, not upstream's 512: we evaluate REASONING models, and a
    thinking model can spend the whole budget on ``reasoning_content`` before emitting
    its answer → empty output (the same reason StructRAG/A-RAG bumped to 32768).
  • **No proactive truncation.** Upstream relies on the gist memory keeping prompts
    small. We send prompts as-is; an oversized prompt surfaces the server's
    ``ContextWindowExceededError``, which the runner records as a genuine model failure
    (folded into the analysis ⁺ view) — faithful to ReadAgent, which does not truncate.
  • Temperature is NOT pinned to upstream's 0.0 — it follows the served model/provider
    default unless a ``--config`` preset overrides it (the harness convention shared by
    structrag/arag, so the method is measured "as served").

Usage is captured DETERMINISTICALLY (``usage_envelope`` per call → ``{total, calls}``)
since ReadAgent fires many SYNC completions (the litellm-callback ``usage_scope`` path
under-counts those). Every call's full request + response is captured to ``full_calls``
→ written to ``calls.json`` by the runner.
"""
from __future__ import annotations

import threading
from typing import Any

from evals.llm.chat import litellm_chat_completion_full
from evals.llm.usage import _merge_numeric, usage_envelope

# Per-call generation budget. Upstream ships 512; we run reasoning models, which can
# burn the budget on reasoning_content before answering. Matches StructRAG/A-RAG (32768).
_MAX_TOKENS = 32768


class ReadAgentLLM:
    """Upstream-``query_model``-compatible ``.complete(prompt) -> str`` seam over the
    harness LiteLLM wrapper. Accumulates per-model ``{total, calls}`` usage AND the full
    request/response of every call (``full_calls`` → ``calls.json``)."""

    def __init__(
        self,
        litellm_kwargs: dict[str, Any],
        *,
        seed: int | None = None,
        completion_params: dict[str, Any] | None = None,
        max_tokens: int = _MAX_TOKENS,
    ):
        self._litellm_kwargs = litellm_kwargs
        self._seed = seed
        self._completion_params = completion_params or {}
        self._max_tokens = max_tokens
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []
        # ReadAgent gists/paginates pages concurrently (run with --gist-workers > 1), so
        # ``complete`` → ``_record`` fires from worker threads. Serialize the shared-state
        # mutations so no call's usage/record is lost to a race.
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        """One single-user-message completion (upstream sends no system message), returning
        the model's text. No truncation — an oversized prompt raises the server's
        context-window error, which the runner records as a failure."""
        response = litellm_chat_completion_full(
            system_prompt="",                 # upstream sends one user message, no system role
            user_prompt=prompt,
            model=self._litellm_kwargs["model"],
            api_base=self._litellm_kwargs.get("api_base"),
            api_key=self._litellm_kwargs.get("api_key"),
            seed=self._seed,
            # max_tokens=self._max_tokens,  # cap removed: send no max_tokens → vLLM fills the remaining window
            **self._completion_params,
        )
        self._record(prompt, response)
        return response.choices[0].message.content or ""

    def _record(self, prompt: str, response: Any) -> None:
        # Build the (read-only) per-call pieces OUTSIDE the lock; only the shared-state
        # mutations are serialized, so concurrent gisting threads can't lose a record.
        env = usage_envelope(response)
        message = getattr(response.choices[0], "message", None)
        entry: dict[str, Any] = {
            "request": {
                "model": self._litellm_kwargs["model"],
                "max_tokens": self._max_tokens,
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
        with self._lock:
            for model, bucket in env["total"].items():
                _merge_numeric(self._total.setdefault(model, {}), bucket)
            self._calls.extend(env["calls"])
            self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` cost record (the TOTAL for the task —
        pagination + gisting + look-up + answer all land here)."""
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
