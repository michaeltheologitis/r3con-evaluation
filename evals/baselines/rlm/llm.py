"""RLM token-usage + call capture seam.

RLM (the ``rlms`` PyPI package) is **not vendored** — it's a normal dependency (the
``evals[rlms]`` extra), and the maintainer's call is to ``uv add`` it rather than ship code
across. So there is no source to re-seam; instead we capture at RLM's LLM-client boundary at
**runtime**, the same maintainer-approved monkeypatch pattern used elsewhere.

Why capture at the client class level (not ``result.usage_summary``): RLM aggregates usage on a
per-completion ``LMHandler``, but recursive ``rlm_query`` sub-calls (and the depth>=max_depth
plain-completion fallback) construct their OWN clients/handlers, whose tokens never reach the
ROOT ``result.usage_summary`` (only ``total_cost`` is propagated upward, and that's ``None`` for a
local vLLM). So to count **every** token across all depths we wrap ``OpenAIClient._track_cost``
(invoked on every completion of every ``OpenAIClient`` instance — root, ``llm_query``, recursive
children, and fallbacks alike). That makes the token total complete and authoritative regardless
of RLM's internal handler routing.

``_track_cost`` sees the full response (usage + message content + ``reasoning_content``); a
threadlocal set around ``completion``/``acompletion`` carries the request messages so each captured
call also records what was sent (best-effort — ``rlm_query_batched`` fans sub-calls across threads,
and the threadlocal is per-thread, so pairing stays correct). RLM's own ``RLMLogger`` trajectory
(saved beside the manifest) remains the authoritative rich log (every turn, code block, REPL
output, and sub-call); this seam adds the harness-standard ``{total, calls}`` usage + ``calls.json``.

Everything routes to the vLLM endpoint via RLM's OpenAI-compatible client (``base_url``) — **no
OpenAI cloud, no litellm** (RLM does not use litellm).
"""
from __future__ import annotations

import threading
from typing import Any

from evals.llm.usage import _merge_numeric


def _to_messages(prompt: Any) -> Any:
    """Mirror ``OpenAIClient.completion``'s prompt→messages coercion for logging, **snapshotting**
    the messages. RLM mutates its ``message_history`` list IN PLACE across turns, so storing it by
    reference would make every captured call's ``request.messages`` show the FINAL history instead
    of what that call actually sent. A shallow copy of the list + each message dict freezes this
    call's prompt (string values are shared — cheap; existing messages aren't mutated after append)."""
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if isinstance(prompt, list):
        return [dict(m) if isinstance(m, dict) else m for m in prompt]
    return prompt


class RLMUsageCapture:
    """Context manager that monkeypatches ``rlm.clients.openai.OpenAIClient`` to record EVERY
    LLM call's ``{total, calls}`` usage + full request/response, then restores the originals.

    Use as ``with RLMUsageCapture() as cap: rlm.completion(...)`` then read ``cap.usage`` /
    ``cap.full_calls``. Thread-safe (RLM may fan sub-calls across threads)."""

    def __init__(self) -> None:
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._tls = threading.local()
        self._orig_completion = None
        self._orig_acompletion = None
        self._orig_track_cost = None
        self._cls = None

    # ----- the patched methods (bound at __enter__ to the real OpenAIClient) -----

    def _record(self, response: Any, model: str) -> None:
        """Record one completed call. Called from inside the patched ``_track_cost`` (which fires
        on every completion of every ``OpenAIClient``), so token counting is complete."""
        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
        message = response.choices[0].message if getattr(response, "choices", None) else None
        per_call = {
            "num_calls": 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        entry = {
            "model": model,
            "request": getattr(self._tls, "request", None),
            "content": getattr(message, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                      "total_tokens": total_tokens},
        }
        with self._lock:
            _merge_numeric(self._total.setdefault(model, {}), per_call)
            self._calls.append({"model": model, "usage": entry["usage"]})
            self._full_calls.append(entry)

    def __enter__(self) -> "RLMUsageCapture":
        from rlm.clients.openai import OpenAIClient  # import here so the dep stays optional

        self._cls = OpenAIClient
        self._orig_completion = OpenAIClient.completion
        self._orig_acompletion = OpenAIClient.acompletion
        self._orig_track_cost = OpenAIClient._track_cost

        cap = self
        orig_completion = self._orig_completion
        orig_acompletion = self._orig_acompletion
        orig_track = self._orig_track_cost

        def completion(self, prompt, model=None):  # noqa: ANN001
            cap._tls.request = {"model": model or getattr(self, "model_name", None),
                                "messages": _to_messages(prompt)}
            try:
                out = orig_completion(self, prompt, model)
                # Endpoint normalization: a vLLM served with a tool-call parser may
                # return content=None (allowed by the OpenAI schema); RLM's
                # find_code_blocks assumes str and crashes on None.
                return "" if out is None else out
            finally:
                cap._tls.request = None

        async def acompletion(self, prompt, model=None):  # noqa: ANN001
            cap._tls.request = {"model": model or getattr(self, "model_name", None),
                                "messages": _to_messages(prompt)}
            try:
                out = await orig_acompletion(self, prompt, model)
                return "" if out is None else out  # same None-content normalization
            finally:
                cap._tls.request = None

        def _track_cost(self, response, model):  # noqa: ANN001
            orig_track(self, response, model)  # preserve RLM's own accounting first
            try:
                cap._record(response, model)
            except Exception:  # noqa: BLE001 — capture must never break the run
                pass

        OpenAIClient.completion = completion
        OpenAIClient.acompletion = acompletion
        OpenAIClient._track_cost = _track_cost
        return self

    def __exit__(self, *exc: Any) -> bool:
        self._cls.completion = self._orig_completion
        self._cls.acompletion = self._orig_acompletion
        self._cls._track_cost = self._orig_track_cost
        return False

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` token record — complete across all depths/sub-calls."""
        with self._lock:
            return {"total": {m: dict(b) for m, b in self._total.items()},
                    "calls": list(self._calls)}

    @property
    def full_calls(self) -> list[dict[str, Any]]:
        """Per-call request + response (content + reasoning) — written to ``calls.json``."""
        with self._lock:
            return list(self._full_calls)
