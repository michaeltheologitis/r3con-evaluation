"""CodeAct (smolagents) token-usage + per-call capture seam.

smolagents is a normal dependency (the ``evals[codeact]`` extra), **not vendored** — its
``CodeAgent`` is used via the public API. The one thing we must add is complete cost capture.

Why a subclass (not the litellm callback): smolagents' ``LiteLLMModel`` routes every call through
``litellm.completion`` — so the project's ``usage_scope`` ``UsageLogger`` callback *would* see them
— BUT the SYNC-completion callback path silently drops ~a third of records (see
``evals/llm/usage.py``; this is why ``usage_envelope`` reads ``response.usage`` directly), and a
``CodeAgent`` run is many sequential SYNC calls. So we capture **deterministically** instead:
``LiteLLMModel.generate`` returns a ``ChatMessage`` whose ``.raw`` is the full litellm response, so
we read ``response.usage`` per call (exact, never dropped) and accumulate the harness-standard
``{total, calls}`` record + a verbatim ``calls.json``.

The model is otherwise **unchanged** — same litellm transport, so ``--model`` / ``--base-url`` /
``--api-key`` and the project's provider prefixes behave identically to every other litellm-routed
baseline; ``seed`` is injected as a model kwarg (applied to every completion, ``run.py``). The usage shape is byte-identical to ``usage_envelope`` / the callback (same
``_usage_to_dict`` + ``_merge_numeric`` helpers), so the manifest and any reader of it consume it
unchanged. See ``PROVENANCE.md``.
"""
from __future__ import annotations

from typing import Any

from smolagents import LiteLLMModel

from evals.llm.usage import _merge_numeric, _usage_to_dict


def _snapshot_messages(messages: Any) -> list[dict[str, Any]]:
    """Freeze a call's request messages for ``calls.json``.

    smolagents builds a fresh ``input_messages`` list each step from memory, so cross-call
    aliasing is unlikely — but we copy defensively (a logged call must show exactly what THAT
    call sent). ``ChatMessage`` dataclasses become ``{role, content}`` dicts; plain dicts are
    shallow-copied. ``content`` may be a string OR a list of content-part dicts (smolagents'
    structured form) — both are JSON-serialisable, passed through as-is.
    """
    out: list[dict[str, Any]] = []
    for m in messages or []:
        if isinstance(m, dict):
            out.append({k: m[k] for k in ("role", "content") if k in m})
        else:
            role = getattr(m, "role", None)
            out.append({"role": getattr(role, "value", role), "content": getattr(m, "content", None)})
    return out


class CodeActModel(LiteLLMModel):
    """A smolagents ``LiteLLMModel`` that records EVERY completion's ``{total, calls}`` usage +
    full request/response. Read :pyattr:`usage` / :pyattr:`full_calls` after the agent run.

    Token counting is deterministic — it reads ``result.raw.usage`` (the full litellm usage
    object, incl. nested ``*_tokens_details`` like ``reasoning_tokens`` / ``cached_tokens``) per
    call — so nothing is dropped across a long multi-step CodeAgent run.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []

    def generate(self, messages: Any, **kwargs: Any) -> Any:  # smolagents calls model.generate(messages, stop_sequences=...)
        result = super().generate(messages, **kwargs)

        raw = getattr(result, "raw", None)  # the full litellm ModelResponse
        usage_dict = _usage_to_dict(getattr(raw, "usage", None))
        # Key the rollup on the model that actually produced the tokens (matches
        # usage_envelope + the callback), falling back to the requested id.
        model_key = getattr(raw, "model", None) or self.model_id or "<unknown>"

        bucket = self._total.setdefault(model_key, {"num_calls": 0})
        bucket["num_calls"] += 1
        if usage_dict:
            _merge_numeric(bucket, usage_dict)
        self._calls.append({"model": model_key, "usage": usage_dict})

        message = raw.choices[0].message if getattr(raw, "choices", None) else None
        req: dict[str, Any] = {"messages": _snapshot_messages(messages)}
        if kwargs.get("stop_sequences"):
            req["stop_sequences"] = kwargs["stop_sequences"]
        self._full_calls.append({
            "model": model_key,
            "request": req,
            "content": getattr(result, "content", None),
            "reasoning_content": getattr(message, "reasoning_content", None),
            "usage": usage_dict,
        })
        return result

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` token record — complete across every step."""
        return {"total": {m: dict(b) for m, b in self._total.items()}, "calls": list(self._calls)}

    @property
    def full_calls(self) -> list[dict[str, Any]]:
        """Per-call request + response (content + reasoning + usage) — written to ``calls.json``."""
        return list(self._full_calls)
