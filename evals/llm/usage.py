"""Per-run LLM call capture via a LiteLLM CustomLogger.

Most baselines here reach their LLM through LiteLLM, whether they call it
directly or through a vendored library. Hooking `litellm.callbacks` with a
``CustomLogger`` is the only baseline-agnostic place that CAN see every such call
(sync + async, completion + embedding) without touching the baseline's own code, so
this module captures the calls uniformly — but that hook is LOSSY for sync completions
(``usage_envelope`` below has the measured number), which is exactly why the
per-baseline seams exist. Every baseline runner opens a ``usage_scope`` around its
``run_one`` as the harness-wide fallback; a baseline whose own LLM/embedding seam
accumulates usage deterministically hands that record back instead, and the runner
prefers it.

We save the per-call **token cost** — not prompts or responses. For each
successful call we record ``{model, usage}``, where ``usage`` is the full
provider usage object (all token fields, incl. nested ``*_tokens_details``).
Plus a ``total`` rollup at the top: per-model recursive numeric sum of every
usage field (so nested detail counters like ``prompt_tokens_details.cached_tokens``
and ``completion_tokens_details.reasoning_tokens`` accumulate too) + ``num_calls``.

So a scope yields::

    {
      "total": { "<model>": { "prompt_tokens", "completion_tokens",
                              "total_tokens", "num_calls",
                              "prompt_tokens_details": {...}, ... } },
      "calls": [ { "model": "<model>", "usage": {...} }, ... ]
    }

Note: ``litellm.success_callback`` (the older list-of-functions API) is NOT
what we want — it dispatches to *named* integrations (Langfuse, Helicone, etc.)
and won't fire for an arbitrary callable. The modern ``litellm.callbacks`` list
with a ``CustomLogger`` instance dispatches the way we need.

Usage::

    from evals.llm.usage import usage_scope

    with usage_scope() as usage:
        run_baseline()      # any LLM activity (sync or async, asyncio)
    # `usage` is now {"total": {...}, "calls": [...]}

The scope is per-task: the runner opens one around each `run_one`. Index-build cost
is accounted separately where a baseline keeps a shared index store — arag persists
its build embeddings to `_indices/<hash>/index_usage.json` so a reused index is not
re-counted per task. Per-call records are delivered to the active scope through
`contextvars.ContextVar`, which propagates across `asyncio.run` and async tasks, so
the async callback (`async_log_success_event`) lands in the same accumulator as the
sync one.
"""
from __future__ import annotations

import contextvars
import threading
from contextlib import contextmanager
from typing import Any, Iterator

import litellm
from litellm.integrations.custom_logger import CustomLogger


# The whole-scope accumulator: {"total": {model: rollup}, "calls": [{model, usage}]}.
RunUsage = dict[str, Any]


# The active per-scope accumulator. None means "no scope active" — the callback
# then silently does nothing (won't leak calls from incidental LLM activity
# outside any tracked baseline invocation).
_current_accumulator: contextvars.ContextVar[RunUsage | None] = contextvars.ContextVar(
    "_evals_usage_accumulator", default=None
)

# `litellm.callbacks` is module-global, and callbacks may fire from arbitrary
# threads (LiteLLM uses thread pools for async dispatch). Guard the accumulator
# update with a lock so concurrent appends don't lose calls.
_lock = threading.Lock()


# ============================================================
# Public API
# ============================================================


@contextmanager
def usage_scope() -> Iterator[RunUsage]:
    """Open a fresh accumulator for the duration of the block.

    Yields ``{"total": {...}, "calls": [...]}``; on exit, every successful
    LiteLLM call that happened inside the block has been appended to ``calls``
    and folded into the per-model ``total`` rollup.

    Nested scopes are isolated (each opens its own dict; the outer scope does
    NOT see the inner scope's calls) — the right thing for test isolation, and it
    keeps a scope opened around one phase (say an index build) from also landing in
    an enclosing per-task scope.
    """
    acc: RunUsage = {"total": {}, "calls": []}
    token = _current_accumulator.set(acc)
    try:
        yield acc
    finally:
        _current_accumulator.reset(token)


def total_tokens(usage: RunUsage) -> int:
    """Sum of ``total_tokens`` across every model in a scope's ``total`` rollup.

    Cheap convenience for log lines. For breakdowns, read ``usage["total"]`` (per
    model) or ``usage["calls"]`` (per call) directly.
    """
    total = (usage or {}).get("total", {}) if isinstance(usage, dict) else {}
    return sum(int(m.get("total_tokens", 0) or 0) for m in total.values())


def usage_envelope(response: Any) -> RunUsage:
    """Build a ``{total, calls}`` usage record from ONE response object, directly.

    The DETERMINISTIC alternative to ``usage_scope`` for a caller that holds the
    response in hand (a single ``litellm.completion``): litellm fires its success
    callback off the scope's thread / after the call returns for SYNC completions, so
    the scope-based capture drops ~a third of records (measured live; even serially).
    Reading ``response.usage`` here is exact and always present. That is why the
    per-baseline LLM / embedding seams accumulate one envelope per call and hand the
    result back, rather than relying on the callback. Output shape is identical to a
    ``usage_scope`` that saw exactly one call (``total[model]`` rollup with
    ``num_calls`` + ``calls[0]``), so the manifest and any reader of it consume it
    unchanged. The model key is ``response.model``
    (what actually produced the tokens — matches what the callback records).

    Returns an empty envelope (``{"total": {}, "calls": []}``) if the response
    carries no usage, mirroring the no-call case.
    """
    model = getattr(response, "model", None) or "<unknown>"
    usage_dict = _usage_to_dict(getattr(response, "usage", None))
    if not usage_dict:
        return {"total": {}, "calls": []}
    bucket: dict[str, Any] = {"num_calls": 1}
    _merge_numeric(bucket, usage_dict)
    return {"total": {model: bucket}, "calls": [{"model": model, "usage": usage_dict}]}


# ============================================================
# Internal: the LiteLLM callback
# ============================================================


def _on_litellm_success(
    kwargs: dict[str, Any],
    response_obj: Any,
    start_time: Any,
    end_time: Any,
) -> None:
    """Capture one successful call's token cost into the active scope (if any).

    Used by both the sync and async paths of ``UsageLogger``. Kept standalone so
    tests can invoke it directly without constructing a CustomLogger. No-ops when
    no ``usage_scope()`` is active.

    Records ``{model, usage}`` per call (usage = the full provider usage object,
    incl. nested token details) and folds the usage into the per-model rollup.
    """
    acc = _current_accumulator.get()
    if acc is None:
        return

    model = kwargs.get("model") or "<unknown>"
    usage_dict = _usage_to_dict(getattr(response_obj, "usage", None))

    with _lock:
        acc["calls"].append({"model": model, "usage": usage_dict})
        bucket = acc["total"].setdefault(model, {"num_calls": 0})
        bucket["num_calls"] += 1
        if usage_dict:
            _merge_numeric(bucket, usage_dict)


class UsageLogger(CustomLogger):
    """LiteLLM ``CustomLogger`` that pipes both sync and async successful
    completion / embedding calls into the active ``usage_scope`` accumulator.

    ``litellm.callbacks`` (the modern API) calls ``log_success_event`` for sync
    completions and ``async_log_success_event`` for async/streaming. The older
    ``litellm.success_callback`` list dispatches only to *named* integrations —
    arbitrary functions there are not invoked, which is why we route through a
    ``CustomLogger`` instance.
    """

    def log_success_event(self, kwargs, response_obj, start_time, end_time):  # type: ignore[override]
        _on_litellm_success(kwargs, response_obj, start_time, end_time)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):  # type: ignore[override]
        _on_litellm_success(kwargs, response_obj, start_time, end_time)


# ============================================================
# Internal helpers
# ============================================================


def _usage_to_dict(usage: Any) -> dict[str, Any] | None:
    """The full usage object as a plain dict (all fields incl. nested
    ``*_tokens_details``), or None if absent."""
    if usage is None:
        return None
    if isinstance(usage, dict):
        return dict(usage)
    dumped = _safe_dump(usage)
    if dumped is not None:
        return dumped
    # Last resort: pull the canonical fields off an opaque object.
    return {f: _get_count(usage, f) for f in ("prompt_tokens", "completion_tokens", "total_tokens")}


def _safe_dump(obj: Any) -> Any:
    """``model_dump()`` a usage object (or pass a dict through), else None.
    Never raises — capture is best-effort and must not break the LLM call."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:  # noqa: BLE001 — capture must never break the call
            return None
    if isinstance(obj, dict):
        return dict(obj)
    return None


def _merge_numeric(into: dict[str, Any], src: dict[str, Any]) -> None:
    """Recursively sum the numeric leaves of ``src`` into ``into``.

    Sums top-level token counts AND nested detail counters
    (``prompt_tokens_details.cached_tokens``, ``completion_tokens_details.
    reasoning_tokens``, …). Non-numeric, non-dict values (None, strings) are
    skipped — the rollup is numbers only; the raw usage lives in ``calls``.

    """
    for k, v in (src or {}).items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            into[k] = (into.get(k) or 0) + v
        elif isinstance(v, dict):
            sub = into.get(k)
            if not isinstance(sub, dict):
                sub = {}
                into[k] = sub
            _merge_numeric(sub, v)


def _get_count(usage: Any, field: str) -> int:
    """Read `field` off `usage` whether it's an object or a dict; coerce to int."""
    if isinstance(usage, dict):
        value = usage.get(field, 0)
    else:
        value = getattr(usage, field, 0)
    return int(value or 0)


# ============================================================
# Registration (module-load side effect)
# ============================================================


def _register() -> None:
    """Insert a ``UsageLogger`` into ``litellm.callbacks`` if not already there.

    Idempotent under module reload (pytest sometimes triggers this). Skipping
    duplicate registration avoids double-counting.
    """
    cb_list = getattr(litellm, "callbacks", None)
    if cb_list is None:
        return  # LiteLLM API surface changed; bail rather than crash on import.
    if any(isinstance(cb, UsageLogger) for cb in cb_list):
        return
    cb_list.append(UsageLogger())


_register()
