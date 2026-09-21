"""HippoRAG LLM seam — routes the vendored pipeline's LLM calls through litellm.

HippoRAG drives EVERY LLM call — OpenIE NER, OpenIE triple extraction, the
recognition-memory triple filter (``rerank.DSPyFilter``, which despite the name is
just a baked-prompt LLM call — no runtime dspy), and the final QA reader — through a
single ``llm_model.infer(messages, **kwargs) -> (text, metadata, cache_hit)`` client.
Upstream that client is ``llm.openai_gpt.CacheOpenAI`` (raw ``openai`` SDK + a sqlite
cache). This seam REPLACES it with the SAME ``infer`` interface but routes through
litellm (OpenAI OR vLLM via ``--base-url``, provider prefixes, transport retries) and
captures usage + the full per-call request/response for ``calls.json`` — so all of
HippoRAG's internal calls are accounted for in one place.

Deviations from ``CacheOpenAI`` (ledgered in PROVENANCE.md):
  * D4 transport: litellm (``num_retries``) instead of the raw ``openai`` client.
  * D5 NO max-token cap: upstream sends ``max_completion_tokens`` (default 400) on
    OpenIE and a hardcoded 512 on the triple filter. A THINKING model spends the early
    budget on reasoning tokens (which count as completion tokens), so a small cap can
    truncate the JSON output before it is emitted. We send NO cap (matching what
    structrag/readagent/raptor do for reasoning models); the model's own default max
    applies, and HippoRAG's parsers regex-extract the JSON block from the reply, so
    reasoning-then-JSON still parses. This also removes the need to scale the cap with
    chunk size.
  * D6 NO sqlite response cache: the harness owns resumption (one run folder per task),
    so upstream's ``@cache_response`` is dropped — every call is a real call.
  * ``--seed`` is threaded onto every call.
"""
from __future__ import annotations

import threading
from typing import Any

import litellm

from evals.llm.usage import _merge_numeric, usage_envelope

_LITELLM_PROVIDER_PREFIXES = ("hosted_vllm/", "ollama_chat/", "ollama/", "openai/")

# Transport-level retries for transient errors (connection/timeout/5xx), matching
# evals.llm.chat's default. Deterministic errors (400s) are not retried.
_NUM_RETRIES = 3


class HippoRAGLLM:
    """Upstream-compatible ``infer()`` seam over litellm.

    Duck-types ``llm.base.BaseLLM`` closely enough for HippoRAG: it exposes
    ``infer(messages, **kwargs) -> (content, metadata, cache_hit)`` (the shape the
    ``@cache_response`` decorator produced upstream — a 3-tuple), plus ``llm_name``.
    Records per-model usage (``self.usage`` = ``{total, calls}``) and the full
    request/response of every call (``self.full_calls`` → ``calls.json``).
    """

    def __init__(self, litellm_kwargs, *, seed=None, completion_params=None):
        self._litellm_kwargs = litellm_kwargs
        self._model = litellm_kwargs["model"]
        self.llm_name = self._model  # some upstream paths read ``.llm_name``
        self._seed = seed
        self._completion_params = completion_params or {}
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []
        # HippoRAG's OpenIE fans NER + triple-extraction calls out across a
        # ThreadPoolExecutor, so `infer` (and thus `_record`) runs from many threads at
        # once. Guard the usage accumulation — the read-modify-write in `_merge_numeric`
        # would otherwise race and UNDERCOUNT tokens.
        self._usage_lock = threading.Lock()

    def infer(self, messages: list[dict], **kwargs) -> tuple[str, dict, bool]:
        """One chat completion over a raw ``messages`` list (system + few-shot +
        user). Returns ``(content, metadata, cache_hit=False)`` — the tuple shape
        HippoRAG's OpenIE / filter / reader expect. NO max-token cap is sent (D5).

        ``kwargs`` may carry upstream's ``model`` / ``max_completion_tokens`` /
        ``max_tokens`` / ``n`` / ``temperature`` / ``seed`` — we IGNORE the cap keys
        (D5) and honor the rest only when they don't conflict with the run's own
        seed.
        """
        request: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "num_retries": _NUM_RETRIES,
        }
        if self._litellm_kwargs.get("api_base") is not None:
            request["api_base"] = self._litellm_kwargs["api_base"]
        if self._litellm_kwargs.get("api_key") is not None:
            request["api_key"] = self._litellm_kwargs["api_key"]
        if self._seed is not None:
            request["seed"] = self._seed
        # Any generation params the run config carries apply to every internal call,
        # indexing and query alike (none are set today — the served defaults stand).
        request.update(self._completion_params)
        # Honor an upstream-requested temperature/n ONLY if the run didn't pin its own
        # in ``completion_params`` (so faithful defaults survive without letting the 512-cap etc.
        # leak back in). The cap keys are intentionally dropped (D5).
        for k in ("temperature", "n"):
            if k in kwargs and k not in request:
                request[k] = kwargs[k]

        response = litellm.completion(**request)
        content = response.choices[0].message.content or ""
        self._record(messages, request, response)

        usage = getattr(response, "usage", None)
        metadata = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "finish_reason": response.choices[0].finish_reason,
        }
        return content, metadata, False

    def _record(self, messages, request, response) -> None:
        env = usage_envelope(response)
        message = getattr(response.choices[0], "message", None)
        entry: dict[str, Any] = {
            "request": {
                "model": self._model,
                "seed": self._seed,
                "completion_params": self._completion_params,
                "messages": messages,
            },
            "content": getattr(message, "content", None),
        }
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            entry["reasoning_content"] = reasoning
        entry["response"] = _safe_dump(response)
        # Single critical section: OpenIE calls this from many threads at once.
        with self._usage_lock:
            for model, bucket in env["total"].items():
                _merge_numeric(self._total.setdefault(model, {}), bucket)
            self._calls.extend(env["calls"])
            self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` inference-cost record for the task
        (OpenIE + filter + reader all folded in). A snapshot under the lock, so a reader
        never sees a half-applied update from a concurrent OpenIE call."""
        with self._usage_lock:
            return {"total": {m: dict(b) for m, b in self._total.items()},
                    "calls": list(self._calls)}

    @property
    def full_calls(self) -> list[dict[str, Any]]:
        """Per-call full request + response, written verbatim to ``calls.json``."""
        with self._usage_lock:
            return list(self._full_calls)


def _safe_dump(response: Any) -> Any:
    try:
        return response.model_dump()
    except Exception:  # noqa: BLE001
        msg = getattr(response.choices[0], "message", None)
        return {"content": getattr(msg, "content", None)}
