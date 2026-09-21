"""StructRAG LLM seam — re-host of upstream ``utils/qwenapi.py`` (truncate-and-answer).

REPLACES upstream's ``QwenAPI`` with the SAME interface (``.response(text) -> str``) so
the vendored router/structurizer/utilizer run UNMODIFIED, but routes calls through the
harness's LiteLLM wrapper (vLLM / OpenAI / Ollama) and records usage + the full per-call
request/response (for ``calls.json``).

CONTEXT-WINDOW HANDLING — faithful to upstream's *behaviour* (truncate an oversized prompt
to fit the window and ANSWER; never surface a context-window error StructRAG wouldn't have
produced), and an IMPROVEMENT over upstream's hardcoded implementation:

  • Upstream proactively counted tokens with a **gpt2** tokenizer and clipped to a
    **hardcoded 128K** — both artifacts of their Qwen2-72B box (gpt2 mis-counts Qwen
    tokens; 128K isn't your model's window). We hardcode neither.
  • We clip **reactively only**: send the prompt; only if the server rejects it for length
    (``ContextWindowExceededError``) do we clip it to fit the *served* window and retry
    (up to 3 re-clips; transport blips are litellm ``num_retries`` — slightly larger
    budgets than upstream's shared 3-try pool, see PROVENANCE D2). The window is taken
    from the server's own error message, else
    the model's HF config, else a **262,144 (256K) fallback** for models with no HF
    tokenizer/config (e.g. OpenAI's gpt-5.4-nano). No proactive per-call token pass —
    simpler, and uniform across models.
  • ``max_tokens`` = **32,768** (Qwen/Alibaba's recommended max generation length), not
    upstream's 4,096 — a thinking model spends the early budget on reasoning before the
    answer, so 4,096 clips it (often to empty). vLLM auto-clamps generation to the room the
    prompt leaves, so we don't pre-reserve it.

Other differences: transport via ``litellm_chat_completion_full`` (litellm ``num_retries``
for transient errors) not raw ``requests.post``; seed from ``--seed`` not 1024; usage →
harness ``{total, calls}`` shape.

Full deviation ledger: evals/baselines/structrag/PROVENANCE.md
"""
from __future__ import annotations

import re
from typing import Any

import litellm

from evals.llm.chat import litellm_chat_completion_full
from evals.llm.usage import _merge_numeric, usage_envelope

_LITELLM_PROVIDER_PREFIXES = ("hosted_vllm/", "ollama_chat/", "ollama/", "openai/")

# Per-call completion budget (sent as `max_tokens`). Qwen/Alibaba's recommended max
# generation length for thinking models; NOT upstream's 4096 (which clips a thinking
# model's answer once reasoning eats the budget). PROVENANCE D7.
_MAX_NEW_TOKENS = 32768

# Fallback context window for models whose size we can't detect (no HF tokenizer/config
# and no number in the server's error) — e.g. OpenAI's gpt-5.4-nano. 256K. PROVENANCE D2.
_DEFAULT_WINDOW = 262144

_CONTEXT_MARGIN = 256        # headroom below the window (chat-template tokens + safety)
_MAX_TRUNCATE_RETRIES = 3    # reactive re-clip attempts on a length rejection (upstream: 3)
_CHARS_PER_TOKEN = 3.0       # conservative chars/token for tokenizer-free clipping (over-counts → safe)


def _strip_provider(model: str) -> str:
    """``hosted_vllm/Qwen/Qwen3.5-35B-A3B`` -> ``Qwen/Qwen3.5-35B-A3B`` (the HF id / local
    path the model's config is loaded from for window detection)."""
    for p in _LITELLM_PROVIDER_PREFIXES:
        if model.startswith(p):
            return model[len(p):]
    return model


def _parse_window(message: str) -> int | None:
    """The model's max context window parsed from a server length-error, e.g.
    *"This model's maximum context length is 262144 tokens…"* → 262144; else None."""
    m = re.search(r"maximum context length is (\d+)", message)
    return int(m.group(1)) if m else None


class StructRAGLLM:
    """Upstream-compatible ``.response()`` seam over the harness LiteLLM wrapper.

    Truncate-and-answer is **reactive only**: an oversized prompt is sent, and only if the
    server rejects it for length is it clipped to fit and retried. Records per-model usage
    (``self.usage`` = ``{total, calls}``) AND the full request/response of every call
    (``self.full_calls`` → written to ``calls.json``; each entry also surfaces the
    generated ``content`` — plus ``reasoning_content`` for a thinking model — as
    explicit fields beside the raw response dump)."""

    def __init__(self, litellm_kwargs, *, seed=None, completion_params=None, max_context_tokens=None):
        self._litellm_kwargs = litellm_kwargs
        self._seed = seed
        self._completion_params = completion_params or {}
        self._model_id = _strip_provider(litellm_kwargs["model"])
        self._window = max_context_tokens          # explicit override; else resolved lazily
        self._window_resolved = max_context_tokens is not None
        self._tokenizer: Any = None                # loaded only if/when we clip (lazy)
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []

    # ---- window resolution (server error → HF config → hardcoded 256K fallback) ----

    def _resolve_window(self) -> int:
        """The served model's context window for clipping. The model's HF config when
        available (e.g. Qwen3.5 = 262,144, nested under ``text_config``); else the 262,144
        fallback for models with no HF tokenizer/config (e.g. gpt-5.4-nano). Cached."""
        if self._window_resolved:
            return self._window or _DEFAULT_WINDOW
        self._window_resolved = True
        self._window = self._detect_window_from_hf() or _DEFAULT_WINDOW
        return self._window

    def _detect_window_from_hf(self) -> int | None:
        """The model's context window from its HF tokenizer/config, or None (no HF model —
        e.g. gpt-5.4-nano). Also caches the tokenizer for precise clipping."""
        try:
            from transformers import AutoConfig, AutoTokenizer  # lazy: evals[structrag]

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
            mml = getattr(self._tokenizer, "model_max_length", None)
            if isinstance(mml, int) and 0 < mml < 10_000_000:
                return mml
            cfg = AutoConfig.from_pretrained(self._model_id)
            for c in (cfg, getattr(cfg, "text_config", None)):  # MoE/MM configs nest it
                if c is None:
                    continue
                for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"):
                    v = getattr(c, attr, None)
                    if isinstance(v, int) and v > 0:
                        return v
        except Exception:  # noqa: BLE001 — no HF tokenizer → caller falls back to _DEFAULT_WINDOW
            self._tokenizer = None
        return None

    def _count_tokens(self, text: str) -> int | None:
        """Token count with the model's own tokenizer if one was loaded, else None."""
        if self._tokenizer is None:
            return None
        try:
            return len(self._tokenizer(text, add_special_tokens=False).input_ids)
        except Exception:  # noqa: BLE001
            return None

    def _clip(self, text: str, target_tokens: int) -> str:
        """Slice the prompt (keep the head) toward ``target_tokens`` — precise when a
        tokenizer is loaded, else a conservative char estimate. ALWAYS shrinks, so the
        retry loop makes progress."""
        if target_tokens <= 0:
            target_tokens = max(1, self._resolve_window() // 2)
        n = self._count_tokens(text)
        if n is not None and n > 0:
            ratio = min(0.95, target_tokens / n)                       # precise
        else:
            ratio = min(0.9, (target_tokens * _CHARS_PER_TOKEN) / max(1, len(text)))  # conservative
        return text[: max(1, int(len(text) * max(0.0, ratio)))]

    # ---- the call (reactive truncate-and-answer) ----

    def response(self, input_text: str, max_new_tokens: int = _MAX_NEW_TOKENS) -> str:
        """Upstream signature: one user-only completion, returns the text. Sends the prompt;
        only on a server context-length rejection does it clip to fit and retry."""
        for attempt in range(_MAX_TRUNCATE_RETRIES + 1):
            try:
                response = litellm_chat_completion_full(
                    system_prompt="",
                    user_prompt=input_text,
                    model=self._litellm_kwargs["model"],
                    api_base=self._litellm_kwargs.get("api_base"),
                    api_key=self._litellm_kwargs.get("api_key"),
                    seed=self._seed,
                    max_tokens=max_new_tokens,
                    **self._completion_params,
                )
                self._record(input_text, max_new_tokens, response)
                return response.choices[0].message.content or ""
            except Exception as exc:  # noqa: BLE001
                window = _parse_window(str(exc))
                is_ctx = window is not None or isinstance(exc, litellm.ContextWindowExceededError)
                if not is_ctx or attempt == _MAX_TRUNCATE_RETRIES:
                    raise  # not a length error, or out of retries → record it as a failure
                window = window or self._resolve_window()
                input_text = self._clip(input_text, window - max_new_tokens - _CONTEXT_MARGIN)
        raise RuntimeError("unreachable")  # the loop always returns or raises

    # ---- bookkeeping ----

    def _record(self, user_prompt: str, max_new_tokens: int, response: Any) -> None:
        env = usage_envelope(response)
        for model, bucket in env["total"].items():
            _merge_numeric(self._total.setdefault(model, {}), bucket)
        self._calls.extend(env["calls"])
        # The generated content (and a thinking model's reasoning_content) as explicit
        # per-call fields — a reader shouldn't have to dig them out of the response dump.
        message = getattr(response.choices[0], "message", None)
        entry: dict[str, Any] = {
            "request": {
                "model": self._litellm_kwargs["model"],
                "max_tokens": max_new_tokens,
                "seed": self._seed,
                "completion_params": self._completion_params,
                "user_prompt": user_prompt,
            },
            "content": getattr(message, "content", None),
        }
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning is not None:
            entry["reasoning_content"] = reasoning
        entry["response"] = _safe_dump(response)
        self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` inference-cost record for the task."""
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
        msg = getattr(response.choices[0], "message", None)
        return {"content": getattr(msg, "content", None)}
