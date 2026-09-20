"""Dynamic, per-model token accounting for the A-RAG agent — the sanctioned
deviation from upstream (PROVENANCE D2).

Upstream's ``BaseAgent`` hardcodes its context stop-gate to ``tiktoken
gpt-4o`` and ``max_token_budget=128_000`` ([agent/base.py:31,33,87] in upstream),
i.e. the running message history is counted in *gpt-4o* tokens and the agent
force-answers at a *fixed* 128k regardless of the served model. That's wrong when
we evaluate other models: a 1M-window model would be capped at 128k for no reason,
and a non-OpenAI model's tokens are mis-counted by the gpt-4o BPE.

So we make it **dynamic by model**:

  • **token counter** — ``tiktoken`` for OpenAI models, the model's **HF tokenizer**
    for open-source models (lazy, optional), else a char/≈4 estimate.
  • **context window** — ``litellm.get_model_info`` (knows OpenAI windows), else the
    model's **HF config** (``max_position_embeddings`` & friends), else a 128k floor.
  • **budget** = window − generation-reserve − safety, so there's room left for the
    answer when the agent stops retrieving.

These resolve a ``token_counter`` + ``max_token_budget`` that are injected into
``AragAgent`` (a thin ``BaseAgent`` subclass overriding only the token accounting);
the vendored ReAct loop is otherwise byte-for-byte. The window/HF detection mirrors
``structrag.llm`` so behaviour is consistent across baselines.
"""
from __future__ import annotations

from typing import Any, Callable

from evals.baselines.arag.upstream.agent.base import BaseAgent

_LITELLM_PROVIDER_PREFIXES = ("hosted_vllm/", "ollama_chat/", "ollama/", "openai/")

# A-RAG's original hardcoded budget, kept only as the FLOOR for models whose window we
# can't detect (no litellm entry, no HF config) — so we never cap *below* what upstream
# would have, and never run away unbounded either.
_FALLBACK_WINDOW = 128_000
# Room reserved below the window for the agent's answer generation, so "stop retrieving"
# leaves space to actually answer. MUST stay >= llm._MAX_TOKENS (the per-call generation
# cap), else a near-budget answer re-clamps on long-context tasks. Both are 32768 (raised
# from 16384 for reasoning models — see llm._MAX_TOKENS).
_GENERATION_RESERVE = 32_768
_BUDGET_SAFETY = 1_024
_CHARS_PER_TOKEN = 4.0  # conservative estimate when no real tokenizer is available


def _strip_provider(model: str) -> str:
    """``hosted_vllm/Qwen/Qwen3.5-35B-A3B`` → ``Qwen/Qwen3.5-35B-A3B`` (the HF id /
    bare model name)."""
    for prefix in _LITELLM_PROVIDER_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix):]
    return model


# ============================================================
# Token counter
# ============================================================


def _tiktoken_encoding(name: str):
    """The tiktoken encoding for an OpenAI model, falling back to ``o200k_base``
    (gpt-4o/5 family) for models tiktoken's registry doesn't know yet."""
    import tiktoken
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _hf_tokenizer(model: str):
    """The model's HF tokenizer, or None (transformers absent / id unresolvable).
    Lazy — only an open-source model path hits this, and only if transformers is
    installed (``evals[arag]``)."""
    try:
        from transformers import AutoTokenizer  # lazy: evals[arag]
        return AutoTokenizer.from_pretrained(_strip_provider(model))
    except Exception:  # noqa: BLE001 — no HF tokenizer → caller falls back to char estimate
        return None


def resolve_token_counter(model: str) -> Callable[[str], int]:
    """A ``str -> int`` token counter for ``model``: tiktoken for OpenAI, the model's
    HF tokenizer for open-source, else a char/≈4 estimate (never gpt-4o-for-everything)."""
    if model.startswith("openai/"):
        enc = _tiktoken_encoding(_strip_provider(model))
        return lambda text: len(enc.encode(text))
    tokenizer = _hf_tokenizer(model)
    if tokenizer is not None:
        return lambda text: len(tokenizer.encode(text))
    return lambda text: max(1, round(len(text) / _CHARS_PER_TOKEN))


# ============================================================
# Context window
# ============================================================


def _litellm_model_info(model: str) -> dict[str, Any] | None:
    """``litellm.get_model_info`` for ``model`` (full or provider-stripped form), or
    None if litellm doesn't know it."""
    import litellm
    for candidate in (model, _strip_provider(model)):
        try:
            info = litellm.get_model_info(candidate)
            if info:
                return info
        except Exception:  # noqa: BLE001 — unknown model → try the next form / give up
            continue
    return None


def _hf_window(model: str) -> int | None:
    """The model's context window from its HF config (``max_position_embeddings`` and
    friends, incl. nested ``text_config`` for MoE/MM), or None. Mirrors
    ``structrag.llm._detect_window_from_hf``."""
    try:
        from transformers import AutoConfig  # lazy: evals[arag]
        cfg = AutoConfig.from_pretrained(_strip_provider(model))
        for sub in (cfg, getattr(cfg, "text_config", None)):
            if sub is None:
                continue
            for attr in ("max_position_embeddings", "n_positions", "seq_length", "max_sequence_length"):
                value = getattr(sub, attr, None)
                if isinstance(value, int) and value > 0:
                    return value
    except Exception:  # noqa: BLE001 — no HF config → caller falls back
        return None
    return None


def resolve_context_window(model: str) -> int:
    """The served model's context window: litellm's registry (OpenAI), else the
    model's HF config (open-source), else the 128k floor."""
    info = _litellm_model_info(model)
    if info:
        window = info.get("max_input_tokens") or info.get("max_tokens")
        if isinstance(window, int) and window > 0:
            return window
    hf = _hf_window(model)
    if hf:
        return hf
    return _FALLBACK_WINDOW


def context_budget(window: int) -> int:
    """The agent's stop-gate budget: the window minus room reserved for the answer."""
    return max(1, window - _GENERATION_RESERVE - _BUDGET_SAFETY)


# ============================================================
# The agent (thin subclass — dynamic token accounting only)
# ============================================================


class AragAgent(BaseAgent):
    """A-RAG's vendored ``BaseAgent`` with a model-aware context stop-gate.

    Overrides ONLY ``_calculate_message_tokens`` (to count with the injected
    per-model ``token_counter`` instead of the hardcoded gpt-4o BPE) and takes a
    dynamic ``max_token_budget``. The ReAct loop, tool dispatch, force-answer, and
    trajectory recording are the vendored upstream behaviour, unchanged.
    """

    def __init__(self, llm_client, tools, *, system_prompt: str,
                 token_counter: Callable[[str], int], max_token_budget: int,
                 max_loops: int = 15, verbose: bool = False):
        super().__init__(
            llm_client, tools, system_prompt=system_prompt,
            max_loops=max_loops, max_token_budget=max_token_budget, verbose=verbose,
        )
        self._token_counter = token_counter

    def _calculate_message_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = self._token_counter(self.system_prompt)
        for message in messages:
            content = message.get("content", "")
            if content:
                total += self._token_counter(str(content))
        return total
