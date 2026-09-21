from __future__ import annotations

import re
from pathlib import Path


# ----- Model registry -----
# This is the project's central registry of DEFAULT model choices.
# CLI flags (`--model`, `--embedding-model`) override per-invocation; the
# values below are the defaults you get without any flag. A few model
# constants live outside this registry by design: a baseline that is
# inseparable from its own model (memagent's RL checkpoint; claude_code,
# which only runs Claude), the embeddings-always-OpenAI pin (hipporag), and
# the r3con pipeline's own `configs/default.yaml`. Anything else hardcoded
# elsewhere belongs here instead.

# The registry's small/cheap/deterministic-ish slot. NOTHING in this repo reads it
# today: all three benchmarks here (loong / corpusqa / dracula) are free-form
# generation graded by an LLM judge, so none of them needs a step that canonicalizes
# a reply onto a fixed answer vocabulary. Kept as the registry's "small model"
# default, and as the reference point JUDGE_MODEL is sized against.
PARSE_MODEL = "openai/gpt-5.4-nano"

# LLM used by each benchmark's judge (Loong / CorpusQA / Dracula — their
# score() / score_details() / score_batch()) to grade a free-form answer against the
# gold answer. Deliberately a notch larger than PARSE_MODEL: rating answer-equivalence
# on free-form text is the harder job. Note that nothing in this repo INVOKES those
# judges — the runners record only the model's raw output; grading happens in whatever
# reads `logs/` afterwards.
JUDGE_MODEL = "openai/gpt-5.4-mini"

# Default completion model: what `--model` falls back to in the runners that carry no
# default of their own (arag, codeact, hipporag, raptor, readagent, structrag). memagent
# and claude_code default to their own model instead (still overridable via `--model`),
# and rlm requires `--model`. gpt-5.4-nano matches PARSE_MODEL (the judge uses the
# slightly larger gpt-5.4-mini) — a cheap OpenAI default lets the project work
# out-of-the-box from a `.env` with just `OPENAI_API_KEY`. Cluster runs typically pass
# `--model hosted_vllm/Qwen/Qwen3-32B` (or similar) to route completions through a
# local vLLM endpoint instead.
DEFAULT_COMPLETION_MODEL = "openai/gpt-5.4-nano"

# Default embedding model for the baselines that embed: arag and raptor take it as the
# `--embedding-model` default; hipporag pins the same OpenAI model in its own connector.
# Per the project's embeddings-always-OpenAI rule, embeddings route to OpenAI by the
# model's own provider (the `OPENAI_API_KEY` in the environment) independently of the
# completion endpoint — so pointing `--model` at a local vLLM server does not move the
# embeddings. text-embedding-3-small is the cheapest production-grade OpenAI option
# (1536-dim, ~$0.02/1M tokens).
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"

# Fixed seed for the deterministic shuffle every benchmark's `get_task_ids`
# applies before returning, so a `limit=N` slice is a stable, representative
# sample (a spread across the dataset) rather than the first N in file order.
# Deliberately NOT exposed as a CLI/API parameter — the order must be
# reproducible everywhere. Used by `evals.benchmarks._sampling.order_task_ids`.
TASK_ID_SHUFFLE_SEED = 1234


def _slug(value: str) -> str:
    """Lowercase, collapse whitespace/hyphens, strip non-word chars, trim ends.

    Used by every code path that turns a model id, benchmark name, or baseline
    name into a filesystem-safe directory segment. Centralised here (rather
    than per-baseline) so the same string maps to the same directory across
    callers — and re-imported by `evals.baselines._common`.
    """
    value = re.sub(r"[^\w\s-]", "-", value.lower())
    return re.sub(r"[-\s]+", "-", value).strip("-_")


# LiteLLM provider prefixes, and the provider-agnostic canonical model id. These
# live here in the model registry (not in `baselines/_common`) so ANY layer can
# canonicalize a model id without an upward dependency — the benchmark judges use
# `canonical_model_id` for the grader slug they stamp on each score result, and the
# baselines for the `model` they record in each `run_config`, which every manifest
# carries — and which the content-addressed baselines (arag, structrag) also hash into
# the log-folder name.
# `baselines/_common` re-exports both for backward compatibility.
LITELLM_PROVIDER_PREFIXES = ("hosted_vllm/", "ollama_chat/", "ollama/", "openai/")


def canonical_model_id(model: str) -> str:
    """Provider-agnostic canonical id for a model.

    Strips the LiteLLM provider prefix and any org/namespace, then slugifies, so
    the same underlying model produces the same logged metadata (and therefore
    the same log folder) whether reached via vLLM, Ollama, or OpenAI.

    Examples:
        Qwen/Qwen3.5-9B              -> qwen3-5-9b
        hosted_vllm/Qwen/Qwen3.5-9B  -> qwen3-5-9b
        ollama_chat/qwen3.5:9b       -> qwen3-5-9b   (collapses with the vLLM form)
        openai/gpt-5-nano            -> gpt-5-nano
    """
    for prefix in LITELLM_PROVIDER_PREFIXES:
        if model.startswith(prefix):
            model = model[len(prefix):]
            break
    return _slug(model.rsplit("/", 1)[-1])


# Manual USD pricing for models LiteLLM doesn't price yet (e.g. a just-released model),
# keyed by the **canonical model id** (`canonical_model_id`) → (input_$_per_token,
# output_$_per_token). NOTHING in this repo reads this dict — there is no cost, scoring or
# reporting layer here. It is kept as the registry's price table for whatever reads
# `logs/` afterwards. Such a reader uses it as the fallback when `litellm.cost_per_token`
# raises on a model it does not price yet, so a new model shows a real cost instead of "?".
# Add a model as a one-liner; drop it once LiteLLM ships the price.
MANUAL_MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2e-6, 10e-6),   # $2 / 1M input tokens, $10 / 1M output tokens
}


class Settings:
    ROOT = Path(__file__).resolve().parent.parent
    PKG_DIR = Path(__file__).resolve().parent
    ENV_FILE = ROOT / ".env"
    LOGS_DIR = ROOT / "logs"
    LOONG_DIR = PKG_DIR / "benchmarks" / "loong" / "data"
    CORPUSQA_DIR = PKG_DIR / "benchmarks" / "corpusqa" / "data"


settings = Settings()
