from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


class Settings:
    # This file is <repo>/evals/r3con/pipeline/settings.py, so parents[3] is the repo root
    # (the runner resolves scripts/ and the child cwd from it).
    ROOT: Path = Path(__file__).resolve().parents[3]
    PKG_DIR: Path = Path(__file__).resolve().parent
    ENV_FILE: Path = ROOT / ".env"
    # Our own log root, a sibling of the baselines' logs/<benchmark>/<baseline>/ trees.
    # Run-folders stay opaque (<UTC-timestamp>_<hex>); the benchmark and the full run
    # config live inside each folder's manifest.json.
    LOGS_DIR: Path = ROOT / "logs" / "r3con"
    # Prompts and configs ship INSIDE the package (they are part of the method, not of
    # the repo layout), so they resolve the same from a checkout or an installed copy.
    PROMPTS_DIR: Path = PKG_DIR / "prompts"
    CONFIGS_DIR: Path = PKG_DIR / "configs"  # experiment configs + the sampling/ presets subfolder

    # Max documents processed concurrently within ONE task (the summaries
    # fan-out and the per-document extraction fan-out). Bounds eval --workers
    # (tasks) × this so total in-flight LLM calls stay under the served
    # endpoint's concurrency cap. Override via R3CON_DOC_WORKERS.
    DOC_WORKERS: int = 16

    # Hard cap on the inference agent's multi-turn loop.
    INFERENCE_MAX_TURNS: int = 30

    # Cap on the proposer's schema-validation retry loop.
    PROPOSER_MAX_ATTEMPTS: int = 5

    # Cap on the extractor's structured-output retry loop (one document).
    EXTRACTOR_MAX_ATTEMPTS: int = 5

    # Transport-level retries passed to every litellm call (exponential backoff
    # via tenacity) on transient errors: connection refused/reset, timeouts, 5xx.
    LLM_NUM_RETRIES: int = 10

    # Content-level re-rolls for an empty structured-output response (a 200 with
    # no JSON — e.g. a reasoning model that spent its budget on thinking). Each
    # re-roll perturbs the seed so a pinned-seed call gets a different roll.
    LLM_EMPTY_CONTENT_RETRIES: int = 3

    # Show the WHOLE parse in the codeact system prompt below this many tokens (tiktoken cl100k_base); above
    # it (a record flood — e.g. a 1735-row catalog) fall back to one sample per field
    # + a prominent note, so the prompt can't blow up. Most parses are tiny (median ~5
    # records), so the whole thing shows; this is just the flood guard.
    CODEACT_PARSE_MAX_TOKS: int = 16000


settings = Settings()

# Load .env into the process environment before anything reads it.
load_dotenv(settings.ENV_FILE)


# ---------- runtime resolvers ----------
#
# These are RUNTIME knobs only — they don't change a correct answer (just parallelism
# and resilience). Everything that shapes the OUTPUT (model, seed, summary rounds,
# prompt versions, sampling params) lives in a RunConfig (see :mod:`evals.r3con.pipeline.config`),
# not here.


def active_doc_workers() -> int:
    """Max documents processed concurrently within one task —
    ``R3CON_DOC_WORKERS`` else the default."""
    raw = os.environ.get("R3CON_DOC_WORKERS")
    return int(raw) if raw else settings.DOC_WORKERS


def settings_snapshot() -> dict[str, Any]:
    """The runtime knobs (parallelism + resilience caps) recorded in each task's
    manifest ``settings`` block — distinct from the output-shaping ``RunConfig``,
    which is recorded in its own ``config`` block."""
    return {
        "doc_workers": active_doc_workers(),
        "inference_max_turns": settings.INFERENCE_MAX_TURNS,
        "proposer_max_attempts": settings.PROPOSER_MAX_ATTEMPTS,
        "extractor_max_attempts": settings.EXTRACTOR_MAX_ATTEMPTS,
        "llm_num_retries": settings.LLM_NUM_RETRIES,
        "llm_empty_content_retries": settings.LLM_EMPTY_CONTENT_RETRIES,
        "codeact_parse_max_toks": settings.CODEACT_PARSE_MAX_TOKS,
    }
