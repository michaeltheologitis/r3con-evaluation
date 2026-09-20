"""Stage 2: per-task schema proposal driver.

Calls the proposer LLM with the task, validates the emitted code
with :func:`evals.r3con.pipeline.stages.extractor.check_schema`, and on failure feeds
the error back to the proposer so it can correct itself. The retry
feedback channel is the whole point of ``check_schema`` raising
:class:`SchemaError` with a shaped message.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from evals.r3con.pipeline.logging_setup import get_logger
from evals.r3con.pipeline.runtime.llm import litellm_chat_completion
from evals.r3con.pipeline.stages.extractor import SchemaError, check_schema
from evals.r3con.pipeline.stages.summaries import render_summaries
from evals.r3con.pipeline.prompts import load_prompt
from evals.r3con.pipeline.runs import StageRun
from evals.r3con.pipeline.settings import settings

_log = get_logger("proposer")


# The proposer prompts instruct the model to wrap the Python schema source
# in ``<schema>...</schema>`` tags (matching the in-codebase convention used
# by the inference agent's ``<code>``/``<observation>`` blocks). XML-style
# tags are less ambiguous than markdown fences (no collision with backticks
# inside string literals or docstrings) and easier for humans to spot in
# transcripts. ``check_schema`` runs on the *body* between the tags.
_SCHEMA_TAG_RE = re.compile(
    r"<schema>\s*\n?(?P<body>.*?)\n?\s*</schema>",
    re.DOTALL | re.IGNORECASE,
)

# Fallback: some models wrap code in markdown fences (e.g. because they
# missed the tag instructions or because they emit both). Search for
# *any* ```python``` (or bare ``` ```) block anywhere in the response —
# the model might prefix it with a `Thought:` line or other prose.
_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<body>.*?)\n\s*```",
    re.DOTALL,
)

# The proposer prompts ask for a brief ``Thought:`` snippet before the
# ``<schema>`` block. We pull it out for the per-stage result.json so the
# schema-design reasoning is readable without reading the raw transcript.
# Captures everything after ``Thought:`` until the schema body delimiter
# (the ``<schema>`` tag, a markdown fence, or end-of-text).
_THOUGHT_RE = re.compile(
    r"Thought:\s*(?P<body>.*?)(?=\n\s*<schema>|\n\s*```|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_schema(text: str) -> str:
    """Extract Python source from the proposer's response.

    Resolution order:

    1. ``<schema>...</schema>`` tag body — what the prompts ask for.
    2. First markdown ```python``` (or bare ``` ```) fence found
       anywhere in the response — fallback when the model uses
       fences (often after a `Thought:` preamble) instead.
    3. Raw text — final fallback.

    The returned string is what ``check_schema`` will ``exec``. The
    ``Thought:`` preamble the prompt asks for (and any other prose
    outside the schema tags / fence) is automatically discarded.
    """
    if m := _SCHEMA_TAG_RE.search(text):
        return m.group("body").strip()
    if m := _FENCE_RE.search(text):
        return m.group("body").strip()
    return text.strip()


def _extract_thought(text: str) -> str | None:
    """Pull the ``Thought:`` snippet out of the proposer's response.

    Returns the prose after ``Thought:`` up to the schema body
    delimiter (the ``<schema>`` tag or a markdown fence), stripped.
    Returns ``None`` when no ``Thought:`` marker is present or when
    the captured body is empty.
    """
    m = _THOUGHT_RE.search(text)
    if not m:
        return None
    body = m.group("body").strip()
    return body or None


@dataclass
class ProposalAttempt:
    schema_code: str
    error: str | None  # ``None`` for the (final) successful attempt
    thought: str | None = None  # Optional 1-2 sentence rationale prefacing the schema


@dataclass
class ProposalResult:
    """Outcome of a successful :func:`propose_schema` call.

    ``attempts`` is the full history including the final successful attempt
    (its ``error`` is ``None``). Useful for logging — the proposer prompt and
    its retries are the most interesting signal when iterating on stage 2.
    """

    schema_code: str
    parse_cls: type[BaseModel]
    attempts: list[ProposalAttempt] = field(default_factory=list)


def _retry_prompt(task: str, prev_code: str, error: str) -> str:
    """Build the next user prompt after a rejected attempt.

    Shows the proposer its previous attempt and the validator's error so it
    can fix the specific failure rather than start over blindly.
    """
    return (
        f"Input:\n<task>\n{task}\n</task>\n"
        "Output:\n\n"
        "# Previous attempt (rejected by validator)\n"
        f"{prev_code}\n\n"
        "# Validator error\n"
        f"{error}\n\n"
        "# Emit a corrected version that addresses the error above. "
        "Output only the Python code defining the schema."
    )


def propose_schema(
    *,
    task: str,
    summaries: list[str] | None = None,
    model: str,
    prompt_version: str,
    max_attempts: int = settings.PROPOSER_MAX_ATTEMPTS,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> ProposalResult:
    """Propose a validated Pydantic schema for ``task``.

    Calls the proposer LLM up to ``max_attempts`` times; after each
    rejection, the prior attempt and the validator error are appended
    to the next user prompt so the model can correct itself. Default
    comes from ``settings.PROPOSER_MAX_ATTEMPTS``.

    Args:
        task: The task the schema must capture information for.
        summaries: The collection's final per-document summaries (from stage 1).
            Rendered into a "## Task-conditioned document summaries" block in the system prompt so
            the schema is grounded in what the documents actually contain, not the
            task's surface words alone. ``None``/empty → no summaries block (the
            proposer still sees only the task).
        model: LiteLLM provider-prefixed model string (e.g. ``"openai/gpt-5.4-nano"``).
        max_attempts: Cap on proposer calls. Defaults to
            ``settings.PROPOSER_MAX_ATTEMPTS``; pass a smaller value
            for a faster fail in tests, or a larger one if you expect
            the model to need more retry cycles.
        **llm_kwargs: Forwarded to :func:`litellm_chat_completion`.

    Returns:
        A :class:`ProposalResult` with the validated source, its ``Parse`` class,
        and the full attempt history.

    Raises:
        SchemaError: if every attempt fails validation. The exception's
            message references the last attempt's error.
        ValueError: if ``max_attempts`` is not positive.
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}. "
            f"Override via settings.PROPOSER_MAX_ATTEMPTS (current default: "
            f"{settings.PROPOSER_MAX_ATTEMPTS})."
        )

    system_prompt = load_prompt("proposer", version=prompt_version, summaries=render_summaries(summaries))
    user_prompt = f"Input:\n<task>\n{task}\n</task>\nOutput:"

    attempts: list[ProposalAttempt] = []
    last_error: str = ""
    for attempt_idx in range(max_attempts):
        _log.info("proposer: schema attempt %d/%d", attempt_idx + 1, max_attempts)
        raw_response = cast(
            str,
            litellm_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                run=run,
                kind="llm_call" if attempt_idx == 0 else "retry",
                **llm_kwargs,
            ),
        )
        schema_code = _extract_schema(raw_response)
        thought = _extract_thought(raw_response)
        try:
            parse_cls = check_schema(schema_code)
        except SchemaError as e:
            last_error = str(e)
            _log.info("proposer: attempt %d rejected — %s", attempt_idx + 1, last_error.splitlines()[0][:120])
            attempts.append(
                ProposalAttempt(schema_code=schema_code, error=last_error, thought=thought)
            )
            user_prompt = _retry_prompt(task, schema_code, last_error)
            continue
        _log.info("proposer: schema accepted on attempt %d (fields: %s)",
                  attempt_idx + 1, ", ".join(parse_cls.model_fields))
        attempts.append(ProposalAttempt(schema_code=schema_code, error=None, thought=thought))
        return ProposalResult(
            schema_code=schema_code,
            parse_cls=parse_cls,
            attempts=attempts,
        )

    raise SchemaError(
        f"propose_schema exhausted {max_attempts} attempts. "
        f"Last validator error: {last_error}"
    )
