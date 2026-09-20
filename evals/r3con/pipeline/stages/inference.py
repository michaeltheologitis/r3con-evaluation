"""Stage 4 — the inference stage: two strategies over a parse.

- :func:`infer_codeact` — the multi-turn CodeAct loop. Delegates the loop to
  :func:`evals.r3con.pipeline.runtime.codeact.run_codeact` with the parse bound as the
  Python variable ``parse`` and the ``inference/codeact`` prompt rendered with
  the parse's schema + sample records. Commits via ``final_answer(x)``.
- :func:`infer_llm` — a single LLM call over the parse JSON + document context
  (the ``inference/llm`` prompt). The "signal-in-the-parse ceiling": how much
  can a plain LLM read out of the parse without a multi-turn loop?

The CodeAct *loop* itself lives in :mod:`evals.r3con.pipeline.runtime.codeact`; this module
holds only the inference-specific glue — prompt rendering, the ``parse``
binding, and the sample-record helper that feeds the prompt.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, cast

from pydantic import BaseModel

from evals.r3con.pipeline.prompts import load_prompt
from evals.r3con.pipeline.runs import StageRun
from evals.r3con.pipeline.runtime.codeact import (
    DEFAULT_EXEC_TIMEOUT_S,
    CodeActResult,
    run_codeact,
)
from evals.r3con.pipeline.runtime.llm import litellm_chat_completion
from evals.r3con.pipeline.settings import settings
from evals.r3con.pipeline.stages.summaries import render_summaries


class _LazyStr:
    """A value that defers an expensive string render until something ``str()``s it.

    Jinja only stringifies a template variable it actually references, so passing
    one of these as a prompt kwarg means the work runs only for prompt versions
    that use that variable. Used for the full-parse JSON dump (``parse_json``),
    which the active samples-based CodeAct prompts never reference.
    """

    __slots__ = ("_render",)

    def __init__(self, render: Callable[[], str]) -> None:
        self._render = render

    def __str__(self) -> str:
        return self._render()


def _sample_record_per_field(parse_dict: Any) -> str:
    """Render one EXAMPLE record per top-level list field — *not* the whole parse.

    The agent has the full parse bound as the Python variable ``parse`` in its
    sandbox executor; dumping the whole JSON into the prompt context (a) bloats
    huge floods like 766-record celebration lists / 1735-record ISO catalogs
    well past useful, and (b) defeats the print-and-reflect loop by giving the
    model the answer in-context.

    This helper renders one sample record per list field (the first record),
    in JSON for unambiguous shape — enough for the model to write code that
    indexes correctly. For non-list top-level fields the value renders
    directly. Empty lists are noted as such.
    """
    if not isinstance(parse_dict, dict):
        return f"(parse is not a JSON object — type={type(parse_dict).__name__})"
    if not parse_dict:
        return "(parse is empty — no top-level fields)"
    # ensure_ascii=False so non-ASCII values (e.g. CJK) reach the model as readable text
    # in the prompt, not \uXXXX escapes (which also degrade non-ASCII-task quality).
    lines: list[str] = []
    for key, value in parse_dict.items():
        if isinstance(value, list):
            if not value:
                lines.append(f"`{key}` (empty list — no example to show)")
            else:
                example = json.dumps(value[0], indent=2, default=repr, ensure_ascii=False)
                lines.append(f"`{key}[0]` (1 of {len(value)} record(s)):\n```json\n{example}\n```")
        elif isinstance(value, dict):
            example = json.dumps(value, indent=2, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (dict):\n```json\n{example}\n```")
        else:
            example = json.dumps(value, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (scalar): {example}")
    return "\n\n".join(lines)


def _tag_source_documents(parse_dict: Any, source_docs: dict[str, list[int]] | None) -> Any:
    """Stamp each list-field record with the 1-based document it was extracted from
    (``"document": N``), from the extractor's ``source_docs`` provenance.

    The pipeline computes this mapping at merge time (each record aligns 1:1 with
    ``source_docs[field]`` by construction) but otherwise drops it at the inference
    boundary, leaving the model unable to say *which* document a fact came from. The label
    matches :func:`evals.r3con.pipeline.stages.summaries.render_summaries`'s "Document N", so the parse
    and the summaries share one document-id space (for Loong legal, Document N is the
    document's own ``《判决文书N》`` header). No-op without ``source_docs`` or for a non-dict
    parse. Mutates and returns ``parse_dict`` (``document`` first, so it reads first)."""
    if not source_docs or not isinstance(parse_dict, dict):
        return parse_dict
    for field, idxs in source_docs.items():
        recs = parse_dict.get(field)
        if not isinstance(recs, list):
            continue
        for k in range(min(len(recs), len(idxs))):
            if isinstance(recs[k], dict) and "document" not in recs[k]:
                recs[k] = {"document": idxs[k] + 1, **recs[k]}
    return parse_dict


@functools.lru_cache(maxsize=1)
def _parse_token_encoding():  # noqa: ANN202 — returns a tiktoken Encoding
    """tiktoken encoding for the parse-size guard — model-agnostic (cl100k_base), cached."""
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    """Approximate token count of ``text`` (tiktoken cl100k_base). It's only a size guard, so
    the exact tokenizer doesn't matter; ``disallowed_special=()`` so arbitrary text never errors."""
    return len(_parse_token_encoding().encode(text, disallowed_special=()))


def _render_parse_for_codeact(parse_dict: Any) -> str:
    """The `parse` view embedded in the codeact system prompt: the WHOLE parse as JSON when it
    fits (the common case), else a prominent "this is only a sample" note + one sample record
    per field. Either way the full parse is also bound as the `parse` variable for the agent to
    compute over. The cap is ``settings.CODEACT_PARSE_MAX_TOKS`` (tiktoken token count; a flood
    guard for huge catalogs)."""
    full = json.dumps(parse_dict, indent=2, ensure_ascii=False, default=repr)
    if _count_tokens(full) <= settings.CODEACT_PARSE_MAX_TOKS:
        return full
    n = sum(len(v) for v in parse_dict.values() if isinstance(v, list)) if isinstance(parse_dict, dict) else 0
    note = (
        "**IMPORTANT — the block below is only a SAMPLE of the parse, NOT the full data.** "
        f"The full parse is large ({n} record(s)), so only ONE example record per field is shown "
        "here, to convey the shape. The COMPLETE parse is bound as the `parse` variable — do not "
        "assume these samples are all the records; read the rest with `print(...)` before answering."
    )
    return note + "\n\n" + _sample_record_per_field(parse_dict)


def infer_codeact(
    *,
    task: str,
    schema_code: str,
    parsed: BaseModel | dict[str, Any],
    model: str,
    prompt_version: str,
    summaries: list[str] | None = None,
    source_docs: dict[str, list[int]] | None = None,
    max_turns: int = settings.INFERENCE_MAX_TURNS,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> CodeActResult:
    """Answer ``task`` over ``parsed`` with the multi-turn CodeAct loop.

    The LLM never sees the long source text — only the task, the schema
    source (so it knows the parse's shape), a sample record per top-level field,
    and the collection's **document summaries** (from stage 1).
    The full parse is bound as the Python variable ``parse`` in the sandbox; the
    agent inspects it via ``print(...)`` across turns and commits via
    ``final_answer(x)``.

    The system prompt carries everything immutable across the loop's turns; the
    user message is the bare task. Loop mechanics live in
    :func:`evals.r3con.pipeline.runtime.codeact.run_codeact`.
    """
    parse_dict = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
    parse_dict = _tag_source_documents(parse_dict, source_docs)

    summaries_block = render_summaries(summaries)
    # The codeact prompt variables — Jinja renders only what the active template references:
    #   parse_block  : the whole parse (or samples + a note if huge) — current (v2)
    #   samples_block: one sample record per field — older (v1)
    #   parse_json   : the full parse dump, lazy so the huge case is computed only on demand
    parse_block = _render_parse_for_codeact(parse_dict)
    samples_block = _sample_record_per_field(parse_dict)
    parse_json = _LazyStr(lambda: json.dumps(parse_dict, indent=2, ensure_ascii=False))
    system_prompt = load_prompt(
        "inference/codeact",
        version=prompt_version,
        task=task,
        schema_code=schema_code,
        summaries=summaries_block,
        parse_json=parse_json,
        samples_block=samples_block,
        parse_block=parse_block,
    )

    return run_codeact(
        system_prompt=system_prompt,
        user_message=f"Input:\n<task>\n{task}\n</task>",
        model=model,
        variables={"parse": parse_dict},
        max_turns=max_turns,
        timeout_s=timeout_s,
        additional_authorized_imports=additional_authorized_imports,
        run=run,
        **llm_kwargs,
    )


def infer_llm(
    *,
    task: str,
    parsed: BaseModel | dict[str, Any],
    model: str,
    prompt_version: str,
    summaries: list[str] | None = None,
    source_docs: dict[str, list[int]] | None = None,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> str:
    """LLM inference: one LLM call. The system prompt carries the document summaries and
    the parsed records (each tagged with its source document) plus the instruction to use
    both; the user message is the bare task.

    Returns the model's response text as-is; the full prompt + response are already
    recorded in ``calls.json`` via ``run``, so we don't return (or duplicate) them.
    """
    parse_dict = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
    parse_dict = _tag_source_documents(parse_dict, source_docs)
    parsed_json = json.dumps(parse_dict, indent=2, ensure_ascii=False)
    system_prompt = load_prompt(
        "inference/llm",
        version=prompt_version,
        summaries=render_summaries(summaries),
        parsed=parsed_json,
    )
    return cast(
        str,
        litellm_chat_completion(
            system_prompt=system_prompt,
            user_prompt=task,
            model=model,
            run=run,
            **llm_kwargs,
        ),
    )
