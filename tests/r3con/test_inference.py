"""Tests for `evals.r3con.pipeline.stages.inference` — the inference-stage wrappers.

The generic CodeAct loop is tested in ``test_codeact.py``; this file covers the
inference-specific glue:
- ``infer_codeact`` renders the ``inference/codeact`` prompt correctly (schema +
  sample records + the document summaries in the system message; the task wrapped
  in `<task>` tags in the user message) and binds the parse as ``parse``.
- ``infer_llm`` renders (parse JSON + document summaries + task) into the user message.

``infer_codeact`` delegates the LLM call to ``evals.r3con.pipeline.runtime.codeact`` (patched
there); ``infer_llm`` calls the LLM directly (patched on the inference module).

Run with:  uv run python tests/unit/test_inference.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from evals.r3con.pipeline.runtime import codeact
from evals.r3con.pipeline.stages import inference as inference_mod
from evals.r3con.pipeline.stages.inference import infer_codeact

SUMMARIES = ["Doc 1 is a 1923 court opinion.", "Doc 2 cites the petitioner Smith."]


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., str]) -> Iterator[list[list[dict[str, str]]]]:
    """Swap the LLM call (which lives in `runtime.codeact`) for a scripted fake."""
    original = codeact.litellm_chat_completion
    seen: list[list[dict[str, str]]] = []

    def wrapper(**kwargs: Any) -> str:
        if kwargs.get("messages") is not None:
            seen.append([dict(m) for m in kwargs["messages"]])
        return fake(**kwargs)

    codeact.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield seen
    finally:
        codeact.litellm_chat_completion = original  # type: ignore[assignment]


def _final(value: str) -> str:
    return f"Thought: commit.\n<code>\nfinal_answer({value!r})\n</code>"


# ----- infer_codeact -----


def test_infer_codeact_binds_parse_and_commits() -> None:
    response = "Thought: sum.\n<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = infer_codeact(
            task="sum?", schema_code="class Parse(BaseModel): items: list[Item]",
            parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]}, model="m", prompt_version="v1",
        )
    assert r.answer == "6"
    assert r.terminated_by == "final_answer"


def test_user_message_wraps_task_in_tags() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        infer_codeact(
            task="What did Alice say?", schema_code="class Parse(BaseModel): statements: list[S]",
            parsed={"statements": [{"speaker": "Alice", "text": "hi"}]}, model="m", prompt_version="v1",
        )
    assert msgs[0][1]["content"] == "Input:\n<task>\nWhat did Alice say?\n</task>"


def test_system_prompt_contains_sample_records() -> None:
    """The parse's field names reach the model via the samples block (one example
    record per top-level field)."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        infer_codeact(
            task="?", schema_code="class Parse(BaseModel): supports: list[X]",
            parsed={"supports": [{"ok": True}], "oppositions": []}, model="m", prompt_version="v1",
        )
    sys_msg = msgs[0][0]["content"]
    assert "`supports" in sys_msg
    assert "`oppositions" in sys_msg


def test_system_prompt_includes_document_summaries() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, summaries=SUMMARIES, model="m", prompt_version="v1")
    sys_msg = msgs[0][0]["content"]
    assert "## Task-conditioned document summaries" in sys_msg
    assert "1923 court opinion" in sys_msg
    assert "petitioner Smith" in sys_msg


def test_system_prompt_omits_summaries_when_none() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as ml_none:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, summaries=None, model="m", prompt_version="v1")
    with _patched_llm(fake) as ml_empty:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, summaries=[], model="m", prompt_version="v1")
    for ml in (ml_none, ml_empty):
        assert "## Task-conditioned document summaries" not in ml[0][0]["content"]


def test_system_uses_samples_block_not_full_parse_dump() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(50)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): records: list[R]", parsed=parsed, model="m", prompt_version="v1")
    sys_msg = msgs[0][0]["content"]
    assert "name_0" in sys_msg
    assert "name_25" not in sys_msg
    assert "name_49" not in sys_msg


def test_codeact_v2_shows_whole_small_parse() -> None:
    """codeact v2 embeds the WHOLE parse in the system prompt when it's small (so the agent
    can read + symbolically manipulate it directly), not just one sample per field."""
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(8)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, model="m", prompt_version="v2")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_7" in sys   # the WHOLE parse, not just the first sample
    assert "manipulate it symbolically" in sys


def test_codeact_v2_falls_back_to_samples_for_huge_parse() -> None:
    """A parse over CODEACT_PARSE_MAX_TOKS falls back to one sample per field + a prominent
    note, so the prompt can't blow up (the full parse is still bound as the `parse` variable)."""
    def fake(**_: Any) -> str:
        return _final("ok")

    # Well over the token cap: ~1500 records of distinct content (>> 16k tokens).
    parsed = {"records": [{"name": f"name_{i}", "blob": f"item {i}: descriptive text {i * 7}"}
                          for i in range(1500)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, model="m", prompt_version="v2")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_1499" not in sys   # sample only, not the whole flood
    assert "only a SAMPLE" in sys and "`parse` variable" in sys   # the prominent note


def test_sample_block_keeps_unicode_readable() -> None:
    """Non-ASCII parse values reach the model (and the logs) as readable text, not
    \\uXXXX escapes — across scalar, list-record, and dict fields."""
    from evals.r3con.pipeline.stages.inference import _sample_record_per_field

    out = _sample_record_per_field({
        "判决文书1_result": "判决结果4",                 # scalar
        "rows": [{"label": "应付账款", "amount": 12}],     # list of records
        "meta": {"机构": "上海"},                          # dict
    })
    assert "判决结果4" in out and "应付账款" in out and "上海" in out
    assert "\\u" not in out


def test_codeact_system_prompt_keeps_unicode_readable() -> None:
    """End-to-end: a CJK parse renders to a codeact system prompt with literal CJK."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        infer_codeact(task="哪个判决结果?", schema_code="class Parse(BaseModel): 判决文书1_result: str",
                      parsed={"判决文书1_result": "判决结果4"}, model="m", prompt_version="v1")
    sys_msg = msgs[0][0]["content"]
    assert "判决结果4" in sys_msg and "\\u5224" not in sys_msg


def test_infer_codeact_surfaces_summaries_for_alias_resolution() -> None:
    parse = {"actions": [
        {"actor": "Mike", "action": "destroyed the bridge"},
        {"actor": "Mike", "action": "burned the library"},
        {"actor": "Jenny", "action": "watched"},
    ]}
    summaries = ["A 1992 noir. Mike is also referred to as 'The Destroyer' throughout."]
    response = (
        "Thought: summaries say The Destroyer is Mike.\n<code>\n"
        "m = [r for r in parse['actions'] if r['actor'] == 'Mike']\n"
        "final_answer(f'The Destroyer (Mike) performed {len(m)} actions.')\n</code>"
    )

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake) as msgs:
        r = infer_codeact(task="How many actions did The Destroyer perform?",
                          schema_code="...", parsed=parse, summaries=summaries, model="m", prompt_version="v1")
    assert r.terminated_by == "final_answer"
    assert "2 actions" in r.answer
    assert "Mike is also referred to as 'The Destroyer'" in msgs[0][0]["content"]


# ----- infer_llm -----


@contextlib.contextmanager
def _patched_inference_llm(fake: Callable[..., str]) -> Iterator[list[dict[str, Any]]]:
    original = inference_mod.litellm_chat_completion
    calls: list[dict[str, Any]] = []

    def wrapper(**kwargs: Any) -> str:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    inference_mod.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        inference_mod.litellm_chat_completion = original  # type: ignore[assignment]


def test_infer_llm_puts_data_in_system_bare_task_in_user() -> None:
    """The summaries + parsed records live in the SYSTEM prompt; the user message is the
    bare task (no <task> tags)."""
    def fake(**_: Any) -> str:
        return "the answer"

    with _patched_inference_llm(fake) as calls:
        out = inference_mod.infer_llm(task="What happened?", parsed={"events": [{"x": 1}]},
                                      summaries=["Doc A summary text."], model="m", prompt_version="v4")
    assert out == "the answer"
    sys = calls[0]["system_prompt"]
    assert "## Parsed information" in sys and '"x": 1' in sys          # parsed in the system prompt
    assert "## Document summaries" in sys and "Doc A summary text." in sys  # summaries in the system prompt
    assert calls[0]["user_prompt"] == "What happened?"                # bare task in the user message
    assert "<task>" not in calls[0]["user_prompt"]


def test_infer_llm_omits_summaries_when_none() -> None:
    def fake(**_: Any) -> str:
        return "x"

    with _patched_inference_llm(fake) as calls:
        inference_mod.infer_llm(task="q", parsed={"events": []}, summaries=None, model="m", prompt_version="v4")
    sys = calls[0]["system_prompt"]
    assert "## Document summaries" not in sys   # omitted when there are no summaries
    assert "## Parsed information" in sys        # parsed still present
    assert calls[0]["user_prompt"] == "q"        # bare task


def test_infer_llm_tags_records_with_source_document() -> None:
    """Each parse record is stamped with the 1-based source document it came from
    (from source_docs), so the LLM can identify which document a fact belongs to."""
    def fake(**_: Any) -> str:
        return "x"

    with _patched_inference_llm(fake) as calls:
        inference_mod.infer_llm(
            task="q", parsed={"docs": [{"a": 1}, {"a": 2}]},
            source_docs={"docs": [2, 0]},  # record 0 ← doc-index 2 → Document 3; record 1 ← index 0 → Document 1
            summaries=None, model="m", prompt_version="v4",
        )
    sys = calls[0]["system_prompt"]
    assert '"document": 3' in sys and '"document": 1' in sys


def test_infer_codeact_tags_records_with_source_document() -> None:
    """The same source-document tag reaches the codeact agent (in the sample block /
    bound parse), regardless of prompt version."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): docs: list[D]",
                      parsed={"docs": [{"a": 1}]}, source_docs={"docs": [4]},  # index 4 → Document 5
                      model="m", prompt_version="v1")
    assert '"document": 5' in msgs[0][0]["content"]


def test_codeact_v3_drops_schema_section_and_conditional_hedge() -> None:
    """v3 no longer prints the proposer schema (we show the full parse instead, and the
    schema lacked the injected `document` field), and it drops the conditional
    'if the parse was too large' hedge — the small-parse case shows the whole parse with
    no such caveat."""
    def fake(**_: Any) -> str:
        return _final("ok")

    schema = "class Rec(BaseModel):\n    UNIQUE_SCHEMA_MARKER: str\nclass Parse(BaseModel): records: list[Rec]"
    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(5)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code=schema, parsed=parsed, model="m", prompt_version="v3")
    sys = msgs[0][0]["content"]
    assert "UNIQUE_SCHEMA_MARKER" not in sys  # the schema source is no longer embedded
    assert "## Schema used to extract" not in sys
    assert "If the parse was too large" not in sys  # the conditional hedge is gone
    assert "manipulate it in code" in sys  # permissive (not prescriptive) symbolic-manipulation note


def test_codeact_v3_shows_whole_small_parse_and_document_note() -> None:
    """v3 still embeds the WHOLE small parse and explains the per-record `document` field."""
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(8)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, source_docs={"records": list(range(8))}, model="m", prompt_version="v3")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_7" in sys      # the WHOLE parse
    assert "`document`" in sys                       # the document-field explanation
    assert '"document": 1' in sys                    # id-injection still stamps records


def test_codeact_v3_falls_back_to_samples_for_huge_parse() -> None:
    """A parse over CODEACT_PARSE_MAX_TOKS still falls back to one sample per field + the
    prominent "this is only a SAMPLE" note."""
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "blob": f"item {i}: descriptive text {i * 7}"}
                          for i in range(1500)]}
    with _patched_llm(fake) as msgs:
        infer_codeact(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, model="m", prompt_version="v3")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_1499" not in sys
    assert "only a SAMPLE" in sys and "`parse` variable" in sys
