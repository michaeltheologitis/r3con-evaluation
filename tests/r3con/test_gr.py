"""Tests for `evals.r3con.pipeline.gr.gr_answer` — orchestration under a RunConfig (offline).

The four stage functions (`summarize_collection`, `propose_schema`, `extract_text`,
`inference.infer_llm`/`infer_codeact`) are monkeypatched so no LLM is called; the
fakes capture the kwargs they receive. Verifies the summaries → proposer → extractor
→ inference flow, that the summaries reach every downstream stage, that the config's
sampling + seed + per-stage prompt versions reach every stage, and that a per-strategy
failure writes a discoverable `error.txt` without aborting the other.

Run with:  uv run python tests/unit/test_gr.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from evals.r3con.pipeline import gr
from evals.r3con.pipeline.config import RunConfig
from evals.r3con.pipeline.gr import gr_answer
from evals.r3con.pipeline.runs import TaskLogger
from evals.r3con.pipeline.stages import inference as inference_mod

_PROMPTS = {
    "summaries": "v3", "proposer": "v3", "extractor": "v2",
    "inference/llm": "v1", "inference/codeact": "v1",
}


def _cfg(**over: Any) -> RunConfig:
    """A minimal RunConfig for the orchestration tests (no LLM is actually called)."""
    base: dict[str, Any] = dict(name="test", model="m", seed=0, summary_rounds=3,
                                prompts=dict(_PROMPTS), sampling={})
    base.update(over)
    return RunConfig(**base)


@contextlib.contextmanager
def _temp_logs_dir() -> Iterator[Path]:
    from evals.r3con.pipeline.settings import settings

    with tempfile.TemporaryDirectory() as tmp:
        original = settings.LOGS_DIR
        settings.LOGS_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            settings.LOGS_DIR = original


@contextlib.contextmanager
def _patched_stages(captured: dict) -> Iterator[None]:
    orig = (
        gr.summarize_collection, gr.propose_schema, gr.extract_text,
        inference_mod.infer_llm, inference_mod.infer_codeact,
    )

    def fake_summarize(*, task, documents, model, rounds, run, **kw):
        captured["summaries"] = {"rounds": rounds, "kw": kw}
        return SimpleNamespace(final=["s0", "s1"], rounds=[["a", "b"], ["s0", "s1"]])

    def fake_propose(*, task, summaries, model, run, **kw):
        captured["proposer"] = {"summaries": summaries, "kw": kw}
        return SimpleNamespace(
            schema_code="class Parse: pass", parse_cls=object,
            attempts=[SimpleNamespace(thought="t", schema_code="x", error=None)],
        )

    def fake_extract(*, documents, schema_code, parse_cls, task, summaries, model, run, **kw):
        captured["extractor"] = {"summaries": summaries, "kw": kw}
        return SimpleNamespace(parse={"records": []}, source_docs={"records": []})

    def fake_llm(*, task, parsed, summaries, model, run, **kw):
        captured["llm"] = {"summaries": summaries, "kw": kw}
        return "llm-ans"

    def fake_codeact(*, task, schema_code, parsed, summaries, model, max_turns, timeout_s, run, **kw):
        captured["codeact"] = {"summaries": summaries, "kw": kw}
        return SimpleNamespace(answer="codeact-ans", terminated_by="final_answer", turns=[])

    gr.summarize_collection = fake_summarize
    gr.propose_schema = fake_propose
    gr.extract_text = fake_extract
    inference_mod.infer_llm = fake_llm
    inference_mod.infer_codeact = fake_codeact
    try:
        yield
    finally:
        (gr.summarize_collection, gr.propose_schema, gr.extract_text,
         inference_mod.infer_llm, inference_mod.infer_codeact) = orig


def test_runs_four_stages_into_the_run_folder() -> None:
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        res = gr_answer(
            task="q", documents=["d1", "d2"], config=_cfg(),
            strategies=("llm", "codeact"), task_logger=TaskLogger("t1"),
        )
        assert res == {"llm": ("llm-ans", None), "codeact": ("codeact-ans", None)}
        t = root / "t1"
        for rel in ("summaries/result.json", "proposer/result.json", "extractor/result.json",
                    "inference/llm/result.json", "inference/codeact/result.json"):
            assert (t / rel).is_file(), rel
        summ = json.loads((t / "summaries/result.json").read_text())
        assert summ["n_rounds"] == 3 and summ["n_docs"] == 2
        assert len(summ["rounds"]) == 2
        assert summ["rounds"][0]["round"] == 1
        assert summ["rounds"][-1]["summaries"] == ["s0", "s1"]


def test_summaries_flow_into_every_downstream_stage() -> None:
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=_cfg(),
                  strategies=("llm", "codeact"), task_logger=TaskLogger("t1"))
    for stage in ("proposer", "extractor", "llm", "codeact"):
        assert captured[stage]["summaries"] == ["s0", "s1"], stage


def test_source_docs_reach_inference() -> None:
    """The extractor's per-record source-document provenance reaches both inference
    strategies (so they can tag records with the document they came from)."""
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=_cfg(),
                  strategies=("llm", "codeact"), task_logger=TaskLogger("t1"))
    assert captured["llm"]["kw"]["source_docs"] == {"records": []}
    assert captured["codeact"]["kw"]["source_docs"] == {"records": []}


def test_summary_rounds_come_from_config() -> None:
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=_cfg(summary_rounds=2), task_logger=TaskLogger("t1"))
    assert captured["summaries"]["rounds"] == 2
    captured.clear()
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=_cfg(summary_rounds=5), task_logger=TaskLogger("t1"))
    assert captured["summaries"]["rounds"] == 5


def test_proposer_uses_settings_max_attempts() -> None:
    """gr_answer must not override the proposer's retry budget — it passes no
    ``max_attempts``, so ``propose_schema`` uses ``settings.PROPOSER_MAX_ATTEMPTS``."""
    captured: dict = {}
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=_cfg(), task_logger=TaskLogger("t1"))
    assert "max_attempts" not in captured["proposer"]["kw"]


def test_config_sampling_and_seed_reach_every_stage() -> None:
    captured: dict = {}
    cfg = _cfg(seed=0, sampling={"temperature": 0.7, "extra_body": {"top_k": 20}})
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=cfg, strategies=("llm", "codeact"),
                  task_logger=TaskLogger("t1"))
    for stage in ("summaries", "proposer", "extractor", "llm", "codeact"):
        kw = captured[stage]["kw"]
        assert kw["temperature"] == 0.7, stage
        assert kw["extra_body"] == {"top_k": 20}, stage
        assert kw["seed"] == 0, stage  # transport kwarg, from config.seed


def test_per_stage_prompt_versions_reach_each_stage() -> None:
    captured: dict = {}
    cfg = _cfg(prompts={"summaries": "vA", "proposer": "vB", "extractor": "vC",
                        "inference/llm": "vD", "inference/codeact": "vE"})
    with _temp_logs_dir(), _patched_stages(captured):
        gr_answer(task="q", documents=["d"], config=cfg, strategies=("llm", "codeact"),
                  task_logger=TaskLogger("t1"))
    assert captured["summaries"]["kw"]["prompt_version"] == "vA"
    assert captured["proposer"]["kw"]["prompt_version"] == "vB"
    assert captured["extractor"]["kw"]["prompt_version"] == "vC"
    assert captured["llm"]["kw"]["prompt_version"] == "vD"
    assert captured["codeact"]["kw"]["prompt_version"] == "vE"


def test_per_strategy_failure_writes_error_txt() -> None:
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        def boom(*, task, parsed, summaries, model, run, **kw):
            raise ValueError("ctx too long")

        inference_mod.infer_llm = boom  # restored by _patched_stages' finally
        res = gr_answer(task="q", documents=["d"], config=_cfg(),
                        strategies=("llm", "codeact"), task_logger=TaskLogger("t1"))
        t = root / "t1"
        err = t / "inference" / "llm" / "error.txt"
        assert err.is_file() and "ValueError" in err.read_text()
        assert not (t / "inference" / "llm" / "result.json").is_file()
        assert res["llm"][0] == "" and "ValueError" in res["llm"][1]
        assert (t / "inference" / "codeact" / "result.json").is_file()
        assert res["codeact"] == ("codeact-ans", None)


def test_no_logger_returns_answers_without_writing() -> None:
    captured: dict = {}
    with _temp_logs_dir() as root, _patched_stages(captured):
        res = gr_answer(task="q", documents=["d"], config=_cfg(), strategies=("codeact",))
        assert res == {"codeact": ("codeact-ans", None)}
        assert not any(root.iterdir())  # nothing written without a task_logger
