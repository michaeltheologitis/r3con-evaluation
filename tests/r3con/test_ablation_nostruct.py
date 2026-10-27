"""Tests for `evals.r3con.harness.ablation_nostruct` — the summaries-only CodeAct ablation (offline).

The point of the ablation is that CodeAct gets the summaries and **nothing else** — no parse in
the prompt, nothing bound in the sandbox. The core test pins exactly that. ``run_codeact`` (the
real LLM loop) is monkeypatched so these stay offline; the ablation prompt is loaded for real.

Run with:  uv run pytest tests/r3con/test_ablation_nostruct.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from evals.r3con.harness import ablation_nostruct as A
from evals.r3con.harness import loong
from evals.r3con.pipeline.config import RunConfig
from evals.r3con.pipeline.runtime.codeact import CodeActResult

_CFG = {
    "name": "default", "model": "m", "seed": 42, "summary_rounds": 2,
    "prompts": {"summaries": "v3", "proposer": "v4", "extractor": "v2",
                "inference/llm": "v4", "inference/codeact": "v4"},
    "sampling": {}, "sampling_preset": None, "overrides": {},
}
_LABEL = RunConfig(**_CFG).label()
_CFG2 = {**_CFG, "seed": 7}  # a different run label


@contextlib.contextmanager
def _patched_run_codeact() -> Iterator[dict]:
    """Capture what infer_codeact_nostruct passes to run_codeact; return a fake result."""
    captured: dict = {}
    orig = A.run_codeact

    def fake(*, system_prompt, user_message, model, variables, **kw):
        captured.update(system_prompt=system_prompt, user_message=user_message,
                        model=model, variables=variables, kw=kw)
        return CodeActResult(answer="ANS", turns=[], terminated_by="final_answer")

    A.run_codeact = fake  # type: ignore[assignment]
    try:
        yield captured
    finally:
        A.run_codeact = orig  # type: ignore[assignment]


def _w(p: Path, data) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data if isinstance(data, str) else json.dumps(data))


def _summaries_json(final: list[str]) -> dict:
    return {"n_rounds": 2, "n_docs": len(final),
            "rounds": [{"round": 1, "summaries": ["r1"] * len(final)}, {"round": 2, "summaries": final}]}


def test_binds_no_parse_and_no_parse_block_in_prompt() -> None:
    with _patched_run_codeact() as cap:
        res = A.infer_codeact_nostruct(task="the task", summaries=["sum one", "sum two"], model="m")
    assert res.answer == "ANS"
    # THE ablation: nothing is bound in the sandbox (no `parse`).
    assert cap["variables"] == {}
    # the summaries ARE in the prompt...
    assert "sum one" in cap["system_prompt"] and "Document summaries" in cap["system_prompt"]
    # ...and the parse view is NOT (no parsed-data block, no bound-`parse` mention).
    assert "## The parsed data" not in cap["system_prompt"]
    assert "bound as the `parse` variable" not in cap["system_prompt"]
    # the task is delivered in the user message, as usual.
    assert "<task>\nthe task\n</task>" in cap["user_message"]


def test_folders_to_ablate_filters() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # f1: has summaries, not yet ablated, label L  -> eligible
        _w(root / "f1" / "manifest.json", {"benchmark": "CorpusQA", "task_id": "t1", "config": _CFG})
        _w(root / "f1" / "summaries" / "result.json", _summaries_json(["A", "B"]))
        # f2: already ablated -> skip
        _w(root / "f2" / "manifest.json", {"benchmark": "CorpusQA", "task_id": "t2", "config": _CFG})
        _w(root / "f2" / "summaries" / "result.json", _summaries_json(["A"]))
        _w(root / "f2" / "inference" / A.STRATEGY / "result.json", {"answer": "x"})
        # f3: no summaries -> skip
        _w(root / "f3" / "manifest.json", {"benchmark": "CorpusQA", "task_id": "t3", "config": _CFG})
        # f4: has summaries, different label -> eligible only without the label filter
        _w(root / "f4" / "manifest.json", {"benchmark": "CorpusQA", "task_id": "t4", "config": _CFG2})
        _w(root / "f4" / "summaries" / "result.json", _summaries_json(["A"]))

        assert A.folders_to_ablate(root) == ["f1", "f4"]
        assert A.folders_to_ablate(root, label=_LABEL) == ["f1"]
        assert A.folders_to_ablate(root, benchmark="CorpusQA") == ["f1", "f4"]
        assert A.folders_to_ablate(root, benchmark="Loong") == []


def test_folders_to_ablate_skips_stripped_or_zero_round_summaries() -> None:
    """The shipped logs keep only the token totals of most runs' summaries, and a
    summary_rounds=0 run has no rounds at all — neither gives a summaries-only agent anything
    to read, so neither is eligible."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _w(root / "stripped" / "manifest.json", {"benchmark": "Loong", "task_id": "t1", "config": _CFG})
        _w(root / "stripped" / "summaries" / "result.json", {"totals": {"tokens": {"prompt": 1}}})
        _w(root / "sr0" / "manifest.json", {"benchmark": "Loong", "task_id": "t2", "config": {**_CFG, "summary_rounds": 0}})
        _w(root / "sr0" / "summaries" / "result.json", {"n_rounds": 0, "n_docs": 2, "rounds": []})
        _w(root / "ok" / "manifest.json", {"benchmark": "Loong", "task_id": "t3", "config": _CFG})
        _w(root / "ok" / "summaries" / "result.json", _summaries_json(["A"]))
        assert A.folders_to_ablate(root) == ["ok"]


def test_ablate_one_writes_result_to_new_strategy_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp, _patched_run_codeact() as cap:
        root = Path(tmp)
        _w(root / "f1" / "manifest.json",
           {"benchmark": "CorpusQA", "task_id": "t1", "question": "the task", "config": _CFG})
        _w(root / "f1" / "summaries" / "result.json", _summaries_json(["final A", "final B"]))

        ans, err = A.ablate_one("f1", model="hosted_vllm/Q", logs_root=root)
        assert err is None and ans == "ANS"
        # used the LAST round's summaries + the routable model
        assert "final A" in cap["system_prompt"] and cap["model"] == "hosted_vllm/Q"
        assert "<task>\nthe task\n</task>" in cap["user_message"]
        # wrote a native result.json into the new strategy subfolder (the analysis reads this)
        rp = root / "f1" / "inference" / A.STRATEGY / "result.json"
        assert rp.is_file()
        rj = json.loads(rp.read_text())
        assert rj["answer"] == "ANS" and rj["terminated_by"] == "final_answer" and "totals" in rj
        # the manifest is not touched
        assert "usage" not in json.loads((root / "f1" / "manifest.json").read_text())


def test_task_text_reloads_a_stripped_question_from_the_adapter() -> None:
    """A shipped manifest has no `question`; the task is re-loaded from the benchmark adapter
    named by `benchmark` + `task_id`, composed exactly as the original run composed it."""
    calls: list[str] = []
    orig = loong.load
    loong.load = lambda tid: (calls.append(tid), loong.TaskInput(task="reloaded task", documents=["d"]))[1]
    try:
        assert A.task_text({"benchmark": "Loong", "task_id": "T-9"}) == "reloaded task"
        assert calls == ["T-9"]
        # a manifest that still carries its question never hits the adapter
        assert A.task_text({"benchmark": "Loong", "task_id": "T-9", "question": "kept"}) == "kept"
        assert calls == ["T-9"]
    finally:
        loong.load = orig
    with pytest.raises(KeyError):
        A.task_text({"benchmark": "Nope", "task_id": "x"})
    with pytest.raises(KeyError):
        A.task_text({"benchmark": "Loong"})


def test_ablate_one_records_a_failure_as_error_txt() -> None:
    """A folder whose summaries were stripped fails inside the guarded block: error.txt in the
    strategy folder, no result.json, and the error is returned rather than raised."""
    with tempfile.TemporaryDirectory() as tmp, _patched_run_codeact():
        root = Path(tmp)
        _w(root / "f1" / "manifest.json",
           {"benchmark": "CorpusQA", "task_id": "t1", "question": "the task", "config": _CFG})
        _w(root / "f1" / "summaries" / "result.json", {"totals": {"tokens": {}}})
        ans, err = A.ablate_one("f1", model="hosted_vllm/Q", logs_root=root)
        assert ans == "" and err is not None and "no rounds" in err
        sdir = root / "f1" / "inference" / A.STRATEGY
        assert (sdir / "error.txt").is_file() and not (sdir / "result.json").is_file()
