"""Tests for `evals.r3con.pipeline.runs.TaskLogger` and `StageRun` (bare-folder logging).

Run with:  uv run python tests/unit/test_runs.py
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from evals.r3con.pipeline.runs import StageRun, TaskLogger, new_run_folder, normalize_model_name


class _Sample(BaseModel):
    name: str
    count: int = Field(description="some count")


def _msgs_assistant(content: str = "ok", finish: str = "stop") -> dict:
    return {"role": "assistant", "content": content, "finish_reason": finish}


# ----- TaskLogger -----


def test_creates_per_task_dir() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("task-1", root=Path(tmp))
        assert log.dir == Path(tmp) / "task-1"
        assert log.dir.is_dir()


def test_new_run_folder_is_unique_timestamped_and_opaque() -> None:
    a = new_run_folder()
    b = new_run_folder()
    # <UTC-timestamp>_<hex> — opaque (no identity), flat, and unique per call.
    assert a != b
    assert "/" not in a
    ts, sep, hexpart = a.partition("_")
    assert sep == "_"
    assert ts.endswith("Z") and len(ts) == len("20260613T142233Z")
    assert len(hexpart) == 8 and all(ch in "0123456789abcdef" for ch in hexpart)


def test_task_logger_flat_folder() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("gpt__r3__abc123", task_id="t1", root=Path(tmp))
        assert log.dir == Path(tmp) / "gpt__r3__abc123"  # one flat level, no nesting
        assert log.dir.is_dir()
        assert log.folder == "gpt__r3__abc123" and log.task_id == "t1"
        # task_id defaults to the folder name when not given
        assert TaskLogger("solo", root=Path(tmp)).task_id == "solo"


def test_write_text_and_json() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_text("answer", "final\n")
        log.write_json("score", {"score": 1, "gold": "yes"})
        assert (log.dir / "answer.txt").read_text() == "final\n"
        assert json.loads((log.dir / "score.json").read_text()) == {"score": 1, "gold": "yes"}


def test_write_json_pydantic_instance_and_class() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("parser", _Sample(name="x", count=3))
        log.write_json("schema", _Sample)
        assert json.loads((log.dir / "parser.json").read_text()) == {"name": "x", "count": 3}
        assert "properties" in json.loads((log.dir / "schema.json").read_text())


def test_write_json_nested_pydantic_and_overwrite() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("p", {"attempt": 1, "result": _Sample(name="x", count=3)})
        assert json.loads((log.dir / "p.json").read_text())["result"] == {"name": "x", "count": 3}
        log.write_text("a", "first")
        log.write_text("a", "second")
        assert (log.dir / "a.txt").read_text() == "second"


def test_unserializable_fallback_to_repr() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        log.write_json("oddity", {"obj": Opaque()})
        assert json.loads((log.dir / "oddity.json").read_text()) == {"obj": "<Opaque>"}


def test_write_json_keeps_unicode_readable() -> None:
    """Non-ASCII (e.g. CJK) is written literally, not as \\uXXXX escapes, and still
    round-trips through json.loads."""
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        p = log.write_json("r", {"label": "应付账款"})
        raw = p.read_text(encoding="utf-8")
        assert "应付账款" in raw and "\\u" not in raw
        assert json.loads(raw) == {"label": "应付账款"}


def test_calls_json_keeps_unicode_readable() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="extractor", task_logger=task)
        run.add_step(kind="extract-d0", messages=[{"role": "user", "content": "文档"}],
                     response=_msgs_assistant("应付账款"))
        run.flush(write_transcript=False)
        raw = (run.dir / "calls.json").read_text(encoding="utf-8")
        assert "应付账款" in raw and "文档" in raw and "\\u" not in raw


def test_write_creates_subdirs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        log = TaskLogger("t", root=Path(tmp))
        p = log.write_json("inference/codeact/result", {"answer": "x"})
        assert p.exists() and p.name == "result.json" and p.parent.name == "codeact"
        y = log.write_yaml("proposer/transcript", {"messages": [{"role": "user", "content": "hi"}]})
        assert y.exists() and y.name == "transcript.yaml"


# ----- StageRun -----


def test_stage_run_makes_subdir_and_flushes_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task, model="m", seed=0)
        assert run.dir == task.dir / "proposer" and run.dir.is_dir()
        run.add_step(
            kind="llm_call",
            messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            response=_msgs_assistant("a"),
            tokens={"prompt": 5, "completion": 1, "total": 6},
        )
        ty = run.flush()
        assert ty.name == "transcript.yaml" and ty.exists()
        assert (run.dir / "calls.json").is_file()


def test_calls_json_parses_doc_from_new_kinds() -> None:
    """calls.json carries per-call provenance — ``doc`` parsed from the new per-doc
    kinds (``extract-d0``, ``summary-r1-d2``); ``chunk`` is None (no chunking)."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="extractor", task_logger=task, model="m", seed=0)
        run.add_step(kind="summary-r1-d2", messages=[{"role": "user", "content": "u"}],
                     response=_msgs_assistant("a"), tokens={"prompt": 5, "completion": 1, "total": 6})
        run.add_step(kind="extract-d0", messages=[{"role": "user", "content": "u2"}],
                     response=_msgs_assistant("b", finish="length"), tokens={"prompt": 9, "completion": 3, "total": 12})
        run.flush(write_transcript=False)
        calls = json.loads((run.dir / "calls.json").read_text())
        assert (calls[0]["kind"], calls[0]["doc"], calls[0]["chunk"]) == ("summary-r1-d2", 2, None)
        assert (calls[1]["kind"], calls[1]["doc"], calls[1]["chunk"]) == ("extract-d0", 0, None)
        assert calls[0]["output"] == "a" and calls[1]["finish_reason"] == "length"


def test_flush_skip_transcript() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="summaries", task_logger=task)
        run.add_step(kind="summary-r1-d0", messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("a"))
        p = run.flush(write_transcript=False)
        assert (run.dir / "calls.json").is_file()
        assert not (run.dir / "transcript.yaml").exists()
        assert p.name == "calls.json"


def test_compute_totals() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="inference/codeact", task_logger=task)
        for pt, ct in [(10, 2), (20, 5), (5, 1)]:
            run.add_step(messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("r"),
                         tokens={"prompt": pt, "completion": ct, "total": pt + ct})
        totals = run.compute_totals()
        assert totals["n_steps"] == 3
        assert totals["tokens"] == {"prompt": 35, "completion": 8, "total": 43}


def test_compute_totals_none_when_no_tokens() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task)
        run.add_step(messages=[{"role": "user", "content": "u"}], response=_msgs_assistant("r"))
        assert run.compute_totals() == {"n_steps": 1, "tokens": None}


def test_transcript_multi_turn_thread() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="inference/codeact", task_logger=task)
        msgs1 = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
        run.add_step(kind="turn-1", messages=msgs1, response=_msgs_assistant("A1"))
        msgs2 = msgs1 + [{"role": "assistant", "content": "A1"}, {"role": "user", "content": "<observation>obs</observation>"}]
        run.add_step(kind="turn-2", messages=msgs2, response=_msgs_assistant("A2 final"))
        transcript = yaml.safe_load(run.flush().read_text())
        contents = [m["content"] for m in transcript["messages"]]
        assert contents == ["S", "U1", "A1", "<observation>obs</observation>", "A2 final"]


def test_transcript_is_final_attempt_for_retry_loop() -> None:
    """For a retry loop (proposer) the transcript is just the final (successful)
    attempt; the earlier failed attempt lives in calls.json, not the transcript."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task)
        run.add_step(kind="llm_call", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "first"}], response=_msgs_assistant("a1"))
        run.add_step(kind="retry", messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "retry-prompt"}], response=_msgs_assistant("a2"))
        contents = [m["content"] for m in yaml.safe_load(run.flush().read_text())["messages"]]
        assert contents == ["S", "retry-prompt", "a2"]  # final attempt only
        # The failed first attempt is preserved in calls.json.
        calls = json.loads((run.dir / "calls.json").read_text())
        assert calls[0]["output"] == "a1" and calls[1]["output"] == "a2"


def test_transcript_single_step() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task)
        run.add_step(messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "U"}], response=_msgs_assistant("A"))
        transcript = yaml.safe_load(run.flush().read_text())
        assert [m["content"] for m in transcript["messages"]] == ["S", "U", "A"]


def test_no_steps_is_safe_to_flush() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task)
        transcript = yaml.safe_load(run.flush().read_text())
        assert transcript["messages"] == []
        assert run.compute_totals()["n_steps"] == 0


def test_add_step_is_thread_safe() -> None:
    """Concurrent add_step (the parallel doc fan-out shares one run) assigns unique,
    contiguous step numbers and loses no steps."""
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="summaries", task_logger=task)

        def worker(i: int) -> None:
            run.add_step(kind=f"summary-r1-d{i}", messages=[{"role": "user", "content": str(i)}], response=_msgs_assistant("x"))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(run.steps) == 50
        assert sorted(s.step for s in run.steps) == list(range(1, 51))


# ----- helpers -----


def test_normalize_model_name() -> None:
    assert normalize_model_name("openai/gpt-5.4-nano") == "gpt-5.4-nano"
    assert normalize_model_name("hosted_vllm/Qwen/Qwen3-8B") == "Qwen3-8B"
    assert normalize_model_name("gpt-5.4-nano") == "gpt-5.4-nano"
    assert normalize_model_name(None) is None


def test_transcript_normalizes_model() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        task = TaskLogger("t", root=Path(tmp))
        run = StageRun(stage="proposer", task_logger=task, model="hosted_vllm/Qwen/Qwen3-8B")
        run.add_step(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], response=_msgs_assistant("a"))
        assert yaml.safe_load(run.flush().read_text())["model"] == "Qwen3-8B"


def test_transcript_block_style_despite_trailing_spaces() -> None:
    from evals.r3con.pipeline.runs import _dump_transcript_yaml

    out = _dump_transcript_yaml({"content": "1. Thorir  \n2. Gudrun  \n3. Amleth"})
    assert "content: |" in out
    assert "\\n" not in out
    assert yaml.safe_load(out)["content"] == "1. Thorir\n2. Gudrun\n3. Amleth"
