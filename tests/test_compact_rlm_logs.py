"""Tests for ``scripts/compact_rlm_logs.py``.

The input trajectories are written by RLM's REAL ``RLMLogger`` from real ``RLMIteration`` /
``REPLResult`` objects, so the on-disk format is the one the existing logs have. These pin: every
``locals`` is dropped and nothing else changes; the metadata line is byte-identical; a compacted
file is exactly what ``TrajectoryLogger`` writes for new runs; re-running is a no-op; only rlm
trajectories are touched.
"""
from __future__ import annotations

import json
from pathlib import Path

from rlm.core.types import CodeBlock, REPLResult, RLMIteration, RLMMetadata
from rlm.logger import RLMLogger

from evals.baselines.rlm.logger import TrajectoryLogger
from scripts.compact_rlm_logs import compact_file, compact_rlm_logs, main

_CONTEXT = "=== Document 1 ===\n" + "Paris is the capital of France. " * 200


def _iterations() -> list[RLMIteration]:
    probe = REPLResult(stdout="6420\n", stderr="", execution_time=0.01,
                       locals={"context": _CONTEXT, "context_0": _CONTEXT, "n": 6420})
    submit = REPLResult(stdout="", stderr="", execution_time=0.01, final_answer="Paris",
                        locals={"context": _CONTEXT, "context_0": _CONTEXT, "n": 6420,
                                "answer": {"content": "Paris", "ready": True}})
    return [
        RLMIteration(prompt=[{"role": "user", "content": "Turn 1/30:"}], response="```repl\nn = len(context)\nprint(n)\n```",
                     code_blocks=[CodeBlock(code="n = len(context)\nprint(n)", result=probe)], iteration_time=1.0),
        RLMIteration(prompt=[{"role": "user", "content": "Turn 2/30: 中文"}], response="submit",
                     code_blocks=[CodeBlock(code='answer["content"] = "Paris"', result=submit)],
                     final_answer="Paris", iteration_time=2.0),
    ]


def _write_trajectory(run_dir: Path, logger_cls=RLMLogger) -> Path:
    logger = logger_cls(log_dir=str(run_dir))
    logger.log_metadata(RLMMetadata(root_model="Qwen/Q", max_depth=1, max_iterations=30, backend="vllm",
                                    backend_kwargs={"model_name": "Qwen/Q"}, environment_type="local",
                                    environment_kwargs={}))
    for it in _iterations():
        logger.log(it)
    return Path(logger.log_file_path)


def _without_locals_and_timestamps(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        entry.pop("timestamp")
        for block in entry.get("code_blocks", []):
            block["result"].pop("locals", None)
        out.append(entry)
    return out


def test_compact_drops_locals_and_keeps_everything_else(tmp_path) -> None:
    path = _write_trajectory(tmp_path / "loong" / "rlm" / "run1")
    original = path.read_bytes()
    expected = _without_locals_and_timestamps(path)

    before, after, rewritten = compact_file(path)

    assert rewritten and after < before == len(original)
    assert b'"locals"' not in path.read_bytes()
    assert _without_locals_and_timestamps(path) == expected          # only `locals` went away
    assert path.read_bytes().splitlines()[0] == original.splitlines()[0]  # metadata byte-identical
    assert not path.with_name(path.name + ".compacting").exists()


def test_compacted_file_matches_what_new_runs_write(tmp_path) -> None:
    old = _write_trajectory(tmp_path / "old")
    compact_file(old)
    new = _write_trajectory(tmp_path / "new", logger_cls=TrajectoryLogger)
    strip_ts = lambda p: [{k: v for k, v in json.loads(l).items() if k != "timestamp"}  # noqa: E731
                          for l in p.read_text().splitlines()]
    assert strip_ts(old) == strip_ts(new)


def test_rerun_is_a_noop(tmp_path) -> None:
    path = _write_trajectory(tmp_path / "corpusqa" / "rlm" / "run1")
    compact_file(path)
    compacted = path.read_bytes()
    before, after, rewritten = compact_file(path)
    assert not rewritten and before == after == len(compacted)
    assert path.read_bytes() == compacted


def test_only_rlm_trajectories_and_benchmark_filter(tmp_path) -> None:
    lo = _write_trajectory(tmp_path / "loong" / "rlm" / "r1")
    cq = _write_trajectory(tmp_path / "corpusqa" / "rlm" / "r2")
    other = tmp_path / "loong" / "codeact" / "r3" / "rlm_x.jsonl"   # not an rlm run folder
    other.parent.mkdir(parents=True)
    other.write_text('{"code_blocks": [{"result": {"locals": {}}}]}\n')

    summary = compact_rlm_logs(tmp_path, benchmark="loong", workers=1)

    assert b'"locals"' not in lo.read_bytes() and b'"locals"' in cq.read_bytes()
    assert b'"locals"' in other.read_bytes()
    assert summary["loong"]["rewritten"] == 1 and "corpusqa" not in summary


def test_main_cli_prints_summary(tmp_path, capsys) -> None:
    path = _write_trajectory(tmp_path / "dracula" / "rlm" / "r1")
    main(["--logs-dir", str(tmp_path), "--workers", "1"])
    assert b'"locals"' not in path.read_bytes()
    assert "dracula: rewrote 1/1" in capsys.readouterr().out
