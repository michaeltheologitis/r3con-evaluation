"""Tests for `evals.r3con.harness.runner` — the subprocess launcher + CLI scaffolding.

The launcher is driven by a fake Popen so no real process spawns. Run with:
uv run python tests/unit/test_runner.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

from evals.r3con.pipeline.config import RunConfig
from evals.r3con.harness import runner
from evals.r3con.harness.runner import (
    LaunchSummary,
    _run_task_argv,
    completed_task_ids,
    print_launch_summary,
)
from evals.r3con.pipeline.settings import settings

_PROMPTS = {"summaries": "v3", "proposer": "v3", "extractor": "v2",
            "inference/llm": "v1", "inference/codeact": "v1"}


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = dict(name="default", model="m", seed=0, summary_rounds=3,
                                prompts=dict(_PROMPTS), sampling={})
    base.update(over)
    return RunConfig(**base)


@contextlib.contextmanager
def _temp_logs_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        original = settings.LOGS_DIR
        settings.LOGS_DIR = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            settings.LOGS_DIR = original


@contextlib.contextmanager
def _pinned_env(*keys: str) -> Iterator[None]:
    prior = {k: os.environ.get(k) for k in keys}
    try:
        yield
    finally:
        for k, v in prior.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _FakeProc:
    live = 0
    max_live = 0

    def __init__(self, argv, *, rc=0, stdout=None, finish_after=1):
        self.argv = argv
        self._rc = rc
        self._stdout = stdout
        self._finish_after = finish_after
        self._polls = 0
        self.returncode = None
        _FakeProc.live += 1
        _FakeProc.max_live = max(_FakeProc.max_live, _FakeProc.live)

    def poll(self):
        self._polls += 1
        if self._polls < self._finish_after:
            return None
        if self._stdout is not None:
            tid = self.argv[self.argv.index("--task-id") + 1]
            self._stdout.write(f"{tid} OK ran\n")
            self._stdout.flush()
        self.returncode = self._rc
        _FakeProc.live -= 1
        return self.returncode

    @classmethod
    def reset(cls):
        cls.live = 0
        cls.max_live = 0


def _make_fake_popen(*, fail_tids=(), finish_after=1):
    def factory(argv, *, stdout=None, stderr=None, cwd=None, **_kw):
        tid = argv[argv.index("--task-id") + 1]
        rc = 2 if tid in fail_tids else 0
        return _FakeProc(argv, rc=rc, stdout=stdout, finish_after=finish_after)

    return factory


# ---------- _run_task_argv ----------


def test_run_task_argv_collapses_strategies_and_passes_config() -> None:
    cfg = _cfg(name="default", overrides={"seed": 7, "model": "openai/x", "sampling": "qwen-no-thinking"})
    argv = _run_task_argv(task_id="T", config=cfg, strategies=("llm", "codeact"),
                          run_folder="RF", api_base="http://e", api_key="k")
    assert argv[1].endswith("scripts/r3con/loong/run_task.py")
    assert argv[argv.index("--task-id") + 1] == "T"
    assert argv[argv.index("--config") + 1] == "default"
    assert argv[argv.index("--inference") + 1] == "both"
    assert argv[argv.index("--run-folder") + 1] == "RF"
    # config overrides become CLI flags so the child reconstructs the same RunConfig
    assert argv[argv.index("--seed") + 1] == "7"
    assert argv[argv.index("--model") + 1] == "openai/x"
    assert argv[argv.index("--sampling") + 1] == "qwen-no-thinking"  # preset re-passed to the child
    assert "--base-url" in argv and "--api-key" in argv


def test_run_task_argv_no_overrides_omits_optional() -> None:
    argv = _run_task_argv(task_id="T", config=_cfg(name="default"), strategies=("codeact",),
                          run_folder="RF", api_base=None, api_key=None)
    assert argv[argv.index("--inference") + 1] == "codeact"
    assert argv[argv.index("--config") + 1] == "default"
    assert "--seed" not in argv and "--model" not in argv
    assert "--base-url" not in argv and "--api-key" not in argv


def test_run_task_argv_custom_script() -> None:
    """The per-task child script is parameterizable (default Loong) so a second benchmark
    harness (e.g. CorpusQA) can reuse the same launcher pointed at its own child."""
    default = _run_task_argv(task_id="T", config=_cfg(name="default"), strategies=("codeact",),
                             run_folder="RF", api_base=None, api_key=None)
    assert default[1].endswith("scripts/r3con/loong/run_task.py")
    custom = _run_task_argv(task_id="T", config=_cfg(name="default"), strategies=("codeact",),
                            run_folder="RF", api_base=None, api_key=None,
                            run_task_script="scripts/r3con/corpusqa/run_task.py")
    assert custom[1].endswith("scripts/r3con/corpusqa/run_task.py")


# ---------- run (the launcher) ----------


def test_run_counts_ok_and_failures() -> None:
    with _temp_logs_dir():
        _FakeProc.reset()
        summary = runner.run(["t1", "t2", "t3"], _cfg(), strategies=("codeact",),
                             workers=2, _popen=_make_fake_popen(fail_tids={"t2"}))
    assert (summary.n, summary.ok, summary.failed) == (3, 2, 1)
    assert summary.failures == [("t2", 2)]


def test_run_respects_worker_cap() -> None:
    with _temp_logs_dir():
        _FakeProc.reset()
        runner.run(["t1", "t2", "t3", "t4", "t5"], _cfg(), strategies=("codeact",),
                   workers=2, _popen=_make_fake_popen(finish_after=3))
    assert 1 <= _FakeProc.max_live <= 2


def test_run_writes_per_task_log_in_opaque_run_folder() -> None:
    """The per-task log lands in the flat, opaque run-folder the runner stamped — not a
    run-dir, and not a bare task-id folder."""
    with _temp_logs_dir() as root:
        _FakeProc.reset()
        runner.run(["t1"], _cfg(), strategies=("codeact",), workers=1, _popen=_make_fake_popen())
        folders = [p for p in root.iterdir() if (p / "run_task.log").is_file()]
        assert len(folders) == 1  # one opaque <timestamp>_<hex> folder
        assert "t1 OK ran" in (folders[0] / "run_task.log").read_text()
        assert not (root / "t1").exists()  # not a bare task-id folder


def test_run_prints_minimal_tick_cross() -> None:
    """Per-task console output is a minimal ✓/✗ + id — no answer echo, no log tail."""
    buf = io.StringIO()
    with _temp_logs_dir(), contextlib.redirect_stdout(buf):
        _FakeProc.reset()
        runner.run(["t1", "t2"], _cfg(), strategies=("codeact",), workers=2,
                   _popen=_make_fake_popen(fail_tids={"t2"}))
    out = buf.getvalue()
    assert "✓ t1" in out and "✗ t2" in out and "(exit 2)" in out
    # the child's stdout ("t1 OK ran") goes to the LOG, never echoed to the console
    assert "OK ran" not in out


def test_run_terminates_children_on_interrupt() -> None:
    """Ctrl-C (KeyboardInterrupt) mid-run must SIGTERM every inflight child rather than
    leave orphaned subprocesses making LLM calls in the background, then re-raise."""
    procs: list = []
    state = {"raised": False}

    class _IProc:
        def __init__(self, argv):
            self.argv = argv
            self.returncode = None
            self.terminated = False
            self.killed = False

        def poll(self):
            if not state["raised"]:  # first poll anywhere simulates the Ctrl-C
                state["raised"] = True
                raise KeyboardInterrupt
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.killed = True
            self.returncode = -9

        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = -15
            return self.returncode

    def factory(argv, *, stdout=None, stderr=None, cwd=None, **_kw):
        p = _IProc(argv)
        procs.append(p)
        return p

    with _temp_logs_dir():
        raised = False
        try:
            runner.run(["t1", "t2"], _cfg(), strategies=("codeact",), workers=2, _popen=factory)
        except KeyboardInterrupt:
            raised = True
    assert raised  # the interrupt propagates after cleanup
    assert len(procs) == 2 and all(p.terminated for p in procs)  # every inflight child stopped


def test_run_closes_logs_on_launch_failure() -> None:
    opened: list = []

    def factory(argv, *, stdout=None, stderr=None, cwd=None, **_kw):
        opened.append(stdout)
        if argv[argv.index("--task-id") + 1] == "t2":
            raise OSError("popen boom")
        return _FakeProc(argv, stdout=stdout, finish_after=1)

    with _temp_logs_dir():
        _FakeProc.reset()
        raised = False
        try:
            runner.run(["t1", "t2", "t3"], _cfg(), strategies=("codeact",), workers=1, _popen=factory)
        except OSError:
            raised = True
        assert raised
        assert opened and all(f.closed for f in opened)


# ---------- CLI scaffolding ----------


def test_build_common_argparser_flags() -> None:
    p = runner.build_common_argparser("desc")
    a = p.parse_args([])
    assert a.config == "default" and a.inference == "codeact" and a.workers == 10
    # overrides default to None (don't override the config) until passed
    assert a.model is None and a.seed is None and a.summary_rounds is None and a.sampling is None
    assert a.doc_workers is None  # unset → the env var / default
    assert p.parse_args(["--doc-workers", "8"]).doc_workers == 8
    b = p.parse_args(["--config", "default", "--seed", "7", "--inference", "both", "--sampling", "qwen-no-thinking"])
    assert b.config == "default" and b.seed == 7 and b.inference == "both" and b.sampling == "qwen-no-thinking"


def test_resolve_config_applies_overrides() -> None:
    args = argparse.Namespace(config="default", model="openai/zzz", seed=7, summary_rounds=None, sampling=None)
    cfg = runner.resolve_config(args)
    assert cfg.name == "default" and cfg.model == "openai/zzz" and cfg.seed == 7
    assert cfg.overrides == {"model": "openai/zzz", "seed": 7}
    assert cfg.label().startswith("default[model=zzz,seed=7,sr=2,prompts=(")  # resolved identity, sanitized model


def test_resolve_config_applies_sampling_override() -> None:
    args = argparse.Namespace(config="default", model=None, seed=None, summary_rounds=None, sampling="qwen-no-thinking")
    cfg = runner.resolve_config(args)
    assert cfg.sampling_preset == "qwen-no-thinking"
    assert cfg.overrides.get("sampling") == "qwen-no-thinking"
    assert ",sampling=qwen-no-thinking," in cfg.label()


def test_apply_verbose() -> None:
    with _pinned_env("R3CON_LOG_LEVEL"):
        os.environ.pop("R3CON_LOG_LEVEL", None)
        runner.apply_verbose(argparse.Namespace(verbose=False))
        assert "R3CON_LOG_LEVEL" not in os.environ
        runner.apply_verbose(argparse.Namespace(verbose=True))
        assert os.environ["R3CON_LOG_LEVEL"] == "INFO"


def test_apply_doc_workers() -> None:
    with _pinned_env("R3CON_DOC_WORKERS"):
        os.environ.pop("R3CON_DOC_WORKERS", None)
        runner.apply_doc_workers(argparse.Namespace(doc_workers=None))  # unset → leaves env alone
        assert "R3CON_DOC_WORKERS" not in os.environ
        runner.apply_doc_workers(argparse.Namespace(doc_workers=8))
        assert os.environ["R3CON_DOC_WORKERS"] == "8"


def test_strategies_from_arg() -> None:
    assert runner.strategies_from_arg("llm") == ("llm",)
    assert runner.strategies_from_arg("codeact") == ("codeact",)
    assert runner.strategies_from_arg("both") == ("llm", "codeact")


def test_print_launch_summary() -> None:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        print_launch_summary(LaunchSummary(n=3, ok=2, failed=1, failures=[("t2", -9)]))
    out = buf.getvalue()
    assert "2/3" in out and "t2" in out and "signal 9" in out and "logs/r3con" in out


# ---------- resume scan ----------


def _run_folder(root, name: str, *, label_cfg: dict, task_id: str, strategies: dict) -> None:
    """A fake run folder: manifest.json + one inference/<strategy>/ per entry.

    ``strategies`` maps a strategy name to "result" (answered), "ctx" (a settled
    context-window error) or anything else (a transient error).
    """
    folder = root / name
    (folder / "inference").mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(
        json.dumps({"task_id": task_id, "benchmark": "Loong", "config": label_cfg})
    )
    for strategy, kind in strategies.items():
        d = folder / "inference" / strategy
        d.mkdir(parents=True, exist_ok=True)
        if kind == "result":
            (d / "result.json").write_text(json.dumps({"answer": "a"}))
        elif kind == "ctx":
            (d / "error.txt").write_text("litellm.ContextWindowExceededError: too long")
        else:
            (d / "error.txt").write_text("TimeoutError: flaky")


def test_completed_task_ids_resume_semantics(tmp_path) -> None:
    from evals.r3con.pipeline.config import load_config

    cfg = load_config("default")
    label, block = cfg.label(), cfg.model_dump()
    other = cfg.model_copy(update={"seed": cfg.seed + 1})

    # answered under both strategies -> done
    _run_folder(tmp_path, "a", label_cfg=block, task_id="t-done",
                strategies={"llm": "result", "codeact": "result"})
    # a settled context-window error counts as done (re-running fails identically)
    _run_folder(tmp_path, "b", label_cfg=block, task_id="t-ctx",
                strategies={"llm": "result", "codeact": "ctx"})
    # answered under only ONE of the two requested strategies -> not done
    _run_folder(tmp_path, "c", label_cfg=block, task_id="t-partial",
                strategies={"llm": "result"})
    # a transient error is retried, not skipped
    _run_folder(tmp_path, "d", label_cfg=block, task_id="t-flaky",
                strategies={"llm": "result", "codeact": "boom"})
    # a different run identity must not count, however complete it is
    _run_folder(tmp_path, "e", label_cfg=other.model_dump(), task_id="t-other-seed",
                strategies={"llm": "result", "codeact": "result"})
    # an unreadable manifest is skipped rather than crashing the scan
    (tmp_path / "f").mkdir()
    (tmp_path / "f" / "manifest.json").write_text("{ not json")

    assert completed_task_ids(tmp_path, label, ("llm", "codeact")) == {"t-done", "t-ctx"}
    # asking for one strategy only widens what counts as finished
    assert completed_task_ids(tmp_path, label, ("llm",)) == {
        "t-done", "t-ctx", "t-partial", "t-flaky"
    }
    # a missing log root is empty, not an error
    assert completed_task_ids(tmp_path / "nope", label, ("llm",)) == set()
