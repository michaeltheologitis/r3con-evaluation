"""Tests for `evals.r3con.harness.loong.run_task.run_task` (offline — gr_answer + loong faked).

run_task writes into a flat ``logs/<run-folder>/`` (the folder is chosen by the
caller / runner, not derived from the config) and records the full parameter snapshot:
a ``config`` block (the experiment identity) + a ``settings`` block (runtime knobs).

Run with:  uv run pytest tests/r3con/test_run_task.py
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any, Iterator

from evals.r3con.pipeline.config import RunConfig
from evals.r3con.harness import loong
from evals.r3con.harness.loong import run_task as rt_mod
from evals.r3con.harness.loong.run_task import run_task
from evals.r3con.pipeline.runs import StageRun, TaskLogger
from evals.r3con.pipeline.settings import settings

_PROMPTS = {"summaries": "v3", "proposer": "v3", "extractor": "v2",
            "inference/llm": "v1", "inference/codeact": "v1"}


def _cfg(**over: Any) -> RunConfig:
    base: dict[str, Any] = dict(name="test", model="m", seed=0, summary_rounds=3,
                                prompts=dict(_PROMPTS), sampling={})
    base.update(over)
    return RunConfig(**base)


@contextlib.contextmanager
def _patched(*, answer_or_raise) -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as tmp:
        orig = (settings.LOGS_DIR, loong.load, loong.gold, rt_mod.gr_answer)
        settings.LOGS_DIR = Path(tmp)
        loong.load = lambda tid: loong.TaskInput(task="the task", documents=["d1", "d2"])
        loong.gold = lambda tid: "GOLD"

        def fake_gr(**kw):
            if isinstance(answer_or_raise, Exception):
                raise answer_or_raise
            return answer_or_raise

        rt_mod.gr_answer = fake_gr
        try:
            yield Path(tmp)
        finally:
            settings.LOGS_DIR, loong.load, loong.gold, rt_mod.gr_answer = orig


def test_writes_full_manifest_in_run_folder() -> None:
    with _patched(answer_or_raise={"codeact": ("ans", None)}) as root:
        outcome = run_task("t1", config=_cfg(), run_folder="RF1", strategies=("codeact",))
        assert outcome.results["codeact"] == ("ans", None)
        assert not outcome.failed
        man = json.loads((root / "RF1" / "manifest.json").read_text())
        # identity card
        assert man["benchmark"] == "Loong" and man["task_id"] == "t1"
        assert man["run_folder"] == "RF1"
        assert man["question"] == "the task" and man["gold"] == "GOLD" and man["n_docs"] == 2
        assert man["strategies"] == ["codeact"]
        assert "created" in man
        # config block = the experiment identity
        assert man["config"]["model"] == "m" and man["config"]["seed"] == 0
        assert man["config"]["summary_rounds"] == 3
        assert man["config"]["prompts"]["summaries"] == "v3"
        # run_label is NOT persisted — the board recomputes it from the config block
        assert "run_label" not in man
        assert RunConfig(**man["config"]).label() == "test[model=m,seed=0,sr=3,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v1)]"
        # settings block = runtime knobs (distinct from the config)
        assert man["settings"]["proposer_max_attempts"] == settings.PROPOSER_MAX_ATTEMPTS
        assert man["settings"]["llm_num_retries"] == settings.LLM_NUM_RETRIES


def test_run_label_recomputes_from_config_with_overrides() -> None:
    with _patched(answer_or_raise={"codeact": ("a", None)}) as root:
        run_task("t1", config=_cfg(seed=7, overrides={"seed": 7}), run_folder="RF2", strategies=("codeact",))
        man = json.loads((root / "RF2" / "manifest.json").read_text())
        assert "run_label" not in man
        assert RunConfig(**man["config"]).label() == "test[model=m,seed=7,sr=3,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v1)]"


def test_manifest_sanitizes_model_name() -> None:
    with _patched(answer_or_raise={"codeact": ("a", None)}) as root:
        full = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
        run_task("t1", config=_cfg(model=full, overrides={"model": full}), run_folder="RFm",
                 strategies=("codeact",))
        man = json.loads((root / "RFm" / "manifest.json").read_text())
        # provider/route prefix dropped in the persisted record (it's transport, not identity)
        assert man["config"]["model"] == "Qwen3.5-35B-A3B"
        assert man["config"]["overrides"]["model"] == "Qwen3.5-35B-A3B"


def test_passes_config_and_transport_through_to_gr() -> None:
    captured: dict = {}
    with _patched(answer_or_raise={"codeact": ("a", None)}):
        def capturing_gr(**kw):
            captured.update(kw)
            return {"codeact": ("a", None)}

        rt_mod.gr_answer = capturing_gr  # restored by _patched's finally
        run_task("t1", config=_cfg(model="zzz"), run_folder="RF3", strategies=("codeact",),
                 api_base="http://x", api_key="k")
    assert captured["config"].model == "zzz"
    assert captured["api_base"] == "http://x" and captured["api_key"] == "k"
    assert captured["strategies"] == ("codeact",)


def _gr_that_burns_tokens(*, raises: Exception | None = None):
    """A fake gr_answer that records one LLM call against the task logger (as every
    real stage does) and then either answers or blows up."""

    def fake(**kw):
        run = StageRun(stage="summaries", task_logger=kw["task_logger"], model="m")
        run.add_step(
            kind="summary-r1-d0", messages=[], response={"role": "assistant", "content": "s"},
            tokens={"prompt": 30, "completion": 4, "total": 34}, model="m",
            usage={"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34,
                   "completion_tokens_details": {"reasoning_tokens": 2}},
        )
        if raises is not None:
            raise raises
        return {"codeact": ("ans", None)}

    return fake


def test_manifest_gains_usage_block() -> None:
    """The manifest is re-written at the end of the run with the task's complete token
    cost, in the same ``{total, calls}`` shape the baselines write — and the identity
    fields written at t=0 survive the rewrite."""
    with _patched(answer_or_raise={"codeact": ("ans", None)}) as root:
        rt_mod.gr_answer = _gr_that_burns_tokens()  # restored by _patched's finally
        run_task("t1", config=_cfg(), run_folder="RFu", strategies=("codeact",))
        man = json.loads((root / "RFu" / "manifest.json").read_text())
    assert man["usage"]["total"]["m"] == {
        "num_calls": 1, "prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34,
        "completion_tokens_details": {"reasoning_tokens": 2},
    }
    assert man["usage"]["calls"][0]["model"] == "m"
    assert man["task_id"] == "t1" and man["gold"] == "GOLD" and "config" in man


def test_manifest_usage_present_even_when_the_pipeline_raises() -> None:
    """A crashed task still reports what it burned before it died."""
    with _patched(answer_or_raise={"codeact": ("ans", None)}) as root:
        rt_mod.gr_answer = _gr_that_burns_tokens(raises=ValueError("proposer blew up"))
        outcome = run_task("t1", config=_cfg(), run_folder="RFx", strategies=("codeact",))
        man = json.loads((root / "RFx" / "manifest.json").read_text())
    assert outcome.failed
    assert man["usage"]["total"]["m"]["num_calls"] == 1
    assert man["usage"]["total"]["m"]["total_tokens"] == 34


def test_usage_rollup_failure_does_not_sink_a_task_that_answered() -> None:
    """Cost accounting is bookkeeping: if the roll-up (or the manifest rewrite) blows up,
    the task must still report its answer, and the failure must be discoverable on disk.
    Otherwise a run that genuinely answered gets recorded as a crash."""
    with _patched(answer_or_raise={"codeact": ("ans", None)}) as root:
        orig_rollup = TaskLogger.usage_rollup
        TaskLogger.usage_rollup = lambda self: (_ for _ in ()).throw(RuntimeError("rollup boom"))
        try:
            outcome = run_task("t1", config=_cfg(), run_folder="RFb", strategies=("codeact",))
        finally:
            TaskLogger.usage_rollup = orig_rollup
        man = json.loads((root / "RFb" / "manifest.json").read_text())
        err = (root / "RFb" / "usage_error.txt").read_text()

    assert not outcome.failed                       # the answer survived
    assert outcome.results["codeact"] == ("ans", None)
    assert "usage" not in man                       # t=0 manifest intact, no usage block
    assert man["task_id"] == "t1"
    assert "rollup boom" in err                     # and the failure is on disk


def test_manifest_usage_is_empty_when_no_call_was_made() -> None:
    with _patched(answer_or_raise={"codeact": ("ans", None)}) as root:
        run_task("t1", config=_cfg(), run_folder="RFe", strategies=("codeact",))
        man = json.loads((root / "RFe" / "manifest.json").read_text())
    assert man["usage"] == {"total": {}, "calls": []}


def test_top_level_failure_writes_error_txt() -> None:
    with _patched(answer_or_raise=ValueError("proposer blew up")) as root:
        outcome = run_task("t1", config=_cfg(), run_folder="RF4", strategies=("llm", "codeact"))
        assert outcome.failed
        for strat in ("llm", "codeact"):
            err = root / "RF4" / "inference" / strat / "error.txt"
            assert err.is_file() and "ValueError" in err.read_text()
            assert outcome.results[strat][0] == "" and "ValueError" in outcome.results[strat][1]
