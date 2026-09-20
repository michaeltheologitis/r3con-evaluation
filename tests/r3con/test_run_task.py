"""Tests for `evals.r3con.harness.loong.run_task.run_task` (offline — gr_answer + loong faked).

run_task writes into a flat ``logs/<run-folder>/`` (the folder is chosen by the
caller / runner, not derived from the config) and records the full parameter snapshot:
a ``config`` block (the experiment identity) + a ``settings`` block (runtime knobs).

Run with:  uv run python tests/unit/test_run_task.py
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


def test_top_level_failure_writes_error_txt() -> None:
    with _patched(answer_or_raise=ValueError("proposer blew up")) as root:
        outcome = run_task("t1", config=_cfg(), run_folder="RF4", strategies=("llm", "codeact"))
        assert outcome.failed
        for strat in ("llm", "codeact"):
            err = root / "RF4" / "inference" / strat / "error.txt"
            assert err.is_file() and "ValueError" in err.read_text()
            assert outcome.results[strat][0] == "" and "ValueError" in outcome.results[strat][1]
