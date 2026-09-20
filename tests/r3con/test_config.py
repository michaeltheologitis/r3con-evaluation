"""Tests for `evals.r3con.pipeline.config` — the RunConfig (experiment identity) + its loaders.

The happy-path tests run against the real ``configs/`` folder; validation / error
cases use a temp configs dir (with a ``sampling/`` subfolder). Run with:
uv run python tests/unit/test_config.py
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator

import yaml

from evals.r3con.pipeline.config import available_configs, available_sampling_presets, load_config
from evals.r3con.pipeline.settings import settings

_FULL = {
    "model": "openai/gpt-5.4-nano",
    "seed": 42,
    "summary_rounds": 3,
    "prompts": {
        "summaries": "v3", "proposer": "v3", "extractor": "v2",
        "inference/llm": "v1", "inference/codeact": "v1",
    },
}


@contextlib.contextmanager
def _temp_configs(files: dict) -> Iterator[Path]:
    """Point ``settings.CONFIGS_DIR`` at a tmp dir holding ``files`` (``name -> dict``).
    A ``"sampling/<name>"`` key writes into the sampling subfolder."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, data in files.items():
            p = root / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(data))
        original = settings.CONFIGS_DIR
        settings.CONFIGS_DIR = root
        try:
            yield root
        finally:
            settings.CONFIGS_DIR = original


# ---------- available_* (real configs/) ----------


def test_available_configs_and_presets() -> None:
    assert "default" in available_configs()
    assert "qwen-no-thinking" in available_sampling_presets()


# ---------- load_config happy path (real configs/) ----------


def test_load_default_config() -> None:
    c = load_config("default")
    assert c.name == "default"
    assert c.model == "openai/gpt-5.4-nano"
    assert c.seed == 42 and c.summary_rounds == 2
    assert c.sampling == {} and c.sampling_preset is None
    assert c.prompts["summaries"] == "v3" and c.prompts["inference/codeact"] == "v4"
    assert c.prompts["inference/llm"] == "v4"
    assert c.label() == "default[model=gpt-5.4-nano,seed=42,sr=2,prompts=(sum=v3,prop=v4,ext=v2,llm=v4,cod=v4)]"


# ---------- CLI overrides ----------


def test_overrides_apply_and_show_in_label() -> None:
    c = load_config("default", seed=7, model="openai/gpt-x")
    assert c.seed == 7 and c.model == "openai/gpt-x"
    assert c.overrides == {"seed": 7, "model": "openai/gpt-x"}
    # the label renders the RESOLVED identity (seed 7, bare model) — not the override mechanics
    assert c.label().startswith("default[model=gpt-x,seed=7,sr=2,prompts=(")


def test_label_sanitizes_model_override() -> None:
    """The label shows the bare model name (provider/route prefix dropped — it's transport),
    while the live config keeps the full string for LiteLLM routing."""
    c = load_config("default", model="hosted_vllm/Qwen/Qwen3.5-35B-A3B")
    assert c.model == "hosted_vllm/Qwen/Qwen3.5-35B-A3B"  # live model unchanged (routing)
    assert c.label().startswith("default[model=Qwen3.5-35B-A3B,seed=42,sr=2,prompts=(")


def test_none_overrides_are_ignored() -> None:
    c = load_config("default", seed=None, model=None, summary_rounds=None)
    assert c.overrides == {} and c.seed == 42
    assert c.label().startswith("default[model=gpt-5.4-nano,seed=42,sr=2,prompts=(")


def test_sampling_override_resolves_and_labels() -> None:
    """`--sampling <preset>` picks a preset by name at run time (no bundled config): it resolves the
    params, records the override (so the runner re-passes it), and shows in the label — but only
    when sampling is set (no `sampling=` component for API-default runs)."""
    c = load_config("default", sampling="qwen-no-thinking")
    assert c.sampling_preset == "qwen-no-thinking"
    assert c.sampling["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert c.overrides.get("sampling") == "qwen-no-thinking"
    assert ",sampling=qwen-no-thinking," in c.label()
    # not given → no sampling component in the label
    assert "sampling=" not in load_config("default").label()


def test_label_is_full_resolved_identity() -> None:
    """The label is built from the resolved identity (model, seed, summary_rounds, prompts) — so
    two runs differing in ANY of these are distinct cells, even under the same config name."""
    bumped = {**_FULL, "prompts": {**_FULL["prompts"], "inference/codeact": "v9"}}
    with _temp_configs({"exp": _FULL, "exp2": bumped}):
        base = "exp[model=gpt-5.4-nano,seed=42,sr=3,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v1)]"
        assert load_config("exp").label() == base
        # seed and summary_rounds are part of the identity (always, not just when overridden)
        assert load_config("exp", seed=7).label() == \
            "exp[model=gpt-5.4-nano,seed=7,sr=3,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v1)]"
        assert load_config("exp", summary_rounds=5).label() == \
            "exp[model=gpt-5.4-nano,seed=42,sr=5,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v1)]"
        # bumping ONE prompt stage changes the label too
        assert load_config("exp2").label() == \
            "exp2[model=gpt-5.4-nano,seed=42,sr=3,prompts=(sum=v3,prop=v3,ext=v2,llm=v1,cod=v9)]"


def test_model_dump_round_trips_for_manifest() -> None:
    d = load_config("default", seed=9).model_dump()
    assert d["model"] == "openai/gpt-5.4-nano" and d["seed"] == 9
    assert d["overrides"] == {"seed": 9} and "prompts" in d and "sampling" in d


# ---------- sampling field: name / inline / none (temp configs) ----------


def test_sampling_field_name_inline_and_none() -> None:
    files = {
        "sampling/myp": {"temperature": 0.3},
        "named": {**_FULL, "sampling": "myp"},
        "inline": {**_FULL, "sampling": {"top_p": 0.9}},
        "nosamp": _FULL,
    }
    with _temp_configs(files):
        named = load_config("named")
        assert named.sampling == {"temperature": 0.3} and named.sampling_preset == "myp"
        inline = load_config("inline")
        assert inline.sampling == {"top_p": 0.9} and inline.sampling_preset is None
        nosamp = load_config("nosamp")
        assert nosamp.sampling == {} and nosamp.sampling_preset is None


# ---------- validation / errors ----------


def test_unknown_config_raises() -> None:
    try:
        load_config("does-not-exist-xyz")
    except KeyError as e:
        assert "does-not-exist-xyz" in str(e)
    else:
        raise AssertionError("expected KeyError")


def test_unknown_sampling_preset_raises() -> None:
    with _temp_configs({"bad": {**_FULL, "sampling": "ghost"}}):
        try:
            load_config("bad")
        except KeyError as e:
            assert "ghost" in str(e)
        else:
            raise AssertionError("expected KeyError")


def test_missing_model_raises() -> None:
    bad = {k: v for k, v in _FULL.items() if k != "model"}
    with _temp_configs({"bad": bad}):
        try:
            load_config("bad")
        except ValueError as e:
            assert "model" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_missing_prompt_stage_raises() -> None:
    with _temp_configs({"bad": {**_FULL, "prompts": {"summaries": "v3"}}}):
        try:
            load_config("bad")
        except ValueError as e:
            assert "prompt versions" in str(e)
        else:
            raise AssertionError("expected ValueError")
