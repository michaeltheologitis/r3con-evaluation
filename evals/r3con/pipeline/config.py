"""The run config: every knob that shapes the OUTPUT, in one place.

A :class:`RunConfig` is the **experiment identity** — the model, seed, number of
summary rounds, the prompt version of each stage, and the sampling params. It is
loaded from ``configs/<name>.yaml`` (with optional CLI overrides) and flows through
the whole pipeline as one object, so nothing output-affecting is threaded ad-hoc.
Runtime-only knobs (parallelism, paths, retry caps, timeouts) stay in
:mod:`evals.r3con.pipeline.settings` — they don't change a correct answer.

Sampling presets — the generation params (``temperature``, ``top_p``, ``extra_body`` …)
— live in their own folder ``configs/sampling/<name>.yaml`` and are referenced **by
name** from a config's ``sampling:`` field, so one preset is reusable across configs.

Config file shape (``configs/<name>.yaml``)::

    model: openai/gpt-5.4-nano
    seed: 42
    summary_rounds: 2
    prompts:
      summaries: v3
      proposer: v3
      extractor: v2
      inference/llm: v1
      inference/codeact: v1
    sampling: qwen-no-thinking   # a preset name (configs/sampling/<name>.yaml), or omit for none
"""

from __future__ import annotations

from typing import Any

import yaml
from pydantic import BaseModel, Field

from evals.r3con.pipeline.settings import settings

# The pipeline stages that carry a versioned prompt — a config must pin all of them.
PROMPT_STAGES: tuple[str, ...] = (
    "summaries",
    "proposer",
    "extractor",
    "inference/llm",
    "inference/codeact",
)

DEFAULT_CONFIG = "default"

# Compact per-stage abbreviations for the prompts tag in RunConfig.label().
_PROMPT_STAGE_ABBR: dict[str, str] = {
    "summaries": "sum",
    "proposer": "prop",
    "extractor": "ext",
    "inference/llm": "llm",
    "inference/codeact": "cod",
}


def normalize_model_name(model: str | None) -> str | None:
    """Reduce a LiteLLM model string to the bare model name — drop the provider/route
    prefix, which is *transport*, not identity::

        openai/gpt-5.4-nano             -> gpt-5.4-nano
        hosted_vllm/Qwen/Qwen3.5-35B-A3B -> Qwen3.5-35B-A3B
        gpt-5.4-nano                    -> gpt-5.4-nano   (no prefix → unchanged)
        None                            -> None

    Used wherever the model is **persisted or shown** (the run label, the manifest's
    recorded config block, the run header, transcript headers). The *live* call site
    keeps the full provider-prefixed string — LiteLLM needs it to route."""
    if not model:
        return model
    return model.split("/")[-1]


class RunConfig(BaseModel):
    """Everything that shapes the output of one run (the experiment identity)."""

    name: str = DEFAULT_CONFIG  # the config it was loaded from (the run label's base)
    model: str
    seed: int = 42
    summary_rounds: int = 2  # keep in sync with configs/default.yaml (the shipped default)
    prompts: dict[str, str]  # {stage: version} for every stage in PROMPT_STAGES
    sampling: dict[str, Any] = Field(default_factory=dict)  # RESOLVED generation params
    sampling_preset: str | None = None  # the preset name `sampling` resolved from (None if none/inline)
    overrides: dict[str, Any] = Field(default_factory=dict)  # CLI overrides applied (recorded in the manifest)

    def _identity_parts(self) -> list[str]:
        """The output-shaping components that make up the run identity, in label order, each a
        ``key=value`` string. **This is the one place to extend the identity** — add a line here
        and the new axis flows into the run label *and* the resume key. Values are
        the *resolved* config (not how they were set), so seed 42 reads the same whether it came
        from the config file or ``--seed 42``. The model is reduced to its bare name (the
        provider/route prefix is transport — see :func:`normalize_model_name`)."""
        prompts = ",".join(
            f"{_PROMPT_STAGE_ABBR.get(s, s)}={self.prompts[s]}"
            for s in PROMPT_STAGES
            if s in self.prompts
        )
        parts = [
            f"model={normalize_model_name(self.model)}",
            f"seed={self.seed}",
            f"sr={self.summary_rounds}",
        ]
        # Sampling is part of the identity only when set (no `sampling=` for API-default runs):
        # the preset name when there is one, else a compact repr of the inline params.
        if self.sampling_preset:
            parts.append(f"sampling={self.sampling_preset}")
        elif self.sampling:
            parts.append("sampling={" + ",".join(f"{k}={self.sampling[k]}" for k in sorted(self.sampling)) + "}")
        parts.append(f"prompts=({prompts})")
        return parts

    def label(self) -> str:
        """Readable run label — the run's identity, and the key a re-run resumes on. The config
        name plus the run-identity components (:meth:`_identity_parts`), e.g.
        ``default[model=gpt-5.4-nano,seed=42,sr=3,prompts=(sum=v3,prop=v4,ext=v2,llm=v4,cod=v3)]``.
        Two runs are the same run iff their labels match, so **every** output-shaping axis must
        appear in ``_identity_parts``."""
        return f"{self.name}[" + ",".join(self._identity_parts()) + "]"


def available_configs() -> list[str]:
    """The experiment-config names under ``configs/`` (the top-level YAML stems)."""
    if not settings.CONFIGS_DIR.is_dir():
        return []
    return sorted(p.stem for p in settings.CONFIGS_DIR.glob("*.yaml"))


def available_sampling_presets() -> list[str]:
    """The sampling-preset names under ``configs/sampling/`` (the YAML stems)."""
    d = settings.CONFIGS_DIR / "sampling"
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def _resolve_sampling(spec: Any) -> tuple[dict[str, Any], str | None]:
    """A config's ``sampling`` field → ``(params, preset_name | None)``.

    The field may be a **preset name** (string → ``configs/sampling/<name>.yaml``),
    null / omitted (→ no params), or an inline dict (used as-is, no preset name).
    """
    if spec is None or spec == "":
        return {}, None
    if isinstance(spec, str):
        path = settings.CONFIGS_DIR / "sampling" / f"{spec}.yaml"
        if not path.is_file():
            raise KeyError(
                f"Unknown sampling preset {spec!r} (no {path}). Known: {available_sampling_presets()}"
            )
        return (yaml.safe_load(path.read_text()) or {}), spec
    if isinstance(spec, dict):
        return dict(spec), None
    raise ValueError(f"`sampling` must be a preset name, a dict, or null — got {type(spec).__name__}.")


def load_config(name: str = DEFAULT_CONFIG, **overrides: Any) -> RunConfig:
    """Load ``configs/<name>.yaml`` into a :class:`RunConfig`, resolving its sampling
    preset, then apply any non-``None`` CLI overrides (``model`` / ``seed`` /
    ``summary_rounds`` / ``sampling``). The ``sampling`` override is a **preset name**
    (``configs/sampling/<name>.yaml``) — it replaces the config's sampling and shows in the
    run label, so you can pick e.g. ``qwen-no-thinking`` at run time without a bundled config.

    Raises ``KeyError`` for an unknown config name or sampling preset, ``ValueError`` if the
    config omits a required field or a prompt version for any stage.
    """
    path = settings.CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise KeyError(f"Unknown config {name!r} (no {path}). Known: {available_configs()}")
    raw = yaml.safe_load(path.read_text()) or {}
    if "model" not in raw:
        raise ValueError(f"config {name!r} must define a `model`.")

    sampling, preset = _resolve_sampling(raw.get("sampling"))
    prompts = dict(raw.get("prompts") or {})
    missing = [s for s in PROMPT_STAGES if s not in prompts]
    if missing:
        raise ValueError(f"config {name!r} is missing prompt versions for stage(s): {missing}")

    cfg = RunConfig(
        name=name,
        model=raw["model"],
        seed=int(raw.get("seed", 42)),
        summary_rounds=int(raw.get("summary_rounds", 2)),
        prompts=prompts,
        sampling=sampling,
        sampling_preset=preset,
    )

    # CLI overrides, recorded in `overrides` (for the manifest + so the runner can re-pass them
    # to each child). Scalars are set directly; `sampling` is a preset name → resolve it.
    applied = {k: v for k, v in overrides.items() if v is not None and k in {"model", "seed", "summary_rounds"}}
    update: dict[str, Any] = dict(applied)
    if overrides.get("sampling") is not None:
        params, p = _resolve_sampling(overrides["sampling"])
        update["sampling"], update["sampling_preset"] = params, p
        applied["sampling"] = overrides["sampling"]  # the preset name, for the manifest + runner re-pass
    if update:
        cfg = cfg.model_copy(update={**update, "overrides": applied})
    return cfg
