"""Named sampling-parameter overrides for completion models.

Each entry is a bundle of generation params handed to ``litellm.completion`` when
selected via ``--config <name>`` on a runner. The **model is NOT here** — it's
passed separately via ``--model`` (and the endpoint via ``--base-url`` / ``--api-key``);
a config is purely the parameters that *override* the default completion behaviour.
When a config is selected its params land in the run's ``run_config``, so they are
hashed (a different sampling config is a different inference) and recorded in the
manifest.

Param placement follows the OpenAI/litellm split:
- top-level: the OpenAI-standard params (``temperature``, ``top_p``, ``presence_penalty``);
- ``extra_body``: the vLLM-specific ones (``top_k``, ``min_p``, ``repetition_penalty``)
  plus ``chat_template_kwargs`` (e.g. ``enable_thinking``).

Add a config by pasting a new entry.
"""
from __future__ import annotations

# Only the Instruct (non-thinking) preset lives here — it's the one setting that
# DEVIATES from the served vLLM defaults (it disables the thinking template + uses
# the instruct sampling values). Thinking mode IS the model's served default, so for
# it just omit --config. These values match Qwen3.5 MoE across the 122B-A10B and
# 35B-A3B sizes.
MODEL_SAMPLING_CONFIG: dict[str, dict] = {
    "Qwen3.5-MoE-Instruct": {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
}
