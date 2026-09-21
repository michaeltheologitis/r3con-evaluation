"""RAPTOR LLM seams — re-host of upstream's summarization + QA models over the
harness's LiteLLM transport (PROVENANCE D2).

RAPTOR uses an LLM in TWO roles, both of which we re-seam onto ``litellm.completion``
(so vLLM / OpenAI / Ollama all work via provider prefixes + ``--base-url`` — RAPTOR
itself only ships raw-``openai`` clients with no vLLM path; the LiteLLM seam IS the
vLLM connection):

  • ``RaptorSummarizationModel`` (``BaseSummarizationModel``) — the recursive
    cluster-summary LLM at TREE-BUILD time (one call per cluster, per layer).
  • ``RaptorQAModel`` (``BaseQAModel``) — the single final-answer call over the
    retrieved context.

Both subclass RAPTOR's own ABCs and are injected via ``RetrievalAugmentationConfig``,
so the vendored tree builder / retriever run UNMODIFIED. The PROMPTS are copied
VERBATIM from upstream's ``GPT3TurboSummarizationModel`` / ``GPT3TurboQAModel`` (only
the transport changed) — including RAPTOR's QuALITY-flavored QA wording ("…the best
full answer amongst the option to question…"), kept as-is for faithfulness even though
Loong is free-form.

DEVIATIONS (vs upstream, ledgered in PROVENANCE.md):
  • Transport: ``litellm.completion`` (+ ``num_retries`` transport retries, ``--seed``)
    instead of a raw ``openai`` client + tenacity.
  • ``max_tokens`` (D10/D12 — run RAPTOR on a reasoning model): the summary's ~N-token length
    control is a PROMPT hint, and the summarize call sends **no max_tokens** (v4) so reasoning has the
    full window; empty (runaway) summaries are retried. The QA call also sends NONE (uncapped),
    exactly like upstream. Runs use the served model's own sampling.
  • QA ``temperature=0`` is kept: upstream pins it in ``QAModels.py`` (three places) while
    its SUMMARIZER passes no temperature at all — the seam mirrors both halves of that split.
  • Usage is accumulated DETERMINISTICALLY (``usage_envelope`` per call, like
    structrag/arag) — the build fires many SYNC calls, which the
    litellm-callback ``usage_scope`` path under-counts. Thread-safe: RAPTOR builds the
    tree with ``use_multithreading=True``, so ``summarize`` is called from worker
    threads; a lock guards the shared accumulator.

Full deviation ledger: evals/baselines/raptor/PROVENANCE.md
"""
from __future__ import annotations

import threading
from typing import Any

import litellm

from evals.baselines.raptor.upstream.QAModels import BaseQAModel
from evals.baselines.raptor.upstream.SummarizationModels import BaseSummarizationModel
from evals.llm.usage import _merge_numeric, usage_envelope

# Transport-level retries (litellm backs off on transient connection/5xx; a 400
# context-window error is NOT retried). Matches evals.llm.chat's default.
_NUM_RETRIES = 3

# RAPTOR's summary call is the only one upstream caps small (summarization_length=100); on a reasoning
# model that cap is consumed by reasoning_content → EMPTY summary. D10 moved the ~N-token length control
# into the PROMPT (a hint). As of v4 (D12) the summarize call sends **no max_tokens at all** — letting
# vLLM use the full remaining window so the model reasons then writes a complete summary. (`_SUMMARY_MAX_TOKENS`
# below is kept only for provenance — it is NOT sent.) Runaway reasoning can still return empty content
# (rare), so we retry a couple of times with a varied seed. The QA call is unchanged — uncapped, exactly
# like upstream `GPT3TurboQAModel`.
_SUMMARY_MAX_TOKENS = 32768  # historical (no longer sent — see ``summarize``); kept for provenance
_SUMMARY_EMPTY_RETRIES = 2


class _LiteLLMSeam:
    """Shared litellm transport + thread-safe ``{total, calls}`` usage accumulation +
    per-call full request/response capture (→ ``calls.json``). Subclassed for the two
    RAPTOR roles, which differ only in the prompt + per-call kwargs."""

    def __init__(self, litellm_kwargs: dict[str, Any], *, seed: int | None = None,
                 completion_params: dict[str, Any] | None = None):
        self._litellm_kwargs = litellm_kwargs
        self._seed = seed
        self._completion_params = completion_params or {}
        self._lock = threading.Lock()
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._full_calls: list[dict[str, Any]] = []
        # Optional live-progress hook (duck-typed ``.summary_done()``), attached by run.py for the
        # summarizer only so a long tree build's per-summary tick shows up in progress.json.
        self._progress: Any = None

    def _complete(self, role: str, messages: list[dict[str, Any]], **call_kwargs: Any) -> str:
        request: dict[str, Any] = {
            "model": self._litellm_kwargs["model"],
            "messages": messages,
            "num_retries": _NUM_RETRIES,
            **self._completion_params,   # run-config generation params (caller kwargs win below)
            **call_kwargs,               # role-specific (max_tokens / temperature)
        }
        if self._litellm_kwargs.get("api_base") is not None:
            request["api_base"] = self._litellm_kwargs["api_base"]
        if self._litellm_kwargs.get("api_key") is not None:
            request["api_key"] = self._litellm_kwargs["api_key"]
        # A per-call seed (e.g. summarize's empty-retry, passed in call_kwargs) wins; else the
        # seam's --seed; else send none at all (never seed=None).
        if request.get("seed") is None:
            if self._seed is not None:
                request["seed"] = self._seed
            else:
                request.pop("seed", None)
        response = litellm.completion(**request)
        message = response.choices[0].message
        self._record(role, request, response, message)
        return message.content or ""

    def _record(self, role: str, request: dict[str, Any], response: Any, message: Any) -> None:
        env = usage_envelope(response)
        with self._lock:
            for model, bucket in env["total"].items():
                _merge_numeric(self._total.setdefault(model, {}), bucket)
            self._calls.extend(env["calls"])
            entry: dict[str, Any] = {
                "role": role,
                "request": {
                    "model": self._litellm_kwargs["model"],
                    "seed": request.get("seed"),
                    "max_tokens": request.get("max_tokens"),
                    "temperature": request.get("temperature"),
                    "completion_params": self._completion_params,
                    "messages": request["messages"],
                },
                "content": getattr(message, "content", None),
            }
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning is not None:
                entry["reasoning_content"] = reasoning
            entry["response"] = _safe_dump(response)
            self._full_calls.append(entry)

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` completion-cost record."""
        with self._lock:
            return {"total": {m: dict(b) for m, b in self._total.items()},
                    "calls": list(self._calls)}

    @property
    def full_calls(self) -> list[dict[str, Any]]:
        """Per-call full request + response, written verbatim to ``calls.json``."""
        with self._lock:
            return list(self._full_calls)


class RaptorSummarizationModel(_LiteLLMSeam, BaseSummarizationModel):
    """RAPTOR's cluster-summary LLM over litellm. Upstream's summary sentence is kept VERBATIM;
    (D10) the incoming ``max_tokens`` is RAPTOR's ``summarization_length`` (the intended ~N-token
    summary), used here as a LENGTH HINT in the prompt — not the generation cap. The call sends **no
    max_tokens** (v4/D12) so a reasoning model has the full window. Empty content (rare runaway
    reasoning) is retried with a varied seed; a persistent empty falls through to the embedder's
    space-guard (D3)."""

    def summarize(self, context: str, max_tokens: int = 150, stop_sequence: Any = None) -> str:
        # Upstream prompt VERBATIM, plus the length hint (D10). ``max_tokens`` = summarization_length.
        user = (f"Write a summary of the following, including as many key details as possible: "
                f"{context}:\n\nIMPORTANT: keep your answer below {max_tokens} tokens.")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
        ]
        out = ""
        for attempt in range(_SUMMARY_EMPTY_RETRIES + 1):
            # Vary the seed across attempts so a retry isn't a deterministic re-roll of the same
            # empty generation. None when no --seed (sampling is already stochastic).
            seed = None if self._seed is None else self._seed + attempt
            # max_tokens deliberately NOT sent (was _SUMMARY_MAX_TOKENS): a reasoning model would
            # reserve/consume the cap → empty summary; sending none lets vLLM use the remaining window
            # so the model reasons, then writes a complete summary. (D10/D12.)
            out = self._complete("summarize", messages, seed=seed)
            if out.strip():
                break
        if self._progress is not None:
            self._progress.summary_done()  # live progress.json tick (one per cluster summarized)
        return out


class RaptorQAModel(_LiteLLMSeam, BaseQAModel):
    """RAPTOR's final-answer LLM over litellm. Prompt + ``temperature=0`` VERBATIM from
    upstream ``GPT3TurboQAModel._attempt_answer_question``, which sends NO ``max_tokens``
    (uncapped) — we keep that (no reasoning-model bump; RAPTOR runs non-thinking)."""

    def answer_question(self, context: str, question: str,
                        max_tokens: int | None = None, stop_sequence: Any = None) -> str:
        messages = [
            {"role": "system", "content": "You are Question Answering Portal"},
            {"role": "user",
             "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}"},
        ]
        # Upstream sends NO max_tokens; default None → omit it. temperature defaults to 0 (upstream);
        # a temperature in _completion_params would override it.
        kwargs: dict[str, Any] = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if "temperature" not in self._completion_params:
            kwargs["temperature"] = 0
        return self._complete("qa", messages, **kwargs).strip()


def _safe_dump(response: Any) -> Any:
    """The full litellm response as a dict (best-effort)."""
    try:
        return response.model_dump()
    except Exception:  # noqa: BLE001
        message = getattr(response.choices[0], "message", None)
        return {"content": getattr(message, "content", None)}
