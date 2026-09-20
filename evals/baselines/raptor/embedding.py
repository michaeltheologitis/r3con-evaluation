"""RAPTOR embedding seam — OpenAI embeddings in place of RAPTOR's local default
(PROVENANCE D3).

RAPTOR embeds every tree node (leaf chunks + each cluster summary) and the query, via
a ``BaseEmbeddingModel.create_embedding(text) -> list[float]`` (one text at a time). Its
shipped options are ``OpenAIEmbeddingModel`` (raw ``openai`` client) and
``SBertEmbeddingModel`` (local sentence-transformers). Per the project's
**embeddings-always-openai** rule (same as arag) we keep RAPTOR's
method (per-node single-text embedding, the vendored builder/retriever call
``create_embedding`` UNMODIFIED) and swap ONLY the backend to ``text-embedding-3-small``
via litellm — including the verbatim ``text.replace("\\n", " ")`` upstream applied.

Thread-safe: RAPTOR builds leaf nodes with ``use_multithreading=True``, so
``create_embedding`` is called from worker threads; a lock guards the usage accumulator.

Routing is by the embedding model's own provider prefix (``openai/…`` → OpenAI via
``OPENAI_API_KEY`` from the environment), deliberately INDEPENDENT of the completion
endpoint — the completion model may be a local vLLM (``--base-url``), but the embedding
stays an OpenAI call (forwarding the vLLM ``api_base`` would 404). Same as arag.

Full deviation ledger: evals/baselines/raptor/PROVENANCE.md
"""
from __future__ import annotations

import threading
from typing import Any

import litellm

from evals.baselines.raptor.upstream.EmbeddingModels import BaseEmbeddingModel
from evals.llm.usage import _merge_numeric, usage_envelope

_NUM_RETRIES = 3


class RaptorEmbeddingModel(BaseEmbeddingModel):
    """OpenAI embeddings via litellm, with RAPTOR's ``create_embedding(text) -> vector``
    interface the vendored tree builder + retriever rely on. Accumulates ``{total,
    calls}`` usage (thread-safe)."""

    def __init__(self, model: str):
        self.model = model
        self._lock = threading.Lock()
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []

    def create_embedding(self, text: str) -> list[float]:
        # `text.replace("\n", " ")` is verbatim upstream; the `or " "` is OURS (D3): an empty node
        # text (e.g. a reasoning model that emitted only reasoning and no summary) would make OpenAI
        # 400 with "input cannot be an empty string" and crash the whole (expensive) build. We embed
        # a single space instead — a degenerate node, but it must not kill a 462K-call run.
        text = text.replace("\n", " ") or " "
        # No api_base/api_key: litellm routes the `openai/…` model to OpenAI using
        # OPENAI_API_KEY from the environment (loaded from .env by _common).
        response = litellm.embedding(model=self.model, input=[text], num_retries=_NUM_RETRIES)
        self._record(response)
        item = response.data[0]
        return item["embedding"] if isinstance(item, dict) else item.embedding

    def _record(self, response: Any) -> None:
        env = usage_envelope(response)
        with self._lock:
            for model, bucket in env["total"].items():
                _merge_numeric(self._total.setdefault(model, {}), bucket)
            self._calls.extend(env["calls"])

    @property
    def usage(self) -> dict[str, Any]:
        with self._lock:
            return {"total": {m: dict(b) for m, b in self._total.items()},
                    "calls": list(self._calls)}
