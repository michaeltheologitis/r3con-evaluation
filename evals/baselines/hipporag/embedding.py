"""HippoRAG embedding seam — OpenAI ``text-embedding-3-small``.

HippoRAG embeds three stores (passages/chunks, entities, facts) via
``embedding_model.batch_encode(texts, **kwargs) -> np.ndarray``. Upstream's
``embedding_model.OpenAI.OpenAIEmbeddingModel`` imports torch + transformers at module
top (unused on the OpenAI path) and — critically — its ``batch_encode`` wraps each
request in ``except: import ipdb; ipdb.set_trace()``, which HANGS a headless run on ANY
embedding error. This seam reproduces its behaviour (same normalized-vector contract,
same optional query instruction) but:

  * D7: uses a plain ``openai`` client for ``text-embedding-3-small`` — the
    embeddings-always-OpenAI rule. Even when the completion LLM is served by vLLM
    (``--base-url``), embeddings go to OpenAI (its own ``OPENAI_API_KEY``), never the
    vLLM endpoint.
  * D8: removes the ``ipdb.set_trace()`` failure trap — an embedding error RAISES
    (so the task records a clean ``error.json``) instead of dropping into a debugger.
  * D9: tiktoken-safe — each input is truncated to the model's 8,191-token cap before
    the call (our chunk sizes are ≤ 8,000, so this only ever guards a pathological
    input); requests are batched under OpenAI's per-request token budget.

Injected via a monkeypatch of ``embedding_model._get_embedding_model_class`` (see
run.py), so the vendored ``EmbeddingStore`` uses it unmodified.
"""
from __future__ import annotations

import threading
from typing import Any, List

import numpy as np

from evals.baselines.hipporag.upstream.hipporag.embedding_model.base import BaseEmbeddingModel
from evals.llm.usage import _merge_numeric, usage_envelope

# text-embedding-3-small: 1536-dim, 8,191-token max PER INPUT, 300,000-token max PER REQUEST.
_EMBED_DIM = 1536
_MAX_INPUT_TOKENS = 8000          # headroom under the hard 8,191 per-input cap (D9)
_MAX_REQUEST_TOKENS = 120000      # tiktoken-estimated budget (the API's own count can run
                                  # higher on dense/numeric text, so this is the first line of
                                  # defense; _embed_request recursively splits on the real cap)
_MAX_REQUEST_ITEMS = 512          # OpenAI also caps the array length
# Retries for the embeddings API. HippoRAG bursts a lot of embedding tokens (a big
# CorpusQA task embeds ~1M+ tokens of passages/entities/facts), so under high
# --max-workers the OpenAI per-minute token limit is easy to hit. The SDK retries 429s
# with exponential backoff + honors Retry-After, so a generous cap rides out bursts
# instead of failing the (expensive) task. Bigger than the SDK default of 2.
_EMBED_MAX_RETRIES = 10
_PROVIDER_PREFIXES = ("openai/", "hosted_vllm/", "ollama_chat/", "ollama/")


def _strip_provider(model: str) -> str:
    for p in _PROVIDER_PREFIXES:
        if model.startswith(p):
            return model[len(p):]
    return model


def _is_token_limit_error(exc: Exception) -> bool:
    """True for OpenAI's per-request token-cap error (so we split-and-retry rather than
    fail the task) — matched on the API's own code/message, not the exception class."""
    msg = str(getattr(exc, "message", "") or exc).lower()
    return "max_tokens_per_request" in msg or "tokens per request" in msg


class HippoRAGOpenAIEmbedder(BaseEmbeddingModel):
    """OpenAI embedding seam with HippoRAG's ``batch_encode`` contract.

    Constructed by the (monkeypatched) factory as
    ``HippoRAGOpenAIEmbedder(global_config=…, embedding_model_name=…)``.
    """

    embedding_dim = _EMBED_DIM

    def __init__(self, global_config: Any = None, embedding_model_name: str | None = None) -> None:
        super().__init__(global_config=global_config)
        name = embedding_model_name or getattr(global_config, "embedding_model_name", "text-embedding-3-small")
        self.embedding_model_name = _strip_provider(name)
        self._normalize = getattr(global_config, "embedding_return_as_normalized", True)
        self._encoder = None      # tiktoken encoder, lazy
        self._client = None       # openai client, lazy
        # Index-build embedding cost — captured so the run's TOTAL usage folds in the
        # passage/entity/fact embeddings, not just the LLM calls (thread-safe: HippoRAG
        # may embed from worker threads).
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []
        self._usage_lock = threading.Lock()

    # ---- lazy resources ----

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            # reads OPENAI_API_KEY — always OpenAI (D7). max_retries rides out 429 bursts
            # (exponential backoff + Retry-After) so a rate limit doesn't fail the task.
            self._client = OpenAI(max_retries=_EMBED_MAX_RETRIES)
        return self._client

    def _enc(self):
        """The tiktoken encoder for this embedding model (cl100k for text-embedding-3),
        loaded once."""
        if self._encoder is None:
            import tiktoken
            try:
                self._encoder = tiktoken.encoding_for_model(self.embedding_model_name)
            except Exception:  # noqa: BLE001 — unknown model → cl100k
                self._encoder = tiktoken.get_encoding("cl100k_base")
        return self._encoder

    def _truncate(self, text: str) -> str:
        """Clip one input to the 8,191-token cap (D9). Empty → a single space (the
        embeddings API rejects empty strings)."""
        if not text:
            return " "
        ids = self._enc().encode(text)
        if len(ids) <= _MAX_INPUT_TOKENS:
            return text
        return self._enc().decode(ids[:_MAX_INPUT_TOKENS])

    # ---- the contract HippoRAG calls ----

    def encode(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts, sub-batching so NO request exceeds OpenAI's per-request
        token budget (300K) or array-length cap — bulletproof regardless of how big a list
        the caller passes (OpenIE over large passages can hand us many long entity/fact
        strings at once). Each input is truncated to the 8,191 per-input cap first (D9)."""
        texts = [self._truncate(t.replace("\n", " ")) for t in texts]
        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        vecs: list[np.ndarray] = []
        batch: list[str] = []
        batch_tokens = 0
        for t in texts:
            n = len(self._enc().encode(t))
            if batch and (batch_tokens + n > _MAX_REQUEST_TOKENS or len(batch) >= _MAX_REQUEST_ITEMS):
                vecs.append(self._embed_request(batch))
                batch, batch_tokens = [], 0
            batch.append(t)
            batch_tokens += n
        if batch:
            vecs.append(self._embed_request(batch))
        return np.concatenate(vecs, axis=0)

    def _embed_request(self, texts: List[str]) -> np.ndarray:
        """One embeddings API request over an already-truncated, budget-bounded batch.

        The tiktoken estimate in ``encode`` can undercount the API's own token count on
        dense/numeric text (e.g. financial tables), so a request can still trip the
        300K-tokens-per-request cap. When it does, split the batch in half and retry each
        half — this adapts to the REAL count and is guaranteed to converge (a single
        input is ≤ 8,191 tokens after truncation, well under the cap)."""
        try:
            response = self._get_client().embeddings.create(
                input=texts, model=self.embedding_model_name
            )
        except Exception as exc:  # noqa: BLE001
            if len(texts) > 1 and _is_token_limit_error(exc):
                mid = len(texts) // 2
                return np.concatenate(
                    [self._embed_request(texts[:mid]), self._embed_request(texts[mid:])],
                    axis=0,
                )
            raise
        self._record(response)
        return np.array([v.embedding for v in response.data], dtype=np.float32)

    def _record(self, response) -> None:
        """Fold this embedding call's usage into the ``{total, calls}`` record."""
        env = usage_envelope(response)
        with self._usage_lock:
            for model, bucket in env["total"].items():
                _merge_numeric(self._total.setdefault(model, {}), bucket)
            self._calls.extend(env["calls"])

    @property
    def usage(self) -> dict[str, Any]:
        """The accumulated ``{total, calls}`` embedding cost for the index build."""
        with self._usage_lock:
            return {"total": {m: dict(b) for m, b in self._total.items()},
                    "calls": list(self._calls)}

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Embed ``texts`` → ``(N, 1536)``. Honors an optional ``instruction`` kwarg
        (prepended, mirroring upstream's OpenAI embedder) and L2-normalizes when the
        config asks (HippoRAG scores by cosine)."""
        if isinstance(texts, str):
            texts = [texts]
        instruction = kwargs.get("instruction", "")
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        if not texts:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        # encode() sub-batches internally by token budget, so a huge list is safe.
        results = self.encode(texts)

        if self._normalize:
            norms = np.linalg.norm(results, axis=1, keepdims=True)
            results = results / np.clip(norms, 1e-12, None)
        return results
