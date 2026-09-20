"""A-RAG embedding seam — OpenAI embeddings in place of A-RAG's local Qwen3-Embedding
(PROVENANCE D3).

A-RAG's ``semantic_search`` is **sentence-level** dense retrieval: ``scripts/build_index.py``
splits each chunk into sentences and embeds them into ``sentence_index.pkl``; at query time
``SemanticSearchTool`` embeds the query and cosine-ranks sentences, grouping the top ones
back to chunks. The paper/repo embed with a local ``Qwen3-Embedding-0.6B`` (sentence-
transformers, GPU).

Per the project's **embeddings-always-openai** rule (same as arag), we keep that METHOD
(sentence split → embed → top-k group, byte-for-byte in the vendored ``execute``) and swap
ONLY the embedding backend to OpenAI ``text-embedding-3-small`` via litellm. Two pieces:

  • ``OpenAIEmbedder`` — exposes the slice of ``SentenceTransformer.encode`` the vendored
    code uses (``encode(list[str], normalize_embeddings=True) -> np.ndarray``), batched over
    litellm, L2-normalized (the vendored cosine is a plain ``np.dot``), usage accumulated.
  • ``build_sentence_index`` — replaces the ``scripts/build_index.py`` SCRIPT (not imported
    upstream) with the SAME sentence split + pickle shape, embedding via ``OpenAIEmbedder``.
  • ``OpenAISemanticSearchTool`` — a thin ``SemanticSearchTool`` subclass that injects the
    OpenAI embedder + loads the index, inheriting the vendored ``execute`` / ``get_schema``
    UNCHANGED (so the retrieval logic is exactly upstream's).

Full deviation ledger: evals/baselines/arag/PROVENANCE.md
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import litellm
import numpy as np

from evals.baselines.arag.upstream.tools.semantic_search import SemanticSearchTool
from evals.llm.usage import _merge_numeric, usage_envelope

_NUM_RETRIES = 3
_EMBED_BATCH = 256  # inputs per OpenAI embedding call

_INDEX_FILE = "sentence_index.pkl"


def _split_sentences(text: str) -> list[str]:
    """Sentence split, byte-for-byte from upstream ``build_index.split_sentences``:
    split on ``[.!?\\n]+`` and drop fragments of length ≤ 10."""
    sentences = re.split(r"[.!?\n]+", text)
    return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]


def _load_chunks(rows: list[str]) -> list[dict[str, str]]:
    """``["id:text", …]`` → ``[{"id","text"}]`` (upstream ``build_index.load_chunks``)."""
    if rows and isinstance(rows[0], dict):
        return rows  # already in {id,text} form
    chunks: list[dict[str, str]] = []
    for item in rows:
        if isinstance(item, str):
            parts = item.split(":", 1)
            if len(parts) == 2:
                chunks.append({"id": parts[0], "text": parts[1]})
    return chunks


class OpenAIEmbedder:
    """OpenAI embeddings via litellm, with the ``SentenceTransformer.encode`` interface
    the vendored index build + tool use. Accumulates ``{total, calls}`` usage.

    Routing is by the **embedding model's own provider prefix** (``openai/…`` →
    OpenAI via ``OPENAI_API_KEY`` from the environment), exactly like graphrag's
    embedding ``ModelConfig`` — deliberately INDEPENDENT of the completion endpoint.
    The completion model may be a local vLLM (``--base-url``), but per the project's
    embeddings-always-openai rule the embedding is a separate OpenAI call; forwarding
    the vLLM ``api_base`` here would 404 (the vLLM server doesn't serve
    text-embedding-3-small). A self-hosted embedding ENDPOINT would need its own
    base-url flag (none today; graphrag punts the same way)."""

    def __init__(self, model: str):
        self.model = model
        self._total: dict[str, Any] = {}
        self._calls: list[dict[str, Any]] = []

    def encode(self, texts, normalize_embeddings: bool = True,
               batch_size: int = _EMBED_BATCH, show_progress_bar: bool = False, **_) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            # No api_base/api_key: litellm routes the `openai/…` model to OpenAI using
            # OPENAI_API_KEY from the environment (loaded from .env by _common).
            response = litellm.embedding(model=self.model, input=batch, num_retries=_NUM_RETRIES)
            self._record(response)
            for item in response.data:
                vectors.append(item["embedding"] if isinstance(item, dict) else item.embedding)
        arr = np.asarray(vectors, dtype=np.float32)
        if normalize_embeddings and len(arr):
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr

    def _record(self, response: Any) -> None:
        env = usage_envelope(response)
        for model, bucket in env["total"].items():
            _merge_numeric(self._total.setdefault(model, {}), bucket)
        self._calls.extend(env["calls"])

    @property
    def usage(self) -> dict[str, Any]:
        return {"total": self._total, "calls": self._calls}


def build_sentence_index(rows: list[str], embedder: OpenAIEmbedder, index_dir: Path | str) -> None:
    """Build A-RAG's ``sentence_index.pkl`` from the ``["id:text", …]`` chunks — same
    sentence split + pickle shape as ``scripts/build_index.py``, embedding via ``embedder``."""
    chunks = _load_chunks(rows)
    chunk_lookup = {c["id"]: c for c in chunks}
    sentences: list[str] = []
    sentence_to_chunk: list[str] = []
    for chunk in chunks:
        for sentence in _split_sentences(chunk["text"]):
            sentences.append(sentence)
            sentence_to_chunk.append(chunk["id"])

    embeddings = (embedder.encode(sentences, normalize_embeddings=True)
                  if sentences else np.zeros((0, 1), dtype=np.float32))

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    with open(index_dir / _INDEX_FILE, "wb") as f:
        pickle.dump({
            "sentences": sentences,
            "embeddings": embeddings,
            "sentence_to_chunk": sentence_to_chunk,
            "chunks": chunk_lookup,
            "model_name": embedder.model,
        }, f)


class OpenAISemanticSearchTool(SemanticSearchTool):
    """``SemanticSearchTool`` with the OpenAI embedder injected (no SentenceTransformer).

    Bypasses the vendored ``__init__`` (which would construct a ``SentenceTransformer``)
    and sets the same attributes its ``execute`` / ``_load_index`` rely on, so the
    retrieval logic + tool schema are inherited UNCHANGED."""

    def __init__(self, chunks_file: str, index_dir: str, embedder: OpenAIEmbedder):
        import tiktoken
        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.model_name = embedder.model
        self.device = None
        self.embedding_model = embedder  # .encode(list, normalize_embeddings=True) -> np.ndarray
        self._load_index()               # sets sentences / embeddings / sentence_to_chunk / chunks
        self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
