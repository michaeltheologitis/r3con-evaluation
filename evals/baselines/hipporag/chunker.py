"""HippoRAG chunker — our own by-token splitter (the deviation that lets HippoRAG
run on whole-document benchmarks).

HippoRAG ships NO chunker: ``index(docs)`` treats each element of ``docs`` as one
passage and runs a per-passage OpenIE (NER + triple extraction) LLM pass. Its own
datasets are pre-chunked ~100-token Wikipedia paragraphs. Our benchmarks supply whole
large documents (financial reports, court judgments, research papers, 19th-century
journals and letters), so we must split them into passages ourselves — a choice the
authors never specify, so these sizes are a sanctioned deviation (PROVENANCE D1), not
an upstream recipe.

Per-benchmark token sizes (``CHUNK_SIZES``), chosen to keep the OpenIE-call count
tractable while staying under the OpenAI embedder's 8,191-token cap:
  * corpusqa    → 8,000  (capped under the embed limit; ≈ ⌈corpus/8000⌉ passages/task)
  * loong       → 3,000
  * dracula     → 3,000  (Loong's value — matching token profile)

A passage never spans two documents (each document is split independently, then the
passages are pooled) — so the graph's passage nodes stay document-scoped, as in
HippoRAG's native corpora. Split is by token (tiktoken ``cl100k_base``), no overlap
(HippoRAG's native passages are discrete, non-overlapping).
"""
from __future__ import annotations

from functools import lru_cache

# Per-benchmark chunk sizes (tokens). A benchmark absent here has no HippoRAG wiring.
CHUNK_SIZES: dict[str, int] = {
    "corpusqa": 8000,
    "loong": 3000,
    "dracula": 3000,
}

# Bump when the chunking logic changes in a way that alters passage boundaries.
CHUNKER_VERSION = "v1"


@lru_cache(maxsize=1)
def _encoder():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def _split_one(text: str, chunk_size: int) -> list[str]:
    """Split ONE document into ≤``chunk_size``-token passages (by token, no overlap)."""
    enc = _encoder()
    ids = enc.encode(text)
    if len(ids) <= chunk_size:
        return [text] if text.strip() else []
    passages = []
    for i in range(0, len(ids), chunk_size):
        piece = enc.decode(ids[i:i + chunk_size]).strip()
        if piece:
            passages.append(piece)
    return passages


def chunk_documents(documents: list[str], chunk_size: int) -> list[str]:
    """Pool a document bundle into passages: split each document independently into
    ≤``chunk_size``-token pieces, then concatenate (a passage never spans two docs).
    Returns the flat passage list that becomes HippoRAG's ``index(docs=…)`` input."""
    passages: list[str] = []
    for doc in documents:
        passages.extend(_split_one(doc, chunk_size))
    return passages
