"""~1000-token, sentence-aligned corpus chunker — the corpus-prep step A-RAG itself
does not ship.

A-RAG consumes a PRE-CHUNKED corpus (``chunks.json`` = ``["0:text", "1:text", …]``)
and does NO document chunking — its released ``rag_test`` corpora arrive pre-chunked.
Our benchmarks are whole documents, so the connector must chunk them. To stay faithful
we MATCH the granularity of A-RAG's released corpus rather than invent one:

  • Paper (arXiv 2602.03442): *"we partition the corpus into chunks of approximately
    1,000 tokens each, ensuring that chunk boundaries align with sentence boundaries."*
  • Verified against the released corpus: ``medical`` (225 chunks) + ``musique`` (1354)
    measured with tiktoken ``gpt-4o`` cluster at mean/median ≈ 1075 tokens.

Chunking uses **`semantic-text-splitter`** — a well-tested splitter that recursively
respects semantic boundaries (paragraph → sentence → word) while filling each chunk up
to a token capacity — rather than a hand-rolled regex. Capacity is measured with
**tiktoken gpt-4o**; see the note below on why that tokenizer is fixed.

────────────────────────────────────────────────────────────────────────────────────
WHY gpt-4o HERE IS NOT THE `model_budget` MISTAKE.
The chunk size is a property of **A-RAG's CORPUS**, not of the model we serve. A-RAG's
authors built their released corpus by counting ~1000 tokens with gpt-4o's tokenizer, so
matching that tokenizer reproduces their exact chunk granularity for ANY backbone we then
evaluate. This is the OPPOSITE situation from ``model_budget``: there, the agent's context
stop-gate must track the *served* model (counting a Qwen prompt in gpt-4o tokens would
mis-cap it — the deviation you flagged), so that counter is DYNAMIC. Here the served model
only *reads* the chunks; their size is fixed corpus prep, so the tokenizer is fixed to
A-RAG's (gpt-4o). If a future experiment wants model-consistent chunking instead,
`semantic-text-splitter` also takes an HF tokenizer (`from_huggingface_tokenizer`) — flip
``_CHUNK_TOKENIZER`` and bump ``CHUNKER_VERSION``.
────────────────────────────────────────────────────────────────────────────────────

``CHUNKER_VERSION`` is folded into the index hash + the inference identity, so changing
the recipe (size, tokenizer, splitter) lands on a fresh index dir + re-runs inferences
instead of silently reusing stale ones.
"""
from __future__ import annotations

from functools import lru_cache

from semantic_text_splitter import TextSplitter

# Bump on ANY recipe change that alters chunk CONTENT. v2: switched from a hand-rolled
# regex greedy-packer to semantic-text-splitter (gpt-4o token capacity).
CHUNKER_VERSION = "v2"

# A-RAG's corpus-prep token budget + tokenizer (see the module note — fixed to A-RAG's
# corpus, NOT the served model). semantic-text-splitter fills each chunk up to this many
# tokens, so the median lands just under it — squarely the paper's "approximately 1,000".
_CHUNK_TOKENS = 1000
_CHUNK_TOKENIZER = "gpt-4o"


@lru_cache(maxsize=None)
def _splitter(capacity: int) -> TextSplitter:
    """A tiktoken-gpt-4o-capacity splitter (cached per capacity — building one is not free)."""
    return TextSplitter.from_tiktoken_model(_CHUNK_TOKENIZER, capacity)


def chunk_text(text: str, target_tokens: int = _CHUNK_TOKENS) -> list[str]:
    """Split ``text`` into sentence-aligned chunks of ≤ ``target_tokens`` gpt-4o tokens,
    via semantic-text-splitter (paragraph → sentence → word boundaries, no overlap)."""
    return [chunk for chunk in _splitter(target_tokens).chunks(text) if chunk.strip()]


def build_chunks(documents: list[str], target_tokens: int = _CHUNK_TOKENS) -> list[str]:
    """The A-RAG ``chunks.json`` payload for a document SET: chunk each document
    independently (a chunk never spans two documents), then number all chunks
    sequentially in document order → ``["0:text", "1:text", …]`` (A-RAG's format,
    keeping ``read_chunk``'s ±1-adjacent strategy meaningful)."""
    rows: list[str] = []
    index = 0
    for document in documents:
        for chunk in chunk_text(document, target_tokens=target_tokens):
            rows.append(f"{index}:{chunk}")
            index += 1
    return rows
