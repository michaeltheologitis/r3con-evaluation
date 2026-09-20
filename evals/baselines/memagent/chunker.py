"""MemAgent's token-window chunker — faithful to upstream ``quickstart.py``.

MemAgent processes the context as a stream of **fixed-size token windows** (upstream
``RECURRENT_CHUNK_SIZE = 5000``), tokenized with the served MODEL's own tokenizer so the
chunk boundaries align with what the model sees. This is a pure token window — a chunk
may span two documents (upstream's context is a single blob; there is no
document-boundary rule, unlike readagent's pagination). See PROVENANCE.md.

``split_into_chunks(text, tokenizer, chunk_size, max_context_len)`` reproduces
``quickstart.async_query_llm``'s chunking exactly: encode → (optionally) clip to a
head+tail window → slice into ``chunk_size``-token pieces → decode each back to text.
"""
from __future__ import annotations

from typing import Any


def split_into_chunks(
    text: str,
    tokenizer: Any,
    chunk_size: int,
    max_context_len: int = 0,
) -> list[str]:
    """Tokenize ``text`` and split into ``chunk_size``-token chunks (decoded to strings).

    - ``max_context_len`` > 0 clips an over-long context to its **head+tail halves**
      (``ids[:n/2] + ids[-n/2:]``) BEFORE chunking — upstream's demo default was 120,000.
      We default to ``0`` (**no clip**): MemAgent is linear-time / window-independent by
      design (the paper runs 3.5M-token contexts), and on our "leave no document behind"
      benchmarks clipping would silently drop the middle documents (PROVENANCE). Pass a
      positive value to restore the upstream clip.
    - ``tokenizer`` needs ``encode(str) -> list[int]`` and ``decode(list[int]) -> str``
      (a HuggingFace ``AutoTokenizer`` of the served model; a fake with the same shape in
      tests).

    An empty ``text`` yields ``[]`` (no chunks → the loop leaves the memory empty and the
    final answer runs on ``NO_MEMORY``).
    """
    input_ids = tokenizer.encode(text)
    if max_context_len and len(input_ids) > max_context_len:
        head = max_context_len // 2
        input_ids = input_ids[:head] + input_ids[-(max_context_len - head):]
    return [
        tokenizer.decode(input_ids[i:i + chunk_size])
        for i in range(0, len(input_ids), chunk_size)
    ]
