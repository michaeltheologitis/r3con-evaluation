"""RAPTOR leaf chunker — CJK-aware re-host of upstream ``utils.split_text`` (PROVENANCE D9).

RAPTOR chunks the pooled document into ~100-token leaves with ``utils.split_text``, which
splits a "sentence" on the delimiters ``. ! ? \\n`` and an over-long sentence on the clause
delimiters ``, ; :`` — **all ASCII**. Chinese text uses FULL-WIDTH punctuation
(``。！？，；：、``), none of which upstream recognizes, so a paragraph of Chinese prose is
treated as ONE "sentence": when it exceeds ``max_tokens`` and has no ASCII clause delimiter
either, ``split_text`` appends it whole → a single **oversized** leaf (observed up to ~1735
tiktoken tokens vs the 100-token target on Loong ZH ``legal``).

Why that crashes the build (not just bloats it): RAPTOR re-clusters any cluster whose total
text exceeds ``max_length_in_cluster`` = 3500 tokens (``cluster_utils``), and its recursion
base-case stops only at ONE node. With oversized leaves a cluster of just **2–4** of them
exceeds 3500 → RAPTOR recurses on a tiny set → ``global_cluster_embeddings`` sets
``n_neighbors = int(√(len−1)) ≤ 1`` → UMAP raises ``ValueError: n_neighbors must be greater
than 1``. Live on the Qwen3.5-MoE-Instruct (non-thinking) run this crashed ~50% of Loong ZH
``legal`` tasks (and the whole ZH half is at risk) — NOT the documented thinking-model
problem, a separate English-centric-chunker clash.

The fix is the chunking RAPTOR *intends*: this is upstream ``split_text`` copied **verbatim**
with the two delimiter sets extended to the full-width CJK equivalents, so Chinese prose
splits into ~100-token leaves like English does. It is a strict **superset** — ASCII text
contains none of these code points, so for EN/financial documents the output is
**byte-identical** to upstream (verified in tests); only CJK text chunks differently (i.e.
correctly). Mirrors ReadAgent's "CJK-aware pagination" deviation (PROGRESS 2026-06-15).

Full deviation ledger: evals/baselines/raptor/PROVENANCE.md (D9).
"""
from __future__ import annotations

import re

# Upstream ``utils.split_text`` splits on these ASCII sets. We extend each with its
# full-width CJK counterpart (sentence-enders 。！？ ; clause delimiters ，；：、 — the
# Chinese enumeration comma 、 included), keeping upstream's two-tier sentence-then-clause
# structure. ASCII-only text never contains these, so it is unaffected.
_SENTENCE_DELIMITERS = [".", "!", "?", "\n", "。", "！", "？"]
_CLAUSE_DELIMITERS_RE = r"[,;:，；：、]"


def split_text(text: str, tokenizer, max_tokens: int, overlap: int = 0) -> list[str]:
    """CJK-aware copy of upstream ``raptor.utils.split_text``.

    IDENTICAL to upstream except the two delimiter sets are the module-level CJK-extended
    versions above; everything else (the sentence loop, the over-long sub-split, the
    ``overlap`` handling, the ``" ".join`` reassembly) is verbatim. See the module docstring.
    """
    # Split the text into sentences using multiple delimiters
    regex_pattern = "|".join(map(re.escape, _SENTENCE_DELIMITERS))
    sentences = re.split(regex_pattern, text)

    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue

        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(_CLAUSE_DELIMITERS_RE, sentence)

            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chuk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) for sub_sentence in filtered_sub_sentences]

            sub_chunk = []
            sub_length = 0

            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:

                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])

                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count

            if sub_chunk:
                chunks.append(" ".join(sub_chunk))

        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count

        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
