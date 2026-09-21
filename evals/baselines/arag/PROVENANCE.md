# A-RAG

**Upstream:** github.com/Ayanami0730/arag @ `a44de6b` · **Paper:** arXiv:2602.03442 ·
**License:** MIT (declared in the README; the repo ships no LICENSE file).

The ReAct loop, its three tools (`keyword_search`, `semantic_search`, `read_chunk`) and its
system prompt are vendored byte-for-byte under `upstream/`. Needs a tool-calling endpoint.

## Deviations

- **Our own chunker** (no upstream analog). A-RAG consumes a pre-chunked corpus and ships no chunker; ours
  splits at ~1,000 tokens on sentence boundaries, matching its released corpus.
- **D3 — embeddings.** OpenAI `text-embedding-3-small` instead of A-RAG's local
  Qwen3-Embedding-0.6B. The retrieval method is untouched.
- **D2 — per-model context budget.** Upstream force-answers at a hardcoded 128k counted with a
  gpt-4o tokenizer, wrong for any other model; both are resolved from the served model.
- **Generation cap raised** 16,384 → 32,768, or a reasoning model spends it all on thinking.

Deviation IDs match the `DEVIATION (Dn)` comments in the code. Only the ones that
affect results are listed here; mechanical ones (import paths, dead-import trims,
logging) are marked at the code site.
