# A-RAG — provenance & deviations

**Upstream:** github.com/Ayanami0730/arag @ `a44de6b` · **Paper:** arXiv:2602.03442 ·
**License:** MIT (declared in the upstream README; the repo ships no LICENSE file).

The ReAct agent loop, its three retrieval tools (`keyword_search`, `semantic_search`,
`read_chunk`) and its system prompt are vendored **byte-for-byte** under `upstream/`.
The connector only replaces the model backends and the batch loop; the task is posed
through the agent's front door and it retrieves over the task's own documents.

## Deviations that matter

- **D1 — import prefix.** `from arag.…` → `from evals.baselines.arag.upstream.…`.
  Mechanical; the logic is byte-for-byte unchanged.
- **D2 — LLM seam + a per-model context budget.** Transport routes through litellm, with
  the agent's own client interface preserved so the vendored loop runs unmodified. Upstream
  force-answers at a hardcoded 128k budget counted with a **gpt-4o** tokenizer — wrong for
  any other model — so both the counter and the budget are resolved from the served model.
- **D3 — embeddings.** Dense retrieval embeds with OpenAI `text-embedding-3-small` instead
  of A-RAG's local Qwen3-Embedding-0.6B. The retrieval method itself is untouched.
- **Chunker (no upstream analog).** A-RAG consumes a pre-chunked corpus and ships no
  chunker; ours splits at ~1,000 tokens on sentence boundaries, matching the granularity of
  A-RAG's own released corpus.
- **Generation cap raised** 16,384 → 32,768: a reasoning model can spend the whole 16k on
  reasoning tokens and emit nothing.

Needs a **tool-calling endpoint** — A-RAG drives the model with OpenAI tool-calling.
