# StructRAG — provenance & deviations

**Upstream:** github.com/icip-cas/StructRAG @ `82e2804c` · **Paper:** ICLR 2025,
arXiv:2410.08815 · **License:** ⚠️ **none stated upstream** — see [THIRD_PARTY.md](../../../THIRD_PARTY.md).

Router, structurizer, utilizer and all six prompts are vendored **byte-for-byte** under
`upstream/`. Per task the router picks one knowledge structure, the structurizer rebuilds
every document into it, and the utilizer decomposes the question, extracts per-subquestion
knowledge and merges an answer. No retriever, no persistent index.

## Deviations that matter

- **D2 — LLM seam, and how an oversized prompt is handled.** Transport routes through
  litellm. Upstream proactively counted tokens with a **gpt2** tokenizer and clipped to a
  hardcoded **128K** — both artifacts of the box it was written on. We keep upstream's
  *behaviour* (an oversized prompt is truncated and answered, never errored) but clip
  **reactively**: only on the server's own context-length error, to the served model's real
  window.
- **D7 — generation cap raised** 4,096 → 32,768: on a thinking model 4,096 is consumed by
  reasoning before the answer, often yielding empty output.
- **Loong input and query.** The documents are re-wrapped into upstream's own title-marker
  format so the vendored splitter runs unmodified, and the Loong query is built from
  upstream's own prompt template.

## Upstream quirks kept on purpose

The router is effectively 3-way (its prompt only ever offers table / graph / chunk);
structure matching is substring-based; and `do_decompose` splits on a bare newline, so a
thinking model's reply yields two empty subqueries per task. All upstream behaviour,
reproduced rather than fixed.
