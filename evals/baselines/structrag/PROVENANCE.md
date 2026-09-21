# StructRAG

**Upstream:** github.com/icip-cas/StructRAG @ `82e2804c` · **Paper:** ICLR 2025,
arXiv:2410.08815 · **License:** none stated upstream.

Router, structurizer, utilizer and all six prompts are vendored byte-for-byte under
`upstream/`. No retriever, no persistent index.

## Deviations

- **D2 — oversized prompts are clipped reactively.** Upstream counted tokens with a gpt2
  tokenizer and clipped to a hardcoded 128K. We keep the behaviour (truncate and answer,
  never error) but clip only on the server's own context-length error, to the real window.
- **D7 — generation cap raised** 4,096 → 32,768, or a thinking model emits nothing.

## Upstream quirks kept on purpose

The router is effectively 3-way (its prompt only offers table / graph / chunk), structure
matching is substring-based, and `do_decompose` splits on a bare newline, so a thinking
model's reply yields two empty subqueries per task. Reproduced rather than fixed.

Deviation IDs match the `DEVIATION (Dn)` comments in the code. Only the ones that
affect results are listed here; mechanical ones (import paths, dead-import trims,
logging) are marked at the code site.
