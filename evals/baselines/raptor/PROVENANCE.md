# RAPTOR

**Upstream:** github.com/parthsarthi03/raptor @ `7da1d48a` · **Paper:** ICLR 2024,
arXiv:2401.18059 · **License:** MIT (kept at `upstream/LICENSE.txt`).

Vendored byte-for-byte with zero edits; all adaptation lives in connectors outside
`upstream/`. Posed through RAPTOR's own `add_documents` → `answer_question` front door.

## Deviations

**D12 — hyperparameters re-scaled to the corpus** is the load-bearing one. RAPTOR's default leaf
chunk is 100 tokens, validated on ~6K-token documents and demonstrated only to ~78K (~780
leaves). Our bundles are 20–160× larger, where chunk=100 means ~10K leaves and ~2,300
reasoning summaries per task, and recursion on tiny clusters crashes UMAP.

| knob | paper | here |
| --- | --- | --- |
| leaf chunk | 100 | 2000 |
| recluster threshold | 3500 | 40000 |
| summary length hint | 100 | 400 |
| retrieval top-k / budget | 10 / 3500 | 20 / 16000 |

Read these numbers as "RAPTOR, hyperparameter-scaled to the corpus", not paper-faithful.

- **D9 — CJK-aware leaf chunking.** RAPTOR splits only on ASCII sentence punctuation, so Chinese
  prose becomes one oversized leaf and the cluster step crashes. The delimiter set is
  extended to full-width equivalents — a strict superset, so ASCII output is unchanged.
- **D10 — no summary cap.** RAPTOR's small cap is eaten by reasoning tokens, giving empty
  summaries; the length target moves into the prompt, with a retry if one still comes back
  empty.
- **D3 — OpenAI embeddings**, and litellm model backends injected via RAPTOR's own ABCs (its
  prompts copied verbatim). This is also the vLLM path — RAPTOR ships only raw-openai
  clients.

Deviation IDs match the `DEVIATION (Dn)` comments in the code. Only the ones that
affect results are listed here; mechanical ones (import paths, dead-import trims,
logging) are marked at the code site.
