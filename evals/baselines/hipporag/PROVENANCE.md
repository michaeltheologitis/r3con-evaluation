# HippoRAG 2 — provenance & deviations

**Upstream:** github.com/OSU-NLP-Group/HippoRAG @ `ad30fc3` · **Paper:** ICML 2025,
arXiv:2502.14802 · **License:** MIT (kept at `upstream/LICENSE`).

Vendored **byte-for-byte** under `upstream/hipporag/`; the task is posed through HippoRAG's
own index + `rag_qa` front door, so only the model backends are seamed. Offline, an LLM runs
OpenIE on **each passage** to build a phrase/passage graph; online, the query is linked to
triples, pruned by a recognition-memory filter, ranked by Personalized PageRank, and the top
passages go to a reader LLM. Each edit inside `upstream/` carries a `DEVIATION (Dn)` comment
at the site.

## The load-bearing deviation

**D1 — our own chunker.** HippoRAG ships none: `index(docs)` treats each element as ONE
passage, and its datasets are pre-chunked ~100-token paragraphs. Our benchmarks supply whole
documents, so we must split them: **corpusqa 8,000 / loong 3,000 / dracula 3,000 tokens**, no
overlap, never spanning two documents (corpusqa capped under the embedder's 8,191-token
limit). This is **coarser than HippoRAG's native granularity** — it lowers OpenIE recall per
passage and coarsens retrieval — so these numbers are "HippoRAG 2, chunked to the corpus",
not a reproduction of the paper's setup. Coarse is deliberate: indexing costs ~2 LLM calls
per passage, and on a reasoning model the call *count* is what makes the method un-runnable.

## The rest, briefly

- **D2 / D3 / D10 — import trims.** A dead `transformers` import removed; the eager
  local-model backends (vllm / gritlm / transformers embedders) dropped from the import path
  since our seams replace them; the template-import anchor re-pointed at the vendored
  package. No behaviour change.
- **D4 — LLM seam.** One litellm client drives OpenIE, the triple filter and the reader.
  (That filter uses no runtime dspy — it is a baked prompt.)
- **D5 — no generation cap**, which would truncate a reasoning model's JSON.
- **D6 — no sqlite response cache**; the harness owns resumption, so every call is real.
- **D7 — OpenAI embeddings**, even when completions run on vLLM.
- **D8 — no `ipdb` trap.** Upstream drops into an interactive debugger on an embedding
  error, which hangs a headless run; ours raises so the task records a clean failure.
- **D9 — embedding inputs truncated** to the 8,191-token cap and batched under it.

Expensive (per-passage OpenIE, no cross-task reuse) → run a stratified `--limit` subset.
