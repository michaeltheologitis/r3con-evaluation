# HippoRAG 2

**Upstream:** github.com/OSU-NLP-Group/HippoRAG @ `ad30fc3` · **Paper:** ICML 2025,
arXiv:2502.14802 · **License:** MIT (kept at `upstream/LICENSE`).

Vendored byte-for-byte under `upstream/hipporag/`; posed through HippoRAG's own index +
`rag_qa` front door. Offline, an LLM runs OpenIE on each passage to build a phrase/passage
graph; online, the query is linked to triples, filtered, ranked by Personalized PageRank,
and the top passages go to a reader LLM. Each edit inside `upstream/` carries a
`DEVIATION (Dn)` comment at the site.

## Deviations

- **D1 — our own chunker** (the load-bearing one). HippoRAG ships none — `index(docs)` treats
  each element as one passage, and its datasets are pre-chunked ~100-token paragraphs. Our
  benchmarks supply whole documents, so we split: corpusqa 8,000 / loong 3,000 / dracula
  3,000 tokens, no overlap, never spanning two documents. That is coarser than HippoRAG's
  native granularity, which lowers OpenIE recall per passage and coarsens retrieval, so
  these are "HippoRAG 2, chunked to the corpus", not the paper's setup. Coarse is
  deliberate: indexing costs ~2 LLM calls per passage, and the call count is what makes the
  method un-runnable on a reasoning model.
- **D7 — OpenAI embeddings**, even when completions run on vLLM.
- **D5 — no generation cap**, which would truncate a reasoning model's JSON.

Deviation IDs match the `DEVIATION (Dn)` comments in the code. Only the ones that
affect results are listed here; mechanical ones (import paths, dead-import trims,
logging) are marked at the code site.
