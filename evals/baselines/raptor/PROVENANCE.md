# RAPTOR — provenance & deviations

**Upstream:** github.com/parthsarthi03/raptor @ `7da1d48a` · **Paper:** ICLR 2024,
arXiv:2401.18059 · **License:** MIT (kept at `upstream/LICENSE.txt`).

Vendored **byte-for-byte** (zero edits — its imports are already relative); all adaptation
lives in connectors outside `upstream/`. Chunk the text into leaves, embed them, then
recursively cluster (UMAP + GMM) and LLM-summarize each cluster into the next tree level; at
query time collapse the tree and answer over the top-k nodes. Posed through RAPTOR's own
`add_documents` → `answer_question` front door.

## The load-bearing deviation

**D12 — hyperparameters re-scaled to the corpus (`run_version` v4).** RAPTOR's default leaf
chunk is **100 tokens**, validated on ~6K-token documents and demonstrated only up to ~78K
(~780 leaves). Our bundles are 20–160× larger, where chunk=100 means ~10K leaves and ~2,300
reasoning summaries per task — days per task, and recursion on tiny clusters crashes UMAP.
So the leaf granularity and the knobs that depend on it are re-scaled; the method itself is
unchanged.

| knob | paper | here | why |
|---|---|---|---|
| leaf chunk | 100 | **2000** | ~500 leaves at 1M tokens — back inside the validated ≤780-leaf range |
| recluster threshold | 3500 | **40000** | lets natural clusters stand; removes the tiny-cluster crash |
| summary length hint | 100 | **400** | clusters are ~20× bigger; keeps compression in RAPTOR's own band |
| retrieval top-k / budget | 10 / 3500 | **20 / 16000** | sized to the bigger nodes; still ~1.6% of a 1M corpus |

Documented as "RAPTOR, hyperparameter-scaled to the corpus", not paper-faithful: with paper
defaults extrapolated 13×, it neither finishes nor stays crash-free on a reasoning model.

## The rest, briefly

- **D1 — model backends.** RAPTOR ships only raw-openai clients; we subclass its ABCs and
  inject litellm ones (prompts copied verbatim), which is also the vLLM path. **D2** covers
  transport and thread-safe usage capture, **D3** OpenAI embeddings (plus upstream's
  empty-node guard).
- **D8 — parallel cluster summaries.** RAPTOR supports multithreading but never enables it;
  the connector flips its own flag. Output-identical — a throughput knob only.
- **D9 — CJK-aware leaf chunking.** RAPTOR splits only on ASCII sentence punctuation, so
  Chinese prose becomes one oversized leaf and the cluster step crashes; the delimiter set is
  extended to its full-width equivalents (a strict superset — ASCII output is byte-identical).
- **D10 — reasoning summaries.** RAPTOR's small summary cap is eaten by reasoning tokens,
  giving empty summaries; the length target moves into the prompt as a hint and no cap is
  sent, with a retry if a summary still comes back empty.
- **D11 — tiktoken special tokens.** RAPTOR encodes node text without `disallowed_special`,
  so a document containing a literal `<|endoftext|>` crashes it (some Loong papers do);
  patched for the task without editing `upstream/`.
- **Multi-document pooling** into one blob before chunking, and the same query composition
  every other baseline uses.

Expected to be a retrieval **foil** on leave-no-document-behind benchmarks, and the most
call-heavy baseline here — use a stratified subset.
