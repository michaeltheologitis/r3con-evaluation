# MemAgent — provenance & deviations

**Upstream:** github.com/BytedTsinghua-SIA/MemAgent @ `ef4219b` (a verl fork) ·
**Paper:** arXiv:2507.02259 · **License:** Apache-2.0.

Upstream ships an inference demo (`quickstart.py`, vendored under `upstream/` as the
byte-for-byte anchor) and a verl-coupled eval harness — not a library — so the connector
**reproduces** the loop with both prompt templates copied verbatim: read the context in
fixed 5,000-token chunks, fold each chunk into a running fixed-size memory, then answer
from the final memory. Linear-time and context-window-independent by construction.

**The method IS the RL-trained checkpoint** (`BytedTsinghua-SIA/RL-MemoryAgent-14B`, served
on vLLM). Running the loop with a generic model is not MemAgent, so there is no
generic-model default.

## Deviations that matter

- **The 1,024-token cap is KEPT.** It bounds the fixed-size memory — the cap *is* the
  method — unlike the other baselines here, where caps are dropped so a reasoning model
  isn't cut off. MemAgent's checkpoint is non-thinking, so nothing is eaten by reasoning.
- **The demo's 120k context clip is lifted by default.** Upstream's demo clips an over-long
  context to a head+tail window *before* chunking; on "leave no document behind" benchmarks
  that silently drops the middle (~88% of a 1M-token CorpusQA instance). The method is
  unbounded by design — the paper runs 3.5M-token contexts — so the whole bundle is
  processed. `--max-context-len` restores the demo's clip.
- **Multi-document pooling.** The bundle is joined into one blob, so a chunk may span two
  documents — faithful to upstream, which has no document-boundary rule.
