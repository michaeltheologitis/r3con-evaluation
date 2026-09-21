# MemAgent

**Upstream:** github.com/BytedTsinghua-SIA/MemAgent @ `ef4219b` (a verl fork) ·
**Paper:** arXiv:2507.02259 · **License:** Apache-2.0.

Upstream ships an inference demo (`quickstart.py`, vendored under `upstream/` as the
anchor) and a verl-coupled harness, not a library, so the connector reproduces the loop
with both prompt templates copied verbatim: read the context in 5,000-token chunks, fold
each into a running fixed-size memory, answer from the final memory.

**The method is the RL-trained checkpoint** (`BytedTsinghua-SIA/RL-MemoryAgent-14B`, served
on vLLM). Running the loop with a generic model is not MemAgent, so there is no
generic-model default.

## Deviations

- **The 1,024-token cap is kept.** It bounds the memory, so the cap is the method — unlike
  the other baselines here, where caps are dropped so a reasoning model isn't cut off.
  MemAgent's checkpoint is non-thinking. A thinking model spends the cap on reasoning and
  returns an empty memory.
- **The demo's 120k context clip is lifted by default.** It clips to a head+tail window
  before chunking, silently dropping the middle (~88% of a 1M-token CorpusQA instance). The
  method is unbounded by design. `--max-context-len` restores the clip.
