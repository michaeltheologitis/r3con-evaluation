# Vendored MemAgent artifacts (provenance only — NOT imported)

Source: **github.com/BytedTsinghua-SIA/MemAgent** @ `ef4219b23499a069cb00e5daff4c426d4c600851`
(a fork of **verl**, the RL training framework). License: **Apache-2.0**.
Paper: *MemAgent: Reshaping Long-Context LLM with Multi-Conv RL based Memory Agent*
(arXiv 2507.02259).

- **`quickstart.py`** — the canonical **inference** recipe, byte-for-byte
  (sha256 `f76f3e7b…`). It defines the whole method at inference time: the two prompt
  templates (`TEMPLATE` memory-update / `TEMPLATE_FINAL` answer), the fixed token-window
  chunking (`RECURRENT_CHUNK_SIZE = 5000`), the head+tail context clip
  (`RECURRENT_MAX_CONTEXT_LEN = 120000`), and the per-call cap (`RECURRENT_MAX_NEW = 1024`).
  The connector (`../prompts.py`, `../chunker.py`, `../llm.py`, `../run.py`) **reproduces**
  this loop — copying the prompts verbatim and routing the calls through the harness's
  litellm seam — because upstream ships it as a demo script + a verl-based eval harness
  (`taskutils/memory_eval/`), not an importable library.

The identical loop (with `extract_solution` on the memory output) lives upstream at
`taskutils/memory_eval/utils/recurrent_boxed.py`; the RL training code is the rest of the
verl fork and is not used here (inference needs only a served checkpoint + the loop).

Trained checkpoints (public, ungated, on HF): `BytedTsinghua-SIA/RL-MemoryAgent-14B`
(the default) and `-7B` — RL-fine-tuned `Qwen/Qwen2.5-14B-Instruct` / `-7B-Instruct`,
so a standard `vllm serve <repo>` serves them directly.

Nothing in this directory is imported by the connector; it is kept as the provenance
anchor for the reproduction. See `../PROVENANCE.md` for the deviation ledger.
