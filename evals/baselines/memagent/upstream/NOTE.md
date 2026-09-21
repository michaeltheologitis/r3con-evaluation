# Vendored MemAgent artifact (not imported)

**Source:** github.com/BytedTsinghua-SIA/MemAgent @ `ef4219b` (a verl fork) ·
**License:** Apache-2.0 · **Paper:** arXiv:2507.02259.

`quickstart.py` is upstream's inference recipe, byte-for-byte. It defines the method at
inference time: the two prompt templates, the 5,000-token chunking, the 120k head+tail
clip and the 1,024-token per-call cap. Upstream ships it as a demo plus a verl-coupled
harness rather than a library, so the connector reproduces the loop from it and this file
is kept only as the anchor. See `../PROVENANCE.md`.

Checkpoints are public on HF: `BytedTsinghua-SIA/RL-MemoryAgent-14B` (the default) and
`-7B`, RL-fine-tuned Qwen2.5-Instruct, served directly by `vllm serve <repo>`.
