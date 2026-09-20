"""MemAgent baseline — recurrent fixed-size memory over token chunks (long-context).

MemAgent (BytedTsinghua-SIA / Seed, arXiv 2507.02259) reads the context in fixed-size
token chunks, folding each into a running, overwriting **memory** (produced by an
RL-trained model — RLVR/DAPO on HotpotQA), then answers from the final memory. Linear-time,
context-window-independent. The method IS the trained checkpoint (default
``BytedTsinghua-SIA/RL-MemoryAgent-14B``, Qwen2.5-14B-Instruct-based), served on vLLM.

Not vendored as a library — upstream ships a demo (``quickstart.py``) + a verl-based eval
harness — so this package REPRODUCES the recurrent loop faithfully (prompts copied verbatim
in ``prompts.py``; ``quickstart.py`` vendored under ``upstream/`` for provenance). Every
deviation is recorded in ``PROVENANCE.md``. Behind the ``evals[memagent]`` extra
(``transformers`` — the served model's tokenizer for the token-window chunker).

SIMPLE NO-REUSE logging (the readagent/rlm layout): one self-contained folder per task run
holding the memory trajectory, the TOTAL cost in the manifest, no reuse; the runner resumes
(skips tasks already done for the config). Wired for Loong + CorpusQA (NOT MINTEval yet).
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
