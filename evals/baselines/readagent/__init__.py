"""ReadAgent baseline — gist-memory reading agent for very long contexts.

ReadAgent (Google DeepMind, ICML 2024; arXiv 2402.09727) is an LLM-ONLY long-context
method (no embeddings, no retriever): paginate the document(s) into pages, compress each
page into a gist, then for a question look up a few pages' full text from the gist memory
and answer. We wire **ReadAgent-P** (one batched look-up call) — the only look-up variant
upstream implements in code (ReadAgent-S is a prompt-only template upstream; PROVENANCE D5).

Upstream ships only demo artifacts (a HuggingFace Space ``app.py`` + the project-page
notebook), vendored under ``upstream/`` for provenance; this package REPRODUCES the three
prompting stages faithfully (the prompts are copied verbatim). Every deviation is recorded
in ``PROVENANCE.md``. No extra dependencies — pure stdlib + litellm (the harness core).

SIMPLE NO-INDEX-REUSE logging (the flat per-run-folder layout): one self-contained folder per task
run, the gist memory inside it, the TOTAL cost in the manifest, no content-addressing, no
gist-memory reuse. The runner DOES resume — it skips tasks already completed for the same
config (so a re-run doesn't redo finished experiments). Wired for Loong + CorpusQA.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
