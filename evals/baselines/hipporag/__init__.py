"""HippoRAG 2 baseline (vendored upstream + litellm/OpenAI connector).

OpenIE knowledge graph + Personalized PageRank retrieval. Vendored byte-for-byte under
``upstream/hipporag`` (MIT); the connector injects a litellm LLM seam + an OpenAI
embedding seam and adds an own chunker (HippoRAG ships none). Deviation ledger:
PROVENANCE.md.
"""
from evals.baselines.hipporag.run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
