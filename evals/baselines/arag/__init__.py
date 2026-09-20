"""A-RAG baseline — agentic RAG over hierarchical retrieval interfaces.

A-RAG (github.com/Ayanami0730/arag @ ``a44de6b``, arXiv 2602.03442) is a ReAct
agent that loops over three retrieval tools (``keyword_search`` / ``semantic_search``
/ ``read_chunk``) on a per-task corpus until it answers. Upstream is vendored
**byte-for-byte** under ``upstream/`` (only the intra-package import prefix is
rewritten — D1); everything else here is a thin connector with a documented
``DEVIATIONS`` header. The full deviation ledger is in ``PROVENANCE.md``.

Heavy/optional deps live in ``evals[arag]`` (tiktoken + numpy; transformers is
lazy/optional). The harness layer must not import this package at module load.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
