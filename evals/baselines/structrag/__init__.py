"""``structrag`` baseline — StructRAG (ICLR 2025) inference-time hybrid
information structurization, hooked into the harness.

Per task: a router picks a knowledge-structure type (table / graph / algorithm /
catalogue / chunk), a structurizer rebuilds each document into that structure,
and a utilizer decomposes the question, extracts per-subquestion knowledge, and
merges a final answer. It is an *inference-time* method — no retriever, no
persistent index, no training in the default mode — so packaging-wise it is as
light as a single-call baseline while internally being a multi-call pipeline.

Upstream code (github.com/icip-cas/StructRAG @ 82e2804c) is vendored
**byte-for-byte** under ``upstream/`` (router / structurizer / utilizer + the 6
prompt files); everything else here is a thin connector with a documented
``DEVIATIONS`` header. The full deviation ledger is in ``PROVENANCE.md``.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
