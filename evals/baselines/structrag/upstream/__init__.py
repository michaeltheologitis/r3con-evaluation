"""Vendored StructRAG code — github.com/icip-cas/StructRAG @ 82e2804c.

``router.py`` / ``structurizer.py`` / ``utilizer.py`` and ``prompts/*.txt`` are
copied **byte-for-byte** from upstream; the ONLY change is that the three modules
resolve their prompt files relative to this package instead of the upstream
cwd-relative ``open("prompts/...")`` (each carries a ``DEVIATIONS`` header). The
upstream ``main.py`` (batch loop, sharding, JSONL I/O) and ``utils/qwenapi.py``
(raw-requests LLM seam) are NOT vendored — their roles are filled by the harness
runner and ``structrag/llm.py`` respectively. See ``../PROVENANCE.md``.
"""
