# Vendored ReadAgent artifacts (not imported)

These two files are the only code ReadAgent's authors released, vendored byte-for-byte as
the provenance anchor. Neither is importable, so the connector reproduces the three
prompting stages from them; see `../PROVENANCE.md`.

- **`app.py`** — the HuggingFace Space demo (`ReadAgent/read-agent`, `ecadb03`). The full
  QuALITY ReadAgent-P pipeline and its prompt templates.
- **`read_agent_demo.ipynb`** — the project-page notebook (`read-agent.github.io`,
  `569dff3`). The same pipeline plus an appendix with the verbatim paper prompts and both
  look-up variants.

**Paper:** Lee et al., ICML 2024, arXiv:2402.09727. There is no full eval-grade release:
ReadAgent-S and the QMSum/NarrativeQA implementations exist only as prompts, so only
ReadAgent-P is wired.
