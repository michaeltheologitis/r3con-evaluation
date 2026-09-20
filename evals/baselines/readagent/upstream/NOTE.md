# ReadAgent upstream — provenance anchor

These two files are the **only** code ReadAgent's authors released. They are vendored
here **byte-for-byte** as the provenance anchor; they are **not imported** by the
connector (neither is an importable library — `app.py` is a Gradio demo app, the other a
notebook). The connector REPRODUCES ReadAgent's three prompting stages faithfully from
them; see `../PROVENANCE.md` for exactly which prompt/function each connector piece comes
from and every deviation.

- **`app.py`** — the HuggingFace Space demo
  (`https://huggingface.co/spaces/ReadAgent/read-agent`, commit `ecadb03`). Implements the
  full QuALITY ReadAgent-P pipeline: `quality_pagination`, `quality_gisting`,
  `quality_parallel_lookup`, and the pagination/gisting/look-up/answer prompt templates.
- **`read_agent_demo.ipynb`** — the project-page notebook
  (`https://github.com/read-agent/read-agent.github.io`, `assets/read_agent_demo.ipynb`,
  commit `569dff3`). The same QuALITY pipeline PLUS an appendix (cells 11–14) giving the
  **verbatim paper prompts** for QuALITY / QMSum / NarrativeQA and **both** look-up
  variants — ReadAgent-P (parallel) and ReadAgent-S (sequential).

**Paper:** "A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts",
Lee et al., ICML 2024 — arXiv:2402.09727. There is no official full-eval-grade code
release; QMSum/NarrativeQA implementations and the ReadAgent-S implementation are NOT
shipped (only their prompts). The connector therefore wires **only ReadAgent-P**, the
single look-up variant upstream actually implements in code (see `../PROVENANCE.md` D5).
