"""LLM-driven stages of the R3Con pipeline.

- :mod:`evals.r3con.pipeline.stages.summaries` — task-conditioned, cross-document summaries.
- :mod:`evals.r3con.pipeline.stages.proposer` — per-task schema design (over task + summaries).
- :mod:`evals.r3con.pipeline.stages.extractor` — per-document extraction into the schema.
- :mod:`evals.r3con.pipeline.stages.inference` — single-LLM and multi-turn code-loop inference.
"""
