"""Execution machinery shared by the stages.

- :mod:`evals.r3con.pipeline.runtime.llm` — LiteLLM wrapper + structured-output
  schema enforcement.
- :mod:`evals.r3con.pipeline.runtime.python_executor` — vendored sandboxed
  in-process Python interpreter used by the CodeAct inference loop.
"""
