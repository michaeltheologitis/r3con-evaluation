"""CodeAct baseline — smolagents ``CodeAgent`` over the grounded documents.

CodeAct (Wang et al., 2024 — *executable code actions*) is the agent paradigm where each
step the model takes an action by **writing and running Python code** (rather than emitting a
JSON tool call). smolagents' ``CodeAgent`` is its reference implementation: a ReAct loop where
the LM writes a code blob, a local executor runs it, the agent observes stdout, and it iterates
until it calls ``final_answer(...)``.

Posed like RLM (the closest sibling baseline): the document bundle is **offloaded into the code
sandbox** as a ``documents`` variable (a list of strings) — NOT stuffed into the prompt — so the
agent scans / slices / aggregates it with code; the composed task/question is the agent's task
string. smolagents is a normal dependency (the ``evals[codeact]`` extra), **not vendored** — used
via its public API; "faithful" here means posing the task through the method's front door, not
reproducing internals (see ``PROVENANCE.md``). The LLM routes through smolagents' ``LiteLLMModel``
(subclassed in ``llm.py`` for complete, deterministic token-cost capture).

SIMPLE NO-REUSE logging (the readagent/rlm layout): one self-contained folder per task
run holding the CodeAct trajectory (``trajectory.json``), ``manifest.json`` (the TOTAL token cost
across every step), ``calls.json``, and (at score time) ``score.json``. The runner DOES resume —
it skips tasks already completed for the same config. Wired for **loong + corpusqa**.
"""
from .run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
