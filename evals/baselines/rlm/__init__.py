"""RLM baseline — Recursive Language Models (a code-REPL reasoning agent).

RLM (Recursive Language Models, alexzhang13/rlm, arXiv 2512.24601) is a task-agnostic inference
paradigm: instead of stuffing a long context into the prompt, the context is offloaded as a
``context`` **variable in a Python REPL**, and a root LM writes ``repl`` code (CodeAct — not JSON
tool-calling) to examine/decompose it and launch recursive sub-LM calls (``llm_query`` /
``rlm_query``). We pose Loong/CorpusQA via RLM's canonical front door: the document bundle becomes
the REPL ``context`` variable, the composed task/question becomes the ``root_prompt`` the root LM sees.

RLM is a normal dependency (the ``evals[rlms]`` extra; ``uv add``-ed, not vendored). It is
**token-heavy by design**, so it runs against a local **vLLM** endpoint only — never OpenAI, and it
uses no litellm and no embeddings. Token/cost is captured completely at RLM's OpenAI-compatible
client boundary (every call at every recursion depth); RLM's own ``RLMLogger`` trajectory is saved
alongside as the rich CodeAct log. Deviation ledger: ``PROVENANCE.md``.

SIMPLE NO-REUSE logging (the readagent/rlm layout): one self-contained folder per task run,
the trajectory inside it, the TOTAL token cost in the manifest. The runner resumes (skips tasks
already done for the config). Wired for Loong + CorpusQA.
"""
from evals.baselines.rlm.run import SUPPORTED_BENCHMARKS, run_one

__all__ = ["SUPPORTED_BENCHMARKS", "run_one"]
