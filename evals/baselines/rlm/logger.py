"""RLM trajectory logger that does not persist REPL-variable snapshots.

RLM's native ``RLMLogger`` writes one JSON line per iteration from ``RLMIteration.to_dict()``, and
for every code block that includes ``REPLResult.to_dict()["locals"]`` — the WHOLE REPL namespace
serialized, which always holds ``context`` and ``context_0`` (the entire document bundle, twice) plus
every slice the agent derived from it. Re-dumped after every block, that was 94-99.6% of each
trajectory file (a single CorpusQA run: 224 MB, of which <1 MB was the actual trajectory), for
content that is recoverable anyway: the context from ``benchmark.get_documents``, the derived
variables by replaying the logged code. The model never sees those values either — only the
variable NAMES, which stay in the next turn's prompt (``REPL variables: [...]``).

So the one rule: drop ``locals``, keep everything else — the prompt (message history), response,
code, stdout/stderr, execution time, sub-calls (``rlm_calls``) and final answer.

This observes only: RLM builds the next prompt from the live ``REPLResult`` object, not from the
logged dict, so nothing the model sees changes. ``drop_repl_locals`` is shared with
``scripts/compact_rlm_logs.py``, which applies the same rule to trajectories logged before this.
"""
from __future__ import annotations

from typing import Any

from rlm.logger import RLMLogger


def drop_repl_locals(entry: dict[str, Any]) -> dict[str, Any]:
    """Remove ``locals`` from every code-block result of one logged iteration, in place.

    Recurses into sub-call trajectories (``rlm_calls[*].metadata.iterations``), which a child RLM
    attaches when ``max_depth > 1``. Returns ``entry`` for convenience."""
    for block in entry.get("code_blocks") or []:
        result = block.get("result") or {}
        result.pop("locals", None)
        for call in result.get("rlm_calls") or []:
            for child in (call.get("metadata") or {}).get("iterations") or []:
                drop_repl_locals(child)
    return entry


class _WithoutLocals:
    """Presents an ``RLMIteration`` to ``RLMLogger.log``, which only calls ``to_dict()``."""

    def __init__(self, iteration: Any) -> None:
        self._iteration = iteration

    def to_dict(self) -> dict[str, Any]:
        return drop_repl_locals(self._iteration.to_dict())


class TrajectoryLogger(RLMLogger):
    """``RLMLogger`` minus the REPL-variable snapshots — same file, same line format, same
    in-memory trajectory (``get_trajectory``), just without ``locals``."""

    def log(self, iteration: Any) -> None:
        super().log(_WithoutLocals(iteration))
