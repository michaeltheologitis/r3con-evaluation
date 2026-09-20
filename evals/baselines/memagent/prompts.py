"""MemAgent prompt templates — copied VERBATIM from upstream, kept faithful.

MemAgent (BytedTsinghua-SIA / Seed, arXiv 2507.02259) ships its inference as a demo
script (``quickstart.py``) + an eval harness (``taskutils/memory_eval/``), both built on
verl (its RL training framework). There is no importable inference library, so this
connector REPRODUCES the recurrent-memory loop — the two prompt templates below are
copied verbatim from ``quickstart.py`` (identical to ``taskutils/memory_eval/utils/
recurrent_boxed.py``). Both are vendored under ``upstream/`` for provenance; every
adaptation is ledgered in ``PROVENANCE.md``.

The method: read the context in fixed-size chunks; for each chunk, ask the (RL-trained)
model to fold it into a running ``memory`` (a fixed-size overwriting summary); then ask
the model to answer the problem from the final memory. The ``max_tokens`` cap on each
call is load-bearing — it bounds the memory, which is the whole point (fixed-window,
linear-time processing).
"""
from __future__ import annotations

# ------------------------------------------------------------------------------------
# (1) Memory update — VERBATIM from upstream quickstart.py `TEMPLATE` (the per-chunk
#     memory-overwrite step). Slots {prompt} (the problem) / {memory} (the running
#     memory) / {chunk} (the current section). The trailing space after `<problem>` is
#     upstream's, preserved.
# ------------------------------------------------------------------------------------
MEMORY_UPDATE_TEMPLATE = r"""You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""


# ------------------------------------------------------------------------------------
# (2) Final answer — VERBATIM from upstream quickstart.py `TEMPLATE_FINAL` (answer the
#     problem from the final memory, in \boxed{}). Slots {prompt} / {memory}. The
#     \boxed{} answer contract is upstream's; the benchmark's own answer-format
#     instruction (e.g. CorpusQA's "The answer is: …") rides in {prompt} — the judges
#     are format-robust either way (PROVENANCE).
# ------------------------------------------------------------------------------------
FINAL_ANSWER_TEMPLATE = r"""You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \boxed{}.

<problem>
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

# The initial memory (upstream `NO_MEMORY`), used before the first chunk is folded in.
NO_MEMORY = "No previous memory"


def memory_update_prompt(problem: str, memory: str, chunk: str) -> str:
    """Fill the memory-update template. Uses ``str.replace`` (not ``str.format``) so
    document / memory text containing braces is substituted literally — the repo's
    convention for injecting user content into a template (cf. readagent/prompts.py,
    loong/judge.py). Behaviour-identical to upstream's ``.format`` on brace-clean data."""
    return (
        MEMORY_UPDATE_TEMPLATE
        .replace("{prompt}", problem)
        .replace("{memory}", memory)
        .replace("{chunk}", chunk)
    )


def final_answer_prompt(problem: str, memory: str) -> str:
    """Fill the final-answer template (leaves the literal ``\\boxed{}`` intact)."""
    return (
        FINAL_ANSWER_TEMPLATE
        .replace("{prompt}", problem)
        .replace("{memory}", memory)
    )
