"""MemAgent baseline pipeline — connects the recurrent-memory method to the harness.

MemAgent (BytedTsinghua-SIA / Seed, arXiv 2507.02259) is a long-context method: read the
context in fixed-size **token chunks**, folding each into a running **memory** (a
fixed-size, overwriting summary produced by an RL-trained model), then answer the problem
from the final memory. This gives linear-time, context-window-independent processing —
the model was RL-trained (RLVR/DAPO on HotpotQA) end-to-end for this exact workflow, so
the method IS the trained checkpoint (default ``BytedTsinghua-SIA/RL-MemoryAgent-14B``,
Qwen2.5-14B-Instruct-based). Not vendored as a library — upstream ships only a demo
(``quickstart.py``) + eval harness built on verl — so we REPRODUCE the loop faithfully
(prompts verbatim in ``prompts.py``; ``quickstart.py`` vendored under ``upstream/`` for
provenance). Deviation ledger: evals/baselines/memagent/PROVENANCE.md

Per task: pool the document bundle into one context → token-window into chunks →
recurrently update the memory over the chunks → answer. All calls route through one
litellm seam (``MemAgentLLM`` → vLLM), so ``usage`` is the TOTAL (every memory update +
the final answer). SIMPLE NO-REUSE logging (the readagent/rlm layout): each task run gets
its OWN folder ``logs/{benchmark}/memagent/{run_tag}/`` holding the memory trajectory
(``memory_trajectory.json`` — every intermediate memory state), ``manifest.json`` (TOTAL
cost), ``calls.json``, and a live ``progress.json``. Nothing here grades — the raw answer
is saved as-is for the external scoring repo that reads these logs. The runner resumes
(skips tasks done for the config).

SUPPORTED_BENCHMARKS = {loong, corpusqa, dracula} — per-instance multi-doc bundles.
"""
from __future__ import annotations

import json
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from evals.baselines.memagent import chunker as chunker_mod
from evals.baselines.memagent import prompts
from evals.baselines.memagent.llm import MemAgentLLM

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Run-config version — bump on a change that alters a run's output (prompt, loop, framing).
_RUN_VERSION = "v1"

# The memory trajectory (every intermediate memory state) is persisted here inside the run
# folder, so a deep-dive can read exactly how the memory evolved chunk-by-chunk (where the
# needed fact was retained or dropped — the key artifact for a memory method).
_TRAJECTORY_FILE = "memory_trajectory.json"

# Live progress for a long-running task — MemAgent writes the trajectory + manifest only at
# the END, and a big instance fires hundreds of sequential memory-update calls over minutes,
# so this tiny file is updated per chunk so a mid-flight task's position is visible.
_PROGRESS_FILE = "progress.json"


class _Progress:
    """Writes/updates a tiny live ``progress.json`` (stage + chunk counter + elapsed) as the
    task runs. The loop is single-threaded (recurrent), so no lock is needed; written
    atomically (tmp + replace) so a reader never sees a half-written file."""

    def __init__(self, path: Path, llm: Any, task_id: str, n_docs: int):
        self._path = Path(path)
        self._llm = llm
        self._state: dict[str, Any] = {
            "task_id": str(task_id), "n_docs": n_docs, "stage": "starting",
            "n_chunks": None, "chunks_done": 0, "started_at": round(time.time(), 1),
        }
        self._flush()

    def stage(self, name: str, **extra: Any) -> None:
        self._state["stage"] = name
        self._state.update(extra)
        self._flush()

    def chunk_done(self) -> None:
        self._state["chunks_done"] += 1
        self._flush()

    def _flush(self) -> None:
        now = time.time()
        self._state["llm_calls"] = len(self._llm.full_calls)
        self._state["updated_at"] = round(now, 1)
        self._state["elapsed_s"] = round(now - self._state["started_at"], 1)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))
        tmp.replace(self._path)


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _strip_provider(model: str) -> str:
    """The HuggingFace repo id for the tokenizer — strip the litellm provider prefixes the
    harness uses (``hosted_vllm/``, ``openai/``), keeping ``org/name``. E.g.
    ``hosted_vllm/BytedTsinghua-SIA/RL-MemoryAgent-14B`` → ``BytedTsinghua-SIA/RL-MemoryAgent-14B``."""
    for prefix in ("hosted_vllm/", "openai/"):
        if model.startswith(prefix):
            model = model[len(prefix):]
    return model


@lru_cache(maxsize=4)
def _load_tokenizer(hf_model_id: str):
    """The served model's tokenizer (for the token-window chunker), cached per process.
    Lazy ``transformers`` import (the ``evals[memagent]`` extra) so the module imports
    without it; only a live run needs it. Tests monkeypatch this to inject a fake."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(hf_model_id)


def _build_problem(benchmark, task_id: str) -> str:
    """The ``problem`` fed to the memory-update + final-answer prompts — the SAME components
    every other baseline poses (documents are the context, dropped from the problem).

    - **loong**: ``instruction`` (the whole task when ``question`` is empty) else
      ``instruction\\n\\nquestion``.
    - **corpusqa**: ``question\\n\\n{output-requirements}`` (the output-requirements block,
      with the ``The answer is:`` contract the judge extracts, MUST reach the model).
    """
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 46-doc corpus is the context folded into memory.
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"memagent has no problem assembly for benchmark {name!r}")


def _pool_context(documents: list[str]) -> str:
    """Pool the per-instance multi-doc bundle into ONE context blob (``\\n\\n``-joined),
    matching upstream's single-context-blob input; the token-window chunker then streams
    over it (a chunk may span two documents, faithful to upstream — there is no
    document-boundary rule). A single doc passes through unchanged."""
    return "\n\n".join(documents)


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Per-task MemAgent pipeline, ALL inside ``run_dir``: pool docs → token-window chunk →
    recurrently update memory → answer. Returns the harness record — ``raw_answer`` + the
    **TOTAL** ``usage`` (every memory update + the final answer) + ``calls_full`` + a light
    ``trace``. Nothing is reused."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = litellm_kwargs["model"]
    if model.startswith("hosted_vllm/") and not litellm_kwargs.get("api_base"):
        # MemAgent's default is the RL-trained checkpoint, served on vLLM — it needs a
        # --base-url. (A non-vLLM --model, e.g. openai/…, is allowed and skips this.)
        raise ValueError(
            "memagent's RL model needs a vLLM endpoint: pass --base-url (and --api-key). "
            f"Got model={model!r} with no base URL."
        )

    documents = benchmark.get_documents(task_id)
    problem = _build_problem(benchmark, task_id)
    context = _pool_context(documents)

    chunk_size = int(run_config["chunk_size"])
    max_context_len = int(run_config.get("max_context_len", 0))
    tokenizer = _load_tokenizer(_strip_provider(model))
    chunks = chunker_mod.split_into_chunks(context, tokenizer, chunk_size, max_context_len)

    # ONE seam for the whole task → its usage IS the total. max_new caps the memory + answer.
    llm = MemAgentLLM(
        litellm_kwargs,
        seed=run_config.get("seed"),
        completion_params=run_config.get("completion_params"),
        max_new=int(run_config["max_new"]),
    )
    progress = _Progress(run_dir / _PROGRESS_FILE, llm, task_id, len(documents))

    # Recurrent memory loop: fold each chunk into the running memory (sequential — each
    # update depends on the previous). Keep every intermediate memory for the trajectory.
    progress.stage("memory", n_chunks=len(chunks))
    memory = prompts.NO_MEMORY
    memories: list[str] = []
    for chunk in chunks:
        memory = llm.complete(prompts.memory_update_prompt(problem, memory, chunk))
        memories.append(memory)
        progress.chunk_done()

    # Final answer from the built memory.
    progress.stage("answering")
    raw_answer = llm.complete(prompts.final_answer_prompt(problem, memory))
    progress.stage("answered")

    (run_dir / _TRAJECTORY_FILE).write_text(json.dumps({
        "problem": problem,
        "n_docs": len(documents),
        "n_chunks": len(chunks),
        "chunk_size": chunk_size,
        "max_context_len": max_context_len,
        "final_memory": memory,
        "memories": memories,   # the memory state after each chunk (the trajectory)
    }, ensure_ascii=False, indent=2))

    return {
        "raw_answer": raw_answer,
        "usage": llm.usage,                # TOTAL: every memory update + the final answer
        "calls_full": llm.full_calls,      # → calls.json (every internal call, in order)
        "trace": {
            "problem": problem,
            "n_docs": len(documents),
            "n_chunks": len(chunks),
            "chunk_size": chunk_size,
            "max_context_len": max_context_len,
            "final_memory": memory,
        },
    }
