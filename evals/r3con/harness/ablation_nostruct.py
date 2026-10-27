"""The "w/o structuring" ablation: re-run CodeAct over a finished run's summaries, WITHOUT the parse.

It recomputes nothing upstream. For a run folder that already has its ``summaries/result.json``,
it runs a CodeAct loop conditioned on the **summaries only** — no structured ``parse`` in the
prompt and nothing bound in the sandbox — to measure the parse's marginal contribution to the
coding agent. This is the ``codeact_ablation_nostruct`` strategy the analysis reads
(``analysis/scoreboard/report.py``, the "w/o structuring" row).

It changes **nothing in the pipeline core**: the "no parse" CodeAct call is a small *duplicate*
of :func:`evals.r3con.pipeline.stages.inference.infer_codeact` with the two parse lines removed
(no ``parse_block`` in the prompt, ``variables={}`` in the sandbox), reusing the generic
``run_codeact`` loop, ``render_summaries``, ``load_prompt`` and ``StageRun`` unchanged. The
prompt is ``pipeline/prompts/inference/codeact/ablation-nostruct.yaml``.

Output goes into each run folder as a **new strategy** subfolder,
``inference/codeact_ablation_nostruct/`` (``result.json`` + ``calls.json`` + ``transcript.yaml``,
or ``error.txt``), so it sits beside ``llm`` / ``codeact`` under the same run identity and the
analysis discovers it like any other strategy. The folder's ``manifest.json`` is left untouched:
its ``usage`` block stays the cost of the run that produced the summaries and the parse; the
ablation's own cost is in its ``result.json`` ``totals``, which is what the analysis sums.

The manifest's ``config.model`` is the bare name (not routable), so a routable ``model`` is
passed for the call.

Two things differ from the code the paper's runs were made with, both so the script can be
pointed at the shipped, stripped ``logs/``: the task text comes from the manifest's
``question`` when present and is otherwise re-loaded from the benchmark adapter
(:func:`task_text`), and a folder whose summaries were stripped to their token totals is
skipped rather than attempted (:func:`has_summaries`).
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any

from evals.r3con.harness import corpusqa, dracula, loong
from evals.r3con.pipeline.config import RunConfig
from evals.r3con.pipeline.prompts import load_prompt
from evals.r3con.pipeline.runs import StageRun, TaskLogger
from evals.r3con.pipeline.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S, CodeActResult, run_codeact
from evals.r3con.pipeline.settings import settings
from evals.r3con.pipeline.stages.summaries import render_summaries

# The ablation's prompt (a version of the inference/codeact stage) and the strategy name, which is
# also the output subfolder and the analysis's strategy id.
ABLATION_PROMPT = "ablation-nostruct"  # pipeline/prompts/inference/codeact/ablation-nostruct.yaml
STRATEGY = "codeact_ablation_nostruct"

# The benchmark adapters, keyed by the display name each ``run_task`` stamps into its manifest —
# how :func:`task_text` re-loads a task whose text the manifest no longer carries.
_ADAPTERS: dict[str, Any] = {loong.NAME: loong, corpusqa.NAME: corpusqa, dracula.NAME: dracula}


def _read_summaries(folder: Path) -> dict | None:
    try:
        return json.loads((folder / "summaries" / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def has_summaries(folder: Path) -> bool:
    """True iff the folder has a manifest and computed summaries the ablation can read — a
    ``summaries/result.json`` with at least one round. A folder stripped to the stage's token
    totals, or a ``summary_rounds=0`` run, has nothing for a summaries-only agent to read."""
    if not (folder / "manifest.json").is_file():
        return False
    s = _read_summaries(folder)
    return bool(s and s.get("rounds"))


def folders_to_ablate(logs_root: Path, *, label: str | None = None, benchmark: str | None = None) -> list[str]:
    """Folder names under ``logs_root`` that have summaries but are missing the ablation strategy.
    ``label`` restricts to folders whose recomputed :meth:`RunConfig.label` matches it exactly;
    ``benchmark`` restricts by the manifest's ``benchmark`` field (``"Loong"`` / ``"CorpusQA"`` /
    ``"Dracula"``). Folders with an unparseable manifest are skipped."""
    out: list[str] = []
    if not logs_root.is_dir():
        return out
    for f in sorted(p for p in logs_root.iterdir() if p.is_dir() and not p.name.startswith("_")):
        if not has_summaries(f):
            continue
        if (f / "inference" / STRATEGY / "result.json").is_file():
            continue  # already ablated
        if label is not None or benchmark is not None:
            try:
                man = json.loads((f / "manifest.json").read_text(encoding="utf-8"))
                if benchmark is not None and man.get("benchmark") != benchmark:
                    continue
                if label is not None and RunConfig(**man["config"]).label() != label:
                    continue
            except Exception:  # noqa: BLE001 — unparseable/legacy manifest -> skip
                continue
        out.append(f.name)
    return out


def infer_codeact_nostruct(
    *,
    task: str,
    summaries: list[str] | None,
    model: str,
    prompt_version: str = ABLATION_PROMPT,
    max_turns: int = settings.INFERENCE_MAX_TURNS,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> CodeActResult:
    """CodeAct over the **summaries only** — the ablation of
    :func:`evals.r3con.pipeline.stages.inference.infer_codeact`. The two differences from the
    original are the whole point: the prompt carries **no** ``parse_block``, and **nothing** is
    bound in the sandbox (``variables={}``). Everything else (the loop, allowed imports,
    ``final_answer``) is the generic ``run_codeact``."""
    system_prompt = load_prompt(
        "inference/codeact", version=prompt_version, task=task, summaries=render_summaries(summaries)
    )
    return run_codeact(
        system_prompt=system_prompt,
        user_message=f"Input:\n<task>\n{task}\n</task>",
        model=model,
        variables={},  # <-- the ablation: no `parse` (or anything) bound in the sandbox
        max_turns=max_turns,
        timeout_s=timeout_s,
        run=run,
        **llm_kwargs,
    )


def task_text(manifest: dict) -> str:
    """The task the run was asked: the manifest's ``question`` when it carries one, else the same
    composed task re-loaded from the benchmark adapter named by ``benchmark`` + ``task_id`` (the
    shipped logs were stripped of their questions). The adapters compose the task exactly as the
    original run did, so the ablation sees what the run saw."""
    q = manifest.get("question")
    if q:
        return q
    adapter = _ADAPTERS.get(manifest.get("benchmark"))
    if adapter is None or not manifest.get("task_id"):
        raise KeyError(
            "manifest has no `question`, and no `benchmark` + `task_id` to re-load it from "
            f"(benchmark={manifest.get('benchmark')!r}, task_id={manifest.get('task_id')!r})."
        )
    return adapter.load(manifest["task_id"]).task


def _summaries_of(folder: Path) -> list[str]:
    """The final-round per-document summaries from a folder's ``summaries/result.json``."""
    s = _read_summaries(folder) or {}
    rounds = s.get("rounds") or []
    if not rounds:
        raise ValueError(
            f"{folder.name}: summaries/result.json has no rounds — either a summary_rounds=0 run "
            "or a folder stripped to its token totals; nothing for a summaries-only agent to read."
        )
    return rounds[-1]["summaries"]


def ablate_one(
    folder_name: str,
    *,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    logs_root: Path | None = None,
) -> tuple[str, str | None]:
    """Run the summaries-only CodeAct ablation for one run folder; write native logs into
    ``inference/codeact_ablation_nostruct/``. ``model`` is the routable model string (the
    manifest's bare name can't route). Returns ``(answer, error_or_None)``; a failure leaves
    ``error.txt`` in the strategy folder, like a per-strategy failure of a normal run."""
    root = logs_root or settings.LOGS_DIR
    folder = root / folder_name
    man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    cfg = RunConfig(**man["config"]).model_copy(update={"model": model})  # routable model for the call

    log = TaskLogger(folder_name, task_id=man.get("task_id"), root=root)
    run = StageRun(stage=f"inference/{STRATEGY}", task_logger=log, model=model, seed=cfg.seed)
    llm_kwargs: dict[str, Any] = {**cfg.sampling, "seed": cfg.seed}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key

    try:
        summaries = _summaries_of(folder)
        result = infer_codeact_nostruct(
            task=task_text(man), summaries=summaries, model=model, run=run, **llm_kwargs
        )
        run.flush()
        log.write_json(
            f"inference/{STRATEGY}/result",
            {
                "answer": result.answer,
                "terminated_by": result.terminated_by,
                "n_turns": len(result.turns),
                "turns": [
                    {
                        "response": t.response, "raw_response": t.raw_response,
                        "code": t.code, "observation": t.observation,
                        "error": t.error, "is_final_answer": t.is_final_answer,
                    }
                    for t in result.turns
                ],
                "totals": run.compute_totals(),
            },
        )
        return (result.answer, None)
    except Exception as e:  # noqa: BLE001 — record + report (the analysis reads error.txt)
        err_dir = log.dir / "inference" / STRATEGY
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / "error.txt").write_text(traceback.format_exc())
        return ("", f"{type(e).__name__}: {e}")
