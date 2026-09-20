"""Per-task structured logging at ``logs/<run-folder>/``.

Each task-run gets **one flat, opaque, uniquely-named folder** — no ``run-dir/task-id``
nesting. ``<run-folder>`` (:func:`new_run_folder`) is ``<UTC-timestamp>_<hex>`` and
carries **no identity at all**: the benchmark, task id, and the full run config live
*inside* it in ``manifest.json``. Every run is its own folder (timestamp + a random
suffix never collide), so re-runs accumulate rather than overwrite. Each manifest
records the run's identity (its ``config`` block, from which
:meth:`evals.r3con.pipeline.config.RunConfig.label` is recomputed).

Layout per ``logs/<run-folder>/``:

- ``manifest.json`` — the identity card + the full parameter snapshot: benchmark,
  task_id, question, gold; a ``config`` block (the resolved :class:`RunConfig` —
  model, seed, summary_rounds, prompts, sampling, …); and a ``settings`` block (the
  runtime knobs). Written first, so a crashed task is still discoverable. The run label
  is **not** stored — it is recomputed from the ``config`` block
  (:meth:`evals.r3con.pipeline.config.RunConfig.label`), so a label-format change never
  invalidates a folder already on disk.
- ``summaries/`` — ``result.json`` (``{n_rounds, n_docs, rounds: [{round, summaries}],
  totals}`` — per round, ``summaries[i]`` aligns to ``documents[i]``; the last round
  feeds downstream) + ``calls.json`` (one per ``summarize_one``, tagged
  ``summary-r{round}-d{doc}``).
- ``proposer/`` — ``result.json`` (``{schema_code, thought, attempts, totals}``) +
  ``calls.json`` + ``transcript.yaml``.
- ``extractor/`` — ``result.json`` (``{parsed, source_docs, totals}``) + ``calls.json``
  (one entry per ``extract-d{doc}`` call).
- ``inference/llm/`` and ``inference/codeact/`` — ``result.json`` (``{answer, …,
  totals}``) + ``calls.json`` (+ ``transcript.yaml`` for codeact) + ``error.txt``
  (on a per-strategy failure).
- ``run_task.log`` — the child process's stdout/stderr (written by the launcher).

Two layers in code:

- ``TaskLogger`` writes structured artifacts under one run-folder. ``write_json("a/b",
  data)`` writes ``<folder>/a/b.json`` and auto-creates any missing parent directories
  — so callers spell stage paths inline (``"proposer/result"``,
  ``"inference/codeact/result"``).
- ``StageRun`` accumulates per-LLM-call ``StepRecord`` instances and on ``flush()``
  writes ``calls.json`` (the full per-call record — ``{step, kind, doc, chunk,
  tokens, finish_reason, prompt, output}``) and, unless skipped, ``transcript.yaml``
  (the readable message thread). Aggregate token totals are available via
  ``compute_totals()`` so callers fold them into the stage's ``result.json``.
"""

from __future__ import annotations

import datetime
import json
import re
import secrets
import threading
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# ``normalize_model_name`` now lives in ``evals.r3con.pipeline.config`` (it's a model-identity helper
# used by ``RunConfig.label()`` too); re-exported here so ``evals.r3con.pipeline.runs.normalize_model_name``
# keeps working for the transcript-header write below and existing importers.
from evals.r3con.pipeline.config import normalize_model_name
from evals.r3con.pipeline.settings import settings


# Pulls the source-document index (and, for legacy logs, a chunk index) out of a
# call kind so calls.json carries per-call provenance. Matches ``extract-d0`` and
# ``summary-r2-d3`` (chunk group absent) as well as legacy ``extract-d12c5``.
_DOC_CHUNK_RE = re.compile(r"-d(\d+)(?:c(\d+))?\b")


def new_run_folder() -> str:
    """A fresh, opaque, unique log-folder name for one task-run:
    ``<UTC-timestamp>_<hex>`` (e.g. ``20260613T142233Z_a1b2c3d4``).

    Carries **no identity** — the benchmark, task id, and the full run config live
    inside the folder in ``manifest.json``. Every call is unique (timestamp + a random
    suffix), so runs **accumulate** rather than overwrite, and a re-run reads each
    manifest to tell what is already done. The runner stamps one folder per task and hands
    it to the child via ``--run-folder`` (so parent and child write to the same place).
    """
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(4)}"


class TaskLogger:
    """Writes per-stage artifacts under one flat run-folder ``logs/<folder>/``.

    Callers spell stage paths inline:

    - ``write_json("manifest", {...})`` → ``<folder>/manifest.json``
    - ``write_json("proposer/result", {...})`` → ``<folder>/proposer/result.json``
    - ``write_yaml("inference/codeact/transcript", msgs)`` →
      ``<folder>/inference/codeact/transcript.yaml``

    ``folder`` is the unique run-folder name (see :func:`new_run_folder`); ``task_id``
    is the benchmark task id recorded in transcript headers (defaults to ``folder``).
    Parent directories are auto-created. The extension is appended from the method
    name. ``data`` may be a Pydantic ``BaseModel`` / class, a ``dict``/``list``, or
    any JSON-serializable value; non-serializable values fall back to ``repr``.
    """

    def __init__(self, folder: str, *, task_id: str | None = None, root: Path | None = None) -> None:
        self.folder = folder
        self.task_id = task_id if task_id is not None else folder
        base = root or settings.LOGS_DIR
        self.dir = base / folder
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, ext: str) -> Path:
        path = self.dir / f"{name}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, data: Any) -> Path:
        path = self._path(name, "json")
        # ensure_ascii=False keeps non-ASCII (e.g. CJK) readable in the logs rather than
        # as \uXXXX escapes; the explicit utf-8 write makes that safe on any host locale.
        path.write_text(json.dumps(_jsonable(data), indent=2, default=repr, ensure_ascii=False), encoding="utf-8")
        return path

    def write_yaml(self, name: str, data: Any) -> Path:
        path = self._path(name, "yaml")
        path.write_text(_dump_transcript_yaml(data), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._path(name, "txt")
        path.write_text(text, encoding="utf-8")
        return path


@dataclass
class StepRecord:
    """One step in a stage run — typically one LLM call (or retry, or loop turn)."""

    step: int
    kind: str  # "llm_call" | "retry" | "turn-N" | "summary-r2-d3" | "extract-d0" | ...
    messages: list[dict[str, Any]]
    response: dict[str, Any]  # {"role": ..., "content": ..., "finish_reason": ...}
    schema: dict[str, Any] | None = None
    tokens: dict[str, int] | None = None  # {"prompt": int, "completion": int, "total": int}


class StageRun:
    """Accumulates per-LLM-call records for one stage; flushes calls.json (+ transcript).

    Each stage constructs one of these, calls ``add_step`` per LLM call (and per
    loop turn for the CodeAct loop), and ``flush`` at the end. ``add_step`` is
    thread-safe so a stage that fans calls out across documents in parallel
    (summaries rounds, per-doc extraction) can share one run.
    """

    def __init__(
        self,
        *,
        stage: str,
        task_logger: TaskLogger,
        model: str | None = None,
        seed: int | None = None,
    ) -> None:
        self.stage = stage
        self.task_logger = task_logger
        self.model = model
        self.seed = seed
        self.steps: list[StepRecord] = []
        self._lock = threading.Lock()
        self.dir = task_logger.dir / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    def add_step(
        self,
        *,
        kind: str = "llm_call",
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        schema: dict[str, Any] | None = None,
        tokens: dict[str, int] | None = None,
    ) -> StepRecord:
        """Record one step (thread-safe). Step numbers are assigned under a lock, in
        completion order; each call's ``kind`` carries its own identity."""
        with self._lock:
            step = StepRecord(
                step=len(self.steps) + 1,
                kind=kind,
                messages=list(messages),
                response=dict(response),
                schema=schema,
                tokens=tokens,
            )
            self.steps.append(step)
        return step

    def flush(self, *, write_transcript: bool = True) -> Path:
        """Write ``calls.json`` (always) and, unless ``write_transcript=False``,
        ``transcript.yaml``; return the path of the primary file written.

        ``write_transcript=False`` is used by stages whose many parallel
        independent calls (summaries, extractor) make a flattened transcript
        confusing — their ``calls.json`` is the readable source.
        """
        calls_path = self._write_calls()
        if not write_transcript:
            return calls_path
        transcript = {
            "stage": self.stage,
            "task_id": self.task_logger.task_id,
            "model": normalize_model_name(self.model),
            "seed": self.seed,
            "messages": self._build_transcript_messages(),
        }
        transcript_path = self.dir / "transcript.yaml"
        transcript_path.write_text(_dump_transcript_yaml(transcript), encoding="utf-8")
        return transcript_path

    def _write_calls(self) -> Path:
        """Write ``calls.json`` — one entry per LLM call: ``{step, kind, doc, chunk,
        tokens, finish_reason, prompt, output}``. ``doc``/``chunk`` are parsed from the
        call ``kind`` (e.g. ``extract-d0``, ``summary-r1-d2``) when present."""
        calls: list[dict[str, Any]] = []
        for s in self.steps:
            m = _DOC_CHUNK_RE.search(s.kind)
            resp = s.response if isinstance(s.response, dict) else {}
            calls.append(
                {
                    "step": s.step,
                    "kind": s.kind,
                    "doc": int(m.group(1)) if m else None,
                    "chunk": int(m.group(2)) if (m and m.group(2)) else None,
                    "tokens": s.tokens,
                    "finish_reason": resp.get("finish_reason"),
                    "prompt": s.messages,
                    "output": resp.get("content"),
                }
            )
        path = self.dir / "calls.json"
        path.write_text(json.dumps(calls, indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        return path

    def compute_totals(self) -> dict[str, Any]:
        """Aggregate per-step tokens / step-count for the stage's ``result.json``."""
        prompt = completion = total = 0
        any_tokens = False
        for s in self.steps:
            if s.tokens is not None:
                any_tokens = True
                prompt += s.tokens.get("prompt", 0)
                completion += s.tokens.get("completion", 0)
                total += s.tokens.get("total", 0)
        return {
            "n_steps": len(self.steps),
            "tokens": {"prompt": prompt, "completion": completion, "total": total} if any_tokens else None,
        }

    def _build_transcript_messages(self) -> list[dict[str, Any]]:
        """The readable conversation = the final step's full message thread + its
        response. For a multi-turn loop (codeact) the last step's ``messages`` already
        contain the whole grown conversation; for a retry loop (proposer) they are the
        final (successful) attempt — earlier attempts live in ``calls.json``. Empty
        run → ``[]``. (Stages with many parallel calls flush with
        ``write_transcript=False`` and never reach here.)"""
        if not self.steps:
            return []
        last = self.steps[-1]
        thread = list(last.messages)
        if last.response.get("content"):
            thread.append(last.response)
        return thread


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Render strings with newlines as literal ``|`` block scalars (readable
    transcripts); rstrip each line so trailing whitespace doesn't force PyYAML's
    escaped single-line fallback."""
    if "\n" in data:
        cleaned = "\n".join(line.rstrip() for line in data.split("\n"))
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


class _BlockDumper(yaml.SafeDumper):
    """Local SafeDumper subclass so the literal-block representer stays local."""


_BlockDumper.add_representer(str, _str_representer)


def _dump_transcript_yaml(payload: Any) -> str:
    return yaml.dump(
        _jsonable(payload),
        Dumper=_BlockDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=10000,
    )


def _jsonable(data: Any) -> Any:
    """Coerce Pydantic instances → ``model_dump``, classes → JSON schema; recurse
    lists/dicts; leave everything else for ``json.dumps`` (``default=repr`` net)."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, type) and issubclass(data, BaseModel):
        return data.model_json_schema()
    if isinstance(data, dict):
        return {k: _jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_jsonable(v) for v in data]
    return data
