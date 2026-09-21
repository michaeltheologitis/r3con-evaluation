# R3Con

Just-in-time reasoning over a collection of documents. This directory is the method and
what it takes to run it on the three benchmarks.

```
evals/r3con/
  pipeline/    the method — benchmark-agnostic, never imports a benchmark
    stages/    the four stages
    runtime/   the LLM seam, the CodeAct loop, the sandbox executor
    prompts/   versioned per stage (see below)
    configs/   experiment configs + sampling presets
  harness/     one adapter per benchmark + the subprocess launcher
scripts/r3con/<benchmark>/run.py        runs it
scripts/r3con/<benchmark>/run_task.py   runs one task by hand
```

## The pipeline

Four stages, everything built per task at the moment it is asked. No chunking — each
document is assumed to fit in context. No fixed schema, no pre-built index.

```
(task, documents)
  summaries   task-conditioned, 2 rounds by default; docs fan out in parallel
              R1: summary(task, doc)
              R2: summary(task, doc, other docs' R1 summaries)
  proposer    schema(task, all summaries) -> a Pydantic `Parse` class
  extractor   one call per whole document, in parallel, against that schema;
              merged into one Parse, each record tagged with its source document
  inference   `llm` (one call over parse + summaries) and/or
              `codeact` (a loop running Python over the parse in a sandbox,
               committing via final_answer(...))
```

Both inference strategies run over the same extraction, so `--inference both` costs one
upstream pass. `pipeline.gr_answer(task=..., documents=..., config=...)` is the entry point.

## Running

```bash
uv run python scripts/r3con/loong/run.py --set 1 --n 5 --inference both
uv run python scripts/r3con/corpusqa/run.py --limit 5 --inference both
uv run python scripts/r3con/dracula/run.py --limit 1 --inference both
```

Against a served endpoint:

```bash
uv run python scripts/r3con/loong/run.py --all --inference both \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 \
  --api-key <key> --workers 20
```

One subprocess per task. `--workers` is tasks in parallel, `--doc-workers` is documents in
parallel within a task, so peak load is roughly the product. `--verbose` streams per-stage
progress into each run's `run_task.log`.

Re-running resumes: tasks already finished for this benchmark under the same run identity
and every requested strategy are skipped. `--force` re-runs them.

## Run config

A `RunConfig` holds everything that shapes the output — model, seed, summary rounds, each
stage's prompt version, sampling params — loaded from `pipeline/configs/<name>.yaml` and
threaded through as one object. Pick one with `--config`; override fields with `--model`,
`--seed`, `--summary-rounds`, `--sampling`.

`RunConfig.label()` renders it, and two runs differing in any of it are different runs,
which is what makes resume safe:

```
default[model=gpt-5.4-nano,seed=42,sr=2,prompts=(sum=v3,prop=v4,ext=v2,llm=v4,cod=v4)]
```

To add an axis, add a line to `RunConfig._identity_parts()`. Parallelism and retry caps
live in `pipeline/settings.py` — they can't change a correct answer, so they are not part
of the identity.

Prompts are versioned at `pipeline/prompts/<stage>/<version>.yaml`, pinned per stage in
the config. To iterate, add the next `vN.yaml` and point a config at it. Never edit a
version in place: a manifest records the version string, so editing one destroys the
record of what ran.

## What a run leaves behind

One folder per task-run under `logs/r3con/`:

```
logs/r3con/<UTC-timestamp>_<hex>/
  manifest.json          benchmark, task_id, question, gold, n_docs, the resolved
                         config and settings, and `usage` (the task's whole token
                         cost, added when the run ends)
  summaries/result.json  every round's per-document summary
  proposer/result.json   the proposed schema and the reasoning behind it
  extractor/result.json  the merged parse and which document each record came from
  inference/{llm,codeact}/   the answer, plus the full transcript for codeact
  */calls.json           every LLM call: prompt, response, tokens
  run_task.log
```

The folder name carries no identity — benchmark and run config live in the manifest. Runs
accumulate rather than overwrite.

## Tests

```bash
uv run pytest tests/r3con -q
```

No API calls.
