# R3Con — the method

Just-in-time reasoning over a collection of documents. This directory is the method and
what it takes to run it on the three benchmarks — nothing more. It does not grade, score,
rank, or tidy anything; it runs, and it writes down everything it did.

```
evals/r3con/
  pipeline/    the method (benchmark-agnostic — it never imports a benchmark)
  harness/     the three adapters + the subprocess launcher
```

## The pipeline

Four stages over a collection of documents, each assumed to fit in the model's context
(there is no chunking). Everything is built **per task**, at the moment the task is asked
— no fixed schema, no pre-built index.

```
(task, documents)
      │
      ▼
SUMMARIES   task-conditioned, iterative (default 2 rounds; docs fan out in parallel)
            R1: summary(task, doc)                          # no other-doc context
            R2: summary(task, doc, OTHER docs' R1 summaries)
      ▼
PROPOSER    schema(task, all summaries) -> a Pydantic `Parse` class
      ▼
EXTRACTOR   one call per whole document, in parallel, against that schema
            merged into one Parse, each record tagged with its source document
      ▼
INFERENCE   `llm` (one call over parse + summaries) AND/OR
            `codeact` (a multi-turn loop that runs Python over the parse in a
             sandbox and commits via final_answer(...))
```

Both inference strategies run over the *same* extraction, so `--inference both` costs one
upstream pass. `pipeline.gr_answer(task=..., documents=..., config=...)` is the whole
entry point.

## Running it

```bash
uv run python scripts/r3con/loong/run.py --set 1 --n 5 --inference both
uv run python scripts/r3con/corpusqa/run.py --limit 5 --inference both
uv run python scripts/r3con/dracula/run.py --limit 1 --inference both
```

Against a served endpoint instead of OpenAI:

```bash
uv run python scripts/r3con/loong/run.py --all --inference both \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 \
  --api-key <key> --workers 20
```

One subprocess per task (true concurrency + crash isolation). `--workers` is tasks in
parallel, `--doc-workers` is documents in parallel *within* a task — peak load on the
endpoint is roughly the product. `--verbose` streams per-stage progress into each run's
`run_task.log`. `scripts/r3con/<benchmark>/run_task.py` runs a single task by hand.

Re-running picks up where it left off: tasks already finished for this exact run identity
under every requested strategy are skipped, so an interrupted 1,600-task run doesn't pay
twice. `--force` re-runs them anyway.

## The run config is the experiment identity

A `RunConfig` bundles everything that shapes the output — model, seed, summary rounds, the
prompt version of each stage, and the sampling params — loaded from
`pipeline/configs/<name>.yaml` and threaded through the whole pipeline as one object. Pick
one with `--config <name>`; override single fields with `--model` / `--seed` /
`--summary-rounds` / `--sampling`.

`RunConfig.label()` renders that identity:

```
default[model=gpt-5.4-nano,seed=42,sr=2,prompts=(sum=v3,prop=v4,ext=v2,llm=v4,cod=v4)]
```

Two runs differing in *any* output-shaping knob — including one prompt version — are
different runs, which is also what makes resume safe. To add an axis, add a line to
`RunConfig._identity_parts()`; don't put it in a path or an env var.

Runtime-only knobs (parallelism, retry caps, the turn cap) live in `pipeline/settings.py`
— they can't change a correct answer, so they are deliberately *not* part of the identity.

## Prompts are versioned per stage

`pipeline/prompts/<stage>/<version>.yaml`, with the active version pinned per stage in the
run config. To iterate, add the next `vN.yaml` and point a config at it — **never edit a
version in place**: a run's manifest records the version string, so an in-place edit
destroys the record of what actually ran.

## What a run leaves behind

One folder per task-run under `logs/r3con/`, holding every intermediate the method built:

```
logs/r3con/<UTC-timestamp>_<hex>/
  manifest.json                  benchmark, task_id, question, gold, n_docs, a `config`
                                 block (the resolved RunConfig) and a `settings` block
  summaries/result.json          every round's per-document summary
  proposer/result.json           the proposed schema + the reasoning behind it
  extractor/result.json          the merged parse + which document each record came from
  inference/{llm,codeact}/       the answer, the full turn-by-turn transcript for codeact
  */calls.json                   every LLM call: prompt, response, tokens
  run_task.log
```

The folder name carries no identity — the benchmark and the full run config live inside
the manifest. Runs accumulate rather than overwrite. Reading one of these folders is the
point: it shows what the method found relevant, what structure it proposed for the
question, and how it reasoned over that structure.

## Tests

```bash
uv run --extra test pytest tests/r3con -q
```

All fake-driven — no API calls, no network.
