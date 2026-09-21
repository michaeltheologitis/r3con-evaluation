# r3con-evaluation

Three grounded-reasoning benchmarks, nine baselines, and R3Con (the method under
evaluation), all on one shared contract. This repo runs methods and records what they
produced. Scoring and figures live in a separate repo.

## Where things are

```
evals/
  benchmarks/loong/       1,600 multi-doc QA tasks, EN + ZH
  benchmarks/corpusqa/    329 tasks over ~1M-token bundles
  benchmarks/dracula/     1 hand-curated task over 46 in-world documents
  baselines/<name>/       one package per baseline (see BASELINES.md)
  r3con/                  the method (see evals/r3con/README.md)
  llm/                    LiteLLM seam + token accounting
  settings.py             model ids and paths
scripts/                  log cleaners, and the method's run entry points
logs/                     run outputs (gitignored)
tests/
```

Each baseline package holds the same files: `run.py` (the per-task work), `runner.py`
(the CLI), `PROVENANCE.md` (upstream and what was changed), and `upstream/` when the
method's source is vendored.

Other docs:

- [BASELINES.md](BASELINES.md) — what each baseline is and how it handles documents.
- [THIRD_PARTY.md](THIRD_PARTY.md) — every vendored upstream and its license.
- [evals/r3con/README.md](evals/r3con/README.md) — the method's pipeline and its logs.
- [evals/benchmarks/loong/CHANGES.md](evals/benchmarks/loong/CHANGES.md) — how our Loong differs from upstream.
- [evals/benchmarks/dracula/INFO.md](evals/benchmarks/dracula/INFO.md) — the Dracula corpus and its gold answer.

## Install

```bash
uv sync --extra test                                  # harness + tests
uv sync --extra test --extra r3con                    # ...plus the method
uv sync --extra test --extra structrag --extra arag   # ...plus specific baselines
```

Each baseline that needs dependencies has its own extra (`structrag`, `arag`, `raptor`,
`hipporag`, `memagent`, `codeact`, `rlms`); `readagent` and `claude-code` need none. All
are co-installable.

Put `OPENAI_API_KEY` in `.env`. The judges and every embedding call go to OpenAI;
completions can go anywhere.

## Data

Dracula is in the repo. The other two download once:

```bash
python -m evals.benchmarks.loong.download_docs                  # ~34 MB
python -m evals.benchmarks.corpusqa.download_data --set 1m      # ~1 GB
```

## Running

A baseline is the module you run:

```bash
python -m evals.baselines.codeact --benchmark loong --limit 5

python -m evals.baselines.structrag --benchmark corpusqa \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 --api-key <key>
```

Flags: `--limit N` (next N pending tasks), `--seed`, `--max-workers`. Runs use the served
model's own sampling. Each task runs once in its own process and writes either
`manifest.json` or `error.json`, so re-running only does what is missing.

`rlm` and `memagent` require `--base-url` (vLLM only). `arag` needs a tool-calling
endpoint. `claude-code` shells out to the `claude` CLI and runs on a Max login, not the
served model.

The method has its own entry points:

```bash
python scripts/r3con/loong/run.py --set 1 --n 5 --inference both
python scripts/r3con/corpusqa/run.py --limit 5 --inference both
python scripts/r3con/dracula/run.py --limit 1 --inference both
```

## Logs

Baselines write to `logs/{benchmark}/{baseline}/`, one folder per task-run holding the
manifest, whatever the method built (trajectory, index, memory), and `calls.json`.
`arag` and `structrag` nest that folder under `inferences/{hash}/`; `arag` also keeps its
index in a sibling `_indices/`, reused across runs over the same documents.

R3Con writes to `logs/r3con/`, one folder per task-run with every stage's intermediates.

Either way each task gets one `usage` record in the same `{total, calls}` shape, covering
every completion and embedding, so baseline and R3Con costs are comparable.

## Tests

```bash
uv run pytest -q          # 744 tests, no API calls
```

Tests that hit a real model are marker-gated and skipped by default (`pytest -m arag`,
`-m judge`, …).
