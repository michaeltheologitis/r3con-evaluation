# r3con-evaluation

The **evaluation repo**: three grounded-reasoning benchmarks, nine published
baselines, and one shared contract so any method can be measured against them
without re-plumbing anything.

- **What each baseline is**, conceptually and as wired → [BASELINES.md](BASELINES.md).
- **Per-baseline deviations from upstream** → `evals/baselines/<name>/PROVENANCE.md`.

```
evals/
  benchmarks/{loong,corpusqa,dracula}/   the datasets + their judges
  baselines/<name>/                      one self-contained package per baseline
  r3con/                                 the method (see evals/r3con/README.md)
  llm/                                   the LiteLLM seam + token accounting
scripts/                                 log cleaners
analysis/                                empty — scoring and figures live in a separate repo
logs/                                    run outputs — gitignored, never committed
tests/
```

## Install

```bash
uv sync --extra test                                  # harness + tests
uv sync --extra test --extra structrag --extra arag   # ...plus specific baselines
```

Every baseline that needs one has its own extra (`structrag`, `arag`, `raptor`,
`hipporag`, `memagent`, `codeact`, `rlms`); `readagent` and `claude-code` need
none. All are co-installable in one environment.

Put `OPENAI_API_KEY` in `.env` (the judges and every embedding call route to
OpenAI). Completions can route anywhere — see the run examples below.

## Data

Dracula is vendored in-repo (nothing to download). The other two fetch once:

```bash
python -m evals.benchmarks.loong.download_docs                  # ~34 MB doc pool
python -m evals.benchmarks.corpusqa.download_data --set 1m      # ~1 GB (1m is the only wired tier)
```

## Benchmarks

| benchmark | what it asks | tasks | scoring |
| --- | --- | --- | --- |
| `loong` | extended multi-doc QA, EN + ZH; evidence scattered across a per-instance bundle ("leave no document behind") | 1,600 | LLM judge, **1–100** (Avg Score + Perfect Rate) |
| `corpusqa` | corpus-level analytical reasoning over ~1M-token bundles; computation-heavy, NL2SQL gold | 329 | LLM judge, **0/1** answer-equivalence |
| `dracula` | the hand-curated running example: *Dracula* decompiled into its 46 in-world documents | 1 | LLM judge, **0/1** (strict + lenient) |

Each exposes the same surface — `get_task_ids` / `get_task` / `get_documents` /
`get_task_answer` / `get_task_metadata` / `score` — so a baseline never learns a
benchmark's shape beyond "a question and a set of documents".

## Running a baseline

Every baseline is its own runner; the baseline *is* the module you run.

```bash
# against OpenAI (default model: gpt-5.4-nano)
python -m evals.baselines.codeact --benchmark loong --limit 5

# against a local vLLM endpoint
python -m evals.baselines.structrag --benchmark corpusqa \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 --api-key <key>

# rlm and memagent are vLLM-only by design (--base-url required)
python -m evals.baselines.rlm --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B --base-url http://localhost:8555/v1 --api-key <key>
```

Shared flags: `--limit N` (run the next N pending tasks), `--seed`, `--config`
(a named sampling preset), `--max-workers`. Each task runs in its own child
process exactly once — no retries — and ends with either a `manifest.json` or an
`error.json`, so a re-run only does what is missing.

`arag` needs a tool-calling endpoint (`--enable-auto-tool-choice
--tool-call-parser hermes` on vLLM); `claude-code` shells out to the `claude` CLI
and runs on a Max login, not the served model.

## Output

This repo **runs methods and records what they produced** — it does not grade or
aggregate. Each benchmark still exposes `score` / `score_details` (its judge), for
whatever reads these logs later; scoring, tables and figures live outside this repo.

Logs live at `logs/{benchmark}/{baseline}/`, one folder per run holding the
manifest, the method's own artifacts (trajectory, index, memory — whatever it
builds) and `calls.json`. **Token cost is captured completely**:
every completion and every embedding a task makes, in one `usage` record.

## Tests

```bash
uv run pytest -q          # 557 tests, all fake-driven (no API calls)
```

Tests that hit a real model are marker-gated and deselected by default
(`pytest -m structrag`, `-m arag`, `-m judge`, …).
