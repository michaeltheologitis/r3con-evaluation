# r3con-evaluation

The **evaluation repo**: three grounded-reasoning benchmarks, nine baselines
(eight published methods plus a frontier-agent reference point), and one shared
contract so any method can be measured against them without re-plumbing anything.

- **What each baseline is**, conceptually and as wired → [BASELINES.md](BASELINES.md).
- **Per-baseline deviations from upstream** → `evals/baselines/<name>/PROVENANCE.md`.
- **Every vendored upstream and its license** → [THIRD_PARTY.md](THIRD_PARTY.md).
- **This repo's own code is MIT** → [LICENSE](LICENSE).

```
evals/
  benchmarks/{loong,corpusqa,dracula}/   the datasets + their judges
  baselines/<name>/                      one self-contained package per baseline
  r3con/                                 the method (see evals/r3con/README.md)
  llm/                                   the LiteLLM seam + token accounting
scripts/                                 log cleaners + the method's run entry points
analysis/                                empty — scoring and figures live in a separate repo
logs/                                    run outputs — gitignored, never committed
tests/
```

## Install

```bash
uv sync --extra test                                  # harness + tests
uv sync --extra test --extra r3con                    # ...plus the method itself
uv sync --extra test --extra structrag --extra arag   # ...plus specific baselines
```

The method declares its own extra, `r3con` (jinja2 / pyyaml / tiktoken) — what
`evals/r3con/` itself needs, named explicitly rather than leaned on
transitively. Every baseline that needs one has its own extra too (`structrag`,
`arag`, `raptor`, `hipporag`, `memagent`, `codeact`, `rlms`); `readagent` and
`claude-code` need none. `full` adds the notebook and plotting tooling on top of
`test`. All are co-installable in one environment.

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

Shared flags: `--limit N` (run the next N pending tasks), `--seed`,
`--max-workers`. Every run uses the served model's own default sampling. Each
task runs in its own child process exactly once — no retries — and ends with
either a `manifest.json` or an `error.json`, so a re-run only does what is
missing.

`arag` needs a tool-calling endpoint (`--enable-auto-tool-choice
--tool-call-parser hermes` on vLLM); `claude-code` shells out to the `claude` CLI
and runs on a Max login, not the served model.

## Running the method

R3Con — the method under evaluation — lives in `evals/r3con/` and has its own
per-benchmark entry points (install the `r3con` extra first):

```bash
python scripts/r3con/loong/run.py --set 1 --n 5 --inference both
python scripts/r3con/corpusqa/run.py --limit 5 --inference both
python scripts/r3con/dracula/run.py --limit 1 --inference both
```

The pipeline stages, the run-config that is the experiment identity, the
per-stage versioned prompts, and what a run leaves on disk are documented in
[evals/r3con/README.md](evals/r3con/README.md).

## Output

This repo **runs methods and records what they produced** — it does not grade or
aggregate. Each benchmark still exposes `score` / `score_details` (its judge), for
whatever reads these logs later; scoring, tables and figures live outside this repo.

Baseline logs live at `logs/{benchmark}/{baseline}/`, one folder per task-run
holding the manifest (or an `error.json` if it failed), the method's own
artifacts (trajectory, index, memory — whatever it builds) and `calls.json`.
Two are content-addressed: `arag` and `structrag` nest that per-task folder
under `inferences/{hash}/`, and `arag`'s index is not in it at all — it sits in
a sibling `_indices/`, keyed by the document set and the embedding/chunker
config (not by the agent's LLM), so runs over the same bundle reuse one build.
R3Con writes its own layout — one folder per task-run under `logs/r3con/`, with
every stage's intermediates inside (see
[evals/r3con/README.md](evals/r3con/README.md)). **Token cost is captured
completely** either way: every completion and every embedding a task makes lands
in one `usage` record per task, in the same `{total, calls}` shape — so a
baseline's cost and R3Con's are comparable field for field.

## Tests

```bash
uv run pytest -q          # 747 tests, all fake-driven (no API calls)
```

Tests that hit a real model are marker-gated and deselected by default
(`pytest -m structrag`, `-m arag`, `-m judge`, …).
