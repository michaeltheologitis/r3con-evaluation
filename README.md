# r3con-evaluation

Three benchmarks, nine baselines, and R3Con — the method under evaluation.

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
uv sync --all-extras
```

Put `OPENAI_API_KEY` in `.env` — the judges and every embedding call go to OpenAI.

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

## Paper experiments

Every run below uses `Qwen/Qwen3.5-35B-A3B` on the served endpoint, with the model's own
default sampling. `$KEY` is the API key passed to `vllm serve`.

```bash
# ============ serve the model ============
vllm serve Qwen/Qwen3.5-35B-A3B \
  --port 8555 --api-key "$KEY" \
  --trust-remote-code --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder


# ============ fetch the data (once) ============
python -m evals.benchmarks.loong.download_docs
python -m evals.benchmarks.corpusqa.download_data --set 1m


# ============ baselines ============
# Omitting --limit runs the whole benchmark: 1,600 Loong tasks, 329 CorpusQA.
# Re-running resumes, so any of these can be interrupted and restarted.

# ReadAgent on Loong
python -m evals.baselines.readagent --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# ReadAgent on CorpusQA
python -m evals.baselines.readagent --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# StructRAG on Loong
python -m evals.baselines.structrag --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# StructRAG on CorpusQA
python -m evals.baselines.structrag --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# A-RAG on Loong   (needs the tool-call parser, which the serve command above enables)
python -m evals.baselines.arag --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# A-RAG on CorpusQA
python -m evals.baselines.arag --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# CodeAct on Loong
python -m evals.baselines.codeact --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# CodeAct on CorpusQA
python -m evals.baselines.codeact --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# RLM on Loong
python -m evals.baselines.rlm --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# RLM on CorpusQA
python -m evals.baselines.rlm --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# RAPTOR on Loong   (call-heavy: a reasoning summary per cluster. Add --limit N for a subset)
python -m evals.baselines.raptor --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# RAPTOR on CorpusQA
python -m evals.baselines.raptor --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# HippoRAG on Loong   (call-heavy: ~2 calls per passage, no reuse. Add --limit N for a subset)
python -m evals.baselines.hipporag --benchmark loong \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"

# HippoRAG on CorpusQA
python -m evals.baselines.hipporag --benchmark corpusqa \
  --model Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY"


# ============ the method ============
# R3Con on Loong   (takes the provider-prefixed model id, unlike the baselines)
python scripts/r3con/loong/run.py --all --inference both \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY" --workers 20

# R3Con on CorpusQA
python scripts/r3con/corpusqa/run.py --inference both \
  --model hosted_vllm/Qwen/Qwen3.5-35B-A3B \
  --base-url http://localhost:8555/v1 --api-key "$KEY" --workers 20


# ============ the two that do not use the served model ============
# MemAgent is an RL-trained checkpoint, not a prompting method, so it needs its own
# server. Its 1,024-token cap is the method, which means a reasoning model spends the
# budget on thinking and returns an empty memory.
vllm serve BytedTsinghua-SIA/RL-MemoryAgent-14B --port 8556 --api-key "$KEY"

# MemAgent on Loong
python -m evals.baselines.memagent --benchmark loong \
  --base-url http://localhost:8556/v1 --api-key "$KEY"

# MemAgent on CorpusQA
python -m evals.baselines.memagent --benchmark corpusqa \
  --base-url http://localhost:8556/v1 --api-key "$KEY"

# Claude Code runs a Claude model through the `claude` CLI on a Max login, so it is a
# reference point rather than a same-model comparison. It takes no endpoint flags.

# Claude Code on Loong
python -m evals.baselines.claude_code --benchmark loong

# Claude Code on CorpusQA
python -m evals.baselines.claude_code --benchmark corpusqa
```
