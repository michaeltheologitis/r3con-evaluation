# Status

## What this repo is

The evaluation repo for the ICLR submission: **three benchmarks**, **nine
published baselines**, one shared contract, and the logging/scoring machinery
that makes the numbers comparable across methods. It was cut from an internal harness and pruned to exactly what the paper
reports.

Conceptual reference for the methods: [BASELINES.md](BASELINES.md). How to run
anything: [README.md](README.md).

## Where things stand (2026-09-20)

**Landed — the repo runs.** `uv run pytest -q` → **557 passed**, 32 deselected
(the marker-gated live tests). Every baseline imports, its runner parses, and its
fake-driven connector tests drive the real vendored pipeline.

**What's in:**

| | |
| --- | --- |
| benchmarks | `loong`, `corpusqa`, `dracula` |
| baselines (representation) | `hipporag`, `structrag`, `raptor`, `readagent`, `memagent` |
| baselines (coding / tool-calling agents) | `claude_code`, `codeact`, `rlm`, `arag` |
| harness | `evals/llm` (LiteLLM seam + token accounting), `evals/analysis` (scoring → `score.json`, results CLI), `evals/baselines/_common.py` (hashing, paths, resumption, manifests) |
| ops | per-baseline log cleaners + `compact_rlm_logs.py` under `scripts/` |
| licensing | MIT ([LICENSE](LICENSE)), copyright held as *Anonymous Authors* for the blind copy; vendored upstreams inventoried in [THIRD_PARTY.md](THIRD_PARTY.md) |

**What was deliberately left out, and why:**

- **LinearRAG** — **GPL-3.0**. Vendoring it would impose GPL-3.0 on this whole
  distribution, which is unacceptable for a public artifact. Not shipped, not
  wired, not depended on. Noted in BASELINES.md § "Not included in this repo".
- **`direct-llm`, `graphrag`, `amem`** — not part of the paper's baseline set.
- **`longbenchv2`, `loogle`, `casefacts`, `minteval`, `longhealth`** — the five
  other benchmarks of the source harness. Dropped everywhere: import map,
  `SUPPORTED_BENCHMARKS`, task-assembly branches, chunk-size tables, runner
  flags that existed only for them (readagent's `--task`), tests, extras,
  package-data, pytest markers.
- **Logs** — nothing copied. `logs/` exists and is gitignored. The source repo
  holds ~39k finished loong/corpusqa runs; whether to carry any across is an
  open decision (see below). The log layout here is unchanged, so they stay
  readable if carried.

## TODOs

### Open work items

- [ ] **The method is not wired yet.** It plugs in like any other baseline: a
  package under `evals/` exposing `SUPPORTED_BENCHMARKS`, `run_one(benchmark,
  task_id, run_config, run_dir, litellm_kwargs) -> {raw_answer, usage, trace,
  …}`, its own `runner.py` (argparse + parent fan-out + a `--task-id` child
  mode) and `__main__.py`, plus a log cleaner and fake-driven tests. Nothing
  about it is assumed here — naming, layout and packaging are open.
- [ ] **Decide what happens to the existing runs.** ~39k loong/corpusqa
  manifests live in the source repo (structrag 6.4k, codeact 4.8k, claude-code
  3.8k, arag 3.2k, rlm 1.6k, …). They were produced by this exact code and log
  layout, so they are reusable — but nothing has been copied.
- [ ] **Put the real copyright holder in LICENSE at camera-ready.** It currently
  reads *Anonymous Authors* so the blind copy carries no name. (StructRAG's and
  ReadAgent's upstreams state no license of their own; both are vendored anyway —
  noted in [THIRD_PARTY.md](THIRD_PARTY.md), not treated as a blocker.)
- [ ] **Keep the public copy anonymous.** The tracked tree is clean: the copyright
  line is *Anonymous Authors*, and there are no personal paths, emails or account URLs
  anywhere (every `github.com/...` reference is an upstream project). Agent/working
  files (`CLAUDE.md`, `.claude/`, …) and `.env` are gitignored, so they stay local.
  Commit metadata is not a concern — the anonymous copy is a one-shot clone that does
  not carry history. Re-run the sweep before submitting:
  `git grep -niE "<name>|<email>|/Users/|/home/" -- . ':!evals/benchmarks/dracula/raw/*'`

### Code improvements (clean-as-you-go)

- [ ] **Prose pass over the carried-over docs.** The per-method `*_info.txt`
  files and some `PROVENANCE.md` sections still cross-reference baselines and
  benchmarks this repo does not ship (e.g. "like direct-llm", "unlike
  linearrag"). The functional claims (scope lines, `SUPPORTED_BENCHMARKS`) were
  corrected; the narrative cross-references were not.
- [ ] **CLAUDE.md still describes the source repo's workflow**, including a
  `PROGRESS.md` that does not exist here — this repo keeps its log in STATUS.md
  instead. Its reading path also names dropped baselines.

## Log

Newest first. Keep entries short; this is the trail, not a design doc.

### 2026-09-20 — MIT license + third-party inventory

Added [LICENSE](LICENSE) (MIT, scoped to this repo's own code) and
[THIRD_PARTY.md](THIRD_PARTY.md), which inventories every vendored upstream with
its pinned commit and license. Verifying those licenses turned up a real problem:
**StructRAG and ReadAgent state none at all**, which is now the open item above.
HippoRAG and RAPTOR ship LICENSE files (both vendored); A-RAG declares MIT in its
README but ships no LICENSE file; MemAgent is Apache-2.0.

### 2026-09-20 — repo created

Cut from `grounded-reasoning-eval@55611e48` and pruned to the paper's scope.
Copied: the three benchmarks, the nine baselines (with their vendored
`upstream/` trees and PROVENANCE ledgers), `evals/llm`, `evals/analysis`,
`_common.py`, the matching cleaners and tests. Then: benchmark import map cut to
three; every baseline's `SUPPORTED_BENCHMARKS` and task-assembly branches pruned
to `{loong, corpusqa, dracula}`; readagent's LongHealth-only `--task` flag
removed; hipporag's per-benchmark chunk table and readagent's regime/run-version
tables pruned; `pyproject.toml` extras cut to the shipped baselines (markers and
package-data with them); the `--tasks` variant filter (hipporag, raptor) kept —
it is generic, and CorpusQA ids carry an `@`-suffix — with its help text
re-worded off LongHealth. Tests that exercised dropped features were deleted;
tests that merely used a dropped name as a fixture label were renamed. Suite
green at 557.
