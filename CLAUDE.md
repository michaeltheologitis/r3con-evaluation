# CLAUDE.md

> **This repo.** `r3con-evaluation` was cut from `grounded-reasoning-eval` and pruned to the
> paper's scope: three benchmarks (loong / corpusqa / dracula) and nine baselines. Two
> deviations from the workflow described below, which was written for the source repo:
> there is **no `PROGRESS.md`** here — the trail lives in [STATUS.md](STATUS.md) under
> `## Log` — and the reading path in "Diving into the code" still names baselines this repo
> does not ship (`direct_llm`, `graphrag`, `linearrag`). Read
> [BASELINES.md](BASELINES.md) + [STATUS.md](STATUS.md) first.

## Where to start

For direction (what we're building, which approaches are in/out,
why) read `STATUS.md` first. It is the live source of truth and
is intentionally allowed to diverge from "what's currently in the
code"; when those disagree, `STATUS.md` is the intent and the
code is the lagging snapshot. The `## TODOs` section in
`STATUS.md` is the canonical open-items list — pick from there
before inventing new threads of work.

`PROGRESS.md` is the append-only commit log — read it for the
*trail* of decisions, not for the current direction.

The two documents must not be mixed:

- **Don't park TODOs in PROGRESS entries.** They get lost there
  because nobody re-reads old PROGRESS entries looking for open
  work.
- **Don't put landed-work logs in STATUS.md.** STATUS is rewritten
  in place; using it as a history erases past entries.
- **Don't summarize old PROGRESS into STATUS** to "clean up". The
  point of PROGRESS being append-only is that the trail is intact.
- **Don't bury direction changes in commit messages.** A reader of
  `STATUS.md` should see the current direction without having to
  reconstruct it from `git log`.

STATUS is the intent. PROGRESS is the trail.

## Diving into the code (read this before wiring a new baseline)

If you are new to this repo and about to add or change a baseline harness,
**read the actual Python yourself, top to bottom — do not delegate the
understanding.** Do NOT spawn subagents / fan-out / a Workflow to "explore" or
summarize the code: the whole job of this repo is plumbing baselines onto a
shared contract, and the cost of getting that contract subtly wrong (a dropped
token cost, a mis-saved log, a benchmark fed the wrong components) is far higher
than the tokens you save skimming. **Budget real tokens to read the files into
your own context** and build the mental model first-hand. One agent, reading
carefully, beats a swarm summarizing.

**Reading path (in order — read each file, don't grep-and-guess).** For each,
the thing to extract is in parentheses.

1. `STATUS.md` (current direction + the eight wired baselines) and the tail of
   `PROGRESS.md` (recent decisions).
2. The benchmark surface — read `evals/benchmarks/loong/loader.py` and
   `evals/benchmarks/corpusqa/loader.py` in full, plus `loong/judge.py` and
   `longbenchv2/parse.py` (the public surface every baseline consumes:
   `get_task_ids` / `get_task` / **`get_documents`** / `get_task_answer` /
   `get_task_metadata` / `score`+`score_details` / `SCORER` / `GRADER_MODEL` /
   `PERFECT_SCORE` / `STARTER_FILTER`). `get_documents(task_id)` is the one that
   matters most for a retrieval baseline — it returns the doc SET to index, and
   for Loong/CorpusQA it is a **multi-doc bundle**.
3. The LLM + token-cost layer — `evals/llm/chat.py` and `evals/llm/usage.py`
   (the `{total, calls}` usage shape; `usage_envelope` for a single deterministic
   response vs `usage_scope` for many calls — **this is how you capture cost; get
   it right or the numbers lie**).
4. The shared baseline mechanism — `evals/baselines/_common.py` (hashing, the
   log-store paths, the resumption scan, manifest/error serialization,
   `FAILURE_ERROR_TYPES`, provider-prefix routing).
5. The baselines themselves, as worked examples — read `run.py` + `runner.py`
   of each you'll model on: **`direct_llm`** (the simplest, one call),
   **`arag`** (per-task content-addressed `_indices/` index + LLM-independent
   hash + `calls.json` + a vendored upstream), **`graphrag`** (shared-corpus vs
   per-task index lifecycles), and **`linearrag`** (the *simplest logging* — one
   self-contained folder per run, index inside, total cost, no reuse).
6. Analysis + cleaning — `evals/analysis/{score.py,aggregate.py}`,
   `scripts/_clean_common.py`, and one `scripts/clean_*_logs.py`.
7. If vendoring external code, read a `PROVENANCE.md`
   (`arag`/`structrag`/`linearrag`) for the deviation-ledger discipline — and
   see "Faithful vendoring" in the project memory.

**Three things to understand cold before writing anything:**

- **Logging — save EVERYTHING, especially every token cost.** Per task a
  baseline writes a `manifest.json` (the model's *raw* output + `config` +
  `usage` + a `trace` + an `index_ref` if it indexes) OR an `error.json`;
  grading is deferred to a cached `score.json`; multi-call baselines also dump
  the full request/response of every internal call to `calls.json`. Token cost
  must be attributed completely: index-build embeddings, the query embedding,
  and the reader/agent completions — none dropped, none double-counted. There
  are **two layouts**: the content-addressed one (`inferences/{hash}/` +
  shared `_indices/{hash}/`, reused across configs/modes — graphrag/arag/
  structrag/direct) and `linearrag`'s **one-folder-per-run, index-inside,
  TOTAL-cost, no-reuse, no-resumption** layout. Pick the one that fits; the
  analysis CLI reads both.
- **Cleaning.** Every baseline gets a `scripts/clean_<name>_logs.py` matching
  its layout, reusing `_clean_common.classify_inference` (crash / broken /
  incomplete / transient / real_error, keyed on the shared `FAILURE_ERROR_TYPES`
  whitelist). A new baseline isn't done until it has a cleaner + tests.
- **Loong and CorpusQA are the important benchmarks — wire to them first.** Both
  are **per-task multi-doc**: `get_documents` returns the instance's bundle, so
  you pool/index the whole bundle and pose the query as the task (Loong:
  `instruction[+question]`; CorpusQA: `question + output-requirements` — the
  output-requirements block MUST reach the model). Loong is judge-scored
  **1–100** (Avg Score + Perfect Rate) and is **bilingual (EN + 905 ZH)** — if
  your method is language-sensitive (e.g. NER), route per the `language`
  metadata. CorpusQA is judge-scored **0/1**, un-bakes a frozen prompt, and its
  tiers are huge (128k is downloaded; 1m/4m are multi-GB — use child mode or
  scope a tier for local tests). Both are "leave no document behind", so
  retrieval baselines are expected to be **foils** vs `direct-llm` — that gap is
  the measurement, not a bug.

**Adding a new baseline — the contract.** A baseline package
`evals/baselines/<name>/` exposes `SUPPORTED_BENCHMARKS: frozenset[str]`,
`run_one(benchmark, task_id, run_config, base_dir_or_run_dir, litellm_kwargs)
-> {raw_answer, usage, trace, index_ref?, calls_full?}` (purely the model's
output — never grade in `run_one`), a `runner.py` (its own argparse + parent
fan-out + a `--task-id` child mode for crash isolation), and `__main__.py`.
Route the LLM through a litellm seam (capture usage + `calls.json`); keep
**embeddings always OpenAI**; if connecting external code, **vendor it
byte-for-byte under `upstream/` and wire, don't improve** (a `DEVIATIONS` header
per connector + a `PROVENANCE.md` ledger). Then: add the `evals[<name>]` extra,
fake-driven tests + a cleaner, and **wire-test live on Loong EN + ZH + CorpusQA
with `gpt-5.4-nano`, then open the actual log files and confirm every token cost
and field is saved** before declaring it done.

## TODOs workflow

All open work items live in `STATUS.md` under `## TODOs`,
organized into three subsections:

- **Open work items** — actively on the table; pick one, work it,
  log results.
- **Code improvements (clean-as-you-go)** — loose code noticed
  while working other items. See "Code hygiene as you go" below
  for the full discipline.
- **Parked (don't start yet)** — gated structural moves. The
  **Gate:** is named on the bullet so a future agent knows when
  to unpark.

This is the single place to look for what's in flight. **Don't
park items in PROGRESS retrospectives, in commit messages, or
scattered through other doc sections** — those have other
purposes and items get lost when mixed in.

**Adding a TODO.** `- [ ]` bullet with a one-line description
**grounded in what you actually observed** (not what you might
want). Don't add items for hypothetical future requirements;
don't add items so vague they need re-investigation to act on.

**Closing a TODO.** When the work lands, **delete the bullet
from `STATUS.md`** in the same commit (or a follow-up doc commit).
Don't check the box and leave it — `PROGRESS.md` captures the
landed change with its commit SHA, so `STATUS.md` doesn't need to
carry TODO history. If a TODO turns out to be wrong/obsolete
without work landing, delete it the same way and note the
reasoning in the deletion commit's message.

**Characterizations vs TODOs.** If your `STATUS.md` has a section
like `## Known patterns` or `## Failure-mode taxonomy` — pure
*characterization* of named patterns the project has observed —
don't put TODOs there. If observing a pattern prompts a concrete
action, write the characterization in that section *and* the
action in `## TODOs`. They serve different readers.

## Versioning artifacts on disk

When the project has artifacts that get iterated on —
configuration files, templates, schemas, fixtures, parameter
sets, prompts, anything whose content changes meaningfully over
time and whose previous versions are still useful to compare
against — keep them in an **explicitly versioned layout on disk**,
side by side.

```
<artifact-kind>/
  v1.<ext>
  v2.<ext>
  v3.<ext>
  ...
```

The active version is chosen at runtime by a single piece of
central config (an env var, a settings module, a config file) —
*not* by symlink or filename convention. A reader finds the
active default in one well-known place. Per-invocation overrides
(CLI flag, env var, argument) let an experiment use a non-default
version without flipping anyone else's default.

**Why this beats overwriting in place + relying on git:**

- A/B comparing two versions requires no git surgery — both files
  exist simultaneously, addressable by name.
- Reverting is a one-line config change, not a `git revert` dance.
- Multiple versions can be evaluated against the same downstream
  state in a single batch.

**Workflow for a new version:**

1. `cp <artifact>/v7.<ext> <artifact>/v8.<ext>`.
2. Edit `v8.<ext>` only — leave older files alone, they are now
   historical record.
3. Evaluate `v8` against the same input set the active version was
   measured on. Reuse cached downstream state so the comparison
   is cheap.
4. If `v8` clears the relevant bar — flip the active default in
   the central config, update any doc that names the active
   version, commit. If `v8` regresses, **leave the file on disk
   anyway** as a recorded A/B data point — don't delete it.

**Simpler is better when scores tie.** A new version that is
statistically indistinguishable from the active version but
**meaningfully smaller / simpler / clearer** is a real win — ship
it. Equal signal at lower complexity is not a tie.

**Cache against the full version-chain.** If your evaluation
depends on upstream artifacts (other versioned files, cached
intermediates), make the cache key a function of the **entire
chain of versions**, not just the artifact you're iterating on.
A hash of the upstream chain in the cache directory name works
well — re-running the same `(version, upstream-chain, input)`
becomes a no-op, and a changed upstream produces a fresh
directory rather than overwriting.

**Don't over-prescribe inside the artifact.** When the versioned
artifact is *guidance* (a prompt, a runbook, a checklist):

- Prefer principle-shaped guidance ("for a short list you can
  usually print all of it; for a long one, sample") over
  hardcoded thresholds ("if list ≤ 20, print all").
- Don't predefine the shape of the solution step-by-step ("step 1
  do X, step 2 do Y" — let the actor take 2 or 20 steps as the
  input warrants).
- Don't describe what to do in code-shaped terms that will get
  literal-copied onto unrelated inputs. Show reasoning in prose;
  let any code in the artifact be a microscope on the reasoning,
  not a literal template.

## Measurements: noise, cadence, and the deep-dive

This is the single most load-bearing discipline in the workflow,
for any project that has a measurable signal — a benchmark score,
a test pass-rate, a latency number, an error count, a user metric.

### Treat small deltas as noise

Any measurement has run-to-run variance. Before celebrating a
delta as a "win", know your project's **noise band** — the size
of run-to-run fluctuation when nothing has changed.

- A delta **inside the noise band** is "within noise"; don't ship
  structural changes based on it.
- A delta **outside the noise band on one run** is a candidate
  signal, not a confirmed one.
- A delta **outside the noise band that holds on a second
  independent re-measurement** (different seed, different sample,
  different partition — whatever your project's notion of
  "independent re-measurement" is) is a real win.

The bar for "shippable" can be lower than "confirmed" if the
change has independent value (smaller / simpler / clearer code —
see "simpler is better when scores tie" above). The bar for
"declarable milestone" is the higher one.

### Iteration cadence

The per-iteration measurement should be **cheap** — runnable in
minutes against a stable held-out set, against the same upstream
state used by the current active version. Cache aggressively so
re-running an unchanged `(version, upstream-chain, input)` is a
no-op; this makes A/Bs of past versions effectively free reads.

The expensive "confirmation" measurement (different seed,
different sample, more inputs — whatever costs real
wall-clock / money / external API calls) is **not** a per-ship
check. Run it at natural phase boundaries:

- A whole component has converged (several versions explored, a
  new active version has settled).
- Cross-component compounding — multiple components changed and
  you want to confirm the joint stack holds together.
- Before declaring a milestone (a major PR, a writeup, a
  checkpoint someone will cite later).
- After a long stretch without one — rough rule: every 4–6 ships,
  or one calendar week of active iteration.

In between, rely on the cheap per-iteration measurement and the
noise-band reasoning above.

### After every meaningful measurement: deep-dive what is going on

**This is the most important discipline in the project.** A score
is a weak signal — it tells you ±N units moved, not *why*. The
strong signal — the one that grounds every subsequent iteration
in reality — is the **deep-dive write-up**: read the artifacts of
each failed case, read the actual outputs, compare across
baselines / modes when relevant, and write down what you saw.
Without this you don't know which component to iterate on next —
you'll pick the wrong one and waste the next iteration.

A deep-dive write-up is **not** a one-line "score moved from X to
Y." It contains:

**1. A per-failure attribution table.** For every failure: which
component is responsible, and a one-line root cause grounded in
the artifacts.

| failure  | component         | root cause |
| -------- | ----------------- | ---------- |
| `<id>`   | <component name>  | one-line description grounded in the artifacts |

The point of the table is **structural visibility**: when ten of
twelve failures name the same component, you see something you
would not see reading them one at a time.

**2. Cross-mode / cross-baseline comparison when something is
suspicious.** If you have multiple modes of running the system —
a full pipeline, a simplified pipeline, a baseline that bypasses
your code entirely — and the result is surprising (a regression,
a stagnation, an unexpected gain), run the same input set through
each mode and compare. Patterns:

- **Simpler beats more complex** → the complex path is destroying
  information the simpler path had access to.
- **All modes look identical** → the signal is dominated by
  something outside the axes you've varied; you're iterating on
  the wrong one.

Cross-mode comparison is **the move that catches false
attributions** based on single-mode reasoning. Use it whenever
the single-mode picture is suspicious.

**3. Manual review of the failing inputs.** Read the actual
inputs and intermediate state for the failures. Answer the
question yourself, by hand, from the same material the system
had. If you can answer it from the material but the system
couldn't, the failure is **downstream** of that material. If you
can't answer it from the material, the failure is **upstream** —
and you've now confirmed which upstream component by inspecting
what's missing.

This step is the one people skip because it's slow. **It is the
most informative step.** Metrics can't replace it.

**4. Read the system's own reasoning if it produces any.** If the
system emits a trace, log, or explanation of its own behavior,
read it for the failing cases. You will sometimes find the
system's own output announcing one thing while it actually does
another — invisible in the score, visible in the trace.

**5. A short narrative conclusion.** A few paragraphs at the end:
what pattern do these failures share, what's the highest-leverage
next move, what was ruled out. This is what the next person reads
first.

The deep-dive belongs in the `PROGRESS.md` entry for that
measurement. The headline score can be noted alongside, but the
deep-dive is the load-bearing artifact.

**Why this matters:**

- It catches **stage drift** — "the last three measurements all
  blamed the same component" is invisible in raw scores.
- It prevents **working on the wrong thing** — iterating on
  component A when the bottleneck is component B is the most
  common waste-of-time failure mode; the deep-dive makes it hard
  to make.
- It surfaces **structural patterns** — three failures of the
  same shape is a structural fix; one is a curiosity. Only
  visible by looking at all failures together.
- It catches **false attributions** — particularly via cross-mode
  comparison.

## Code hygiene as you go

When you're working a TODO or any other focused item and you
naturally need to read or edit a file, **if you see obviously
loose code in that file**, log it under `## TODOs` →
`### Code improvements (clean-as-you-go)` in `STATUS.md` so the
project might clean it up later. That subsection is the canonical
landing pad — don't park such items in PROGRESS retrospectives or
scatter them through other doc sections.

The bar is "you'd flag this in code review without thinking" —
not "this could theoretically be cleaner." Good candidates: dead
parameters all callers pass dummy values for, silently-swallowed
exceptions, obvious 10-line duplicated blocks, names that lie,
dead code paths reachable only by config flags no one uses. Bad
candidates: vague restructuring ideas, style preferences,
"we should use library X here."

## Selective unit tests

Don't run the full test suite for every change. Run only the test
file(s) for the modules you touched. Full suite is appropriate
for:

- **Shared infrastructure changes** — files many other modules
  depend on (core runtime helpers, shared config, base classes,
  data-access layer — varies by project).
- **Before a multi-commit chunk lands** — when several related
  commits have stacked up and you want one confirmation that
  nothing slipped across the seam.
- **When you suspect a hidden cross-cutting impact** — surfaced
  by a file having non-obvious dependents (reflection, plugin
  systems, dynamic dispatch, generated code).

Outside those cases, targeted-tests-only is the default. The
question is **which** tests, not **whether** — every change
should be covered by some test run before it lands.

## Iteration practices

**Don't block on long-running tasks.** If a measurement, build,
or batch job takes more than a couple of minutes, kick it off in
the background and switch to safe parallel work — reading code,
writing docs, sketching the next iteration. Safe parallel work
doesn't modify any input the running job depends on, doesn't
compete for the same resource, and doesn't gate on the long task.
Editing the artifact a measurement is currently consuming
**taints the run** — wait for it to finish, or kill it
explicitly if the edit is more valuable than the in-flight
measurement.

**Run independent work in parallel.** When you have multiple
independent units of work — independent inputs, independent
measurements, no shared mutable state — run them in parallel.
Outer parallelism across independent units, inner sequentiality
preserved within each unit where state legitimately threads
forward. Watch for hidden shared state (a global config that
gets mutated, a cache shared across workers, a logger writing to
a single file) — it breaks the independence assumption and
produces results that look correct but aren't reproducible.

**Steer with examples, not rules.** When a piece of guidance —
a prompt, a checklist, a runbook, even a code comment — isn't
producing the behavior you want, the instinct is to add a rule
("don't do X", "always do Y"). Resist this. Rules tend to overfit
the case that motivated them, get ignored when they conflict
with another rule, and bake judgment into the wrong layer. **Add
good examples instead.** A well-chosen example shows the *shape*
of the behavior; the actor can pattern-match for cases the
example doesn't literally cover, in a way they can't with a rule.

**Match commit shape to work shape.** A commit is a unit of
*change*; a PROGRESS entry is a unit of *understanding*. They're
often 1-to-1 but they're not the same thing — a measurement run
that produced findings warrants a PROGRESS entry containing the
deep-dive, even if no code changed.

**Don't edit prior PROGRESS entries.** If later understanding
contradicts an earlier entry, write a *new* entry that supersedes
the old one. The append-only property is what lets a future
reader reconstruct the actual trajectory of the project's
thinking — including the wrong turns. Erasing wrong turns erases
the lesson. Small factual fixes (a wrong SHA, a typo) are fine;
conceptual reversals stay and get a new entry explaining the
reversal.

## PROGRESS.md log

This repo keeps an append-only `PROGRESS.md` at the root logging
substantial changes — anchored to the commit that landed them.

**When to update.** After landing a non-trivial commit — a new
feature, a real refactor, a bug fix that required actual
reasoning, a meaningful change to configuration / schemas /
interfaces. **Skip** trivial commits: typos, whitespace,
single-line config tweaks, lockfile churn, scratch / notebook
output.

**Workflow.**

1. Make and land the work commit first.
2. Update `PROGRESS.md` with an entry referencing that commit's
   short SHA.
3. Commit the `PROGRESS.md` update on its own (commit subject:
   `progress: <short headline>`).

Create `PROGRESS.md` if it does not yet exist. Append at the
bottom; **never reorder or rewrite past entries**.

**Entry format:**

```markdown
## YYYY-MM-DD — <short headline>

- **Commit:** `<short SHA>` — <commit subject line>
- **What changed:** one or two sentences on what was
  added/changed.
- **Why:** the motivation, if it is not already obvious from the
  commit message.
```

If a single chunk of work spans multiple commits, write one entry
per commit unless they are tightly coupled — in which case a
single entry may list multiple commits under **Commit:**.

For measurement-driven entries, the body is the **deep-dive
write-up** (see above), not just the score.
