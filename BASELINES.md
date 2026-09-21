# Baselines

What each baseline **is** — conceptually (the published method or idea) and in
practice (how it's wired onto this harness, its seams, its key deviations, its
expected behavior on the benchmarks). This is the document to read when you want
to understand a method end-to-end before touching its code.

**How this doc relates to the others.** It is a *stable conceptual reference*,
not a status board. Volatile specifics live elsewhere and are linked, not
restated:

- **Per-baseline deviation ledgers** → each `evals/baselines/<name>/PROVENANCE.md`
  (`NOTES.md` for `claude_code`). When this doc says "vendored byte-for-byte" or
  "the one deviation is…", the authoritative list is there.

If this file and the code disagree, **the code wins** — this is prose *about* the
code, not a spec the code owes anything to; and the authority on any single
baseline's deviation from its upstream is the `PROVENANCE.md` sitting next to it.
Score tables are kept **out** of here so it can't drift.

---

## The spine: the contract every baseline plugs into

The whole job of this repo is plumbing baselines onto one small shared contract
so benchmarks and methods compose without re-plumbing. Understand the spine
first; every baseline below is a variation on it.

- **Benchmark surface** (`evals/benchmarks/<name>/`). A benchmark exposes
  `get_task_ids(**filters)` → `get_task(task_id)` → `get_documents(task_id)` →
  `get_task_answer` / `get_task_metadata` → `score` / `score_batch` /
  `score_details` (+ `SCORER` / `GRADER_MODEL`, and `PERFECT_SCORE` for the
  1–100 judges). The crucial discipline: **`get_task` returns named components**
  (instruction, question, docs — Dracula has no instruction, so it returns the
  other two) — *never* an assembled prompt. The benchmark owns the components;
  the baseline owns *composition*.
  `get_documents(task_id)` is the one that matters most for a retrieval baseline:
  it returns the document **set** to index, and for Loong/CorpusQA it is a
  per-instance multi-doc bundle.

- **The per-task entrypoint.**
  `run_one(benchmark, task_id, run_config, base_dir_or_run_dir, litellm_kwargs)`
  returns `{raw_answer, usage, trace, index_ref?, calls_full?}`. The single most
  load-bearing invariant: the record is **purely the model's output** —
  `run_one` **never grades** — grading happens outside this repo. (See [evals/baselines/_common.py](evals/baselines/_common.py)
  for the contract and serialization.)

- **The runner.** Each baseline owns a self-contained runner
  (`python -m evals.baselines.<name>`): its own argparse, parent fan-out, and a
  `--task-id` child mode. The parent runs **one subprocess per task** for crash
  isolation; each child writes **exactly one** of `manifest.json` /
  `error.json`. A deterministic failure (context-window-exceeded) is *recorded,
  not retried*; `--limit N` runs the next N pending; re-running resumes. There is
  no central god-runner — the runners share only the stateless mechanism in
  `_common`.

- **Two log layouts.** (1) *Content-addressed, decoupled* (structrag,
  arag): `inferences/{hash}/` for the model output, plus — for the one baseline
  that builds a reusable index (arag) — a shared `_indices/{hash}/` index store
  reused across runs. (2) *Flat one-folder-per-run, index-inside, **TOTAL** cost,
  no-reuse* (readagent, raptor, rlm, codeact, claude-code, hipporag, memagent). Both are
  self-describing on disk under `logs/`, so whatever grades them later reads either.

- **Cost is captured completely, never dropped.** Every token — index-build
  embeddings, query embeddings, reader/agent completions, recursive sub-calls —
  is attributed. The capture *path* differs per baseline (single-response
  `usage_envelope`, a seam-level accumulator, a monkeypatch, a model subclass,
  the CLI's own usage object, with a `usage_scope` callback as the runner's
  fallback) — see the cross-cutting table at the end — but the **shape is always**
  `{total, calls}` (see [evals/llm/usage.py](evals/llm/usage.py)). Index-build
  cost lives with the index and travels on reuse, recorded once per index.

- **Grading is somebody else's job.** Each benchmark owns its judge and exposes
  `score` / `score_details`, but nothing here calls them: a run records the model's
  raw answer and its cost, and scoring, tables and figures happen in a separate,
  unified repo that reads these logs.

---

## What they're measured against (and why Loong is the sharp test)

The benchmarks all share one shape: **reasoning that must be grounded in a
supplied input**, not produced from priors. This repo ships three, all
per-instance multi-doc: **Loong**, its computation-heavy cousin **CorpusQA**, and
**Dracula** (the hand-curated running example).

**Loong** (EMNLP 2024, Alibaba; arXiv 2406.17419) is the load-bearing benchmark
for understanding baseline behavior, so it's worth a paragraph of its own. Its
defining principle is **"leave no document behind"**: each instance's evidence is
*scattered across a per-instance bundle of relevant documents*, and ignoring any
one of them makes the answer wrong. That single property makes Loong
**structurally hostile to retrieval** — any method that retrieves a *subset*
drops documents and fails — so RAG baselines are expected **foils** here, and the
gap versus a method that reads everything is the *measurement*, not a bug. It is
1,600 instances: EN 695 + ZH 905 (`get_task_ids(languages=…)` selects), across
three domains (`paper` EN-only, `financial` EN+ZH, `legal` ZH-only), four task
types of rising difficulty (`level` 1 Spotlight-Locating → 4 Chain-of-Reasoning),
and four length buckets (`set` 1 ~10–50K → 4 ~200–250K tokens, dialed by adding
*more documents*, not noise). Some instances have an **empty `question`** (the
whole task is the `instruction`, e.g. paper Chain-of-Reasoning), which every
baseline's task-assembly handles. Answers are sometimes JSON (e.g. the citation
task's `{"Reference": [...], "Citation": [...]}`), not strings. Scoring is
**judge-only, 1–100** (`gpt-5.4-mini`, structured-output `Rating`); the two
metrics are **Avg Score** (mean) and **Perfect Rate** (fraction == 100) — and
because Loong's paper *defines EM as the judge-perfect rate*, our Perfect Rate
**is** the paper's EM and Avg Score **is** the paper's LLM Score. The harness also
fixes one real upstream bug — the level-4 legal "match each judgment to its
verdict" task leaks the verdict into the input because upstream checks the wrong
field (`instruction` instead of `question`); StructRAG vendored that code
unchanged, so its published Loong numbers inherit the 100% leak. Full details:
[evals/benchmarks/loong/loader.py](evals/benchmarks/loong/loader.py),
[loong/judge.py](evals/benchmarks/loong/judge.py),
[loong/CHANGES.md](evals/benchmarks/loong/CHANGES.md).

Read each baseline below with one question in mind: **does it see all the
documents, or does it drop some?** That, more than any architectural detail,
predicts how it does on Loong.

---

## Not included in this repo

Three baselines that exist in the harness this repo was cut from are deliberately
absent here, and one of them for a licensing reason worth stating plainly:

- **LinearRAG** (ICLR 2026, relation-free graph RAG) — **excluded because it is
  GPL-3.0.** Every other vendored baseline here is MIT or Apache-2.0, and vendoring
  GPL-3.0 source into this repo would impose GPL-3.0 on the whole distribution. It
  is not shipped, wired, or depended on in any form.
- **GraphRAG** and **A-MEM** — out of scope for this evaluation; dropped rather
  than carried along untested.

---

## The baselines

**Emoji legend (facet tags — a method can carry more than one):**
🔍 retrieves a subset · 🧩 restructures the documents · 📖 reads the prose ·
🐍 acts by writing code · 🤖 runs an autonomous agent loop.

The mechanism tag (🔍 / 🧩 / 📖 / 🐍) says *how it handles the documents*; 🤖 is the
orthogonal *agentic* tag (does it run its own decide-then-act loop?). So A-RAG
(🔍🤖) is a *retrieval* agent, the code trio (🐍🤖) are *code* agents, and StructRAG
/ ReadAgent are *non-agentic* distill-then-reason methods. Listed roughly in order
of increasing machinery — from reading the text as-is, to distilling it, to
retrieving / restructuring it, to turning the model loose as an agent.

### 📖 `readagent`

[evals/baselines/readagent/](evals/baselines/readagent/) — ReadAgent (Google
DeepMind, ICML 2024; arXiv 2402.09727). **LLM-only** — no embeddings, no
retriever, no code. It reads the way a person skims a long book: (1) **paginate**
the document(s) into pages at LLM-chosen natural break points; (2) **gist** each
page into a compressed natural-language memory (one call per page); (3) for the
question, do a **ReadAgent-P look-up** — one batched call names the 1–2 pages
worth re-reading in full — then swap those gists for their full text and answer.
Only **ReadAgent-P** is wired, because it is the only look-up variant upstream
actually implements in code (ReadAgent-S is a prompt template with no
implementation), so there is no variant flag. Despite "Agent" in the name it is a
**fixed pipeline** with a couple of bounded LLM choices, not an open-ended
autonomous loop — hence 📖, not 🤖. For a task's document bundle it paginates each
document independently and pools the pages in document order (a page never spans
two docs), with a CJK-aware word count so Chinese paginates correctly. Because the
answer is built from gists plus a tiny full-text window, it is expected to be a
**foil** on leave-no-document-behind tasks. It is the slowest baseline, and its
cost profile is well understood: a deep-dive found **pagination is ~70% of
wall-clock** (an un-parallelizable serial chain within each document) while
gisting is ~26% (it parallelizes), so the `--gist-workers` knob (default 8) only
speeds the smaller share. It needs **no extra** (pure stdlib + core litellm, like
the harness core), and writes a live `progress.json` so a long mid-flight task is
observable.

### 🔍 `raptor`

[evals/baselines/raptor/](evals/baselines/raptor/) — RAPTOR (Sarthi et al., ICLR
2024; arXiv 2401.18059), vendored byte-for-byte (MIT, zero-edit — all its imports
are already relative). It is an **LLM-built** retrieval index, but
the structure is a **recursive summary tree** instead of an entity graph: chunk
the documents into ~100-token leaves, embed them, then **recursively cluster
(UMAP + GMM) and LLM-summarize** each cluster into the next tree level — so leaves
are raw chunks and higher levels are progressively more abstract summaries. At
query time it "collapses" the tree (all levels → one pool) and retrieves the
top-k nodes up to a token budget, then answers over that mix of raw chunks and
summaries. Posed through RAPTOR's own front door
(`RetrievalAugmentation.add_documents` → `answer_question`), with the method
unchanged — we only subclass its `BaseSummarization/QA/EmbeddingModel` ABCs to
inject the litellm + OpenAI-embedding seams (RAPTOR ships only raw-openai clients
with **no vLLM path**, so the litellm seam *is* the vLLM connection); the
summary/QA prompts are kept verbatim and its hardcoded `RANDOM_SEED=224`
clustering is left untouched. Simple flat per-run-folder layout (the tree built
fresh per run, **no reuse**, trading index reuse for one self-contained folder;
TOTAL cost = every cluster summary + every embedding + the QA answer, in one
number). A 🔍 retrieval **foil** on Loong (top-k node subset), but expected to be
a *less severe* one than plain RAG — its higher tree levels summarize across
clusters, preserving global context that a purely local retrieval
drops. Supports **loong / corpusqa / dracula**. **Runs THINKING** (D10): the summary's
~100-token length control moves from the `max_tokens` cap (which a reasoning model
spends on `reasoning_content` → empty summaries) to a prompt hint + a 32768 budget,
with an empty-summary retry — so it's comparable on the same thinking model as every
other baseline, but **very expensive** (each cluster summary pays full reasoning), so
run a stratified subset. Behind the heavy `evals[raptor]` extra
(faiss/umap/sentence-transformers, vendored).

### 🧩 `structrag`

[evals/baselines/structrag/](evals/baselines/structrag/) — StructRAG (ICLR 2025),
vendored byte-for-byte. Not a retriever at all: it **restructures the documents**
at inference time. Three stages: a **router** picks one knowledge-structure type
for the task (table / graph / algorithm / catalogue / chunk) from the query plus
the document titles; a **structurizer** rebuilds the documents into that chosen
structure; a **utilizer** decomposes the question into subqueries, extracts
per-subquestion knowledge from the structured form, and merges the final answer.
(Note the **chunk** option is the *free-form fallback* — when the router decides
the question doesn't benefit from a rigid structure, it keeps the raw text, i.e.
plain RAG-style chunks.) That is many sequential LLM calls and **no persistent
index** (each task's KB round-trips through an ephemeral temp dir). Because it
restructures *every* document, it only fits a benchmark whose per-question document
set is small enough to rebuild: Loong and CorpusQA are per-instance bundles, and
Dracula's shared corpus is only 46 documents; a large shared corpus would have to be
restructured per query, which is infeasible. Context-window handling is faithful to upstream's
*behavior* but a better implementation: oversized prompts are truncated-to-fit
and answered **reactively only** (clip on a server length rejection, never via
upstream's proactive gpt2/128K clip). Every internal call's full request/response
is logged to `calls.json`. Measured on Loong, it comes out
net-zero-to-negative versus a plain long-context call — it *wins* on
`paper`/`Clustering` (graph structurization matches citation tasks) but
*destroys* multi-hop `legal`/`ChainOfReasoning`, and its wins shrink once the
model can think. The read on it: a reasoning crutch, because structuring is
lossy — it helps when the loss is the noise and hurts when the loss is the
signal. The `evals[structrag]` extra (transformers) is optional — it only
sharpens the truncation.

### 🔍 🤖 `arag`

[evals/baselines/arag/](evals/baselines/arag/) — A-RAG (arXiv 2602.03442),
vendored byte-for-byte. **Agentic, ReAct-style RAG** — the only baseline that is
*retrieval **and** agent*. An agent loops (up to 15 iterations) over three
hierarchical tools — `keyword_search` (lexical), `semantic_search`
(sentence-level dense, over OpenAI embeddings), and `read_chunk` (fetch a full
chunk by id) — deciding for itself what to search and read until it stops calling
tools and answers. The vendored `BaseAgent.run` *is* the loop (search → read →
evaluate → repeat, with a force-answer when the token budget or max-loops is
hit); the harness's one sanctioned deviation is a **dynamic per-model context
stop-gate** (`AragAgent` overrides *only* the token accounting — a tiktoken/HF/char
counter and the model's real window — replacing upstream's hardcoded gpt-4o
tokenizer and fixed 128K). Its content-addressed index is **LLM-independent**
(chunking + embedding are deterministic), so `index_hash` excludes the completion
model/seed — one index serves many completion models. A-RAG is the one baseline
that **needs a tool-calling endpoint** (native on OpenAI; vLLM needs the
tool-call parser enabled). Its actions are *retrieval tool calls*, not code —
hence 🔍🤖, not 🐍🤖. Supports loong / corpusqa / dracula; behind the
`evals[arag]` extra.

### 🐍 🤖 `codeact`

[evals/baselines/codeact/](evals/baselines/codeact/) — CodeAct (Wang et al.,
ICML 2024; arXiv 2402.01030) via smolagents' `CodeAgent`, its reference
implementation; a normal dependency. The agent acts by **writing and running
Python** in a ReAct loop (write a code blob → a local executor runs it → observe
stdout → repeat until `final_answer(...)`). Instead of dropping documents to fit
a budget, it can *scan and compute over the whole bundle* — which is exactly the
right shape for leave-no-document-behind, so the code-agent family is expected to
be **strong**. The bundle is offloaded into the code sandbox as a `documents`
variable via `agent.state["documents"]` — deliberately **not** via
`run(additional_args=…)`, because smolagents *also* stringifies that into the
prompt, dumping the whole bundle and defeating the offload. The LLM routes
through a `LiteLLMModel` subclass for deterministic, complete per-call cost
capture (verified to match smolagents' own token tally exactly), so it runs on
**OpenAI or vLLM** with no tool-calling endpoint needed (it emits code, not
JSON). Validated behavior: a capable coder acts on the `documents` offload
immediately (`for doc in documents: print(doc[:2000])`), and scores jump with
model capability (Loong EN 10 → 100 going from gpt-5.4-nano to a thinking Qwen).
Supports loong / corpusqa / dracula; behind the light `evals[codeact]` extra.

### 🐍 🤖 `rlm`

[evals/baselines/rlm/](evals/baselines/rlm/) — Recursive Language Models (arXiv
2512.24601), a normal dependency used via its front door (not vendored). Instead
of stuffing the context into the prompt, the document bundle is offloaded as a
**`context` variable in a Python REPL**, and a root LM writes **CodeAct code**
(not JSON tool-calls) to examine and decompose it and launch recursive sub-LM
calls (`llm_query` / `rlm_query`). The task is posed through RLM's canonical QA
front door (docs → the REPL `context`, the composed question → `root_prompt`) — we
pose the task, we do not author a strategy. It is **token-heavy by design**, so it
runs **against local vLLM only, never OpenAI** (the runner refuses an OpenAI
endpoint), and because it emits code rather than JSON tool-calls it runs on a
plain vLLM with no tool-call parser. Token cost is captured **completely** at
RLM's OpenAI-client boundary via a runtime monkeypatch
(`OpenAIClient._track_cost`), because RLM's own `usage_summary` misses the tokens
of recursive sub-call clients. Like CodeAct, it can programmatically scan every
document → expected strong on multi-doc. Supports loong / corpusqa / dracula;
behind the `evals[rlms]` extra.

### 🐍 🤖 `claude-code`

[evals/baselines/claude_code/](evals/baselines/claude_code/) — the `claude -p`
CLI; **exploratory (v1) and deliberately distinct from every other baseline**. It
is **not a published method** (it's a frontier general-agent reference point) and
**not a same-model comparison** — it runs a **Claude** model (default
`claude-opus-4-8`, `--effort high`) via the operator's **Max** login, not the
served vLLM, so it answers a different question: "what does a SOTA file-navigating
agent score on Loong/CorpusQA, and at what cost?" Per task it writes the bundle as
files in a temp dir outside this repo and runs the headless agent over them (the
docs are offloaded as files, the prompt poses the task plus "read the files in
your cwd"); it then scans/computes over them with its own tools — the code-agent
shape, hence 🐍🤖. It keeps Claude Code's **default toolset minus web and
`AskUserQuestion`** (web would be *retrieving* the answer; AskUserQuestion is a
headless no-op), runs **clean** (scrubbed env so a child launched inside a Claude
Code session matches a fresh terminal) and **macOS-Seatbelt-sandboxed** out of
`$HOME` (it refuses to run unsandboxed). Token cost stays complete even when the
agent fans out — the CLI's `modelUsage` is the cumulative per-model session total
(verified `total_cost_usd == Σ modelUsage.costUSD`), priced token-only at the full
rate with no cache discount for consistency with the other baselines. It needs
**no extra** (it shells out to the installed CLI). Supports loong / corpusqa.

### 📖 `memagent`

[evals/baselines/memagent/](evals/baselines/memagent/) — MemAgent (BytedTsinghua-SIA /
Seed, arXiv 2507.02259). A **recurrent fixed-size memory** method for long context: read
the context as a stream of fixed **5000-token chunks**, folding each into a running
`memory` (a compact, **overwriting** summary) — one LLM call per chunk, `max_tokens=1024`
capping the memory — then answer from the final memory. Linear-time and
window-independent: it never holds the whole context at once. The distinguishing point is
that **the method IS an RL-trained checkpoint** — `BytedTsinghua-SIA/RL-MemoryAgent-14B`
(Qwen2.5-14B-Instruct fine-tuned end-to-end for this exact loop, RLVR/DAPO on HotpotQA) —
so it runs on that served model over vLLM (default `--model`, needs `--base-url`); a
generic model in the loop is not MemAgent. Not vendored as a library (upstream ships a
demo + a verl-based eval harness); the connector **reproduces** the loop with the prompts
copied verbatim (`upstream/quickstart.py` is the byte-for-byte anchor). Two faithful
calls: the demo's 120k head+tail context clip is **lifted by default** so the method
processes the whole leave-no-document-behind bundle (it is unbounded by design — the paper
runs 3.5M tokens), and the `max_tokens=1024` cap is **kept** (it is load-bearing — it
bounds the fixed memory; and the model is non-thinking Qwen2.5, so nothing is eaten by
reasoning). Conceptually it is a **distiller**, like `readagent` (📖) — it compresses the
whole stream into a small query-conditioned memory that keeps "what helps answer the
problem" and discards the rest. That inductive bias fits **single-target recall** (its
HotpotQA/RULER home turf) but is the wrong tool for **aggregation** ("count/rank/compute
over every document" can't survive a 1024-token overwriting memory), so it is **expected
strong on single-target, a foil on aggregation** — weak on Loong Clustering/Chain-of-Reasoning
and most of CorpusQA. Supports loong / corpusqa / dracula.
Behind the light `evals[memagent]` extra (`transformers`, the tokenizer, lazy); writes a
live `progress.json` and the full memory trajectory.

---

### 🔍 `hipporag`

[evals/baselines/hipporag/](evals/baselines/hipporag/) — HippoRAG 2 (Gutiérrez et al.,
ICML 2025) OpenIE knowledge-graph + Personalized-PageRank retrieval. Per task it chunks the
document bundle into passages (our OWN by-token chunker — HippoRAG ships none), builds a graph
via **per-passage OpenIE** (an LLM NER + triple-extraction pass per passage → phrase + passage
nodes, synonym + context edges), then answers via HippoRAG's own retrieve→read front door
(query→triple linking + a recognition-memory triple filter + PPR over the graph → top-k passages
→ a reader LLM). Supports loong / corpusqa / dracula.

**That chunker is the load-bearing deviation.** The authors specify no long-document recipe
(HippoRAG's own corpora are pre-chunked ~100-token Wikipedia paragraphs), so splitting whole
documents ourselves carries a real misrepresentation risk: corpusqa 8000 / loong 3000 /
dracula 3000 tok (corpusqa capped under OpenAI's 8,191 embed limit) is a **sanctioned
deviation**, not an upstream recipe — PROVENANCE D1.
Upstream is vendored byte-for-byte (`ad30fc3`, MIT); one litellm seam
drives every internal call (OpenIE + the filter — which uses NO runtime dspy, just a baked prompt
— + the reader), an OpenAI embedding seam handles the index (usage captured), and the local-model
backends are trimmed off the import path (torch kept for the synonym-edge KNN). A retrieval **foil**
expected on these leave-no-document-behind benchmarks; **very expensive** (per-passage OpenIE, no
cross-task reuse) → run stratified `--limit` subsets. Validated live on gpt-5.4-nano. Behind the
`evals[hipporag]` extra; flat per-run-folder logging (the OpenIE graph under `index/`).

---

## The synthesis

The cleanest lens on the whole set, against Loong's "leave no document behind":

| family | tags | sees all docs? | Loong expectation |
| --- | --- | --- | --- |
| Reading / NL-distill — `readagent` | 📖 | gist + a 1–2 page look-up | **foil** |
| Recurrent memory — `memagent` | 📖 | all, but a fixed-size overwriting memory | **strong on single-target, foil on aggregation** |
| Retrieval / RAG — `raptor`, `arag`, `hipporag` | 🔍 (`arag` +🤖) | **no** — retrieves a subset | **foils** |
| Structuring — `structrag` | 🧩 | yes — restructures all docs | net-zero/negative; a reasoning crutch |
| Code agents — `codeact`, `rlm`, `claude-code` | 🐍🤖 | **yes** — programmatic scan | **expected strong** |

The story the harness is built to tell: on a benchmark engineered to punish
dropping documents, *retrieval sophistication* (arag's agent, hipporag's
PageRank) should **lose** to **code agents that scan everything** — and StructRAG's already-measured result
(wins on citation-structure tasks, destroys multi-hop reasoning) is the first
concrete data point in exactly that shape. The open question is whether the
code-agent family delivers on its promise across the full Loong axes.

---

## Cross-cutting reference

The same nine baselines, along the axes that actually differ between them.

| baseline | tags | what it is | benchmarks | connected | log layout |
| --- | --- | --- | --- | --- | --- |
| `readagent` | 📖 | paginate→gist→lookup→answer | loong, corpusqa, dracula | reproduced from demo | flat per-run, no-reuse |
| `raptor` | 🔍 | LLM recursive-summary tree | loong, corpusqa, dracula | vendored byte-for-byte | flat per-run, no-reuse |
| `hipporag` | 🔍 | OpenIE KG index + PPR | loong, corpusqa, dracula | vendored byte-for-byte | flat per-run, no-reuse |
| `memagent` | 📖 | recurrent fixed-size memory | loong, corpusqa, dracula | reproduced from demo | flat per-run, no-reuse |
| `structrag` | 🧩 | route→structurize→utilize | loong, corpusqa, dracula | vendored byte-for-byte | content-addressed (no index) |
| `arag` | 🔍🤖 | agentic ReAct RAG, 3 tools | loong, corpusqa, dracula | vendored byte-for-byte | content-addressed + `_indices/` |
| `codeact` | 🐍🤖 | smolagents CodeAgent | loong, corpusqa, dracula | dependency (front door) | flat per-run, no-reuse |
| `rlm` | 🐍🤖 | REPL context + recursive sub-LM | loong, corpusqa, dracula | dependency (front door) | flat per-run, no-reuse |
| `claude-code` | 🐍🤖 | `claude -p` frontier agent | loong, corpusqa | shells out to CLI | flat per-run, no-reuse |

| baseline | tags | LLM calls | embeddings | endpoint | cost-capture path |
| --- | --- | --- | --- | --- | --- |
| `readagent` | 📖 | ~2/page + 2 | no | any | seam accumulator (`ReadAgentLLM`), TOTAL |
| `raptor` | 🔍 | many (cluster summaries) + 1 QA | yes (OpenAI) | any | seam (summary+QA) + embedder, TOTAL |
| `hipporag` | 🔍 | many (OpenIE/passage + filter + reader) | yes (OpenAI) | OpenAI or vLLM | seam (`HippoRAGLLM`) + embedder, TOTAL |
| `memagent` | 📖 | ~1/chunk + 1 answer | no | **vLLM only** | seam accumulator (`MemAgentLLM`), TOTAL |
| `structrag` | 🧩 | many sequential | no | any | seam accumulator (`StructRAGLLM`) |
| `arag` | 🔍🤖 | agent loop (≤15) | yes (OpenAI) | **tool-calling** | seam (`AragLLM`) + embedder |
| `codeact` | 🐍🤖 | step loop (≤20) | no | OpenAI or vLLM | `LiteLLMModel` subclass, TOTAL |
| `rlm` | 🐍🤖 | many (recursive) | no | **vLLM only** | client monkeypatch (`_track_cost`), TOTAL |
| `claude-code` | 🐍🤖 | agent turns | no | Max (Claude) | CLI `modelUsage` (cumulative), token-only |

For the precise per-baseline deviations from upstream, read each
`evals/baselines/<name>/PROVENANCE.md` (or `NOTES.md`).
