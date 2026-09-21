# Baselines

Nine baselines, each a connector around a published method (except `claude-code`, which
is a frontier-agent reference point rather than a method). Per-baseline detail — upstream
commit, license, what was changed — is in each `evals/baselines/<name>/PROVENANCE.md`.

The benchmarks scatter a task's evidence across a bundle of documents, so the question
that predicts most about a baseline's score is whether it reads all of them or a subset.

| baseline | what it does | sees all docs |
| --- | --- | --- |
| `readagent` | paginate → gist → re-read 1–2 pages | no (gists) |
| `memagent` | fold 5k-token chunks into a fixed 1k memory | all, but compressed |
| `raptor` | recursive summary tree, retrieve top-k | no |
| `hipporag` | OpenIE graph + Personalized PageRank | no |
| `arag` | ReAct agent over search/read tools | no |
| `structrag` | restructure every doc, then answer | yes |
| `codeact` | writes Python over the documents | yes |
| `rlm` | documents in a REPL, recursive sub-calls | yes |
| `claude-code` | `claude -p` over the docs as files | yes |

Retrieval baselines are expected to lose here; the gap is the measurement, not a bug.

## The methods

**`readagent`** — ReadAgent (ICML 2024, arXiv 2402.09727). Paginates each document at
LLM-chosen break points, compresses each page into a gist, then for a question names the
1–2 pages worth re-reading in full and answers from gists plus that window. LLM-only: no
embeddings, no retriever. Only the ReadAgent-P look-up is wired, since it is the one
variant upstream implements. Pagination is a serial chain per document, which makes this
the slowest baseline.

**`memagent`** — MemAgent (arXiv 2507.02259). Reads the context as a stream of
5000-token chunks, folding each into a running memory capped at 1024 tokens, then answers
from the final memory. Linear time, never holds the whole context. The method is an
RL-trained checkpoint (`BytedTsinghua-SIA/RL-MemoryAgent-14B`); a generic model in the
loop is not MemAgent. vLLM only. The 1024-token cap is load-bearing, so a reasoning model
spends it on thinking and returns an empty memory.

**`raptor`** — RAPTOR (ICLR 2024, arXiv 2401.18059), vendored. Chunks documents into
leaves, then recursively clusters (UMAP + GMM) and summarizes each cluster into the next
tree level. At query time it collapses the tree and retrieves top-k nodes across all
levels, so answers mix raw chunks with summaries. Higher levels preserve some global
context that flat retrieval loses. Expensive on a reasoning model — every cluster summary
pays for thinking.

**`hipporag`** — HippoRAG 2 (ICML 2025), vendored. Builds a knowledge graph by running
OpenIE over every passage, then answers by linking the query to triples, filtering, and
ranking passages with Personalized PageRank. Per-passage OpenIE with no cross-task reuse
makes it expensive; run subsets.

**`arag`** — A-RAG (arXiv 2602.03442), vendored. An agent loops up to 15 times over
`keyword_search`, `semantic_search` (OpenAI embeddings) and `read_chunk`, deciding what
to look at until it answers. Needs a tool-calling endpoint. Its index depends only on the
documents and the embedding model, not the completion model, so one index serves many
runs.

**`structrag`** — StructRAG (ICLR 2025), vendored. Not retrieval: a router picks a
knowledge structure (table / graph / algorithm / catalogue / chunk), a structurizer
rebuilds the documents into it, and a utilizer decomposes the question and merges an
answer. Many sequential calls, no persistent index. Measured on Loong it is roughly
net-zero against a plain long-context call: it wins on citation-style tasks and loses on
multi-hop reasoning, which fits structuring being lossy — it helps when what is lost is
noise and hurts when it is signal.

**`codeact`** — CodeAct (ICML 2024, arXiv 2402.01030) via smolagents' `CodeAgent`. Acts
by writing and running Python: write a code blob, a local executor runs it, observe
stdout, repeat until `final_answer(...)`. The documents are a `documents` variable in the
sandbox, so it can scan and compute over the whole bundle. Runs on OpenAI or vLLM.

**`rlm`** — Recursive Language Models (arXiv 2512.24601). The bundle is a `context`
variable in a Python REPL; a root LM writes code to examine it and launches recursive
sub-LM calls. Token-heavy by design, so vLLM only — the runner refuses an OpenAI
endpoint.

**`claude-code`** — the `claude -p` CLI, exploratory. Not a published method and not a
same-model comparison: it runs a Claude model on a Max login rather than the served
model, so it answers "what does a frontier file-navigating agent score, and at what
cost?". Per task the documents are written as files in a temp dir outside this repo and
the agent reads them with its own tools. Web search is disabled; it runs sandboxed.
