# Third-party components

Each baseline is a *connector* around a published method. Where the authors released
usable source it is vendored **byte-for-byte** under `evals/baselines/<name>/upstream/`, so
the method runs as published; where they did not, the connector reproduces the method from
the artifacts they did release. Per-baseline detail is in each
`evals/baselines/<name>/PROVENANCE.md`.

This repository ships **no license of its own** — see [STATUS.md](STATUS.md).

## Vendored (redistributed here)

| component | upstream | pinned at | license |
| --- | --- | --- | --- |
| HippoRAG 2 | OSU-NLP-Group/HippoRAG | `ad30fc3` | **MIT** — upstream `LICENSE` kept at `hipporag/upstream/LICENSE` |
| RAPTOR | parthsarthi03/raptor | `7da1d48a` | **MIT** — upstream `LICENSE.txt` kept at `raptor/upstream/LICENSE.txt` |
| A-RAG | Ayanami0730/arag | `a44de6b` | **MIT** — declared in the upstream README; the repo ships no LICENSE file, so none could be vendored |
| MemAgent | BytedTsinghua-SIA/MemAgent | `ef4219b` | **Apache-2.0**; one file (`quickstart.py`) kept as a provenance anchor, not imported |
| StructRAG | icip-cas/StructRAG | `82e2804c` | ⚠️ **no license stated** |
| ReadAgent | HF Space `ReadAgent/read-agent` (`ecadb03`) + read-agent.github.io (`569dff3`) | as noted | ⚠️ **no license stated** |

### ⚠️ Two components with no stated license

**StructRAG** (14 vendored files) and **ReadAgent** (two demo artifacts, kept as provenance
anchors and never imported) come from upstreams that publish no LICENSE file and no license
declaration — checked 2026-09-20 at the repository and Space pages. Absent a license,
default copyright applies and redistribution is not granted.

Options: ask the authors to state a license; drop the vendored copies and rely on the
connector plus a citation; or keep those two out of any public copy. Tracked in
[STATUS.md](STATUS.md).

## Dependencies (installed, not redistributed)

| package | used by | license |
| --- | --- | --- |
| `smolagents` | `codeact` — its `CodeAgent` is the reference CodeAct implementation | Apache-2.0 |
| `rlms` | `rlm` — used via its public API | MIT |
| `litellm`, `datasets`, `pydantic`, `rich`, … | the harness | MIT / Apache-2.0 |

`claude-code` shells out to the installed `claude` CLI; nothing of it is redistributed.

## Benchmark data

`loong` and `corpusqa` are downloaded from their upstream sources on first use and are not
redistributed here. `dracula` is built from *Dracula* by Bram Stoker — public domain,
Project Gutenberg #345.
