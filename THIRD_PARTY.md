# Third-party components

Where a method's authors released usable source it is vendored byte-for-byte under
`evals/baselines/<name>/upstream/`; where they did not, the connector reproduces the
method from what they did release. Per-baseline detail is in each
`evals/baselines/<name>/PROVENANCE.md` (`NOTES.md` for `claude_code`).

One vendored file sits outside that path: `evals/r3con/` is the method under evaluation,
not a baseline, and it vendors a single sandbox-executor file from smolagents.

This repository's own code is MIT — see [LICENSE](LICENSE).

## Vendored

| component | upstream | pinned | license |
| --- | --- | --- | --- |
| HippoRAG 2 | OSU-NLP-Group/HippoRAG | `ad30fc3` | MIT, kept at `hipporag/upstream/LICENSE` |
| RAPTOR | parthsarthi03/raptor | `7da1d48a` | MIT, kept at `raptor/upstream/LICENSE.txt` |
| A-RAG | Ayanami0730/arag | `a44de6b` | MIT, declared in the README; no LICENSE file to vendor |
| MemAgent | BytedTsinghua-SIA/MemAgent | `ef4219b` | Apache-2.0 — text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| StructRAG | icip-cas/StructRAG | `82e2804c` | none stated upstream |
| ReadAgent | HF Space `ReadAgent/read-agent` (`ecadb03`) + read-agent.github.io (`569dff3`) | as noted | none stated upstream |
| R3Con's sandbox executor (`evals/r3con/pipeline/runtime/python_executor.py`) | huggingface/smolagents | `v1.25.0` | Apache-2.0 — text at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |

StructRAG and ReadAgent publish no LICENSE file or declaration (checked 2026-09-20). Both
are vendored anyway with their origin and commit recorded; ReadAgent's two files are demo
artifacts kept as anchors and are never imported.

The smolagents executor is carried as one self-contained file rather than the package, and
its header lists every edit. It is vendored, not imported — `codeact` separately installs
the smolagents package and drives `CodeAgent` through its public API.

## Dependencies (installed, not redistributed)

| package | used by | license |
| --- | --- | --- |
| `smolagents` | `codeact` | Apache-2.0 |
| `rlms` | `rlm` | MIT |
| `litellm`, `datasets`, `pydantic` | the harness | MIT / Apache-2.0 |

`claude-code` shells out to the installed `claude` CLI; nothing of it is redistributed.

## Benchmark data

`loong` and `corpusqa` download from their upstream sources on first use and are not
redistributed. `dracula` is built from *Dracula* by Bram Stoker — public domain, Project
Gutenberg #345.
