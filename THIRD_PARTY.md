# Third-party components

Each baseline is a *connector* around a published method. Where the authors released
usable source it is vendored under `evals/baselines/<name>/upstream/` — **byte-for-byte**
apart from the mechanical connector edits (import prefixes, prompt-path resolution,
optional backends left unimported) that each baseline's PROVENANCE ledger records — so the
method runs as published; where they did not, the connector reproduces the method from
the artifacts they did release. Per-baseline detail is in each
`evals/baselines/<name>/PROVENANCE.md` (`NOTES.md` for `claude_code`, which vendors
nothing).

One vendored file sits outside that path. `evals/r3con/` is not a baseline — it is the
method under evaluation — and it vendors a single sandbox-executor file from smolagents;
it is listed in the table below alongside the baselines' upstreams.

This repository's own code is MIT — see [LICENSE](LICENSE).

## Vendored (redistributed here)

| component | upstream | pinned at | license |
| --- | --- | --- | --- |
| HippoRAG 2 | OSU-NLP-Group/HippoRAG | `ad30fc3` | **MIT** — upstream `LICENSE` kept at `hipporag/upstream/LICENSE` |
| RAPTOR | parthsarthi03/raptor | `7da1d48a` | **MIT** — upstream `LICENSE.txt` kept at `raptor/upstream/LICENSE.txt` |
| A-RAG | Ayanami0730/arag | `a44de6b` | **MIT** — declared in the upstream README; the repo ships no LICENSE file, so none could be vendored |
| MemAgent | BytedTsinghua-SIA/MemAgent | `ef4219b` | **Apache-2.0**; one file (`quickstart.py`) kept as a provenance anchor, not imported. It keeps its upstream header (Copyright 2025 Bytedance Ltd. and/or its affiliates); the License text is at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |
| StructRAG | icip-cas/StructRAG | `82e2804c` | ⚠️ **no license stated** |
| ReadAgent | HF Space `ReadAgent/read-agent` (`ecadb03`) + read-agent.github.io (`569dff3`) | as noted | ⚠️ **no license stated** |
| R3Con's sandbox executor — `evals/r3con/pipeline/runtime/python_executor.py`, vendored from `src/smolagents/local_python_executor.py` | huggingface/smolagents | `v1.25.0` | **Apache-2.0** — it keeps its upstream header (Copyright 2024 The HuggingFace Inc. team); the License text is at [`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt) |

### A note on StructRAG and ReadAgent

Neither upstream publishes a LICENSE file or a license declaration (checked
2026-09-20). Both are vendored here anyway, with their origin and pinned commit
recorded above and in their PROVENANCE ledgers: StructRAG's vendored pipeline is
what the StructRAG baseline actually runs, and ReadAgent's two files are demo
artifacts kept purely as provenance anchors — they are never imported.

### A note on the vendored smolagents executor

Like the other vendored files it carries an in-file provenance header, but its edits go
beyond the mechanical rewrites elsewhere: it is carried as a single self-contained file
rather than as the package, and its header enumerates the edits (tool-calling plumbing
removed, `BASE_BUILTIN_MODULES` and `truncate_content` inlined from `smolagents.utils`,
and the three control-flow exceptions rebased onto `BaseException`). It is vendored, not
imported from the installed package — see the next section.

## Dependencies (installed, not redistributed)

| package | used by | license |
| --- | --- | --- |
| `smolagents` | `codeact` — its `CodeAgent` is the reference CodeAct implementation, used through its public API (the `evals[codeact]` extra, `smolagents>=1.26.0`) | Apache-2.0 |
| `rlms` | `rlm` — used via its public API | MIT |
| `litellm`, `pydantic` | the harness core | MIT |
| `datasets`, `tenacity` | the harness core | Apache-2.0 |
| `python-dotenv` | the harness core | BSD-3-Clause |
| each baseline's extra in `pyproject.toml` (`torch`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `faiss-cpu`, `python-igraph`, `transformers`, …) | the deps the vendored upstreams pull at load | each under its own terms; note that `python-igraph` (the `hipporag` extra) is **GPL** |

smolagents appears in both tables, and the two uses are independent: `codeact` installs the
package and drives `CodeAgent` through its public API, while R3Con carries a copy of one
executor file pinned at `v1.25.0` and never imports the package.

`claude-code` shells out to the installed `claude` CLI; nothing of it is redistributed.

## Benchmark data

`loong` and `corpusqa` are downloaded from their upstream sources on first use and are not
redistributed here. `dracula` is built from *Dracula* by Bram Stoker — public domain,
Project Gutenberg #345.
