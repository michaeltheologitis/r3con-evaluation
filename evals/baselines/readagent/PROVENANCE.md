# ReadAgent — provenance & deviations

**Upstream:** the HF Space demo `ReadAgent/read-agent` (`ecadb03`) + the project-page
notebook (`569dff3`) · **Paper:** ICML 2024, arXiv:2402.09727 ·
**License:** ⚠️ **none stated upstream** — see [THIRD_PARTY.md](../../../THIRD_PARTY.md).

The authors released no library — only a Gradio demo and a notebook, both vendored under
`upstream/` as provenance anchors (never imported). The connector **reproduces** the three
stages with every prompt copied verbatim: paginate the documents at LLM-chosen break
points, gist each page into a compressed memory, then look up a few pages' full text from
that memory and answer.

## Deviations that matter

- **D2 — document prep.** Upstream's parser is QuALITY-specific; ours splits into
  paragraphs generically.
- **D3 — free-form prompts.** Our benchmarks are free-form, so the NarrativeQA prompt
  variants are used rather than the multiple-choice ones.
- **D5 — ReadAgent-P only.** Parallel look-up is the only variant upstream implements in
  code; ReadAgent-S appears in the notebook as a prompt template with no implementation, so
  it is not wired and there is no variant flag.
- **D10 — page and gist sizes scaled per benchmark.** ReadAgent's native 600-word pages
  suit Loong and Dracula; CorpusQA's ~1M-token bundles use ×10 sizes (~10× fewer calls).
  Selected automatically from the benchmark — there is no flag to get it wrong.
- **Multi-document pagination.** Each document is paginated on its own and the pages
  pooled, so a page never spans two documents.
- **No generation cap on the gist call** — a small cap is eaten by reasoning tokens,
  yielding empty gists.
