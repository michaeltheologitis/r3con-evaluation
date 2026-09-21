# ReadAgent

**Upstream:** the HF Space demo `ReadAgent/read-agent` (`ecadb03`) + the project-page
notebook (`569dff3`) · **Paper:** ICML 2024, arXiv:2402.09727 ·
**License:** none stated upstream.

The authors released no library, only a demo and a notebook — both vendored under
`upstream/` as anchors, never imported. The connector reproduces paginate → gist → look up
→ answer with every prompt copied verbatim.

## Deviations

- **D5 — ReadAgent-P only.** It is the one look-up variant upstream implements; ReadAgent-S is a
  prompt template with no code, so it is not wired.
- **D10 — page and gist sizes scale with the benchmark.** ReadAgent's native 600-word pages suit
  Loong and Dracula; CorpusQA's ~1M-token bundles use ×10 sizes. Chosen automatically.
- **No generation cap on the gist call**, or a reasoning model returns empty gists.

Deviation IDs match the `DEVIATION (Dn)` comments in the code. Only the ones that
affect results are listed here; mechanical ones (import paths, dead-import trims,
logging) are marked at the code site.
