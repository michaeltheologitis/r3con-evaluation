# RLM (Recursive Language Models) — provenance & deviations

**Upstream:** github.com/alexzhang13/rlm · **PyPI:** `rlms` 0.1.2 ·
**Paper:** arXiv:2512.24601 · **License:** MIT.

Not vendored — a maintained inference engine used through its public API
(`rlm.completion(prompt, root_prompt)`). The document bundle becomes a `context` variable
in a Python REPL and the composed task becomes the root prompt; the root model writes
` ```repl ` code (CodeAct, not JSON tool-calling) to explore that variable and spawn
recursive sub-LM calls. RLM's own defaults stand: `max_iterations=30`, `max_depth=1`,
in-process REPL.

## Deviations that matter

- **vLLM only.** RLM is token-heavy by design, so the runner **requires** `--base-url` and
  rejects an OpenAI endpoint. It uses no litellm and no embeddings, so nothing can leak to
  a paid API.
- **Complete token capture.** RLM's own usage summary sums only the root handler, so
  recursive sub-call clients are undercounted; we count at the client boundary instead,
  across every depth. Observes only — it never alters RLM's behaviour.
- **Trajectory logging without the REPL-variable dumps.** RLM's native logger is used, minus
  the `locals` snapshot it writes after every code block — that is the whole document bundle
  twice, plus every slice derived from it, and it was 94–99% of each trajectory file. Every
  prompt, response, code block, stdout and sub-call is kept (`logger.py`).
- **Multi-document context.** The bundle is concatenated into one `context` string with
  `=== Document N ===` markers so the agent can split it in code.
