# RLM (Recursive Language Models)

**Upstream:** github.com/alexzhang13/rlm · **PyPI:** `rlms` 0.1.2 ·
**Paper:** arXiv:2512.24601 · **License:** MIT.

Not vendored — used through `rlm.completion(prompt, root_prompt)`. The bundle becomes a
`context` variable in a Python REPL and the root model writes REPL code to explore it and
spawn recursive sub-calls. RLM's defaults stand (`max_iterations=30`, `max_depth=1`).

## Deviations

- **vLLM only.** RLM is token-heavy by design, so the runner requires `--base-url` and
  rejects an OpenAI endpoint.
- **Multi-document context.** The bundle is concatenated into one `context` string with
  `=== Document N ===` markers so the agent can split it in code.
