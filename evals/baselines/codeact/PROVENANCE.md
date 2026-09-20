# CodeAct — provenance & deviations

**Framework:** smolagents' `CodeAgent`, the reference CodeAct implementation ·
**Paper:** Wang et al., ICML 2024, arXiv:2402.01030 · **License:** Apache-2.0.

Not vendored — used through its public API. The agent acts by writing and running Python in
a ReAct loop until it calls `final_answer(...)`. smolagents' own system prompt and code
template are untouched, and `max_steps=20` + the local in-process executor are its defaults.

## Deviations that matter

- **D1 — the documents are offloaded into the sandbox, not the prompt.** The obvious
  channel, `run(additional_args=…)`, *also* stringifies the value into the task prompt,
  which would dump the whole bundle into the context window and defeat the offload. So the
  bundle is set directly as a `documents` variable in the executor state and the task
  carries a one-line pointer to it. Note this is a departure from paper-CodeAct, which hands
  the context to the model directly — our CodeAct sits a step closer to RLM.
- **Token capture** reads each call's usage at the model seam, because the litellm callback
  drops roughly a third of sync-completion records.

Runs on OpenAI or vLLM; it emits code, not JSON tool-calls, so no tool-calling endpoint is
needed. The local executor runs model-written code in-process under smolagents' import
allowlist, with no per-cell timeout — each task is an isolated subprocess.
