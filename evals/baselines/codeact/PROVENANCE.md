# CodeAct

**Framework:** smolagents' `CodeAgent` · **Paper:** Wang et al., ICML 2024,
arXiv:2402.01030 · **License:** Apache-2.0.

Not vendored — used through its public API. smolagents' system prompt, code template,
`max_steps=20` and local executor are its defaults.

## Deviations

- **Documents go into the sandbox, not the prompt.** They are bound as a `documents`
  variable in the executor state, with a one-line pointer in the task. The obvious channel,
  `run(additional_args=…)`, also stringifies the value into the prompt, which would dump the
  whole bundle into the context window. This is a departure from paper-CodeAct, which hands
  the context to the model directly.
