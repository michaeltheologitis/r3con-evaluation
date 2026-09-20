# Claude Code baseline — design notes

**What it is:** each task is posed to the Claude Code CLI (`claude -p`) running headless
over the documents, which are written as files in a temp dir; the prompt states the task and
points the agent at the files in its working directory. Documents-as-files is the same
offload idea as RLM's REPL variable and CodeAct's sandbox variable.

**What it is NOT.** Not a published method — it is a frontier general-agent reference point.
And **not a same-model comparison**: it runs a Claude model through a product harness, not
the served open-weight model every other baseline uses. Read it as an upper reference, not
as a peer.

## How it is run

- **Tools:** Claude Code's default toolset, denying only **web** (`WebSearch` / `WebFetch` —
  the sources are real public documents, so web access would be *retrieving the answer*) and
  `AskUserQuestion` (headless: nobody can answer). The default set shifts between CLI
  versions, so the granted tools and the CLI version are recorded in every manifest.
- **Clean:** project settings only, empty MCP config, slash commands disabled, and the
  session environment variables scrubbed so a run launched from inside an agent session
  matches a fresh terminal.
- **Sandboxed:** confined out of `$HOME` via the OS sandbox; `run_one` refuses to run
  unsandboxed unless explicitly overridden.
- **Cost:** the CLI reports a cumulative per-model usage total that already folds in every
  subagent and auxiliary call. We record it as **tokens only** and price the full input at
  the standard rate — no cache discount — so it stays comparable with the other baselines;
  the CLI's own discounted dollar figure rides along raw in the manifest.
- Runs **one task at a time** by default (rate limits), and the knobs are `--model` and
  `--effort` only — the CLI owns auth and transport.

The full `stream-json` trajectory is kept verbatim per run, and the CLI's `result` and
`init` messages are mirrored into the manifest.
