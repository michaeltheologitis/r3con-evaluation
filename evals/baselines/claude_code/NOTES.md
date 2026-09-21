# Claude Code baseline

Each task is posed to the Claude Code CLI (`claude -p`) running headless over the
documents, which are written as files in a temp dir.

**Not a published method** — a frontier general-agent reference point. **Not a same-model
comparison**: it runs a Claude model through a product harness, not the served open-weight
model every other baseline uses. Read it as an upper reference, not a peer.

## How it is run

- Claude Code's default toolset, denying **web** (the sources are real public documents, so
  web access would be retrieving the answer) and `AskUserQuestion` (nobody can answer it
  headless). The granted tools and CLI version are recorded in every manifest, since the
  default set shifts between releases.
- Sandboxed out of `$HOME`; it refuses to run unsandboxed unless explicitly overridden.
- Cost is recorded as tokens only and priced at the full input rate with no cache discount,
  so it stays comparable with the other baselines.
- One task at a time by default (rate limits). The only knobs are `--model` and `--effort`.
