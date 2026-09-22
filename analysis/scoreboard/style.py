"""Per-method display config — one source of truth for the name, colour, and marker used
for each method across every table and plot, so they stay consistent. (Plotting itself happens in
the notebook; this module only holds the lookup data.)

Dict order is the display order. Look styles up by raw method id via :func:`name` /
:func:`color` / :func:`marker`; an unknown method falls back to a neutral grey circle.

The two headline methods have **canonical brand colours** pinned as module constants
(:data:`GROUNDED_COLOR` / :data:`CLAUDE_COLOR`) — edit them here to recolour our method or
Claude Code *everywhere* (every board and figure reads this registry; nothing hardcodes a
hex for them).
"""
from __future__ import annotations

# Our system's display name — the ONE place it's spelled out. Every board, figure, and the
# notebook's Main alias read it (the notebook does `GROUNDED_NAME = style.GROUNDED_NAME`), so
# renaming the system is this single edit.
GROUNDED_NAME = "R3Con"     # \textsc{R3Con} in the paper (LaTeX exports print its \method macro, not this)

# Canonical brand colours — the ONE place to recolour these two methods across the whole repo.
GROUNDED_COLOR = "#7ea6ce"   # light blue: our method (all grounded-* strategies share it)
CLAUDE_COLOR = "#f0a860"     # light orange: Claude Code (every model it runs)

METHODS: dict[str, dict[str, str]] = {
    "raptor":           {"name": "RAPTOR",        "color": "#c44e52", "marker": ">"},
    "readagent":        {"name": "ReadAgent",     "color": "#4c72b0", "marker": "o"},
    "memagent":         {"name": "MemAgent",      "color": "#17a2b8", "marker": "h"},
    "hipporag":         {"name": "HippoRAG2",     "color": "#c9a227", "marker": "D"},
    "linearrag":        {"name": "LinearRAG",     "color": "#DAD88C", "marker": "s"},
    "structrag":        {"name": "StructRAG",     "color": "#8573c2", "marker": "*"},
    "codeact":          {"name": "CodeAgent",     "color": "#937860", "marker": "d"},
    "rlm":              {"name": "RLMs",          "color": "#da8bc3", "marker": "p"},
    "arag":             {"name": "A-RAG",         "color": "#55a868", "marker": "v"},
    "direct-llm":       {"name": "Direct-LLM",    "color": "#8c8c8c", "marker": "<"},
    "claude-code":      {"name": "Claude Code",   "color": CLAUDE_COLOR, "marker": "^"},
    "grounded-codeact": {"name": GROUNDED_NAME,   "color": GROUNDED_COLOR, "marker": "X"},
    "grounded-llm":     {"name": GROUNDED_NAME,   "color": GROUNDED_COLOR, "marker": "X"},
    # the `w/o structuring` ablation: the coding agent re-run with no structured parse.
    "grounded-codeact_ablation_nostruct": {"name": GROUNDED_NAME, "color": GROUNDED_COLOR, "marker": "P"},
}

_FALLBACK = {"name": None, "color": "#333333", "marker": "o"}


def name(method: str) -> str:
    return METHODS.get(method, {}).get("name") or method


def color(method: str) -> str:
    return METHODS.get(method, _FALLBACK)["color"]


def marker(method: str) -> str:
    return METHODS.get(method, _FALLBACK)["marker"]


ORDER = list(METHODS)      # raw method ids, in display order
