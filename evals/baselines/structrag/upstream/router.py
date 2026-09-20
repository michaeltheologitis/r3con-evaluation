# ─────────────────────────────────────────────────────────────────────────────
# VENDORED FROM StructRAG — github.com/icip-cas/StructRAG @ 82e2804c (router.py)
#
# DEVIATIONS FROM UPSTREAM (otherwise byte-for-byte upstream):
#   • Prompt files are resolved relative to THIS package (`_PROMPTS_DIR`) instead
#     of the upstream cwd-relative `open("prompts/route.txt")`, which only worked
#     when run from the StructRAG repo root. Behaviour is identical; only the path
#     resolution changed.
# Full deviation ledger: evals/baselines/structrag/PROVENANCE.md
# ─────────────────────────────────────────────────────────────────────────────
import os

_PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")


class Router:
    def __init__(self, llm):
        self.llm = llm

    def do_route(self, query, core_content, data_id):
        print(f"data_id: {data_id}, do_route...")

        raw_prompt = open(os.path.join(_PROMPTS_DIR, "route.txt"), "r").read()

        prompt = raw_prompt.format(
            query=query,
            titles=core_content
        )
        output = self.llm.response(prompt) 

        if "table" in output.lower():
            chosen = "table"
        elif "graph" in output.lower():
            chosen = "graph"
        elif "algorithm" in output.lower():
            chosen = "algorithm"
        elif "catalogue" in output.lower():
            chosen = "catalogue"
        else:
            chosen = "chunk"

        return chosen