"""ReadAgent prompt templates + the response parsers — kept faithful to upstream.

ReadAgent (Google DeepMind, ICML 2024; arXiv 2402.09727) ships its code as a demo
(the HuggingFace Space ``app.py`` + the project-page notebook ``read_agent_demo.ipynb``,
both vendored under ``upstream/`` for provenance). There is no importable upstream
library and no Loong/CorpusQA implementation, so this connector REPRODUCES the three
prompting stages faithfully. Every template below is copied **verbatim** from an
upstream artifact; the source of each, and the few deliberate adaptations for our
free-form benchmarks, are recorded in ``PROVENANCE.md``.

All three benchmarks here (Loong, CorpusQA, Dracula) are FREE-FORM generation graded by an
LLM judge, not QuALITY's multiple-choice, so the lookup/answer prompts use the paper's
free-form templates:
  - pagination + gisting  ← ``app.py`` (the generic, runnable QuALITY demo text)
  - look-up + answer      ← the paper's NarrativeQA free-form templates (notebook)

Only ReadAgent-P (parallel look-up) is wired — it is the ONLY look-up variant upstream
actually implements in code (ReadAgent-S exists upstream as a prompt template with no
implementation). See PROVENANCE.md (D5).
"""
from __future__ import annotations

# ----------------------------------------------------------------------------------
# (1) Episode Pagination — VERBATIM from upstream app.py `prompt_pagination_template`
#     (identical to the notebook QuALITY pagination prompt). Generic ("article,
#     book, ...") so it applies to Loong/CorpusQA documents. Slots: {0} preceding
#     context, {1} the labelled passage, {2} end tag.
# ----------------------------------------------------------------------------------
PAGINATION_TEMPLATE = """
You are given a passage that is taken from a larger text (article, book, ...) and some numbered labels between the paragraphs in the passage.
Numbered label are in angeled brackets. For example, if the label number is 19, it shows as <19> in text.
Please choose one label that it is natural to break reading.
Such point can be scene transition, end of a dialogue, end of an argument, narrative transition, etc.
Please answer the break point label and explain.
For example, if <57> is a good point to break, answer with \"Break point: <57>\n Because ...\"

Passage:

{0}
{1}
{2}

"""


# ----------------------------------------------------------------------------------
# (2) Memory Gisting — two forms, selected per benchmark by the active Regime's
#     ``gist_token_hint`` (PROVENANCE D3 + D10):
#       • NO length clause — upstream app.py `prompt_shorten_template` VERBATIM. This is what
#         Loong and Dracula run as v2; pinned byte-for-byte by a test.
#       • WITH a "should be in {} tokens" clause — the notebook's length-capped gisting variant
#         (same authors), used for CorpusQA (v4, hint 640) whose ×10 pages must gist to a bounded
#         size. Slots are POSITIONAL {} — the substituted values (the token count, the page text)
#         are never re-scanned, so page text containing braces is safe.
# ----------------------------------------------------------------------------------
GISTING_TEMPLATE = """
Please shorten the following passage.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{}

"""

GISTING_TEMPLATE_WITH_HINT = """
Please shorten the following passage. The shortened passage should be in {} tokens.
Just give me a shortened version. DO NOT explain your reason.

Passage:
{}

"""


# ----------------------------------------------------------------------------------
# (3) Look-up (ReadAgent-P, the only upstream-implemented variant) — VERBATIM from the
#     notebook's NarrativeQA free-form `parallel_lookup_prompt_template`. Named slots so
#     user content (which may contain braces) is substituted by str.replace, never
#     str.format.
# ----------------------------------------------------------------------------------
PARALLEL_LOOKUP_TEMPLATE = """
The following text is what you remembered from reading an article and a question related to it.
You may read 1 or 2 page(s) of the article again to refresh your memory to prepare yourselve for the question.
Please respond with which page(s) you would like to read in the order of importance, beginning with the most important page number.
For example, if your only need to read Page 8, respond with \"I want to look up Page [8] to ...\";
if your would like to read Page 12 and 7, respond with \"I want to look up Page [12, 7] to ...\";
DO NOT select more pages if you don't need to.
You don't need to answer the question yet.

Text:
{concatenated_gists}

Question:
{question}

"""


# ----------------------------------------------------------------------------------
# (4) Answer — VERBATIM from the notebook's NarrativeQA free-form
#      `answer_prompt_template`. Slot order: the expanded article (gists with the
#      looked-up pages replaced by their full text), then the question.
# ----------------------------------------------------------------------------------
ANSWER_TEMPLATE = """
{concatenated_pages_and_gists}

Question:
{question}

Answer the question based on the above passage and retrieved pages. Your answer should be short and concise.
"""


# ============================================================
# Prompt builders (substitution helpers)
# ============================================================
# Pagination/gisting use positional str.format on brace-clean templates (the
# substituted values are never re-scanned, so doc text containing braces is safe —
# matching upstream's own `.format` usage). The named-slot lookup/answer templates
# use str.replace (the repo's convention for injecting user content into a template;
# cf. loong/judge.py `_format_question`), which is brace-agnostic by construction.


def pagination_prompt(preceding: str, passage: str, end_tag: str) -> str:
    return PAGINATION_TEMPLATE.format(preceding, passage, end_tag)


def gisting_prompt(page_text: str, *, token_hint: int | None = None) -> str:
    """Build the gisting prompt. ``token_hint=None`` → the original app.py prompt (NO length
    clause, byte-exact to Loong's v2 runs); an int → the notebook's "should be in N tokens"
    length-capped variant (CorpusQA v4). The page text is a substituted value, so braces in it
    are preserved literally (str.format only scans the template)."""
    if token_hint is None:
        return GISTING_TEMPLATE.format(page_text)
    return GISTING_TEMPLATE_WITH_HINT.format(token_hint, page_text)


def parallel_lookup_prompt(concatenated_gists: str, question: str) -> str:
    return (
        PARALLEL_LOOKUP_TEMPLATE
        .replace("{concatenated_gists}", concatenated_gists)
        .replace("{question}", question)
    )


def answer_prompt(concatenated_pages_and_gists: str, question: str) -> str:
    return (
        ANSWER_TEMPLATE
        .replace("{concatenated_pages_and_gists}", concatenated_pages_and_gists)
        .replace("{question}", question)
    )


# ============================================================
# Response parsers
# ============================================================


def parse_pause_point(text: str) -> int | None:
    """The pagination break-point label, e.g. ``"Break point: <57> ..."`` → ``57``.

    VERBATIM from upstream ``parse_pause_point`` (note ``str.strip("Break point: ")``
    strips that CHARACTER SET, an upstream quirk we preserve), with ONE robustness
    guard added: upstream indexes ``text[0]`` unconditionally and would ``IndexError``
    on an empty/whitespace reply; we return ``None`` instead (see PROVENANCE.md).
    """
    text = text.strip("Break point: ")
    if not text or text[0] != "<":
        return None
    for i, c in enumerate(text):
        if c == ">":
            if text[1:i].isnumeric():
                return int(text[1:i])
            return None
    return None


def parse_parallel_pages(response: str, n_pages: int) -> list[int]:
    """The page ids from a parallel-lookup reply, e.g. ``"... Page [12, 7] ..."`` →
    ``[12, 7]``. VERBATIM from upstream ``quality_parallel_lookup``'s bracket parse:
    take the text between the first ``[`` and ``]``, split on commas, keep the numeric
    entries that fall in ``range(n_pages)``."""
    try:
        start = response.index("[")
    except ValueError:
        start = len(response)
    try:
        end = response.index("]")
    except ValueError:
        end = 0
    page_ids: list[int] = []
    if start < end:
        for p in response[start + 1:end].split(","):
            if p.strip().isnumeric():
                page_id = int(p)
                if 0 <= page_id < n_pages:
                    page_ids.append(page_id)
    return page_ids
