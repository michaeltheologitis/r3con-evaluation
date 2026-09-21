"""ReadAgent's gist-memory algorithm — a faithful re-host of the upstream demo.

Three stages, each a thin reproduction of the corresponding upstream function
(``upstream/app.py`` / ``upstream/read_agent_demo.ipynb``), driven through the
injected ``llm`` seam (``llm.complete(prompt) -> str``):

1. **Episode Pagination** (``paginate``) — slide a ``word_limit``-word window over
   the document's paragraphs; once ``start_threshold`` words are accumulated, insert
   ``<j>`` break-candidate labels; ask the LLM for a natural break point. One LLM
   call per page; a sub-``_SHORT_PAGE_WORDS`` remainder becomes one page with no call.
   Faithful to upstream ``quality_pagination``.
2. **Memory Gisting** (``gist_pages``) — one LLM call per page to shorten it into a
   gist. Faithful to upstream ``quality_gisting``.
3. **Look-up + Answer** (``parallel_lookup`` + ``answer``) — from the gist memory the
   agent names pages to re-read in ONE call (ReadAgent-P), those gists are swapped for
   their full page text, then it answers. ReadAgent-P is the only look-up variant
   upstream implements in code (see PROVENANCE.md D5).

DEVIATIONS (all also in PROVENANCE.md):
- **Document prep** (``_to_paragraphs``): upstream's ``quality_gutenberg_parser`` is
  QuALITY-Gutenberg-specific. We generalize to arbitrary documents — split on blank
  lines into paragraphs, collapse intra-paragraph whitespace (as the parser did when
  joining lines), and hard-split any paragraph longer than ``word_limit`` words into
  word windows so pagination always has bounded units.
- **Multi-document** (``paginate_documents``): Loong/CorpusQA give a per-instance
  document BUNDLE and Dracula the same 46-doc corpus for every question; ReadAgent
  assumes a single article. We paginate each document independently and pool the pages
  in document order, so a page never spans two documents (the same
  per-document-then-pool convention the other multi-doc baselines use). Page numbering
  is global across the pooled list.
"""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from evals.baselines.readagent import prompts

# Pagination hyperparameters — the ReadAgent-native ("original") defaults: ~600-word pages, no gist
# length clause. These size the DEFAULT ``Regime`` (below) and the low-level functions' default
# params. The regime is selected AUTOMATICALLY per benchmark in ``run.py`` (loong + dracula use
# these; CorpusQA scales them ×10 with a 640-token gist hint) — nothing to flip before a run.
# See PROVENANCE D10.
_WORD_LIMIT = 600          # target words/page → ~90–126 pages on a Loong doc
_START_THRESHOLD = 280     # words before <j> break-candidate labels start appearing
_SHORT_PAGE_WORDS = 350    # a remaining window below this is one page, no LLM call

# Look-up budget — ReadAgent-P: the paper's free-form (NarrativeQA/QMSum) setting is
# "1 or 2 pages"; we cap at 2 (the prompt text says the same). Documented in PROVENANCE.md.
_MAX_LOOKUP_PAGES = 2


@dataclass(frozen=True)
class Regime:
    """The per-benchmark pagination/gist sizing — selected by benchmark in ``run.py`` (PROVENANCE D10).

    ``word_limit`` / ``start_threshold`` / ``short_page_words`` size the pages (bigger pages → fewer
    pages → fewer LLM calls, for corpora far larger than ReadAgent's validated range). ``gist_token_hint``
    is ``None`` → the gist prompt carries **NO** length clause (the original ReadAgent / Loong-v2
    prompt); an int → the "... should be in N tokens" clause (CorpusQA v4). Immutable + threaded
    explicitly through the pipeline, so there is no mutable global to flip per run.
    """
    word_limit: int = _WORD_LIMIT
    start_threshold: int = _START_THRESHOLD
    short_page_words: int = _SHORT_PAGE_WORDS
    gist_token_hint: int | None = None


# The ReadAgent-native default (= what Loong and Dracula run, labelled v2). ``run.py`` maps
# each benchmark to a regime; this is the fallback when a function is called without one.
DEFAULT_REGIME = Regime()


# CJK ideograph ranges (Unified + Extension A + Compatibility Ideographs) — covers
# essentially all modern Chinese Han text. Used to make text-length measurement
# language-aware (see ``count_words``); the algorithm/prompts/thresholds are unchanged.
_CJK_RE = re.compile(r"[㐀-鿿豈-﫿]")
# CJK sentence-ending punctuation, for splitting an over-long CJK block into pieces.
_CJK_SENT_RE = re.compile(r"(?<=[。！？；])")


def count_words(text: str) -> int:
    """Approximate text length in "words" — language-aware (D2 in PROVENANCE).

    Upstream's ``count_words`` is ``len(text.split())`` (whitespace words), which
    pagination uses to size pages (~``word_limit`` units/page). That is exactly right
    for ReadAgent's English-only datasets but collapses on Chinese — no spaces, so a
    whole document counts as a handful of "words" and pagination never fires. To honor
    the authors' INTENT cross-lingually (a page is ~N units of text) we count each CJK
    ideograph as one unit and whitespace-split the rest.

    For text with NO CJK characters this returns ``len(text.split())`` exactly — so all
    English behaviour (and every English result) is unchanged; only CJK text is affected.
    """
    n_cjk = len(_CJK_RE.findall(text))
    if not n_cjk:
        return len(text.split())                       # English path: upstream-identical
    return n_cjk + len(_CJK_RE.sub(" ", text).split())  # CJK ideographs + any latin words


def _split_cjk_block(block: str, word_limit: int = _WORD_LIMIT) -> list[str]:
    """Pack an over-long CJK block into ≤``word_limit``-unit pieces, preferring CJK
    sentence boundaries (。！？；); a single run longer than the budget (no sentence
    breaks) is hard-split into character windows. Only reached for CJK text — see
    ``_to_paragraphs``."""
    pieces: list[str] = []
    cur, cur_n = "", 0
    for sentence in _CJK_SENT_RE.split(block):
        if not sentence:
            continue
        n = count_words(sentence)
        if n > word_limit:                               # one giant unpunctuated run
            if cur:
                pieces.append(cur)
                cur, cur_n = "", 0
            pieces.extend(sentence[k:k + word_limit] for k in range(0, len(sentence), word_limit))
            continue
        if cur and cur_n + n > word_limit:
            pieces.append(cur)
            cur, cur_n = "", 0
        cur += sentence
        cur_n += n
    if cur:
        pieces.append(cur)
    return pieces


def _to_paragraphs(document: str, word_limit: int = _WORD_LIMIT) -> list[str]:
    """Split a document into pagination units (paragraphs).

    Generalizes upstream's QuALITY-specific ``quality_gutenberg_parser``: split on
    blank lines, collapse each block's internal whitespace (as the parser did when it
    joined lines with spaces), and split an over-long block so a single giant block
    can't become one unbounded page. English over-long blocks word-window exactly as
    before; a CJK block (few whitespace tokens but many ideograph units) splits at CJK
    sentence boundaries (``_split_cjk_block``). The third branch is **CJK-only by
    construction** (for non-CJK text ``count_words == len(words)``), so English
    pagination is byte-for-byte unchanged. ``word_limit`` sizes the units (the active
    ``Regime``'s value; default = the original 600).
    """
    paragraphs: list[str] = []
    for block in re.split(r"\n\s*\n", document):
        block = " ".join(block.split())
        if not block:
            continue
        words = block.split()
        if count_words(block) <= word_limit:
            paragraphs.append(block)
        elif len(words) > word_limit:                    # English over-long: word-window (upstream)
            for k in range(0, len(words), word_limit):
                paragraphs.append(" ".join(words[k:k + word_limit]))
        else:                                            # CJK over-long: sentence-split
            paragraphs.extend(_split_cjk_block(block, word_limit))
    return paragraphs


def paginate(
    paragraphs: list[str],
    llm: Any,
    *,
    word_limit: int = _WORD_LIMIT,
    start_threshold: int = _START_THRESHOLD,
    short_page_words: int = _SHORT_PAGE_WORDS,
    allow_fallback_to_last: bool = True,
    progress: Any = None,
) -> list[list[str]]:
    """Group paragraphs into pages at LLM-chosen natural break points.

    Faithful re-host of upstream ``quality_pagination``'s loop (minus its QuALITY
    plumbing/printing). Each page is the list of paragraphs ``paragraphs[i:pause]``.
    ``word_limit`` / ``start_threshold`` / ``short_page_words`` are the active ``Regime``'s
    sizes (defaults = the original 600 / 280 / 350).
    """
    i = 0
    pages: list[list[str]] = []
    n = len(paragraphs)
    while i < n:
        preceding = "" if i == 0 else "...\n" + "\n".join(pages[-1])
        passage = [paragraphs[i]]
        wcount = count_words(paragraphs[i])
        j = i + 1
        while wcount < word_limit and j < n:
            wcount += count_words(paragraphs[j])
            if wcount >= start_threshold:
                passage.append(f"<{j}>")
            passage.append(paragraphs[j])
            j += 1
        passage.append(f"<{j}>")
        end_tag = "" if j == n else paragraphs[j] + "\n..."

        if wcount < short_page_words:
            pause_point: int | None = n
        else:
            prompt = prompts.pagination_prompt(preceding, "\n".join(passage), end_tag)
            response = llm.complete(prompt).strip()
            pause_point = prompts.parse_pause_point(response)
            if pause_point and (pause_point <= i or pause_point > j):
                pause_point = None  # out-of-range label → fall back
            if pause_point is None:
                if allow_fallback_to_last:
                    pause_point = j
                else:
                    raise ValueError(f"pagination produced no valid break point:\n{response}")

        pages.append(paragraphs[i:pause_point])
        i = pause_point
        if progress is not None:
            progress.page_done()  # live progress.json tick (one per paginated page)
    return pages


def paginate_documents(
    documents: list[str], llm: Any, *, regime: Regime = DEFAULT_REGIME,
    progress: Any = None, max_workers: int = 1,
) -> list[list[str]]:
    """Paginate each document independently and pool the pages in document order.

    A page never spans two documents (the per-document-then-pool convention). An
    empty document contributes no pages. Page indices are global across the pool.
    ``regime`` supplies the page sizes (default = the original 600 / 280 / 350).

    Documents are independent, so with ``max_workers > 1`` they paginate CONCURRENTLY
    (within a document pagination stays sequential — each page's break point sets the next
    page's start). ``executor.map`` preserves document order, so the pooled page list is
    byte-identical to the sequential version — a throughput optimization, not a method change
    (PROVENANCE). Default 1 = the upstream sequential loop.
    """
    def _paginate_one(document: str) -> list[list[str]]:
        paragraphs = _to_paragraphs(document, regime.word_limit)
        if not paragraphs:
            return []
        return paginate(paragraphs, llm, word_limit=regime.word_limit,
                        start_threshold=regime.start_threshold,
                        short_page_words=regime.short_page_words, progress=progress)

    if max_workers <= 1 or len(documents) <= 1:
        per_doc = [_paginate_one(d) for d in documents]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            per_doc = list(ex.map(_paginate_one, documents))
    pages: list[list[str]] = []
    for doc_pages in per_doc:
        pages.extend(doc_pages)
    return pages


def gist_pages(
    pages: list[list[str]], llm: Any, *, gist_token_hint: int | None = None,
    progress: Any = None, max_workers: int = 1,
) -> list[str]:
    """One gist (shortened page) per page — upstream ``quality_gisting``.

    ``gist_token_hint`` (the active ``Regime``'s value) selects the gist prompt: ``None`` → the
    original no-length-clause prompt (Loong / Dracula, v2); an int → the "should be in N tokens"
    variant (CorpusQA v4). Gists are INDEPENDENT per page, so with ``max_workers > 1`` they are
    computed CONCURRENTLY; ``executor.map`` returns them in PAGE ORDER, so the gist list is
    byte-identical to the sequential version — a throughput optimization, not a method change
    (PROVENANCE). Default 1 = the upstream sequential loop.
    """
    def _gist_one(page: list[str]) -> str:
        gist = llm.complete(prompts.gisting_prompt("\n".join(page), token_hint=gist_token_hint)).strip()
        if progress is not None:
            progress.gist_done()  # live progress.json tick (one per gisted page)
        return gist

    if max_workers <= 1 or len(pages) <= 1:
        return [_gist_one(page) for page in pages]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_gist_one, pages))


def build_gist_memory(
    documents: list[str], llm: Any, *, regime: Regime = DEFAULT_REGIME,
    progress: Any = None, gist_workers: int = 1,
) -> dict[str, Any]:
    """The gist memory for a document bundle: paginate (pooled) then gist each page.

    Returns ``{"pages": [[para, ...], ...], "gists": [str, ...], "n_docs": int}``.
    Both lists are aligned (one gist per page). All LLM calls land on ``llm``.
    ``regime`` (selected per benchmark in ``run.py``) sizes the pages AND selects the gist
    prompt (``gist_token_hint``); default = the original Loong/Dracula regime.

    ``gist_workers`` bounds INNER-task concurrency for BOTH phases — pagination across
    documents and gisting across pages (default 1 = the upstream sequential loops). The output
    is order-preserved and IDENTICAL regardless of the value, so it's a pure throughput knob
    (PROVENANCE). When > 1 the seam (``llm``) and ``progress`` are called from worker threads,
    so they must be thread-safe — ``ReadAgentLLM`` and ``run._Progress`` lock their shared state.

    An optional ``progress`` object (duck-typed ``.page_done()`` / ``.gist_done()`` /
    ``.stage(name, **extra)``) receives live ticks so a long run's stage + counts are
    observable mid-flight (see ``run._Progress``); ``None`` (the default) is a no-op."""
    pages = paginate_documents(documents, llm, regime=regime, progress=progress, max_workers=gist_workers)
    if progress is not None:
        progress.stage("gisting", n_pages=len(pages))
    gists = gist_pages(pages, llm, gist_token_hint=regime.gist_token_hint,
                       progress=progress, max_workers=gist_workers)
    return {"pages": pages, "gists": gists, "n_docs": len(documents)}


def format_gist_memory(gists: list[str]) -> str:
    """The gist memory the look-up prompt sees: each gist under a ``<Page i>`` marker
    (the notebook's ``"<Page {}>\\n" + gist`` form), so the model can name page ids."""
    return "\n".join(f"<Page {i}>\n{gist}" for i, gist in enumerate(gists))


def parallel_lookup(
    gists: list[str], question: str, llm: Any, *, max_pages: int = _MAX_LOOKUP_PAGES
) -> tuple[list[int], str | None]:
    """ReadAgent-P: ONE call names all pages to re-read. Returns ``(page_ids, reply)``.

    Faithful to upstream ``quality_parallel_lookup``'s single lookup call + bracket
    parse; the only addition is the ``max_pages`` cap (kept = the prompt's "1 or 2
    pages"), so a misbehaving model can't request an unbounded page set.
    """
    if not gists:
        return [], None
    reply = llm.complete(prompts.parallel_lookup_prompt(format_gist_memory(gists), question)).strip()
    page_ids = prompts.parse_parallel_pages(reply, len(gists))
    return page_ids[:max_pages], reply


def assemble_expanded(pages: list[list[str]], gists: list[str], page_ids: list[int]) -> str:
    """The article the ANSWER prompt sees: the gist memory with each looked-up page
    replaced by its FULL text. Faithful to the notebook's memory expansion (the bare
    gist list, no ``<Page i>`` markers — those are only for the look-up prompt)."""
    expanded = list(gists)
    for page_id in page_ids:
        if 0 <= page_id < len(pages):
            expanded[page_id] = "\n".join(pages[page_id])
    return "\n".join(expanded)


def answer(expanded_article: str, question: str, llm: Any) -> str:
    """The final answer call — upstream ``prompt_answer_template``. Returns the raw
    model output verbatim (grading is deferred to the benchmark judge at score time)."""
    return llm.complete(prompts.answer_prompt(expanded_article, question))
