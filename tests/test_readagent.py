"""Tests for the ReadAgent baseline connector.

ReadAgent (Google DeepMind, ICML 2024; arXiv 2402.09727) ships only demo artifacts
(``app.py`` + a notebook, vendored under ``evals/baselines/readagent/upstream/``); the
connector REPRODUCES its three prompting stages faithfully. The harness-side pieces are
tested here with fakes:

- ``prompts`` — the verbatim parsers (pause point, parallel page list).
- ``gist``   — the method: paragraph prep, pagination (incl. multi-doc pooling), gisting,
               ReadAgent-P look-up, memory expansion.
- ``llm``    — ReadAgentLLM, the single-user-message seam over litellm (usage + calls).
- ``run``    — run_one's pipeline + the per-run-folder (TOTAL-cost, no-reuse) logging.

Only ReadAgent-P (parallel look-up) is wired — it is the only look-up variant upstream
implements in code (see PROVENANCE.md D5). The ``readagent`` marker gates a real end-to-end
run against gpt-5.4-nano (deselected by default; see pyproject addopts).
"""
from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import pytest

from evals.baselines.readagent import gist, prompts


# ============================================================
# prompts — verbatim parsers + brace-safe builders
# ============================================================


def test_parse_pause_point_extracts_label() -> None:
    assert prompts.parse_pause_point("Break point: <57>\n Because ...") == 57


def test_parse_pause_point_non_numeric_or_empty_is_none() -> None:
    # Upstream would IndexError on an empty reply; our guard returns None instead.
    assert prompts.parse_pause_point("") is None
    assert prompts.parse_pause_point("   ") is None
    assert prompts.parse_pause_point("I think we should not break here") is None
    assert prompts.parse_pause_point("<abc>") is None


def test_parse_parallel_pages_bracket_list_in_range() -> None:
    assert prompts.parse_parallel_pages("I want to look up Page [12, 7] to ...", n_pages=20) == [12, 7]
    # Out-of-range and non-numeric entries are dropped (upstream behaviour).
    assert prompts.parse_parallel_pages("Page [2, 99, x, 3]", n_pages=5) == [2, 3]
    assert prompts.parse_parallel_pages("no brackets here", n_pages=5) == []


def test_prompt_builders_dont_break_on_braces_in_content() -> None:
    # Document/question text may contain braces (CorpusQA tables, JSON golds). The named-slot
    # templates use str.replace, so braces in the substituted content are preserved literally.
    gists = "<Page 0>\nrevenue = {a: 1, b: 2}"
    question = "what is f({x})?"
    p = prompts.parallel_lookup_prompt(gists, question)
    assert "{a: 1, b: 2}" in p and "f({x})" in p
    a = prompts.answer_prompt("text with {braces}", question)
    assert "{braces}" in a and "f({x})" in a


def test_gisting_prompt_no_hint_is_byte_exact_loong_v2() -> None:
    # The ORIGINAL ReadAgent / Loong-v2 gist prompt carries NO "should be in N tokens" length
    # clause. Loong's 1542 existing runs used exactly this — pin it byte-for-byte so a future
    # edit can't silently change what a resumed / re-measured Loong run sends.
    assert prompts.gisting_prompt("PASSAGE") == (
        "\nPlease shorten the following passage.\n"
        "Just give me a shortened version. DO NOT explain your reason.\n\n"
        "Passage:\nPASSAGE\n\n"
    )


def test_gisting_prompt_with_hint_adds_length_clause() -> None:
    # The CorpusQA (v4) variant: the SAME prompt plus the notebook's length clause.
    p = prompts.gisting_prompt("PASSAGE", token_hint=640)
    assert "The shortened passage should be in 640 tokens." in p
    assert p.endswith("Passage:\nPASSAGE\n\n")
    # Braces in the page text survive (it's a substituted value, not re-scanned).
    assert "{x: 1}" in prompts.gisting_prompt("data {x: 1}", token_hint=640)


# ============================================================
# gist — the method (driven by a scripted fake LLM)
# ============================================================


class _ScriptLLM:
    """Duck-typed seam: returns canned responses in order, recording prompts."""

    def __init__(self, responses=None):
        self._responses = list(responses or [])
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._responses.pop(0) if self._responses else ""


class _ThreadSafeLLM:
    """Thread-safe duck-typed seam for the parallel tests: returns ``fn(prompt)`` and records every
    prompt under a lock, so a parallel run can assert no call was lost to a race."""

    def __init__(self, fn):
        self._fn = fn
        self._lock = threading.Lock()
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.prompts.append(prompt)
        return self._fn(prompt)


def test_to_paragraphs_splits_blank_lines_and_collapses_whitespace() -> None:
    assert gist._to_paragraphs("alpha\n\nbeta") == ["alpha", "beta"]
    # Intra-paragraph newlines/extra spaces collapse to single spaces.
    assert gist._to_paragraphs("alpha\n  line two\n\ngamma") == ["alpha line two", "gamma"]


def test_to_paragraphs_word_windows_oversized_block() -> None:
    wl = gist._WORD_LIMIT
    big = " ".join(f"w{i}" for i in range(wl * 2 + wl // 2))  # 2.5× word_limit, one block, no blanks
    paras = gist._to_paragraphs(big)
    # Hard-split into <=word_limit windows: wl + wl + wl//2 (relative to the active _WORD_LIMIT).
    assert [len(p.split()) for p in paras] == [wl, wl, wl // 2]


# --- CJK-aware length (D2): Chinese pagination, English provably unchanged ---


def test_count_words_english_unchanged_cjk_counts_ideographs() -> None:
    # English: byte-identical to upstream's whitespace split (every EN result preserved).
    assert gist.count_words("one two three four") == 4
    assert gist.count_words("  spaced   out  ") == 2
    zh = "这是一个测试句子"  # 8 ideographs, no spaces
    assert len(zh.split()) == 1        # upstream's degenerate count (the bug)
    assert gist.count_words(zh) == 8   # CJK-aware: one unit per ideograph
    # Mixed: ideographs + latin words both counted (4 hanzi + {2024, revenue}).
    assert gist.count_words("报告 2024 revenue 数据") == 4 + 2


def test_to_paragraphs_splits_long_cjk_single_block() -> None:
    # A single-block ZH doc (no blank lines) with sentence punctuation — the real Loong-ZH
    # shape. Upstream's whitespace count_words would keep it as ONE giant paragraph; the
    # CJK-aware split must produce several, each within the unit budget.
    sentence = "这是一段很长的测试中文文本用来检验分页功能是否能够正常地工作。"  # ~29 ideographs + 。
    reps = gist._WORD_LIMIT // gist.count_words(sentence) + 5  # enough to exceed _WORD_LIMIT
    doc = sentence * reps  # one block, no blank lines, > _WORD_LIMIT units
    paras = gist._to_paragraphs(doc)
    assert len(paras) > 1
    assert all(gist.count_words(p) <= gist._WORD_LIMIT for p in paras)
    # No content lost (every sentence-ending 。 is preserved across the pieces).
    assert sum(p.count("。") for p in paras) == doc.count("。")


def test_paginate_documents_now_paginates_chinese() -> None:
    sentence = "这是测试句子需要被正确地分页并压缩成简短的摘要内容。"
    reps = gist._WORD_LIMIT // gist.count_words(sentence) + 10  # > _WORD_LIMIT → splits into paragraphs
    doc = sentence * reps  # long single-block Chinese doc
    llm = _ScriptLLM(["Break point: <1>"] * 1000)
    pages = gist.paginate_documents([doc], llm)
    assert len(pages) > 1   # actually paginated (not one un-paginated page)
    assert llm.prompts      # the pagination LLM WAS consulted (no longer short-circuited)


def test_paginate_short_doc_is_one_page_no_llm_call() -> None:
    paras = ["one two three", "four five six"]  # < _SHORT_PAGE_WORDS → no break-point call
    llm = _ScriptLLM()
    pages = gist.paginate(paras, llm)
    assert pages == [paras]
    assert llm.prompts == []  # the short-circuit fired (no LLM)


def test_paginate_multipage_preserves_all_paragraphs_in_order() -> None:
    # Paragraphs totalling > _WORD_LIMIT (relative) force the LLM break-point path (multi-page).
    psize = gist._WORD_LIMIT // 4
    paras = [" ".join(f"p{d}w{i}" for i in range(psize)) for d in range(6)]  # ~1.5× word_limit total
    llm = _ScriptLLM(["Break point: <2>"] * 50)  # always break at an offered label
    pages = gist.paginate(paras, llm)
    assert len(pages) > 1                       # actually paginated
    assert [p for page in pages for p in page] == paras  # no loss / reordering
    assert llm.prompts                          # the LLM was consulted


def test_paginate_documents_pools_without_spanning_docs() -> None:
    docs = ["alpha one two.", "beta three four."]  # each short → one page, no LLM
    llm = _ScriptLLM()
    pages = gist.paginate_documents(docs, llm)
    assert len(pages) == 2
    assert "alpha" in " ".join(pages[0]) and "beta" in " ".join(pages[1])
    # No page mixes the two documents.
    assert not any("alpha" in " ".join(pg) and "beta" in " ".join(pg) for pg in pages)


def test_gist_pages_one_gist_per_page() -> None:
    pages = [["page zero text"], ["page one text"]]
    llm = _ScriptLLM(["gist0", "gist1"])
    assert gist.gist_pages(pages, llm) == ["gist0", "gist1"]
    assert all("shorten" in p for p in llm.prompts)  # used the gisting prompt


def test_gist_pages_threads_token_hint_into_prompt() -> None:
    pages = [["p0"], ["p1"]]
    # Default (no hint) → the original no-clause prompt.
    llm = _ScriptLLM(["g0", "g1"])
    gist.gist_pages(pages, llm)
    assert all("should be in" not in p for p in llm.prompts)
    # gist_token_hint set → the length clause reaches every gist prompt.
    llm2 = _ScriptLLM(["g0", "g1"])
    gist.gist_pages(pages, llm2, gist_token_hint=640)
    assert all("should be in 640 tokens" in p for p in llm2.prompts)


def test_build_gist_memory_applies_regime_sizes_and_gist_hint() -> None:
    # The regime is a single object threaded through the whole build (pagination sizes + gist hint),
    # so there is no mutable global to flip. Small docs short-circuit pagination (no LLM), so the
    # only LLM calls are the gists — assert the regime's gist hint reached them.
    docs = ["short doc one.", "short doc two."]
    default = _ScriptLLM(["g"] * 10)                       # DEFAULT_REGIME → no clause
    gist.build_gist_memory(docs, default)
    gp_default = [p for p in default.prompts if "shorten" in p]
    assert gp_default and all("should be in" not in p for p in gp_default)

    cq = _ScriptLLM(["g"] * 10)                            # CorpusQA-style regime → 640 clause
    regime = gist.Regime(word_limit=6000, start_threshold=2800, short_page_words=3500, gist_token_hint=640)
    gist.build_gist_memory(docs, cq, regime=regime)
    gp_cq = [p for p in cq.prompts if "shorten" in p]
    assert gp_cq and all("should be in 640 tokens" in p for p in gp_cq)


# --- D9: inner-task parallelism — output is order-preserved + identical to sequential ---


def test_gist_pages_parallel_preserves_page_order() -> None:
    # 20 pages, each with a unique marker; the fake echoes the (page-embedding) gisting prompt.
    # With max_workers > 1 the gists must still come back in PAGE order (executor.map), every page
    # gisted exactly once, and the result identical to the sequential run (the faithfulness guarantee).
    pages = [[f"MARKER-{i}"] for i in range(20)]
    par = gist.gist_pages(pages, _ThreadSafeLLM(lambda p: p), max_workers=8)
    assert len(par) == 20
    assert all(f"MARKER-{i}" in par[i] for i in range(20))        # gist[i] ↔ page[i]
    assert par == gist.gist_pages(pages, _ThreadSafeLLM(lambda p: p), max_workers=1)  # == sequential


def test_paginate_documents_parallel_preserves_doc_order() -> None:
    docs = [f"doc{i} alpha beta gamma." for i in range(12)]       # each short → one page, no LLM call
    seq = gist.paginate_documents(docs, _ThreadSafeLLM(lambda p: "Break point: <1>"), max_workers=1)
    par = gist.paginate_documents(docs, _ThreadSafeLLM(lambda p: "Break point: <1>"), max_workers=4)
    assert par == seq                                            # byte-identical, doc order preserved
    assert all(f"doc{i}" in " ".join(par[i]) for i in range(12))


def test_format_gist_memory_marks_pages() -> None:
    assert gist.format_gist_memory(["a", "b"]) == "<Page 0>\na\n<Page 1>\nb"


def test_parallel_lookup_parses_and_caps_pages() -> None:
    llm = _ScriptLLM(["I want to look up Page [1, 0, 2] to ..."])
    page_ids, reply = gist.parallel_lookup(["g0", "g1", "g2"], "q", llm, max_pages=2)
    assert page_ids == [1, 0]               # capped at max_pages, order preserved
    assert reply and len(llm.prompts) == 1


def test_parallel_lookup_empty_memory_skips_call() -> None:
    llm = _ScriptLLM()
    assert gist.parallel_lookup([], "q", llm) == ([], None)
    assert llm.prompts == []


def test_assemble_expanded_replaces_looked_up_pages_with_full_text() -> None:
    pages = [["full page zero line a", "full page zero line b"], ["full page one"]]
    gists = ["gist0", "gist1"]
    expanded = gist.assemble_expanded(pages, gists, [0])
    # Page 0 → its full text (joined); page 1 stays a gist. No <Page i> markers.
    assert "full page zero line a\nfull page zero line b" in expanded
    assert "gist1" in expanded and "gist0" not in expanded


# ============================================================
# llm — ReadAgentLLM seam over litellm
# ============================================================

from evals.baselines.readagent import llm as ra_llm  # noqa: E402


def _fake_response(content, *, model="gpt-5.4-nano", ptoks=11, ctoks=7, reasoning=None):
    usage = SimpleNamespace(
        prompt_tokens=ptoks, completion_tokens=ctoks, total_tokens=ptoks + ctoks,
        model_dump=lambda: {"prompt_tokens": ptoks, "completion_tokens": ctoks,
                            "total_tokens": ptoks + ctoks},
    )
    msg = SimpleNamespace(content=content, reasoning_content=reasoning)
    return SimpleNamespace(
        model=model, usage=usage, choices=[SimpleNamespace(message=msg)],
        model_dump=lambda: {"model": model, "choices": [{"message": {"content": content}}]},
    )


def test_readagentllm_sends_one_user_message_no_system(monkeypatch) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return _fake_response("ok")

    import litellm
    monkeypatch.setattr(litellm, "completion", fake_completion)
    llm = ra_llm.ReadAgentLLM(
        {"model": "openai/gpt-5.4-nano", "api_base": "http://x/v1", "api_key": "k"},
        seed=42, completion_params={"temperature": 0.3},
    )
    assert llm.complete("hello") == "ok"
    # One user message (no system role); no max_tokens sent (cap removed — vLLM fills the
    # remaining window); seed + sampling forwarded.
    assert captured["messages"] == [{"role": "user", "content": "hello"}]
    assert "max_tokens" not in captured
    assert captured["seed"] == 42 and captured["temperature"] == 0.3
    assert captured["api_base"] == "http://x/v1" and captured["api_key"] == "k"


def test_readagentllm_accumulates_total_usage_and_full_calls(monkeypatch) -> None:
    import litellm
    responses = iter([_fake_response("a", ptoks=10, ctoks=5),
                      _fake_response("b", ptoks=20, ctoks=8, reasoning="thinking...")])
    monkeypatch.setattr(litellm, "completion", lambda **kw: next(responses))
    llm = ra_llm.ReadAgentLLM({"model": "openai/gpt-5.4-nano"})
    llm.complete("p1")
    llm.complete("p2")
    usage = llm.usage
    assert usage["total"]["gpt-5.4-nano"]["num_calls"] == 2
    assert usage["total"]["gpt-5.4-nano"]["total_tokens"] == (15 + 28)
    # calls.json content: per-call content + reasoning surfaced explicitly.
    assert [c["content"] for c in llm.full_calls] == ["a", "b"]
    assert llm.full_calls[1]["reasoning_content"] == "thinking..."
    assert llm.full_calls[0]["request"]["prompt"] == "p1"


def test_readagentllm_thread_safe_under_parallel_gisting(monkeypatch) -> None:
    # Drive the REAL ReadAgentLLM from 8 worker threads (parallel gist_pages); the seam's lock must
    # capture EVERY call's usage + record — none lost to a race on the shared lists/rollup.
    import litellm
    monkeypatch.setattr(litellm, "completion", lambda **kw: _fake_response("g", ptoks=3, ctoks=1))
    llm = ra_llm.ReadAgentLLM({"model": "openai/gpt-5.4-nano"})
    pages = [[f"p{i}"] for i in range(40)]
    gists = gist.gist_pages(pages, llm, max_workers=8)
    assert len(gists) == 40
    assert len(llm.full_calls) == 40                              # no record dropped
    assert llm.usage["total"]["gpt-5.4-nano"]["num_calls"] == 40  # no usage lost
    assert llm.usage["total"]["gpt-5.4-nano"]["total_tokens"] == 40 * 4


# ============================================================
# run — run_one pipeline + per-run-folder (no-reuse, TOTAL cost) logging
# ============================================================

from evals.baselines.readagent import run as ra_run  # noqa: E402


def _loong_benchmark():
    """A loong-shaped fake benchmark (small docs → no pagination LLM call)."""
    return SimpleNamespace(
        __name__="evals.benchmarks.loong",
        get_documents=lambda tid: ["Paris is the capital of France.",
                                    "Berlin is the capital of Germany."],
        get_task=lambda tid: ("Answer using the documents.", "What is the capital of France?", []),
        get_task_metadata=lambda tid: {"language": "en"},
    )


def _corpusqa_benchmark():
    """A corpusqa-shaped fake benchmark (small docs → no pagination LLM call). run_one selects the
    CorpusQA (v4) regime for it purely from the module name — the mechanism under test."""
    return SimpleNamespace(
        __name__="evals.benchmarks.corpusqa",
        get_documents=lambda tid: ["Firm A grew revenue 10%.", "Firm B shrank 5%."],
        get_task=lambda tid: ("Output requirements: The answer is: a list.", "Which firms grew?", []),
        get_task_metadata=lambda tid: {"domain": "financial_en"},
    )


class _StageCompletion:
    """Stage-aware fake litellm.completion driving the whole ReadAgent-P pipeline."""

    def __init__(self):
        self.stages: list[str] = []

    def __call__(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "Please shorten the following passage" in prompt:
            stage, content = "gist", "a short gist"
        elif "in the order of importance" in prompt:               # ReadAgent-P look-up
            stage, content = "lookup_p", "I want to look up Page [0] to verify."
        elif "Answer the question based on the above" in prompt:    # answer
            stage, content = "answer", "The answer is: Paris."
        else:
            stage, content = "other", "ok"
        self.stages.append(stage)
        return _fake_response(content)


def _capture_completion(monkeypatch):
    """Monkeypatch litellm.completion with the stage-aware fake that ALSO records every prompt
    sent; returns the running list of prompts."""
    import litellm
    seen: list[str] = []
    stage = _StageCompletion()

    def capture(**kwargs):
        seen.append(kwargs["messages"][-1]["content"])
        return stage(**kwargs)

    monkeypatch.setattr(litellm, "completion", capture)
    return seen


def test_run_one_pipeline_and_per_run_folder_logging(tmp_path, monkeypatch) -> None:
    import litellm
    monkeypatch.setattr(litellm, "completion", _StageCompletion())

    run_config = {"benchmark": "loong", "baseline": "readagent", "model": "gpt-5-4-nano",
                  "seed": 42, "run_version": ra_run.run_version_for("loong")}
    rec = ra_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path,
                         {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None})

    assert rec["raw_answer"] == "The answer is: Paris."
    # No content-addressed index (the flat per-run layout): the record carries no index_ref.
    assert "index_ref" not in rec
    # The gist memory is persisted INSIDE the run folder.
    memory = json.loads((tmp_path / "gist_memory.json").read_text())
    assert memory["n_docs"] == 2 and len(memory["pages"]) == 2 and len(memory["gists"]) == 2
    # TOTAL usage (gisting + look-up + answer all on one seam) + the full call trace.
    assert rec["usage"]["total"] and rec["calls_full"]
    assert rec["trace"]["n_pages"] == 2 and rec["trace"]["n_docs"] == 2
    # The ReadAgent-P look-up named page 0; the trace records it + the reply.
    assert rec["trace"]["looked_up_pages"] == [0]
    assert rec["trace"]["lookup_reply"] is not None
    # The task posed is loong's instruction + question.
    assert "capital of France" in rec["trace"]["task"]
    # Live progress.json was written + reached the final stage, with per-stage counts.
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["stage"] == "answered" and progress["task_id"] == "t1"
    assert progress["n_docs"] == 2 and progress["n_pages"] == 2
    assert progress["pages_paginated"] == 2 and progress["pages_gisted"] == 2
    assert progress["looked_up_pages"] == [0] and progress["llm_calls"] >= 1
    assert "elapsed_s" in progress


def test_run_one_threads_gist_workers_and_records_it(tmp_path, monkeypatch) -> None:
    # gist_workers rides in litellm_kwargs (a runtime knob, NOT run_config); run_one threads it into
    # the build (exercising the parallel path + the seam/progress locks) and records it in the trace.
    import litellm
    monkeypatch.setattr(litellm, "completion", _StageCompletion())
    run_config = {"benchmark": "loong", "baseline": "readagent", "model": "gpt-5-4-nano",
                  "seed": 42, "run_version": ra_run.run_version_for("loong")}
    rec = ra_run.run_one(_loong_benchmark(), "t1", run_config, tmp_path,
                         {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None,
                          "gist_workers": 4})
    assert rec["raw_answer"] == "The answer is: Paris."
    assert rec["trace"]["gist_workers"] == 4
    assert "gist_workers" not in run_config   # never added to the config identity (resumption-safe)


def test_run_one_selects_regime_by_benchmark_end_to_end(tmp_path, monkeypatch) -> None:
    # The crux of the change: run_one picks the pagination/gist regime FROM the benchmark (no manual
    # flip of a module global). loong → the original no-length-clause gist prompt; corpusqa → the
    # 640-token variant. The trace records the auto-selected regime for provenance.
    seen = _capture_completion(monkeypatch)
    loong_cfg = {"benchmark": "loong", "baseline": "readagent", "model": "m", "seed": 42,
                 "run_version": ra_run.run_version_for("loong")}
    rec = ra_run.run_one(_loong_benchmark(), "t1", loong_cfg, tmp_path / "loong",
                         {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None})
    loong_gists = [p for p in seen if "Please shorten" in p]
    assert loong_gists and all("should be in" not in p for p in loong_gists)
    assert rec["trace"]["regime"] == {"word_limit": 600, "start_threshold": 280,
                                      "short_page_words": 350, "gist_token_hint": None}

    seen2 = _capture_completion(monkeypatch)
    cq_cfg = {"benchmark": "corpusqa", "baseline": "readagent", "model": "m", "seed": 42,
              "run_version": ra_run.run_version_for("corpusqa")}
    rec2 = ra_run.run_one(_corpusqa_benchmark(), "t1", cq_cfg, tmp_path / "cq",
                          {"model": "openai/gpt-5.4-nano", "api_base": None, "api_key": None})
    cq_gists = [p for p in seen2 if "Please shorten" in p]
    assert cq_gists and all("should be in 640 tokens" in p for p in cq_gists)
    assert rec2["trace"]["regime"]["word_limit"] == 6000 and rec2["trace"]["regime"]["gist_token_hint"] == 640


# ============================================================
# runner — config shape
# ============================================================

from evals.baselines.readagent import runner as ra_runner  # noqa: E402


def test_build_run_config_shape() -> None:
    args = ra_runner.build_arg_parser().parse_args(["--benchmark", "corpusqa", "--seed", "7"])
    cfg = ra_runner.build_run_config(args)
    assert cfg["benchmark"] == "corpusqa" and cfg["baseline"] == "readagent"
    # corpusqa keeps its "v4" regime label (its existing runs must stay valid).
    assert cfg["seed"] == 7 and cfg["run_version"] == ra_run.run_version_for("corpusqa") == "v4"
    # ReadAgent-P is the only variant: no lookup_method field in the config.
    assert "lookup_method" not in cfg


def test_regime_and_version_selected_by_benchmark() -> None:
    # loong + dracula → the ORIGINAL regime (600/280/350, no gist clause), labeled "v2".
    # HARD CONSTRAINT: loong stays "v2" so its existing runs stay valid.
    for bench in ("loong", "dracula"):
        r = ra_run._regime_for(bench)
        assert (r.word_limit, r.start_threshold, r.short_page_words, r.gist_token_hint) == (600, 280, 350, None)
        assert ra_run.run_version_for(bench) == "v2"
    # corpusqa → the v4 regime (×10 sizes + a 640-token gist hint), label kept at "v4".
    r = ra_run._regime_for("corpusqa")
    assert (r.word_limit, r.start_threshold, r.short_page_words, r.gist_token_hint) == (6000, 2800, 3500, 640)
    assert ra_run.run_version_for("corpusqa") == "v4"
    # No silent gap: every supported benchmark has both a regime and a version.
    for bench in ra_run.SUPPORTED_BENCHMARKS:
        assert ra_run._regime_for(bench) is not None and ra_run.run_version_for(bench)


def test_build_task_dracula_bare_question() -> None:
    from types import SimpleNamespace
    b = SimpleNamespace(__name__="evals.benchmarks.dracula",
                        get_task=lambda t: ("HOW MANY?", ["d1", "d2"]))
    assert ra_run._build_task(b, "t") == "HOW MANY?"


def test_gist_workers_default_8_in_litellm_kwargs_not_config() -> None:
    args = ra_runner.build_arg_parser().parse_args(["--benchmark", "loong"])
    assert args.gist_workers == 8                                 # inner-parallelism default
    assert ra_runner._litellm_kwargs(args)["gist_workers"] == 8   # threaded via the runtime channel
    assert "gist_workers" not in ra_runner.build_run_config(args) # NOT part of the run identity
    cmd = ra_runner.build_child_cmd(args, "t1", "tag")
    assert cmd[cmd.index("--gist-workers") + 1] == "8"            # forwarded to the child


def test_no_lookup_method_flag() -> None:
    # The variant flag is gone (only ReadAgent-P is wired).
    with pytest.raises(SystemExit):
        ra_runner.build_arg_parser().parse_args(["--benchmark", "loong", "--lookup-method", "parallel"])


# --- resumption: a re-run must NOT redo finished tasks (a real bug, pinned here) ---


def _patch_runner_for_resumption(tmp_path, monkeypatch):
    """Stub the benchmark + base dir + _dispatch so main() can be driven without subprocesses;
    _dispatch simulates a successful run by writing the manifest the resumption scan reads back.
    Returns the list that records every task_id dispatched."""
    from evals.baselines import _common as common
    fake = SimpleNamespace(get_task_ids=lambda **k: ["t1", "t2", "t3"], STARTER_FILTER={})
    monkeypatch.setattr(common, "load_benchmark_module", lambda n: fake)
    monkeypatch.setattr(common, "base_dir", lambda b, bl: tmp_path)
    dispatched: list[str] = []

    def fake_dispatch(args, run_config, base, tid, run_tag):
        dispatched.append(tid)
        d = base / run_tag
        d.mkdir(parents=True, exist_ok=True)
        common.write_manifest(d, {"task_id": tid, "config": run_config})  # what the scan reads
        return "ok"

    monkeypatch.setattr(ra_runner, "_dispatch", fake_dispatch)
    return dispatched


def test_runner_resumes_and_does_not_rerun_completed(tmp_path, monkeypatch) -> None:
    dispatched = _patch_runner_for_resumption(tmp_path, monkeypatch)
    ra_runner.main(["--benchmark", "loong"])
    assert sorted(dispatched) == ["t1", "t2", "t3"]   # first run: all 3
    dispatched.clear()
    ra_runner.main(["--benchmark", "loong"])
    assert dispatched == []   # re-run: all done for this config → NOTHING re-run


def test_runner_limit_runs_next_n_pending(tmp_path, monkeypatch) -> None:
    dispatched = _patch_runner_for_resumption(tmp_path, monkeypatch)
    ra_runner.main(["--benchmark", "loong", "--limit", "2"])
    assert len(dispatched) == 2                # first 2 pending
    dispatched.clear()
    ra_runner.main(["--benchmark", "loong", "--limit", "2"])
    assert len(dispatched) == 1                # only the remaining 1 (not a re-run of the first 2)


# ============================================================
# Live wire-test (marker-gated) — real gpt-5.4-nano, 1 loong task
# ============================================================


@pytest.mark.readagent
def test_readagent_end_to_end_loong_real(tmp_path) -> None:
    """Drive the REAL ReadAgent-P pipeline (paginate → gist → look up → answer) on one Loong
    task against gpt-5.4-nano. Asserts a non-empty answer, a persisted gist memory, and
    captured TOTAL usage. Costs a few cents; deselected by default (``-m readagent``)."""
    from evals.baselines import _common
    from evals.baselines.readagent.run import run_one
    from evals.settings import DEFAULT_COMPLETION_MODEL

    benchmark = _common.load_benchmark_module("loong")
    task_id = benchmark.get_task_ids(languages=["en"], limit=1)[0]
    run_config = {"benchmark": "loong", "baseline": "readagent",
                  "model": _common.canonical_model_id(DEFAULT_COMPLETION_MODEL),
                  "seed": 42, "run_version": ra_run.run_version_for("loong")}
    litellm_kwargs = {"model": _common.with_provider_prefix(DEFAULT_COMPLETION_MODEL),
                      "api_base": None, "api_key": None}

    rec = run_one(benchmark, task_id, run_config, tmp_path, litellm_kwargs)

    assert isinstance(rec["raw_answer"], str) and rec["raw_answer"].strip()
    assert (tmp_path / "gist_memory.json").exists()
    assert any("nano" in m for m in rec["usage"]["total"])
    assert rec["calls_full"]
