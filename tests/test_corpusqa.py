"""Tests for `evals.benchmarks.corpusqa`.

CorpusQA ships NOTHING in-repo — its 1m tier is a multi-GB file fetched on demand —
so every test here runs against a **tiny synthetic fixture** written to a tmp data
dir (no network, no multi-GB download; the `_download` path is monkeypatched to
fail loudly so an accidental fetch can't slip through). The fixture mirrors the real
row shape: a frozen `prompt` = `[system persona, user]` where the user message glues
together `# Document N:` sections + a `# Question:` marker + an "Output requirements"
instruction block (the answer-format + conflict rules the gold was computed under).

CorpusQA is **1m-only** (`SETS == {"1m"}`): 1m is the tier we measure on, so it is the
only wired tier and a bare `get_task_ids()` returns the whole benchmark (no
`STARTER_FILTER`, no tier to choose). 128k/4m/10m exist upstream but are not wired.

CorpusQA specifics this file pins down:
- The frozen-prompt **un-baking**: `get_documents` splits on `# Document N:`;
  `get_task` returns `(instruction, question, docs)` with the instruction carrying
  the format/conflict rules (they MUST reach the model — the gold depends on them).
- The **byte-offset index**: only `get_task`/`get_documents` read a row's prompt
  (by seeking to its recorded offset); enumeration/metadata/scoring read the cheap
  index. A multibyte (`zh`) row exercises that the offsets are BYTES, not chars.
- **task_id keys on the FILE tier, not `row["set"]`**: upstream ships every row with
  `set == "128k"` even inside the 1m file, so the id is built from the filename
  (`@1m`), not that lying field (regression test below).
- `get_task_answer` is NOT always a string — native number / string / list,
  including the intentional empty list `[]` ("no rows matched").
- Metadata `{domain, set, language, n_docs}`; `language` is DERIVED from the domain
  (only `financial_zh` is `zh`). Upstream's empty `type` field is not surfaced.
- Judge: reproduces upstream `eval.py`'s ORM equivalence judge — `extract_answer`
  ("The answer is: …") + a structured equivalence verdict → 0/1, NO `PERFECT_SCORE`.

`score()` / `score_batch()` hit a real LLM judge (gpt-5.4-nano) and cost money;
those tests are marked `judge` and de-selected by default. Pure-logic + offset-index
+ un-baking guards run by default (mock LLM where needed).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.benchmarks import corpusqa
from evals.benchmarks._sampling import order_task_ids
from evals.benchmarks.corpusqa import download_data, judge, loader


# ============================================================
# Fixture: a tiny synthetic data dir (no network, no big files)
# ============================================================

_PERSONA = "You are a professional data analyst, please answer the question accurately."

# A synthetic "Output requirements" block carrying the two load-bearing task
# semantics: the conflict-resolution rule and the `The answer is:` output contract.
_OUTPUT_REQ = (
    "Output requirements:\n"
    "1. Your answer can only be one of the following types: a list of strings "
    '(including empty list "[]"), a single short string, a number, or a percentage.\n'
    '2. To output the date, use the format "YYYY-MM"\n'
    "3. If multiple files contain a conflicting value for the same item, please use "
    "the value from the latest file.\n"
    "4. Please place the final answer at the end, and strictly use the format: "
    "The answer is: xxx.\n"
)


def _bake_user(docs: list[str], question: str) -> str:
    """Reproduce the real frozen `user` message: docs, the question marker, the block."""
    parts = [f"# Document {i}:\n{d}\n" for i, d in enumerate(docs, 1)]
    parts.append(f"\n# Question:\n{question}\n\n{_OUTPUT_REQ}")
    return "".join(parts)


def _row(rid: str, domain: str, set_field: str, question: str, answer, docs: list[str]) -> dict:
    """A synthetic row. ``set_field`` is the row's own ``set`` value — upstream ships it
    as ``"128k"`` in EVERY file (the loader ignores it and keys on the filename), so the
    fixtures pass it explicitly to mirror reality where it matters."""
    return {
        "id": rid,
        "domain": domain,
        "set": set_field,
        "type": "",  # upstream ships this empty — the loader deliberately ignores it
        "question": question,
        "answer": answer,
        "doc_files": [f"{rid}_doc{i}.md" for i in range(1, len(docs) + 1)],
        "prompt": [
            {"role": "system", "content": _PERSONA},
            {"role": "user", "content": _bake_user(docs, question)},
        ],
    }


def _write_tier(data_dir: Path, tier: str, rows: list[dict]) -> None:
    path = data_dir / f"{tier}_4domains.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def cqa(tmp_path, monkeypatch) -> Path:
    """A tmp CORPUSQA_DIR with the (only) wired `1m` tier present (so a bare
    get_task_ids() needs no network), the cache cleared, and downloads disabled.

    The rows carry ``set == "128k"`` (the lie upstream ships in the 1m file too) to keep
    the fixture realistic — the loader keys task ids on the filename, so they are ``@1m``."""
    monkeypatch.setattr(loader.settings, "CORPUSQA_DIR", tmp_path)
    monkeypatch.setattr(
        download_data, "_download",
        lambda *a, **k: pytest.fail("unexpected download — the 1m tier file is missing"),
    )
    loader._tier_index.cache_clear()
    _write_tier(tmp_path, "1m", [
        _row("education_en_5", "education_en", "128k", "How many sections?", 282.0,
             ["Doc A body", "Doc B body"]),
        _row("financial_zh_1", "financial_zh", "128k", "哪些公司?",
             ["中信国安", "奋达科技"], ["甲", "乙", "丙"]),  # multibyte → byte-offset check
        _row("real_estate_en_3", "real_estate_en", "128k", "Which months?", [],
             ["Only doc"]),  # empty-list gold, single doc; follows the multibyte row
    ])
    yield tmp_path
    loader._tier_index.cache_clear()


# ============================================================
# Constants
# ============================================================


def test_constants() -> None:
    assert corpusqa.DOMAINS == frozenset(
        {"financial_zh", "financial_en", "education_en", "real_estate_en"}
    )
    assert corpusqa.SETS == frozenset({"1m"})       # 1m-only (128k/4m/10m not wired)
    assert corpusqa.STARTER_FILTER == {}            # 1m-only → bare get_task_ids() IS the benchmark
    assert not hasattr(corpusqa, "PERFECT_SCORE")  # 0/1 accuracy, not a 1–100 rating
    assert not hasattr(corpusqa, "get_corpus")     # per-instance multi-doc, not shared
    assert not hasattr(corpusqa, "parse")          # judge benchmark, nothing to parse


# ============================================================
# get_task_ids — counts, uniqueness, filters, shuffle
# ============================================================


def test_get_task_ids_counts_and_unique(cqa) -> None:
    ids = corpusqa.get_task_ids()
    assert len(ids) == 3                  # all 3 fixture rows live in the 1m tier
    assert len(set(ids)) == 3
    assert all(i.endswith("@1m") for i in ids)
    assert all(isinstance(i, str) for i in ids)


def test_starter_filter_equals_bare_call(cqa) -> None:
    """1m-only: the starter subset IS the whole benchmark (empty STARTER_FILTER)."""
    assert corpusqa.get_task_ids(**corpusqa.STARTER_FILTER) == corpusqa.get_task_ids()


def test_get_task_ids_filters_by_set(cqa) -> None:
    assert len(corpusqa.get_task_ids(sets=["1m"])) == 3
    with pytest.raises(ValueError, match="Unknown set"):  # 128k is no longer wired
        corpusqa.get_task_ids(sets=["128k"])
    with pytest.raises(ValueError, match="Unknown set"):  # 4m never was
        corpusqa.get_task_ids(sets=["4m"])


def test_get_task_ids_filters_by_domain(cqa) -> None:
    assert len(corpusqa.get_task_ids(domains=["education_en"])) == 1
    assert len(corpusqa.get_task_ids(domains=["financial_zh"])) == 1
    assert len(corpusqa.get_task_ids(domains=["financial_en"])) == 0  # none in the fixture


def test_get_task_ids_and_semantics(cqa) -> None:
    both = set(corpusqa.get_task_ids(domains=["education_en"], sets=["1m"]))
    assert both == {"education_en_5@1m"}


def test_task_id_uses_file_tier_not_row_set_field(tmp_path, monkeypatch) -> None:
    """Regression for the tier-labeling bug: CorpusQA's real rows ALL carry
    ``set: "128k"`` regardless of which tier file they live in. The loader must key the
    task id on the FILE tier (the filename), NOT ``row["set"]`` — otherwise every 1m task
    id would be ``@128k``, which isn't a wired tier (``SETS == {"1m"}``), so every lookup
    would fail. Here a 1m file holds a row whose ``set`` field LIES (``"128k"``); its task
    id must be ``@1m`` and read the 1m content."""
    monkeypatch.setattr(loader.settings, "CORPUSQA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "_download",
                        lambda *a, **k: pytest.fail("unexpected download"))
    loader._tier_index.cache_clear()
    # the 1m row carries the WRONG `set` field ("128k"), exactly as the released data does
    row = _row("education_en_1", "education_en", "128k", "Which region?", "South",
               ["doc a", "doc b", "doc c"])
    _write_tier(tmp_path, "1m", [row])
    try:
        ids = corpusqa.get_task_ids(sets=["1m"])
        assert ids == ["education_en_1@1m"]          # FILE tier — NOT the row's @128k
        assert "@128k" not in ids[0]
        assert corpusqa.get_task_metadata("education_en_1@1m")["set"] == "1m"
        assert len(corpusqa.get_documents("education_en_1@1m")) == 3  # the 1m row's docs
    finally:
        loader._tier_index.cache_clear()


def test_get_task_ids_deterministic_shuffle_and_limit(cqa) -> None:
    # Reconstruct the pre-shuffle collection order (sorted tiers, then file order).
    raw: list[str] = []
    for tier in sorted(corpusqa.SETS):
        raw.extend(loader._tier_index(tier).keys())
    out = corpusqa.get_task_ids()
    assert out == corpusqa.get_task_ids()        # deterministic across calls
    assert out == order_task_ids(raw)            # routed through the shared shuffle
    assert corpusqa.get_task_ids(limit=2) == out[:2]
    assert corpusqa.get_task_ids(limit=0) == []
    assert corpusqa.get_task_ids(limit=10**9) == out


def test_get_task_ids_empty_filter_returns_empty(cqa) -> None:
    assert corpusqa.get_task_ids(sets=[]) == []
    assert corpusqa.get_task_ids(domains=[]) == []


def test_get_task_ids_unknown_filters_raise(cqa) -> None:
    with pytest.raises(ValueError, match="Unknown set"):
        corpusqa.get_task_ids(sets=["256k"])
    with pytest.raises(ValueError, match="Unknown domain"):
        corpusqa.get_task_ids(domains=["medical_en"])


# ============================================================
# get_task / get_documents — un-baking the frozen prompt
# ============================================================


def test_get_task_returns_instruction_question_docs(cqa) -> None:
    instruction, question, docs = corpusqa.get_task("education_en_5@1m")
    assert question == "How many sections?"
    assert docs == ["Doc A body", "Doc B body"]
    assert docs == corpusqa.get_documents("education_en_5@1m")
    # The format/conflict task semantics MUST reach the model.
    assert "latest file" in instruction       # conflict-resolution rule
    assert "The answer is:" in instruction     # output contract (kept → judge extracts)


def test_get_documents_splits_and_counts(cqa) -> None:
    docs = corpusqa.get_documents("financial_zh_1@1m")
    assert docs == ["甲", "乙", "丙"]  # split on # Document N:, stripped
    # n_docs (from doc_files) matches the un-baked count
    assert corpusqa.get_task_metadata("financial_zh_1@1m")["n_docs"] == 3


def test_offset_index_is_built(cqa) -> None:
    corpusqa.get_task_ids(sets=["1m"])
    assert (cqa / "1m_index_v2.jsonl").exists()


def test_offset_after_multibyte_row_is_correct(cqa) -> None:
    """real_estate_en_3 follows the multibyte `financial_zh_1` row in the file — its
    byte offset is only correct if the index tracked BYTES, not characters."""
    assert corpusqa.get_task_answer("real_estate_en_3@1m") == []
    assert corpusqa.get_documents("real_estate_en_3@1m") == ["Only doc"]


def test_read_row_roundtrips_via_offset(cqa) -> None:
    tier, rec = loader._locate("education_en_5@1m")
    row = loader._read_row(tier, rec)
    assert row["id"] == "education_en_5" and row["answer"] == 282.0


def test_malformed_doc_count_raises_at_build(tmp_path, monkeypatch) -> None:
    """The build-time validation: a row whose `# Document N:` header count disagrees
    with `doc_files` fails loudly at index build, not silently mid-run."""
    monkeypatch.setattr(loader.settings, "CORPUSQA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "_download", lambda *a, **k: pytest.fail("no network"))
    loader._tier_index.cache_clear()
    bad = {
        "id": "bad_1", "domain": "education_en", "set": "128k", "type": "",
        "question": "q?", "answer": 1, "doc_files": ["a.md", "b.md"],  # 2 files...
        "prompt": [
            {"role": "system", "content": "p"},
            # ...but only ONE # Document header
            {"role": "user", "content": "# Document 1:\nonly one\n\n# Question:\nq?\n\nfoo"},
        ],
    }
    (tmp_path / "1m_4domains.jsonl").write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="headers"):
        loader._tier_index("1m")
    loader._tier_index.cache_clear()


# ============================================================
# get_task_answer / get_task_metadata
# ============================================================


def test_get_task_answer_native_types(cqa) -> None:
    assert corpusqa.get_task_answer("education_en_5@1m") == 282.0               # float
    assert corpusqa.get_task_answer("financial_zh_1@1m") == ["中信国安", "奋达科技"]  # list
    assert corpusqa.get_task_answer("real_estate_en_3@1m") == []               # empty list


def test_get_task_metadata_keys_and_language(cqa) -> None:
    md = corpusqa.get_task_metadata("education_en_5@1m")
    assert set(md) == {"domain", "set", "language", "n_docs"}
    # `domain` is the REPORT grouping (renamed + merged), NOT the raw filter domain.
    assert md == {"domain": "education", "set": "1m", "language": "en", "n_docs": 2}
    assert corpusqa.get_task_metadata("real_estate_en_3@1m")["domain"] == "real estate"
    # financial_en + financial_zh MERGE into one "financial" report bucket (language still splits them)
    assert corpusqa.get_task_metadata("financial_zh_1@1m")["domain"] == "financial"
    assert corpusqa.get_task_metadata("financial_zh_1@1m")["language"] == "zh"  # only zh domain
    # ...but the raw filter vocab is untouched — DOMAINS + get_task_ids stay 4-way.
    assert {"financial_en", "financial_zh"} <= corpusqa.DOMAINS


def test_unknown_id_raises(cqa) -> None:
    for fn in (corpusqa.get_task, corpusqa.get_documents,
               corpusqa.get_task_answer, corpusqa.get_task_metadata):
        with pytest.raises(KeyError):
            fn("nope@1m")            # valid tier, unknown id
    with pytest.raises(KeyError):
        corpusqa.get_task("nope@128k")      # unwired tier → rejected without touching disk
    with pytest.raises(KeyError):
        corpusqa.get_task("totally-bogus")  # no @tier → rejected without touching disk


# ============================================================
# download_data — acquisition logic (no network)
# ============================================================


def test_download_url_is_hugging_face_for_the_1m_tier() -> None:
    # HF is the sole mirror; the wired 1m tier resolves to a HF dataset URL.
    url = download_data._HF.format(tier="1m")
    assert "huggingface.co" in url and "1m_4domains.jsonl" in url


def test_ensure_rejects_unwired_tiers() -> None:
    # Only 1m is wired; 128k/4m are not (reject before any fetch).
    assert download_data.SETS == frozenset({"1m"})
    for tier in ("128k", "4m"):
        assert tier not in download_data.SETS
        with pytest.raises(ValueError, match="Unknown tier"):
            download_data.ensure(tier)


def test_ensure_unknown_tier_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_data.settings, "CORPUSQA_DIR", tmp_path)
    with pytest.raises(ValueError, match="Unknown tier"):
        download_data.ensure("99k")


def test_ensure_noop_when_present(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(download_data.settings, "CORPUSQA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "_download", lambda *a, **k: pytest.fail("downloaded"))
    (tmp_path / "1m_4domains.jsonl").write_text("{}\n")
    assert download_data.ensure("1m") == tmp_path / "1m_4domains.jsonl"


def test_ensure_downloads_via_atomic_publish(tmp_path, monkeypatch) -> None:
    """ensure() fetches a missing tier and publishes atomically (no .part leftovers).
    urllib is mocked — no network."""
    monkeypatch.setattr(download_data.settings, "CORPUSQA_DIR", tmp_path)

    def fake_urlretrieve(url: str, dest: str) -> None:
        with open(dest, "w", encoding="utf-8") as f:
            f.write('{"row": 1}\n')

    monkeypatch.setattr(download_data.urllib.request, "urlretrieve", fake_urlretrieve)
    out = download_data.ensure("1m")
    assert out.read_text().strip() == '{"row": 1}'
    assert not any(p.name.endswith(".part") for p in tmp_path.iterdir())


# ============================================================
# Judge — pure logic (no LLM call, run by default)
# ============================================================


def test_extract_answer_ports_upstream() -> None:
    assert judge.extract_answer("reasoning...\nThe answer is: 42") == "42"
    assert judge.extract_answer("no marker here") == "no marker here"
    assert judge.extract_answer('  The answer is: ["A", "B"]  ') == '["A", "B"]'


def test_format_gold_is_str_of_native() -> None:
    assert judge._format_gold(282.0) == "282.0"
    assert judge._format_gold("South") == "South"
    assert judge._format_gold([]) == "[]"          # empty-list gold, by design
    assert judge._format_gold(["A", "B"]) == "['A', 'B']"


def test_user_prompt_is_orm_template() -> None:
    assert judge._user_prompt("PROB", "A1", "A2") == "\nProblem: PROB\nAnswer 1: A1\nAnswer 2: A2\n"


def test_judge_empty_answer_scores_zero_without_llm(monkeypatch) -> None:
    monkeypatch.setattr(judge, "litellm_chat_completion", lambda **k: pytest.fail("LLM called"))
    assert judge._judge("q", "gold", "")["score"] == 0
    assert judge._judge("q", "gold", "   ")["score"] == 0
    assert judge._judge("q", "gold", "")["rationale"] is None
    assert judge._judge("q", "gold", "")["model"] == "gpt-5-4-mini"  # the grader (JUDGE_MODEL) slug


def test_judge_maps_equivalence_to_0_1_and_extracts(monkeypatch) -> None:
    """The judge runs extract_answer first, passes (extracted, str(gold)) as
    (Answer 1, Answer 2), and maps the structured verdict to 0/1."""
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return judge.Equivalence(explanation="same value", equivalent=True)

    monkeypatch.setattr(judge, "litellm_chat_completion", fake)
    result = judge._judge("How many?", 282.0, "blah blah The answer is: 282.0")
    assert result["score"] == 1 and result["rationale"] == "same value"
    assert "Answer 1: 282.0" in captured["user_prompt"]   # extracted, not the whole reply
    assert "Answer 2: 282.0" in captured["user_prompt"]   # str(gold)

    monkeypatch.setattr(
        judge, "litellm_chat_completion",
        lambda **k: judge.Equivalence(explanation="different", equivalent=False),
    )
    assert judge._judge("q", 282.0, "The answer is: 999")["score"] == 0


def test_score_batch_guards() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        corpusqa.score_batch(["x"], ["a", "b"])
    assert corpusqa.score_batch([], []) == []
    assert corpusqa.score_details([], []) == []


def test_gather_unknown_id_raises(cqa) -> None:
    with pytest.raises(KeyError, match="Unknown CorpusQA task id"):
        judge._gather(["nope@1m"])


# ============================================================
# LLM judge — costs money, run only with `pytest -m judge`
# ============================================================


@pytest.mark.judge
def test_score_gold_answer_is_correct(cqa) -> None:
    """Feeding the gold back (in the output contract) scores 1 — tolerating 282 vs 282.0."""
    gold = corpusqa.get_task_answer("education_en_5@1m")
    assert corpusqa.score("education_en_5@1m", f"The answer is: {gold}") == 1


@pytest.mark.judge
def test_score_nonsense_answer_is_incorrect(cqa) -> None:
    assert corpusqa.score("education_en_5@1m", "The answer is: 999999") == 0


@pytest.mark.judge
def test_score_empty_list_gold_matches_empty_answer(cqa) -> None:
    """A correct "no matches" answer must be judged equivalent to the `[]` gold."""
    assert corpusqa.score("real_estate_en_3@1m", "The answer is: []") == 1


@pytest.mark.judge
def test_score_zh_gold_answer_is_correct(cqa) -> None:
    """End-to-end ZH: the judge grades a Chinese gold/list answer (no code change)."""
    gold = corpusqa.get_task_answer("financial_zh_1@1m")
    assert corpusqa.score("financial_zh_1@1m", f"The answer is: {gold}") == 1
