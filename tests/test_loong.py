"""Tests for `evals.benchmarks.loong`.

Loong specifics this file pins down:
- EN + ZH: the vendored loong.jsonl ships 1,600 instances (EN 695 + ZH 905); the
  loader exposes ALL of them, with `get_task_ids(languages=)` as the selector
  (default both). Three domains: paper (EN only), financial (EN + ZH), legal
  (ZH only). All four task types (level 1–4) and length sets (1–4) span both langs.
- `get_task_ids(sets=, tasks=, languages=)` filters on length bucket (`set`), task
  type (`level`), and `language`, AND-combined.
- `get_task()` returns a 3-tuple `(instruction, question, docs)` — everything the
  task needs, docs being the resolved bundle (a list, identical to get_documents).
- `get_task_answer()` is NOT always a string: Loong golds are sometimes a JSON
  object / list, returned natively.
- `get_documents()` resolves the per-instance filename bundle via the three domain
  rules (financial glob; paper direct read; legal legal.json lookup → content+result,
  positional `《判决文书N》` title). Resolution is tested against tiny fixtures — the
  real ~34 MB doc pool is never downloaded here.
- The legal **bug fix**: the level-4 "match document to verdict" task hides the
  verdict (content only); upstream's guard never fired (it checked `instruction`,
  the marker is in `question`). See `evals/benchmarks/loong/CHANGES.md`.
- `download_docs` recovers the macOS-zip CP437-mojibake Chinese filenames on extract.
- Judge: 1–100 rating via structured output (`Rating`), `PERFECT_SCORE = 100`.
  Pure-logic guards run by default; the real judge call is marked `judge`.

`score()` / `score_batch()` hit a real LLM judge (gpt-5.4-nano) and cost money;
those tests are marked `judge` and de-selected by default.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.benchmarks import loong
from evals.benchmarks._sampling import order_task_ids
from evals.benchmarks.loong import download_docs, judge, loader


# Counts pinned from the vendored loong.jsonl (full EN + ZH benchmark).
TOTAL_N = 1600
BY_TASK = {1: 250, 2: 300, 3: 641, 4: 409}   # upstream `level`
BY_SET = {1: 323, 2: 564, 3: 481, 4: 232}    # upstream `set`
BY_DOMAIN = {"paper": 400, "financial": 700, "legal": 500}
BY_LANGUAGE = {"en": 695, "zh": 905}

EXPECTED_META_KEYS = {"set", "task", "task_name", "domain", "language", "length"}


@pytest.fixture(scope="module")
def ids() -> list[str]:
    return loong.get_task_ids()


# ============================================================
# Constants
# ============================================================


def test_sets_and_tasks_constants() -> None:
    assert loong.SETS == frozenset({1, 2, 3, 4})
    assert loong.TASKS == frozenset({1, 2, 3, 4})


# ============================================================
# Loading (EN + ZH)
# ============================================================


def test_load_includes_en_and_zh() -> None:
    """The loader exposes ALL 1,600 instances (both languages) — language is a
    query-time filter axis, not a load-time cut."""
    from collections import Counter
    rows = loader._load()
    assert len(rows) == TOTAL_N
    assert dict(Counter(r["language"] for r in rows.values())) == BY_LANGUAGE

    raw_lines = (loader.settings.LOONG_DIR / "loong.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == TOTAL_N  # nothing dropped at load time


def test_all_three_domains_present() -> None:
    """All three Loong domains load — including the ZH-only legal domain."""
    domains = {r["type"] for r in loader._load().values()}
    assert domains == {"paper", "financial", "legal"}


# ============================================================
# get_task_ids — counts, uniqueness, filters
# ============================================================


def test_get_task_ids_returns_all_unique_strings(ids) -> None:
    assert len(ids) == TOTAL_N
    assert all(isinstance(i, str) for i in ids)
    assert len(set(ids)) == len(ids)


def test_get_task_ids_no_filter_equals_explicit_none() -> None:
    assert len(loong.get_task_ids(sets=None, tasks=None, languages=None)) == TOTAL_N


@pytest.mark.parametrize("set_val,expected", sorted(BY_SET.items()))
def test_get_task_ids_filters_by_set(set_val, expected) -> None:
    assert len(loong.get_task_ids(sets=[set_val])) == expected


@pytest.mark.parametrize("task_val,expected", sorted(BY_TASK.items()))
def test_get_task_ids_filters_by_task(task_val, expected) -> None:
    assert len(loong.get_task_ids(tasks=[task_val])) == expected


def test_get_task_ids_set_filter_partitions_total() -> None:
    assert sum(len(loong.get_task_ids(sets=[s])) for s in sorted(BY_SET)) == TOTAL_N


@pytest.mark.parametrize("lang,expected", sorted(BY_LANGUAGE.items()))
def test_get_task_ids_filters_by_language(lang, expected) -> None:
    assert len(loong.get_task_ids(languages=[lang])) == expected


def test_get_task_ids_language_partitions_total() -> None:
    assert sum(len(loong.get_task_ids(languages=[l])) for l in sorted(BY_LANGUAGE)) == TOTAL_N


def test_get_task_ids_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="Unknown language"):
        loong.get_task_ids(languages=["fr"])


def test_get_task_ids_language_lines_up_with_domains() -> None:
    """paper is EN-only and legal is ZH-only — the language filter must reflect it."""
    en = set(loong.get_task_ids(languages=["en"]))
    zh = set(loong.get_task_ids(languages=["zh"]))
    assert en.isdisjoint(zh) and len(en) + len(zh) == TOTAL_N
    legal = {t for t in loong.get_task_ids() if loong.get_task_metadata(t)["domain"] == "legal"}
    paper = {t for t in loong.get_task_ids() if loong.get_task_metadata(t)["domain"] == "paper"}
    assert legal <= zh   # all legal is ZH
    assert paper <= en   # all paper is EN


def test_get_task_ids_and_semantics_is_intersection() -> None:
    """sets= and tasks= combine with AND — the result is the intersection."""
    by_set = set(loong.get_task_ids(sets=[1]))
    by_task = set(loong.get_task_ids(tasks=[1]))
    both = set(loong.get_task_ids(sets=[1], tasks=[1]))
    assert both == by_set & by_task
    assert both  # non-empty (financial level-1 instances exist in set 1)


def test_get_task_ids_accepts_any_iterable() -> None:
    assert loong.get_task_ids(tasks=(1,)) == loong.get_task_ids(tasks={1})


def test_get_task_ids_empty_filter_returns_empty_list() -> None:
    assert loong.get_task_ids(sets=[]) == []
    assert loong.get_task_ids(tasks=[]) == []


def test_get_task_ids_deterministically_shuffled_and_limited() -> None:
    """get_task_ids routes through the shared deterministic shuffle + limit."""
    raw = list(loader._load().keys())
    out = loong.get_task_ids()
    assert out == loong.get_task_ids()          # deterministic across calls
    assert out == order_task_ids(raw)           # routed through the shared shuffle
    assert out != raw                           # actually reordered (not file order)
    assert loong.get_task_ids(limit=5) == out[:5]
    assert loong.get_task_ids(limit=0) == []
    assert loong.get_task_ids(limit=10**9) == out


def test_get_task_ids_unknown_set_raises() -> None:
    with pytest.raises(ValueError, match="Unknown set"):
        loong.get_task_ids(sets=[5])


def test_get_task_ids_unknown_task_raises() -> None:
    with pytest.raises(ValueError, match="Unknown task"):
        loong.get_task_ids(tasks=[0])


# ============================================================
# get_task — (instruction, question, docs) 3-tuple
# ============================================================


def test_get_task_returns_instruction_question_docs(tmp_path, monkeypatch) -> None:
    """get_task returns everything needed: (instruction, question, docs), where
    docs is the resolved bundle (a list) — identical to get_documents. Run against
    fixtures so no 34 MB download."""
    paper_tid = next(
        t for t in loong.get_task_ids() if loong.get_task_metadata(t)["domain"] == "paper"
    )
    doc_names = loader._load()[paper_tid]["doc"]
    doc_dir = tmp_path / "doc"
    (doc_dir / "paper").mkdir(parents=True)
    for name in doc_names:
        (doc_dir / "paper" / name).write_text(f"# T {name}\nbody", encoding="utf-8")
    monkeypatch.setattr(loader, "_ensure_doc_dir", lambda: doc_dir)

    instruction, question, docs = loong.get_task(paper_tid)
    assert isinstance(instruction, str) and instruction
    assert isinstance(question, str)  # may be empty for some instances (e.g. paper L4)
    assert isinstance(docs, list) and len(docs) == len(doc_names)
    assert docs == loong.get_documents(paper_tid)  # get_task's docs == get_documents


def test_get_task_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        loong.get_task("not-a-real-task-id")


# ============================================================
# get_task_answer — native dict / list / str
# ============================================================


def test_get_task_answer_returns_native_types(ids) -> None:
    """Loong golds are sometimes a JSON object / list, not a string — returned
    as the native Python type (the judge serializes them)."""
    answer_types = {type(loong.get_task_answer(t)).__name__ for t in ids}
    assert "dict" in answer_types   # e.g. the paper citation task
    assert "str" in answer_types


def test_get_task_answer_unknown_id_raises() -> None:
    with pytest.raises(KeyError):
        loong.get_task_answer("not-a-real-task-id")


# ============================================================
# get_task_metadata
# ============================================================


def test_get_task_metadata_keys_and_domains(ids) -> None:
    md = loong.get_task_metadata(ids[0])
    assert set(md.keys()) == EXPECTED_META_KEYS
    for t in ids:
        m = loong.get_task_metadata(t)
        assert m["domain"] in {"paper", "financial", "legal"}
        assert m["language"] in loong.LANGUAGES
        assert m["set"] in loong.SETS
        assert m["task"] in loong.TASKS


def test_get_task_metadata_task_name_matches_level() -> None:
    names = {loong.get_task_metadata(t)["task"]: loong.get_task_metadata(t)["task_name"]
             for t in loong.get_task_ids()}
    assert names[1] == "Spotlight Locating"
    assert names[3] == "Clustering"
    assert names[4] == "Chain of Reasoning"


def test_metadata_domain_counts_match(ids) -> None:
    from collections import Counter
    counts = Counter(loong.get_task_metadata(t)["domain"] for t in ids)
    assert dict(counts) == BY_DOMAIN


def test_metadata_language_counts_match(ids) -> None:
    from collections import Counter
    counts = Counter(loong.get_task_metadata(t)["language"] for t in ids)
    assert dict(counts) == BY_LANGUAGE


# ============================================================
# get_documents — doc resolution (against tiny fixtures, no download)
# ============================================================


def test_get_documents_is_per_task_no_corpus() -> None:
    """No get_corpus → an indexing baseline indexes Loong per task (per-instance bundles)."""
    assert not hasattr(loong, "get_corpus")


def _row(type: str, level: int, question: str = "") -> dict:
    """A minimal instance row for `_resolve_doc` (it reads type/level/question)."""
    return {"type": type, "level": level, "question": question}


def test_resolve_doc_financial_non_level4_globs_2024_company() -> None:
    """financial level≠4: glob `*2024-{company}*.txt`; title = stem's last
    `-`-segment (upstream's quirky `《j》` heading is preserved verbatim)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc_dir = Path(d)
        (doc_dir / "financial").mkdir()
        (doc_dir / "financial" / "2024-AUDDIA INC.-j.txt").write_text("BODY TEXT", encoding="utf-8")
        out = loader._resolve_doc(doc_dir, _row("financial", 1), 0, "AUDDIA INC.")
    assert out == "《j》\nBODY TEXT\n\n"


def test_resolve_doc_financial_level4_globs_bare_name() -> None:
    """financial level 4 (chain-of-reasoning over annual reports): the doc_name
    already carries the year, so the glob is `*{doc_name}*.txt`."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc_dir = Path(d)
        (doc_dir / "financial").mkdir()
        (doc_dir / "financial" / "2023-form10-k.txt").write_text("REPORT", encoding="utf-8")
        out = loader._resolve_doc(doc_dir, _row("financial", 4), 0, "2023-form10-k")
    assert out == "《k》\nREPORT\n\n"


def test_resolve_doc_financial_zh_company_resolves_same_way() -> None:
    """ZH financial resolves identically — the company name is just Chinese; the
    glob + 《title》 (file stem's last `-`-segment) rule is unchanged from EN."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc_dir = Path(d)
        (doc_dir / "financial").mkdir()
        (doc_dir / "financial" / "report_000635-2024-英力特-2024年一季度报告.txt").write_text(
            "财务报表", encoding="utf-8"
        )
        out = loader._resolve_doc(doc_dir, _row("financial", 1), 0, "英力特")
    assert out == "《2024年一季度报告》\n财务报表\n\n"


def test_resolve_doc_paper_reads_directly_and_titles_from_first_line() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc_dir = Path(d)
        (doc_dir / "paper").mkdir()
        (doc_dir / "paper" / "2402.01739.md").write_text("# Some Paper Title\nAbstract...", encoding="utf-8")
        out = loader._resolve_doc(doc_dir, _row("paper", 3), 0, "2402.01739.md")
    assert out == "Some Paper Title\n# Some Paper Title\nAbstract...\n\n"


def test_resolve_doc_missing_financial_file_raises() -> None:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        doc_dir = Path(d)
        (doc_dir / "financial").mkdir()
        with pytest.raises(FileNotFoundError, match="financial document"):
            loader._resolve_doc(doc_dir, _row("financial", 1), 0, "NoSuchCo")


def test_resolve_doc_unsupported_type_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        loader._resolve_doc(Path("/tmp"), _row("tabular", 1), 0, "x")


# ---- legal (ZH-only): legal.json lookup, content+result, the verdict-leak FIX ----


def _legal_fixture(doc_dir: Path) -> None:
    """Write a tiny `doc/legal/legal.json` with two cases (content + result)."""
    (doc_dir / "legal").mkdir(parents=True)
    cases = {
        "案件甲": {"content": "FACTS-A", "result": "VERDICT-A"},
        "案件乙": {"content": "FACTS-B", "result": "VERDICT-B"},
    }
    (doc_dir / "legal" / "legal.json").write_text(
        json.dumps(cases, ensure_ascii=False), encoding="utf-8"
    )
    loader._load_legal_json.cache_clear()


def test_resolve_doc_legal_content_plus_result_positional_title(tmp_path) -> None:
    """Legal default: doc = content + result (case body + verdict), titled
    positionally 《判决文书{idx+1}》 — so `idx` drives the heading the gold references."""
    _legal_fixture(tmp_path)
    row = _row("legal", 3, question="classify these")
    assert loader._resolve_doc(tmp_path, row, 0, "案件甲") == "《判决文书1》\nFACTS-AVERDICT-A\n\n"
    assert loader._resolve_doc(tmp_path, row, 1, "案件乙") == "《判决文书2》\nFACTS-BVERDICT-B\n\n"


def test_resolve_doc_legal_l4_verdict_match_hides_verdict_THE_FIX(tmp_path) -> None:
    """THE BUG FIX: the level-4 'match document to verdict' task (marker in the
    QUESTION) emits content ONLY, so the verdict can't leak into the input. Upstream
    checked `instruction` (where the marker never is), so it never hid the verdict."""
    _legal_fixture(tmp_path)
    q = loader._LEGAL_VERDICT_MARKER + "{'判决结果1': 'VERDICT-A'}"
    out = loader._resolve_doc(tmp_path, _row("legal", 4, question=q), 0, "案件甲")
    assert out == "《判决文书1》\nFACTS-A\n\n"
    assert "VERDICT-A" not in out


def test_resolve_doc_legal_l4_without_marker_keeps_result(tmp_path) -> None:
    """A level-4 legal task WITHOUT the verdict-matching marker (e.g. the
    match-to-category variant) keeps content + result — we withhold the verdict ONLY
    for the verdict-matching task, faithfully."""
    _legal_fixture(tmp_path)
    out = loader._resolve_doc(tmp_path, _row("legal", 4, question="match categories"), 0, "案件甲")
    assert out == "《判决文书1》\nFACTS-AVERDICT-A\n\n"


def test_resolve_doc_legal_marker_outside_level4_keeps_result(tmp_path) -> None:
    """The fix is gated on level==4 as well — the marker at a non-L4 level keeps
    content+result (only L4 is the verdict-matching task)."""
    _legal_fixture(tmp_path)
    out = loader._resolve_doc(tmp_path, _row("legal", 1, question=loader._LEGAL_VERDICT_MARKER), 0, "案件甲")
    assert out == "《判决文书1》\nFACTS-AVERDICT-A\n\n"


def test_get_documents_resolves_real_paper_bundle(tmp_path, monkeypatch) -> None:
    """End-to-end wiring: get_documents reads the real instance's `doc` list and
    resolves each via the paper rule — against a fixture doc dir (no download)."""
    paper_tid = next(
        t for t in loong.get_task_ids() if loong.get_task_metadata(t)["domain"] == "paper"
    )
    doc_names = loader._load()[paper_tid]["doc"]
    doc_dir = tmp_path / "doc"
    (doc_dir / "paper").mkdir(parents=True)
    for name in doc_names:
        (doc_dir / "paper" / name).write_text(f"# Title {name}\ncontent of {name}", encoding="utf-8")
    monkeypatch.setattr(loader, "_ensure_doc_dir", lambda: doc_dir)

    docs = loong.get_documents(paper_tid)
    assert len(docs) == len(doc_names)
    assert docs[0].startswith(f"Title {doc_names[0]}\n")  # first-line title, # stripped


# ============================================================
# download_docs — the explicit fetch script (no network in tests)
# ============================================================


def test_recovered_name_round_trips_cp437_mojibake() -> None:
    """The macOS zip stores Chinese filenames without the UTF-8 flag, so zipfile
    mis-decodes them as CP437; _recovered_name round-trips them back. ASCII and
    already-UTF-8-flagged names are left untouched."""
    import zipfile
    real = "doc/financial/report-平安银行.txt"
    mojibake = real.encode("utf-8").decode("cp437")  # what zipfile yields with the flag unset
    assert mojibake != real
    info = zipfile.ZipInfo(filename=mojibake); info.flag_bits = 0
    assert download_docs._recovered_name(info) == real

    ascii_info = zipfile.ZipInfo(filename="doc/paper/x.md"); ascii_info.flag_bits = 0
    assert download_docs._recovered_name(ascii_info) == "doc/paper/x.md"

    flagged = zipfile.ZipInfo(filename="doc/financial/平安.txt"); flagged.flag_bits = 0x800
    assert download_docs._recovered_name(flagged) == "doc/financial/平安.txt"  # already UTF-8 → untouched


def test_extract_utf8_skips_macosx_and_guards_traversal(tmp_path) -> None:
    """_extract_utf8 extracts entries, skips the macOS __MACOSX resource forks, and
    never writes outside the destination dir."""
    import zipfile
    zpath = tmp_path / "z.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("doc/paper/x.md", "# T\nbody")
        zf.writestr("__MACOSX/doc/._x.md", "junk")
    dest = tmp_path / "out"; dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        download_docs._extract_utf8(zf, dest)
    assert (dest / "doc" / "paper" / "x.md").read_text() == "# T\nbody"
    assert not (dest / "__MACOSX").exists()


def test_download_ensure_extracts_from_existing_zip(tmp_path, monkeypatch) -> None:
    """ensure() extracts a present doc.zip without downloading — exercises the
    extract path with a tiny synthetic zip (no network)."""
    import zipfile
    monkeypatch.setattr(loader.settings, "LOONG_DIR", tmp_path)
    zip_path = tmp_path / "doc.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("doc/paper/x.md", "# Title\nbody")
        zf.writestr("doc/financial/2024-ACME-j.txt", "report")
    assert not download_docs.is_present()

    out = download_docs.ensure()  # zip present → extracts, never downloads

    assert out == tmp_path / "doc" and download_docs.is_present()
    assert (tmp_path / "doc" / "paper" / "x.md").read_text() == "# Title\nbody"


def test_download_ensure_is_noop_when_pool_present(tmp_path, monkeypatch) -> None:
    """ensure() returns immediately when doc/ already exists — no zip needed,
    so no download is attempted (there's no zip to fall back on)."""
    monkeypatch.setattr(loader.settings, "LOONG_DIR", tmp_path)
    (tmp_path / "doc").mkdir()
    assert download_docs.ensure() == tmp_path / "doc"  # would raise if it tried to fetch


def test_loader_ensure_doc_dir_delegates_to_download_script(monkeypatch) -> None:
    """The loader's lazy path is the SAME code as the explicit script."""
    sentinel = Path("/tmp/sentinel-doc-dir")
    monkeypatch.setattr(download_docs, "ensure", lambda: sentinel)
    assert loader._ensure_doc_dir() == sentinel


def test_download_docs_is_reexported_on_the_loong_package() -> None:
    """`loong.download_docs.ensure()` is a documented entrypoint (README "Warming
    the data caches"), so the submodule must be reachable as a package attribute —
    not only via `from evals.benchmarks.loong import download_docs`. A submodule is
    NOT auto-exposed on its package, so `loong/__init__.py` must explicitly
    `from . import download_docs`; this guards that documented snippet from
    silently breaking with an AttributeError."""
    from evals.benchmarks import loong as loong_pkg

    assert hasattr(loong_pkg, "download_docs"), "loong.download_docs not exposed"
    assert "download_docs" in loong_pkg.__all__
    assert callable(loong_pkg.download_docs.ensure)


def test_download_ensure_leaves_no_temp_artifacts(tmp_path, monkeypatch) -> None:
    """ensure() stages the extract through a temp dir and publishes atomically, so
    after it returns only doc/ + doc.zip (+ the lock file) remain — no leftover
    `.doc_extract_*` dir or `.zip.part`."""
    import zipfile
    monkeypatch.setattr(loader.settings, "LOONG_DIR", tmp_path)
    with zipfile.ZipFile(tmp_path / "doc.zip", "w") as zf:
        zf.writestr("doc/paper/x.md", "# T\nbody")
    download_docs.ensure()
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert not any(n.startswith(".doc_extract_") or n.endswith(".zip.part") for n in leftovers), leftovers


def test_download_ensure_concurrent_callers_are_safe(tmp_path, monkeypatch) -> None:
    """The runner fans out many processes whose first get_documents() races here.
    An exclusive file lock + atomic publish must keep the pool intact under
    contention — smoke-tested with threads (each opens its own lock handle, so they
    genuinely contend). Asserts correctness, not timing, so it isn't flaky."""
    import threading
    import zipfile
    monkeypatch.setattr(loader.settings, "LOONG_DIR", tmp_path)
    with zipfile.ZipFile(tmp_path / "doc.zip", "w") as zf:
        zf.writestr("doc/paper/x.md", "# T\nbody")

    errors: list[str] = []
    results: list[Path] = []

    def worker() -> None:
        try:
            results.append(download_docs.ensure())
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], errors
    assert all(r == tmp_path / "doc" for r in results)
    assert (tmp_path / "doc" / "paper" / "x.md").read_text() == "# T\nbody"
    assert not any(p.name.startswith(".doc_extract_") for p in tmp_path.iterdir())


def test_download_ensure_force_refreshes_via_atomic_replace(tmp_path, monkeypatch) -> None:
    """force=True re-downloads (by design) and atomically replaces an existing doc/,
    exercising the rmtree-old + os.replace path. urlretrieve is mocked to write a
    fresh zip (no network)."""
    import zipfile
    monkeypatch.setattr(loader.settings, "LOONG_DIR", tmp_path)

    def write_zip(body: str) -> None:
        with zipfile.ZipFile(tmp_path / "doc.zip", "w") as zf:
            zf.writestr("doc/paper/x.md", f"# T\n{body}")

    write_zip("v1")
    download_docs.ensure()
    assert (tmp_path / "doc" / "paper" / "x.md").read_text().endswith("v1")

    def fake_urlretrieve(url: str, dest: str) -> None:
        with zipfile.ZipFile(dest, "w") as zf:
            zf.writestr("doc/paper/x.md", "# T\nv2")

    monkeypatch.setattr(download_docs.urllib.request, "urlretrieve", fake_urlretrieve)
    download_docs.ensure(force=True)
    assert (tmp_path / "doc" / "paper" / "x.md").read_text().endswith("v2")
    assert not any(p.name.startswith(".doc_extract_") for p in tmp_path.iterdir())


# ============================================================
# Judge — pure-logic guards (no LLM call, run by default)
# ============================================================


def test_perfect_score_constant_is_100() -> None:
    assert loong.PERFECT_SCORE == 100


def test_analysis_display_config_constants() -> None:
    """Report display config, read via getattr by whatever renders these axes: hide
    the redundant breakdown axes (task_name duplicates task; length ≈ a finer set) and
    label the integer axes so a reader needn't know the 1–4 encoding."""
    assert loong.ANALYSIS_HIDE_AXES == frozenset({"task_name", "length"})
    labels = loong.ANALYSIS_VALUE_LABELS
    assert set(labels["task"]) == {"1", "2", "3", "4"}
    assert "Spotlight Locating" in labels["task"]["1"]
    assert "Chain of Reasoning" in labels["task"]["4"]
    assert set(labels["set"]) == {"1", "2", "3", "4"}
    assert "50K" in labels["set"]["1"] and "250K" in labels["set"]["4"]


def test_format_gold_passes_strings_through_and_json_encodes_objects() -> None:
    assert judge._format_gold("plain answer") == "plain answer"
    encoded = judge._format_gold({"Reference": ["A"], "Citation": []})
    assert json.loads(encoded) == {"Reference": ["A"], "Citation": []}


def test_format_question_fills_slots_and_drops_docs() -> None:
    q = judge._format_question("#Papers:\n{docs}\n\n{instruction}\n\n{question}", "DO X", "WHAT?")
    assert "{docs}" not in q and "{instruction}" not in q and "{question}" not in q
    assert "DO X" in q and "WHAT?" in q
    assert "#Papers:" in q  # template framing kept


def test_judge_empty_answer_scores_worst_without_llm_call() -> None:
    """An empty / whitespace prediction earns the 1–100 floor (1) and issues no
    paid call — _judge short-circuits before touching the LLM. _judge now returns
    a ScoreResult (score + rationale)."""
    result = judge._judge("q", "gold", "")
    assert result["score"] == judge._WORST_SCORE == 1
    assert result["rationale"] is None  # no judge call → no rationale
    assert result["model"] == "gpt-5-4-mini"  # the grader (JUDGE_MODEL) slug
    assert judge._judge("q", "gold", "   ")["score"] == 1


def test_score_details_empty_answer_returns_worst_scoreresult() -> None:
    """`score_details` returns ScoreResult(rating, rationale) per item; an empty
    answer short-circuits to the floor with no rationale and no LLM call."""
    tid = loong.get_task_ids()[0]
    [result] = loong.score_details([tid], [""])
    assert result["score"] == 1
    assert result["rationale"] is None


def test_score_batch_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        loong.score_batch(["x"], ["a", "b"])


def test_score_batch_empty_returns_empty() -> None:
    assert loong.score_batch([], []) == []


def test_score_details_empty_returns_empty() -> None:
    assert loong.score_details([], []) == []


def test_gather_unknown_id_raises() -> None:
    with pytest.raises(KeyError, match="Unknown Loong task id"):
        judge._gather(["not-a-real-task-id"])


# ============================================================
# LLM judge — costs money, run only with `pytest -m judge`
# ============================================================


@pytest.mark.judge
def test_score_gold_answer_rates_high(ids) -> None:
    """Feeding the gold answer back as the model's answer should rate near-perfect."""
    tid = next(t for t in ids if isinstance(loong.get_task_answer(t), str))
    gold = loong.get_task_answer(tid)
    assert loong.score(tid, gold) >= 80


@pytest.mark.judge
def test_score_nonsense_answer_rates_low(ids) -> None:
    tid = next(t for t in ids if isinstance(loong.get_task_answer(t), str))
    nonsense = "Bananas are an igneous rock formed deep underground over millennia."
    assert loong.score(tid, nonsense) <= 40


@pytest.mark.judge
def test_score_returns_value_in_1_100_range(ids) -> None:
    tid = ids[0]
    gold = loong.get_task_answer(tid)
    rating = loong.score(tid, judge._format_gold(gold))
    assert 1 <= rating <= 100


@pytest.mark.judge
def test_score_zh_gold_answer_rates_high() -> None:
    """End-to-end ZH: feeding a Chinese gold answer back rates high — proves the
    judge grades Chinese (the question + gold are zh), no code change needed."""
    zh_str = next(
        t for t in loong.get_task_ids(languages=["zh"])
        if isinstance(loong.get_task_answer(t), str) and loong.get_task_answer(t).strip()
    )
    assert loong.score(zh_str, loong.get_task_answer(zh_str)) >= 70
