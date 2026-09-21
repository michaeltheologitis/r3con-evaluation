"""Dracula showcase mini-benchmark — loader surface + the DUAL (strict/lenient) judge.

The judge tests are fake-driven (no marker, no cost): `litellm_chat_completion` is
monkeypatched, and the fake decides its verdict by looking at WHICH gold answer the
user prompt carries — which is exactly the mechanism under test (one prompt, two
golds, combined in code). The marker-gated `judge` tests at the bottom hit the real
grader with the real near-misses from the measurement grid.
"""
from __future__ import annotations

import json

import pytest

from evals.benchmarks import dracula
from evals.benchmarks.dracula import judge
from evals.benchmarks.dracula.loader import _load

# The canonical lenient-pass from the 90-run grid (readagent, Qwen3.5-35B-A3B, seed 2):
# 12 victims, crew of nine stated, Mrs. Westenra + Lucy + Renfield present, no Swales.
LENIENT_PASS_ANSWER = (
    "Based on the provided text, Dracula killed **12 people** during the voyage to "
    "England and his stay there:\n\n"
    "*   **On the Voyage (*Demeter*):** The entire crew of nine men. (The log lists "
    "five hands, two mates, a cook, and the captain; all were reported missing, "
    "disappeared, or died by the time the ship arrived in Whitby).\n"
    "*   **During His Stay in England:**\n"
    "    *   **Mrs. Westenra**\n    *   **Lucy Westenra**\n    *   **Renfield**"
)

# A real raptor failure: INCLUDES Swales but never commits to nine ("the entire crew").
# It must fail BOTH metrics — this is the leniency-drift guard.
NO_CREW_COUNT_ANSWER = (
    "Based on the provided context, the text does not specify the exact number of crew "
    "members on the *Demeter* who died, only that the crew vanished one by one. The "
    "named victims are Mr. Swales, Mrs. Westenra, Lucy Westenra, and Renfield."
)

# --- v7 regressions: the two REAL grid answers that v6 wrongly passed ---------------
# Both name the complete roster but never state a number — no total, no crew count.
# v6's judge saw the gap and did the arithmetic itself ("the total is effectively the
# same... nine + four = 13"). v7 forbids that: a number not stated is not there.

# raptor, Qwen3.5-35B-A3B, seed 5 — VERBATIM from logs/dracula/raptor/99e8ba6ebce7.
# The word "nine" does not occur anywhere in it, and it gives no total; yet the v8 judge
# passed it claiming "it explicitly names the crew as nine". Kept byte-for-byte so the
# regression is the real artifact, not a paraphrase of it.
ENTIRE_CREW_NO_NUMBER_ANSWER = """Based on the provided text, Dracula killed the following people either directly or indirectly during his voyage to England and his stay there:

1.  **The Entire Crew of the *Demeter* (Voyage to England):** The ship's log records the deaths of the crew members. Specific roles and names mentioned as missing or dead include **Petrofsky** (missing July 16), the **Second Mate** (missing July 29), the **Watch** and **Steersman** (gone July 30), the **Mate** (insane/dead Aug 3), and the **Captain** (alone at wheel, log ends abruptly). The text states the log "records crew deaths," implying the entire crew perished.
2.  **Lucy Westenra (Stay in England):** She was killed directly by Dracula. The text notes she "suffers blood loss, gum shrinkage, throat punctures," and "Lucy dies." Van Helsing identifies her as an "Un-Dead vampire."
3.  **Mrs. Westenra (Stay in England):** She died indirectly from shock. The text states she "dies from shock after seeing a wolf at the window" (which was Dracula).
4.  **Mr. Swales (Stay in England):** He died indirectly. The text notes "Skeptic Mr. Swales dies neck broken" in the context of the Whitby storm and wolf incident associated with Dracula's arrival.
5.  **R.M. Renfield (Stay in England):** He was found dead. The text states "Renfield, found dead from a fall" and "Renfield is found dead from a fall." While the text does not explicitly state Dracula struck the blow, his death is linked to his worship of Dracula and the vampire's influence ("Renfield... had traded worship for control of flies and rats").

**Note:** **Quincey Morris** died during the hunt, but the text specifies he "dies from a gypsy knife," not directly by Dracula. **Jonathan Harker** and **Mina Harker** were attacked but survived.

**Summary:** Dracula killed the **entire crew of the *Demeter***, **Lucy Westenra**, **Mrs. Westenra**, **Mr. Swales**, and **Renfield**."""

# --- v8: the other side of the line — these MUST PASS -------------------------------
# readagent, Qwen3.5-35B-A3B, seed 6. Surfaces both numbers ("at least 13", "9 people"),
# full roster, correctly EXCLUDES the Hampstead children. v7 wrongly rejected it over the
# words "at least"; v8 accepts a qualifier around the correct number. It also enumerates
# individual crew members, which is elaboration of the counted group, not extra victims.
HEDGED_TOTAL_ANSWER = (
    "Based on the provided text, Dracula killed at least **13 people** directly or "
    "indirectly:\n\n"
    "1.  **The Crew of the *Demeter* (9 people):** This includes the Captain (who died by "
    "the ship), the First Mate (who jumped overboard), the Cook, and six other crew members "
    "(including Petrofsky, the watch, and the steersman). The Captain's log explicitly "
    "states, \"I now realize he killed the other men.\"\n"
    "2.  **Mr. Swales (Whitby):** Found dead with a broken neck, seemingly from fright.\n"
    "3.  **Mrs. Westenra (Hillingham):** Died of shock/horror after witnessing Lucy's "
    "condition.\n"
    "4.  **Lucy Westenra (Hillingham):** Died of vampirism/blood loss.\n"
    "5.  **Renfield (Asylum):** Found dead with a crushed skull, after an attack by the "
    "Count's mist form.\n\n"
    "*(Note: Several children in Hampstead were wounded, but the text does not explicitly "
    "confirm their deaths.)*"
)

# raptor, Qwen3.5-35B-A3B, seed 8 — explicitly declines to give the number.
CREW_NUMBER_UNSPECIFIED_ANSWER = (
    "Based on the provided context, the text does not specify the exact number of crew "
    "members on the *Demeter* who died, only stating that the \"crew vanishing one by one "
    "from terror, leaving the Captain and a mad Mate.\" However, the text explicitly names "
    "the following individuals killed by Dracula (directly or indirectly) during his voyage "
    "to England and his stay there:\n\n"
    "1.  **The Crew of the *Demeter*:** The text states the crew \"vanished one by one from "
    "terror\" during the voyage.\n"
    "2.  **Swales:** Found dead with his \"face frozen in fear\" in Whitby.\n"
    "3.  **Mrs. Westenra:** Dies after \"seeing a wolf at her window.\"\n"
    "4.  **Lucy Westenra:** Dies two days after her mother.\n"
    "5.  **Renfield:** Found dead with a \"broken neck\" while attempting to protect Mina.\n\n"
    "**Summary:**\n"
    "*   **Voyage:** The crew of the *Demeter* (number unspecified in text).\n"
    "*   **Stay in England:** Swales, Mrs. Westenra, Lucy Westenra, and Renfield."
)


# ============================================================
# Loader surface
# ============================================================


def test_task_ids_and_task_shape() -> None:
    ids = dracula.get_task_ids()
    assert ids == ["death_toll"]
    question, docs = dracula.get_task("death_toll")
    assert "how many people did Dracula kill" in question
    assert len(docs) == 46  # the full in-world corpus, every question shares it
    assert docs == dracula.get_documents("death_toll")


def test_corpus_order_is_fixed_and_puts_the_demeter_log_early() -> None:
    """The corpus order is a chosen constant, not an accident — pin what it buys.

    A method that reads the corpus as one linear stream meets each document with however
    much it has already absorbed, so POSITION decides how much context precedes the
    evidence. Book order buries the log of the *Demeter* at ~75% (the pooled Harker and
    Seward journals are 105k of 161k words and come first). ``_CORPUS_ORDER_SEED`` pulls
    it to 6th of 46 behind only short correspondence, so the muster the ``death_toll``
    gold turns on is read against almost no prior context.
    """
    docs = dracula.get_documents("death_toll")
    assert docs == dracula.get_documents("death_toll")        # deterministic across calls
    assert docs != sorted(docs)                                # actually re-ordered, not book order

    position = next(i for i, d in enumerate(docs) if d.startswith("LOG OF THE \u201cDEMETER"))
    assert position == 5                                       # 6th of 46
    # ...behind only short correspondence — no journal, no cutting, no memorandum.
    words_before = sum(len(d.split()) for d in docs[:position])
    assert words_before < 700, f"{words_before} words precede the log — the order drifted"
    # The whole log therefore lands inside the first ~5000 tokens of the pooled stream,
    # which is what makes "meets the nine deaths cold" measurable.
    assert "Crew, five hands" in "\n\n".join(docs)[:20_000]


def test_every_document_is_a_whole_standalone_artifact() -> None:
    """One in-world artifact per file, and every file whole.

    Regression for two splitter defects (2026-08-23). (1) The log of the *Demeter*
    is transcribed INSIDE the Dailygraph cutting: the correspondent writes his
    lead-in, says he will "send you a rescript", prints the log, then resumes in his
    own voice with the inquest and the funeral. Splitting at the log's header without
    rejoining left the cutting ending mid-sentence at "time being short." and filed
    854 characters of the reporter's prose at the end of the "log". (2) The
    "_Extra Special._" second edition of the 25 September *Westminster Gazette* has an
    INDENTED italic header and was being swallowed into the first edition.
    """
    docs = dracula.get_documents("death_toll")

    cutting = next(d for d in docs if d.startswith("CUTTING FROM \u201cTHE DAILYGRAPH"))
    log = next(d for d in docs if d.startswith("LOG OF THE \u201cDEMETER"))
    # the source is hard-wrapped, so compare on whitespace-normalized text
    flat = lambda t: " ".join(t.split())
    c, l = flat(cutting), flat(log)

    # the cutting opens AND closes in the correspondent's own voice
    assert "send you a rescript" in c                      # promises the transcript
    assert "saw the dead seaman whilst actually lashed to the wheel" in c
    assert "verdict was an open one" in c                  # resumes after the transcript
    assert c.endswith("\u201cmystery of the sea.\u201d")
    assert "two mates, cook, and myself (captain)" not in c   # carries none of the log

    # the log is the log, whole, ending on the captain's own last entry
    assert "two mates, cook, and myself (captain)" in l    # the muster
    assert "found no one there" in l                       # the 3 August loss
    assert "tie my hands to the wheel" in l                # his final entry
    assert l.endswith("trying to do his duty....")
    assert "public funeral" not in l                       # the correspondent's, not his

    # the two Westminster Gazette editions are two standalone documents
    gazettes = [d for d in docs if d.startswith("_\u201cThe Westminster Gazette,\u201d")]
    assert len(gazettes) == 2
    first = next(d for d in gazettes if "A HAMPSTEAD MYSTERY" in d)
    extra = next(d for d in gazettes if "_Extra Special._" in d)
    assert "THE HAMPSTEAD HORROR" in extra
    assert "THE HAMPSTEAD HORROR" not in first


def test_unknown_task_id_raises() -> None:
    for fn in (dracula.get_task, dracula.get_documents, dracula.get_task_answer):
        with pytest.raises(KeyError):
            fn("not-a-real-task-id")


def test_questions_json_carries_both_golds() -> None:
    """The lenient metric is a SECOND gold in the data, not a second prompt."""
    row = _load()["death_toll"]
    assert row["answer"].startswith("13 —")
    assert "Mr. Swales" in row["answer"]
    assert row["answer_lenient"].startswith("12 —")
    assert "Swales" not in row["answer_lenient"]
    # Everything else is identical between the two rosters.
    for who in ("crew of nine", "Mrs. Westenra", "Lucy Westenra", "Renfield"):
        assert who in row["answer"] and who in row["answer_lenient"]


# ============================================================
# Verdict encoding (the pair rides in the shared ScoreResult's `parsed`)
# ============================================================


@pytest.mark.parametrize("strict,lenient", [(0, 0), (0, 1), (1, 1)])
def test_verdict_encoding_round_trips(strict: int, lenient: int) -> None:
    assert judge.read_verdicts(judge.encode_verdicts(strict, lenient)) == (strict, lenient)


def test_read_verdicts_rejects_foreign_or_missing_values() -> None:
    """A pre-v6 score.json (parsed=None) or a label benchmark's parsed decodes to None."""
    assert judge.read_verdicts(None) is None
    assert judge.read_verdicts("") is None
    assert judge.read_verdicts("D.") is None
    assert judge.read_verdicts("strict=correct") is None  # no lenient half


# ============================================================
# The dual judge — fake-driven
# ============================================================


@pytest.fixture
def fake_judge(monkeypatch):
    """Patch the LLM seam; the fake verdicts are keyed by which gold is in the prompt.

    Returns the list of recorded calls so a test can assert the exact call graph.
    """
    calls: list[dict] = []

    def make(verdict_for_strict: str, verdict_for_lenient: str):
        def fake(*, system_prompt, user_prompt, model, schema, seed, **kwargs):
            is_strict = "13 —" in user_prompt
            calls.append(
                {"system_prompt": system_prompt, "user_prompt": user_prompt,
                 "gold": "strict" if is_strict else "lenient", "seed": seed}
            )
            verdict = verdict_for_strict if is_strict else verdict_for_lenient
            return judge.Verdict(reasoning=f"reasoning-{verdict}", verdict=verdict)

        monkeypatch.setattr(judge, "litellm_chat_completion", fake)
        return calls

    return make


def test_two_calls_per_answer_same_prompt_two_golds(fake_judge) -> None:
    """One answer → exactly TWO calls: same system prompt, the two different golds."""
    calls = fake_judge("incorrect", "correct")
    dracula.score("death_toll", "some answer")

    assert len(calls) == 2
    assert {c["gold"] for c in calls} == {"strict", "lenient"}
    # The SAME prompt drives both — no hand-written "relaxed" prompt exists.
    assert calls[0]["system_prompt"] == calls[1]["system_prompt"] == judge.SYSTEM_PROMPT
    assert all(c["seed"] == judge._JUDGE_SEED for c in calls)
    # Each call carries its own gold, and only its own.
    strict_call = next(c for c in calls if c["gold"] == "strict")
    lenient_call = next(c for c in calls if c["gold"] == "lenient")
    assert "Mr. Swales" in strict_call["user_prompt"]
    assert "12 —" in lenient_call["user_prompt"]


@pytest.mark.parametrize(
    "strict_verdict,alt_verdict,expected",
    [
        ("correct", "incorrect", (1, 1)),   # strict-correct → lenient inherits it (the OR)
        ("incorrect", "correct", (0, 1)),   # Swales-only omission → the lenient rescue
        ("incorrect", "incorrect", (0, 0)),  # wrong for other reasons → fails both
        ("correct", "correct", (1, 1)),
    ],
)
def test_lenient_is_the_or_of_both_golds(fake_judge, strict_verdict, alt_verdict, expected) -> None:
    fake_judge(strict_verdict, alt_verdict)
    assert dracula.score("death_toll", "some answer") == expected


def test_lenient_never_below_strict(fake_judge) -> None:
    """The load-bearing invariant: a strict-correct answer is ALWAYS lenient-correct.

    (Without the `or`, judging a 13-with-Swales answer against the 12-gold would fail
    it — two rival rosters instead of strict+lenient.)
    """
    fake_judge("correct", "incorrect")
    strict, lenient = dracula.score("death_toll", "13 victims, Swales included")
    assert strict == 1 and lenient >= strict


def test_score_details_shape_is_the_shared_scoreresult(fake_judge) -> None:
    """No new generic fields: the pair rides in `parsed`, both rationales in `rationale`."""
    fake_judge("incorrect", "correct")
    (result,) = dracula.score_details(["death_toll"], ["an answer"])

    assert set(result) == {"score", "parsed", "rationale", "model"}
    assert result["score"] == 0                       # `score` is the STRICT metric
    assert judge.read_verdicts(result["parsed"]) == (0, 1)
    assert "[strict gold" in result["rationale"]
    assert "[lenient gold" in result["rationale"]
    assert result["model"] == "gpt-5-4-mini"


def test_score_json_payload_preserves_the_pair(fake_judge, tmp_path) -> None:
    """The generic writer whitelists keys — assert the lenient grade survives anyway.

    Mirrors the payload a grader builds from a manifest.
    """
    fake_judge("incorrect", "correct")
    (result,) = dracula.score_details(["death_toll"], ["an answer"])
    payload = {
        "task_id": "death_toll", "config": {}, "score": result["score"],
        "scorer": dracula.SCORER, "scored_with": "gpt-5-4-mini",
        "parsed": result.get("parsed"), "rationale": result.get("rationale"),
    }
    path = tmp_path / "score.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["score"] == 0
    assert judge.read_verdicts(reloaded["parsed"]) == (0, 1)


def test_score_batch_returns_pairs_in_input_order(fake_judge) -> None:
    fake_judge("incorrect", "correct")
    results = dracula.score_batch(["death_toll", "death_toll"], ["a", ""])
    assert results == [(0, 1), (0, 0)]  # the empty answer fails both, unjudged


def test_empty_answer_fails_both_without_a_paid_call(fake_judge) -> None:
    calls = fake_judge("correct", "correct")
    assert dracula.score("death_toll", "") == (0, 0)
    assert dracula.score("death_toll", "   ") == (0, 0)
    assert calls == []  # no LLM call was issued

    (result,) = dracula.score_details(["death_toll"], [""])
    assert result["rationale"] is None
    assert judge.read_verdicts(result["parsed"]) == (0, 0)


def test_scorer_id_bumped_for_the_dual_mechanism() -> None:
    """A mechanism change must flip the score.json cache key."""
    assert dracula.SCORER == "dracula-judge-v9"


def test_prompt_states_the_one_rule_and_the_output_contract() -> None:
    """v9 is ONE rule + examples; the rule and the two output fields must be stated."""
    prompt = judge.SYSTEM_PROMPT
    assert "Never work a number out for yourself" in prompt
    assert "if you cannot quote it from" in prompt
    # The structured-output contract the Verdict schema enforces.
    assert "`reasoning` first, then `verdict`" in prompt
    assert '"correct" or "incorrect"' in prompt
    # The input sections the judge is shown, matching _user_prompt's headings.
    for heading in ("## Question", "## Golden (reference) answer", "## Model's answer to judge"):
        assert heading in prompt and heading in judge._user_prompt("q", "g", "m")


def test_prompt_examples_carry_no_dracula_content() -> None:
    """The examples must NOT reuse the real golds, or the judge can copy their verdicts.

    They teach the function (grade M against G) on an unrelated scenario; anything from
    this benchmark's own roster leaking in is contamination.
    """
    prompt = judge.SYSTEM_PROMPT.lower()
    for leaked in ("dracula", "demeter", "swales", "westenra", "renfield", "lucy",
                   "whitby", "nine", "thirteen", "13", "12"):
        assert leaked not in prompt, f"example contamination: {leaked!r} in the prompt"
    # ...and the unrelated scenario IS there, with all six worked examples.
    assert "sled drivers" in prompt
    assert prompt.count("═══ example") == 6


def test_prompt_examples_show_both_verdicts_with_reasoning() -> None:
    """Every example must demonstrate the reasoning-then-verdict shape, both outcomes."""
    prompt = judge.SYSTEM_PROMPT
    assert prompt.count("reasoning:") == 6
    assert prompt.count("verdict: correct") == 2      # examples 1 and 4
    assert prompt.count("verdict: incorrect") == 4    # examples 2, 3, 5, 6


def test_guards() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        dracula.score_details(["death_toll"], ["a", "b"])
    assert dracula.score_details([], []) == []
    with pytest.raises(KeyError):
        dracula.score_details(["nope"], ["a"])


# ============================================================
# Real grader — costs money, run only with `pytest -m judge`
# ============================================================


@pytest.mark.judge
def test_live_gold_answer_passes_both() -> None:
    gold = dracula.get_task_answer("death_toll")
    assert dracula.score("death_toll", gold) == (1, 1)


@pytest.mark.judge
def test_live_swales_omission_is_the_lenient_rescue() -> None:
    """The canonical case: readagent seed 2 — right but for Swales → (0, 1)."""
    assert dracula.score("death_toll", LENIENT_PASS_ANSWER) == (0, 1)


@pytest.mark.judge
def test_live_missing_crew_count_fails_both() -> None:
    """The drift guard: includes Swales, never commits to nine → lenient must NOT rescue it."""
    assert dracula.score("death_toll", NO_CREW_COUNT_ANSWER) == (0, 0)


def test_the_entire_crew_answer_really_contains_no_crew_number() -> None:
    """Guards the premise of the regression below — no LLM, so this can never be flaky.

    If this ever fails, the fixture was edited and the live test below is meaningless.
    """
    low = ENTIRE_CREW_NO_NUMBER_ANSWER.lower()
    assert "nine" not in low
    assert "crew of 9" not in low and "(9 " not in low
    assert "entire crew" in low          # it names the group but never sizes it
    for who in ("swales", "mrs. westenra", "lucy westenra", "renfield"):
        assert who in low                 # the roster IS complete — only the number is missing


@pytest.mark.judge
def test_live_entire_crew_without_a_number_fails_both() -> None:
    """THE regression: raptor seed 5 verbatim — full roster, no crew size, no total.

    Must be (0,0): 'the entire crew' does not state how many the Demeter's crew was,
    and the muster arithmetic is the benchmark's discriminating evidence site.

    This case was a coin flip under the v6–v8 rule-list prompts — it flipped on every
    rescore (0,1,0,1), with the judge once asserting "it explicitly names the crew as
    nine" of an answer in which the word never occurs. Under v9's example-steered
    prompt it measured 8/8 stable at (0,0); the xfail marker it used to carry is gone.
    """
    assert dracula.score("death_toll", ENTIRE_CREW_NO_NUMBER_ANSWER) == (0, 0)


@pytest.mark.judge
def test_live_crew_number_unspecified_fails_both() -> None:
    """v7 regression (raptor seed 8): the answer SAYS the number is unspecified.

    v6 passed this too — 'the total 13 is effectively represented'.
    """
    assert dracula.score("death_toll", CREW_NUMBER_UNSPECIFIED_ANSWER) == (0, 0)


@pytest.mark.judge
def test_live_hedged_total_with_both_numbers_passes() -> None:
    """v8 regression (readagent seed 6): 'at least 13' + '(9 people)' + full roster.

    The numbers are BOTH there and the roster is right, so it is correct — v7 rejected
    it over the words 'at least', which was pedantic. It also enumerates individual crew
    members (Petrofsky, the watch, the steersman); those are the counted group, not
    extra victims. Strict passes, so lenient inherits it via the OR.
    """
    assert dracula.score("death_toll", HEDGED_TOTAL_ANSWER) == (1, 1)


@pytest.mark.judge
def test_live_extra_victim_fails_both() -> None:
    """Arthur's father dies of natural causes in-window — an addition, not a victim."""
    answer = (
        "14 people: the Demeter's crew of nine, Mr. Swales, Mrs. Westenra, "
        "Lucy Westenra, Renfield, and Arthur's father Lord Godalming."
    )
    assert dracula.score("death_toll", answer) == (0, 0)


@pytest.mark.judge
def test_live_empty_answer_fails_both_unjudged() -> None:
    assert dracula.score("death_toll", "") == (0, 0)


# --- one live test per rule the prompt states ---------------------------------------


@pytest.mark.judge
def test_live_wrong_crew_count_fails_both() -> None:
    """The compression family's real signature: a stated but WRONG crew size.

    readagent/structrag/rlm produced 6, 12 and 20 from a muster that reads nine.
    A number being present is not enough — it has to be the right one.
    """
    answer = (
        "10 people. On the voyage the *Demeter* lost its crew of six, and in England "
        "Dracula killed Mr. Swales, Mrs. Westenra, Lucy Westenra, and Renfield."
    )
    assert dracula.score("death_toll", answer) == (0, 0)


@pytest.mark.judge
def test_live_missing_a_roster_member_fails_both() -> None:
    """Mrs. Westenra dropped — the golden list is exhaustive, so 12 with a gap fails.

    Distinct from the Swales case: only Swales is optional under the lenient gold.
    """
    answer = (
        "12 people: the Demeter's crew of nine, Mr. Swales, Lucy Westenra, and Renfield."
    )
    assert dracula.score("death_toll", answer) == (0, 0)


@pytest.mark.judge
def test_live_numbers_spelled_out_are_fine() -> None:
    """Formatting is not the test — 'thirteen'/'nine' as words must pass."""
    answer = (
        "Thirteen people: the Demeter's crew of nine men, plus Mr. Swales, "
        "Mrs. Westenra, Lucy Westenra and Renfield."
    )
    assert dracula.score("death_toll", answer) == (1, 1)


@pytest.mark.judge
def test_live_excluding_non_victims_is_allowed() -> None:
    """'Mentioning an item only to exclude it is fine' — the in/out-of-window reasoning.

    Naming Hawkins / Quincey / Skinsky as deliberate EXCLUSIONS is good analysis and
    must not be penalised as extra victims.
    """
    answer = (
        "13 people: the Demeter's crew of nine, Mr. Swales, Mrs. Westenra, Lucy Westenra "
        "and Renfield. Not counted: Mr. Hawkins and Arthur's father (natural deaths), "
        "Quincey Morris and Petrof Skinsky (after Dracula fled England), and the castle "
        "child and its mother (before the voyage). The Hampstead children survived."
    )
    assert dracula.score("death_toll", answer) == (1, 1)


@pytest.mark.judge
def test_live_total_contradicting_its_own_roster_fails_both() -> None:
    """Says 13 but lists only twelve — the total must match the roster presented."""
    answer = (
        "13 people: the Demeter's crew of nine, Mrs. Westenra, Lucy Westenra, and Renfield."
    )
    assert dracula.score("death_toll", answer) == (0, 0)


@pytest.mark.judge
def test_live_superset_of_the_window_fails_both() -> None:
    """The old gold-15 answer: correct roster PLUS the castle pair (out of window)."""
    answer = (
        "15 people: the child at the castle and its mother, the Demeter's crew of nine, "
        "Mr. Swales, Mrs. Westenra, Lucy Westenra, and Renfield."
    )
    assert dracula.score("death_toll", answer) == (0, 0)


@pytest.mark.judge
def test_live_collective_naming_without_enumeration_is_fine() -> None:
    """The counted group may be named collectively — the SIZE is what must be stated."""
    answer = (
        "13 — the Demeter's crew of nine, Mr. Swales, Mrs. Westenra, Lucy Westenra, Renfield."
    )
    assert dracula.score("death_toll", answer) == (1, 1)
