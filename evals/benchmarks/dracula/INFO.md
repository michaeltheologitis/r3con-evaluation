# The Dracula mini-benchmark

*Dracula* (1897, public domain, Project Gutenberg #345) is epistolary: it has no narrator
and presents itself as a dossier of primary documents. `split.py` partitions the Gutenberg
text losslessly back into those **46 in-world documents** (160,756 words), and we pose
hand-curated compositional questions over them — questions whose answers exist in no
single passage.

## The corpus

The five running journals (Jonathan Harker, Mina Murray, Mina Harker, Lucy Westenra,
Dr. Seward) are pooled one document each and hold most of the text; the other 41 are
individual letters, telegrams, newspaper cuttings, memoranda, reports, the ship's log, a
phonograph message and two notes. One in-world artifact per file, every file whole. Where
one document is printed inside another — the log of the *Demeter* inside the newspaper
cutting that transcribes it — both are documents and the container is rejoined around it.

`get_documents` returns the corpus shuffled once with a fixed seed, so document position
carries no signal from the novel's own arrangement and every method sees the same stream.
The seed is chosen, not arbitrary: it places the log of the *Demeter* 6th of 46, so the
muster the gold turns on is read early, against almost no prior context. Changing it
re-orders the corpus for every method.

## The question

> **`death_toll`** — "On the voyage to England and during his stay there, how many people
> did Dracula kill, either directly or indirectly, and who were they?"

**Gold (strict): 13** — the Demeter's crew of nine, Mr. Swales, Mrs. Westenra, Lucy
Westenra, Renfield.

**Gold (lenient): 12** — the same roster without Mr. Swales, the most inferential
attribution in it.

Counting rules, recorded in the question's `notes`:

- The window is two-sided: from setting sail for England (6 July) to fleeing England
  (early October).
- The crew of nine is counted from the log's muster. Identifying them collectively is
  correct — most are never named.
- "Directly or indirectly" is load-bearing: several victims are killed by Dracula's
  instruments (a wolf, terror) rather than his hands.

## The roster and where it comes from

| victim | document | basis |
| --- | --- | --- |
| crew of nine | Log of the *Demeter* (via the Dailygraph cutting) | muster "five hands … two mates, cook, and myself" = 9; all nine gone by the ship's arrival |
| Mr. Swales | Mina Murray's Journal | found dead on the churchyard seat, neck broken, "in some sort of fright"; Dracula had landed at Whitby two days earlier |
| Mrs. Westenra | Lucy Westenra's memorandum + the Pall Mall Gazette zoo interview | a wolf comes through the window, her weak heart fails; the escaped wolf ties back to Dracula |
| Lucy Westenra | Dr. Seward's Diary | successive blood loss and the throat wounds, across the transfusion sequence |
| Renfield | Dr. Seward's Diary | beaten by Dracula in his cell after the bargain breaks |

The full text of every document is in `docs/`, so any of these is checkable directly.

## What must not be counted

- **Before the window** — the castle child and its mother, both killed in Transylvania in
  May–June, before the 6 July sailing. (These two are the difference between an earlier
  gold of 15 and the current 13.)
- **After the window** — Petrof Skinsky, killed at Galatz after Dracula has fled England;
  and Quincey Morris, who is outside the window *and* cut down by the Szgany, not Dracula.
- **Natural deaths inside the window** — Mr. Hawkins, elderly and far from Dracula, with
  no wound or vampiric signature; and Arthur's father, a long decline tracked across
  several documents.

Decoys: the "bloofer lady" children all survive; Mina survives; and Van Helsing's
destruction of un-dead Lucy and the three sisters are releases of beings already dead, not
kills.

## Why it is hard

Reaching 13 chains five operations, each defeating a different strategy:

1. **Arithmetic over a muster line** — the number nine is never stated, it must be
   computed. Retrieval cannot find a passage saying "nine".
2. **State-tracking across eight log entries** to confirm all nine die. Lossy summaries
   produce 6, 12 or 20.
3. **Indirect causal attribution** — Swales (fright), Mrs. Westenra (wolf), the captain
   (terror), the mate (suicide) count only under "directly or indirectly", and no
   kill/murder cue sits anywhere near these deaths.
4. **Two-sided temporal windowing**, which is what excludes four real Dracula kills.
5. **Kill-vs-not disambiguation** across natural deaths, survivors and un-dead releases.

The evidence is dispersed by design: a ship's log, two newspaper cuttings, a private
journal, a dying woman's memorandum and a doctor's diary — and the two halves of one
victim's chain (Mrs. Westenra) sit in different documents written by different people.

## Scoring

An LLM judge runs twice, giving two 0/1 metrics (`score` returns the pair): **strict**
against the 13-victim gold, **lenient** against the 12-victim one, combined as
`lenient = max(strict, alt)` so lenient is always a superset.

Both calls use the same prompt; only the gold differs. The prompt states one rule — every
number in the gold must be quotable from the model's answer, never supplied by the grader
— plus worked examples on an unrelated scenario, so the grader cannot copy an example's
verdict onto the real roster.
