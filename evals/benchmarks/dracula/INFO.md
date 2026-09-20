# The Dracula showcase mini-benchmark

A reference for the `dracula` benchmark: what we built from Bram Stoker's novel, what
the corpus contains, what the question asks, and the complete evidence attribution
behind its gold answer. Every quotation below was read directly from the files in
`docs/` and is cited by **document name and line numbers**; the corresponding filenames
are listed in §3.

---

## 1. What we did, in one paragraph

*Dracula* (1897, public domain — Project Gutenberg #345) is an **epistolary novel**: it
has no narrator, and presents itself as a dossier of primary documents — journals,
letters, telegrams, newspaper cuttings, a ship's log, memoranda — compiled by its
characters. We **decompiled the novel back into those documents**. `split.py` performs a
lossless partition of the Gutenberg text into **46 in-world documents** (160,756 words),
which become a multi-document corpus for grounded-reasoning evaluation. Onto that corpus
we pose hand-curated **compositional questions** whose answers exist in no single passage
and must be assembled across many documents. The first such question, `death_toll`, asks
how many people Dracula killed on the voyage to England and during his stay there, and
who they were. It is designed so that retrieval, summarization, and lexical scanning each
fail in a different, diagnosable way.

---

## 2. The novel and how Stoker structures it

### 2.1 The epistolary frame

*Dracula* is told entirely through documents written by participants, each with a date,
an author, and a medium. There is no omniscient narrator; the reader knows only what some
character wrote down. Consequences that matter for us:

- **Every fact is situated.** It carries an author, a date, and that author's limits.
  Mina records Lucy's throat wounds while believing she caused them with a safety-pin;
  the reader must correct her.
- **No character sees the whole.** Harker knows what happened in the castle but is
  delirious for six weeks; Seward watches Renfield without knowing why he raves; Mina
  reads a newspaper cutting that means nothing to her at the time. The truth exists only
  in the *union* of the documents.
- **Dramatic irony is structural.** Lucy writes "Four days and nights of peace" hours
  before she dies (*Lucy Westenra's Diary*, line 67).
- **The plot is a document-synthesis problem.** The characters defeat Dracula by
  collecting and cross-reading their papers. Mina types Seward's phonograph cylinders and
  "knit[s] together in chronological order every scrap of evidence they have"
  (*Dr. Seward's Diary*, lines 3456–3457); Van Helsing's contingency note instructs
  Seward to "take the papers that are with this, the diaries of Harker and the rest, and
  read them, and then find this great Un-Dead" (*Note left by Van Helsing in his
  portmanteau*, lines 32–35); Mina's memorandum is a formal deduction — "(a)… (b)… a
  process of exclusions" — that locates the fleeing Count from documents alone
  (*Mina Harker's Memorandum*).

That last point is why this novel was chosen: **it is a grounded multi-document reasoning
task that happens to have been written as a novel in 1897.** The heroes win by doing what
we are asking a system to do.

### 2.2 The reconstruction

Stoker's chapter divisions are an artifact of book publishing; the *documents* are the
real units. `split.py` restores them:

- **Lossless partition.** Verified by word-stream conservation (every word of the source
  lands in exactly one document), byte-equal regrouping, and zero foreign headers.
- **Boundary audit.** A line-by-line review of all 97 header-shaped lines in the book.
  This caught two real defects: capitalised `--_continued_` headers failed an all-caps
  test and silently swallowed 7 section breaks, and the italic-keyword list lacked
  "Note", which had merged **Van Helsing's undelivered portmanteau note** into a
  neighbouring document. That note now stands as its own document.
- **Trailing matter removed.** 1920s Grosset & Dunlap publisher advertisements riding
  inside the closing Note were trimmed.
- **Pooling decision.** The five running journals/diaries are each pooled into **one**
  document (a diary is one artifact, kept over months); every other item — each letter,
  telegram, cutting, memorandum, report, the log, both notes — stands alone. A finer
  116-section variant was built and then deliberately dropped: section seams could fall
  mid-entry, and two granularities cost more confusion than the ablation was worth.

### 2.3 The benchmark surface

`dracula` is a normal benchmark package (`get_task_ids` / `get_task` / `get_documents` /
`get_task_answer` / `score`). Every question shares the **same full 46-document corpus**,
returned in a fixed shuffled order (`_CORPUS_ORDER_SEED` in `loader.py`), so document
position carries no signal from the novel's own arrangement. The seed is chosen rather than
arbitrary: for a method that reads the corpus as one linear stream, a document's position
decides how much context precedes it, and book order buries the log of the *Demeter* at
~75% of the stream (the pooled Harker and Seward journals are 105,000 of 161,000 words and
come first). The seed places the log **6th of 46**, behind three short letters and two
telegrams — 502 words — so the muster in §5.1 is read early, against almost no prior
context.
Because all questions share one byte-identical doc-set, an index-building baseline
resolves them to a single index.

---

## 3. The corpus: 46 documents, 160,756 words

| type | n | words | what they are |
|---|---:|---:|---|
| **journal / diary** | 5 | 135,209 | the five running personal records, each pooled into one document |
| **letter** | 20 | 9,620 | private correspondence + business/solicitors' letters |
| **memorandum** | 4 | 6,196 | formal accounts written to be read by others |
| **newspaper cutting** | 4 | 5,509 | public reportage pasted into journals — incl. both editions of the 25 September *Westminster Gazette* |
| **ship's log** | 1 | 1,853 | the *Demeter*'s log, recovered in a bottle — transcribed inside the Dailygraph cutting, kept as its own document |
| **report** | 1 | 996 | a doctor's formal report to another doctor |
| **note** | 2 | 734 | Van Helsing's contingency note; the closing note |
| **phonograph message** | 1 | 410 | Van Helsing dictating into Seward's machine |
| **telegram** | 8 | 229 | terse operational wires, several decisive |
| **TOTAL** | **46** | **160,756** | |

### 3.1 The five journals (the bulk of the text)

| document | words | filename | covers |
|---|---:|---|---|
| **Dr. Seward's Diary** | 61,193 | `05_dr_seward_s_diary.txt` | Renfield; Lucy's illness and death; the hunt. **Dictated to a phonograph** |
| **Jonathan Harker's Journal** | 43,680 | `01_jonathan_harker_s_journal.txt` | Transylvania, the castle, imprisonment; later the London hunt and pursuit. **Shorthand** |
| **Mina Harker's Journal** | 20,056 | `32_mina_harker_s_journal.txt` | the hunt, her infection, the pursuit to Transylvania |
| **Mina Murray's Journal** | 9,336 | `08_mina_murray_s_journal.txt` | Whitby: the wreck, Swales, Lucy's sleepwalking |
| **Lucy Westenra's Diary** | 944 | `16_lucy_westenra_s_diary.txt` | her decline, in her own words |

The media differ — shorthand, wax cylinders, shorthand-then-typescript — which is why
Mina's transcription work is what makes the dossier legible to the group at all.

### 3.2 The other 41 artifacts

**Letters (20).** Mina↔Lucy before anything happens (3) · Quincey to Arthur (1)
— Arthur's reply is a telegram, counted below · **Billington & Son ↔ Carter, Paterson & Co.** — the shipping of the fifty
boxes to Carfax (2) · Sister Agatha, on Harker's brain fever (1) · Mina's marriage letter
and Lucy's reply (2) · the Holmwood / Seward / Van Helsing medical thread (5) · Mina to
Lucy, including the Hawkins-death letter and one marked *"Unopened by her"* (2) · Van
Helsing ↔ Mina after Lucy's death (3) · **Mitchell, Sons & Candy** — who bought 347
Piccadilly (1).

**Telegrams (8).** Arthur to Quincey · Arthur to Seward · Seward's three daily status
wires to Van Helsing in Amsterdam (a clinical chart in three lines) · **Van Helsing,
Antwerp, to Seward** — "do not fail to be at Hillingham to-night", *misaddressed to
Carfax, Sussex and delivered 22 hours late* · Mina to Van Helsing · the Lloyd's wire
reporting the *Czarina Catherine*.

**Newspaper cuttings (4).** *The Dailygraph*, 8 August (the wreck of the *Demeter*) ·
***The Pall Mall Gazette*, 18 September** (the escaped-wolf interview) · *The Westminster
Gazette*, 25 September (the "bloofer lady") — and its *Extra Special* second edition of
the same day (THE HAMPSTEAD HORROR), a separate document.

**Memoranda (4).** *Memorandum left by Lucy Westenra* (her account of the night her mother
died — written as she herself is dying) · *Mina Harker's Memorandum* (the deduction of
Dracula's escape route) · two Van Helsing memoranda from Transylvania.

**Others.** the *Log of the "Demeter"* · *Report from Dr. Patrick Hennessey to Dr. Seward*
(on Renfield) · *Note left by Van Helsing in his portmanteau* (undelivered) ·
*Dr. Seward's Phonograph Diary, Spoken by Van Helsing* · the closing *Note*, seven years
later.

### 3.3 Timeline

| date | event | documents |
|---|---|---|
| 3 May – 30 June | Harker at Castle Dracula | Harker's Journal |
| 6 July | the *Demeter* sails from Varna | Log of the *Demeter* |
| 16 July – 4 Aug | the crew vanish one by one | Log of the *Demeter* |
| 7–8 Aug | the wreck at Whitby; Dracula comes ashore | *The Dailygraph*; Mina Murray's Journal |
| 10 Aug | **Mr. Swales found dead** in the churchyard | Mina Murray's Journal |
| 11 Aug | Lucy attacked on the churchyard seat | Mina Murray's Journal |
| 17 Aug – 17 Sep | Lucy's decline; four transfusions | Seward's Diary; Lucy's Diary; the Holmwood/Van Helsing letters |
| 17–18 Sep | escaped wolf; **Mrs. Westenra dies** | *The Pall Mall Gazette*; Lucy's Memorandum |
| 20 Sep | **Lucy dies** | Seward's Diary |
| 25 Sep | the "bloofer lady" reports | *The Westminster Gazette* |
| 3 Oct | **Renfield dies**; Mina attacked | Seward's Diary; Hennessey's Report |
| early Oct | the boxes sterilised; Dracula flees England | Harker's & Mina's Journals; Mitchell Sons & Candy; the Phonograph Diary |
| Oct–Nov | pursuit; Skinsky killed; Dracula destroyed; Quincey Morris dies | Harker's & Mina's Journals; the Van Helsing memoranda |
| +7 years | the closing Note | *Note* |

---

## 4. The question

> **`death_toll`** — "On the voyage to England and during his stay there, how many people
> did Dracula kill, either directly or indirectly, and who were they?"

**Gold (strict): 13** — (1) the Demeter's crew of nine, (2) Mr. Swales, (3) Mrs.
Westenra, (4) Lucy Westenra, (5) Renfield.

**Gold (lenient): 12** — the same roster with Mr. Swales omitted. Swales is the roster's
most inferential attribution, so an answer that leaves him out is graded correct under
the lenient metric; everything else stays strict. (See §8.)

**Counting rules**, recorded in the question's `notes`:

- The window is two-sided: from **setting sail for England** (6 July) to **fleeing
  England** (early October).
- The crew of nine is counted from the log's muster. **Identifying them collectively is
  correct** — most are never named, so the nine cannot be enumerated.
- "Either directly or indirectly" is load-bearing: several victims are killed by
  Dracula's *instruments* (a wolf, terror) rather than his hands.
- Hawkins and Arthur's father die of natural causes inside the window and must not be
  listed.

---

## 5. Evidence attribution

For each victim the chain has three parts: **(i) that they died**, **(ii) that Dracula
caused it**, **(iii) that it falls inside the window**. Part (ii) is the part a naïve
reading skips.

### 5.1 The Demeter's crew of nine — victims 1–9

**(i) The size of the crew.** *Log of the "Demeter"*, lines 10–12:

> "On 6 July we finished taking in cargo, silver sand and boxes of earth. At noon set
> sail. East wind, fresh. Crew, five hands ... two mates, cook, and myself (captain)."

5 + 2 + 1 + 1 = **9**. This line is also the window's opening boundary.

**(ii) That all nine die**, tracked entry by entry through the *Log of the "Demeter"*:

| log date | lines | event | remaining |
|---|---|---|---|
| 16 Jul | 40–41 | "one of crew, Petrofsky, was missing" | 8 |
| 24 Jul | 82–84 | "Already a hand short … another man lost—disappeared" | 7 |
| 29 Jul | 101–104 | "Another tragedy … Are now without second mate" | 6 |
| 30 Jul | 109–112 | "both man of watch and steersman missing. Only self and mate and two hands left" | 4 |
| 2 Aug | 127–130 | "One more gone. Lord, help us!" | 3 |
| 3 Aug | 138–140 | "I went to relieve the man at the wheel … found no one there" | 2 |
| 3 Aug | 173–175 | the first mate: "he sprang on the bulwark and deliberately threw himself into the sea" | 1 |
| 4 Aug | 187–197 | the captain lashes himself to the wheel: "I shall tie my hands to the wheel when my strength begins to fail" | 0 |

The 30 July entry is where the **cook** is silently absorbed: before it there are six
souls, after it the captain names exactly four ("self and mate and two hands").

**(iii) That Dracula is the cause.** Three independent sightings by three men, inside the
log itself:

- *Log of the "Demeter"*, lines 49–55 — Olgaren "saw a tall, thin man, who was not like
  any of the crew, come up the companion-way, and go along the deck forward, and
  disappear."
- *Log of the "Demeter"*, lines 144–148 — the mate: *"It is here; I know it, now. On the
  watch last night I saw It, like a man, tall and thin, and ghastly pale."*
- *Log of the "Demeter"*, lines 184–185 — the captain: *"in the dimness of the night I saw
  It--Him!"*

**(iv) That the man aboard is Dracula — identified by other accounts.** The log never names
him. The identification is a join across documents written by people who were never
aboard:

1. **The cargo is his.** The log records "boxes of earth" (*Log*, line 10). At the castle,
   Harker had found the Count "in one of the great boxes, of which there were fifty in all,
   on a pile of newly dug earth" (*Jonathan Harker's Journal*, lines 1832–1833), and later
   lying gorged in the same box (lines 1973–1986). The shipment is consigned as **fifty
   boxes** to Carfax, his London house (*Letter, Billington & Son to Carter, Paterson &
   Co.*, lines 8–15).

2. **The newspaper names the ship and the cargo.** *The Dailygraph* cutting, in its
   9 August follow-up, lines 206–208: "the schooner is a Russian from Varna, and is called the *Demeter*. She is
   almost entirely in ballast of silver sand, with only a small amount of cargo--a number
   of great wooden boxes filled with mould."

3. **Harker ties the two together explicitly.** *Jonathan Harker's Journal*, lines
   2224–2226, after the group has cross-read the papers: "Of one thing I am now satisfied:
   that _all_ the boxes which arrived at Whitby from Varna in the _Demeter_ were safely
   deposited in the old chapel at Carfax. There should be fifty of them there." This is
   the **only** mention of the *Demeter* anywhere outside the log and the cutting.

4. **Something leaves the ship the instant it grounds — and Van Helsing says it was him.**
   *The Dailygraph*, lines 141–148: "an immense dog sprang up on deck from below … and
   running forward, jumped from the bow on the sand"; line 263: "No trace has ever been
   found of the great dog." The same cutting reports the coal merchant's mastiff found dead
   near the pier, "its throat was torn away, and its belly was slit open as if with a savage
   claw" (lines 226–230). Van Helsing later states the identification outright, listing the
   Count's powers — *Mina Harker's Journal*, lines 796–797: "He can transform himself to
   wolf, **as we gather from the ship arrival in Whitby, when he tear open the dog**."
   This is the clearest in-world statement in the corpus that Dracula came in on the
   *Demeter*.

5. **He could not have left mid-voyage.** A general rule of his nature, stated during the
   later pursuit: "The Count, even if he takes the form of a bat, cannot cross the running
   water of his own volition, and so cannot leave the ship" (*Jonathan Harker's Journal*,
   lines 3761–3763). *Note: this sentence is said of the* Czarina Catherine *on the return
   voyage, not of the* Demeter. *It applies to the outbound voyage by the same rule, but the
   text does not make that application — the reader does.*

So Dracula's **presence** aboard is stated, not merely inferred — by Harker (the cargo) and
by Van Helsing (the ship arrival). What remains an inference is that the "tall, thin man"
the crew saw, and the "It--Him" the captain faced, **killed** them: no document says
"Dracula killed the crew." That step joins the log's sightings to those later accounts —
written weeks afterwards by people who were never aboard, for other purposes (tracing the
boxes; cataloguing a vampire's powers). Neither Harker nor Van Helsing mentions a single
crewman.

**Corroboration on arrival.** *The Dailygraph*, 8 August, lines 127–129:

> "lashed to the helm was a corpse, with drooping head, which swung horribly to and fro at
> each motion of the ship. No other form could be seen on deck at all."

The inquest reaches no verdict — *The Dailygraph*, 8 August, lines 253–255 (the
correspondent resuming after the transcript): "whether or not the man himself committed
the murders there is now none to say" — which is precisely why the reader must do the
attribution the coroner could not.

**Note the composition this victim requires.** In the cutting the body is an anonymous
"dead seaman" (lines 165–166); it is the log found in his pocket — "myself (captain)"
(line 12) plus "I shall tie my hands to the wheel … and along with them I shall tie that
which He--It!--dare not touch" (lines 187–197) — that identifies him, matching the crucifix
bound into the cords (lines 171–173). **The captain is the only victim in the roster whose
identity is itself a cross-document inference**: the log ends with him alive and writing,
and the newspaper never names him.

### 5.2 Mr. Swales — victim 10

This is the roster's most inferential attribution, and it rests on **place and timing**,
not on a stated cause.

**(i) The death.** *Mina Murray's Journal*, 10 August, lines 407–412:

> "There is an additional cause in that poor old Mr. Swales was found dead this morning on
> our seat, his neck being broken. He had evidently, as the doctor said, fallen back in
> the seat in some sort of fright, for there was a look of fear and horror on his face
> that the men said made them shudder. Poor dear old man! Perhaps he had seen Death with
> his dying eyes!"

**(ii) That Dracula caused it — the chain.**

1. **Swales predicts it while the ship is in sight.** On 6 August, talking to Mina,
   *Mina Murray's Journal*, lines 349–352:

   > "Maybe it's in that wind out over the sea that's bringin' with it loss and wreck, and
   > sore distress, and sad hearts. Look! look! … There's something in that wind and in the
   > hoast beyont that sounds, and looks, and tastes, and smells like death. It's in the
   > air; I feel it comin'."

   In the *same journal entry*, lines 358–368, the coastguard is watching an unidentified
   vessel: "she's a Russian, by the look of her; but she's knocking about in the queerest
   way … She is steered mighty strangely, for she doesn't mind the hand on the wheel."
   That ship is the *Demeter*.

2. **Dracula comes ashore into Swales's churchyard.** The *Demeter* strikes Tate Hill Pier
   that night, and *The Dailygraph*, 8 August, lines 141–148:

   > "an immense dog sprang up on deck from below … and running forward, jumped from the
   > bow on the sand. **Making straight for the steep cliff, where the churchyard hangs
   > over the laneway to the East Pier** … it disappeared in the darkness."

   The churchyard on the East Cliff is where Swales sits every day and where "our seat"
   is — the same seat he is found dead on two days later.

3. **Violence at the landing site.** *The Dailygraph*, lines 226–230: a mastiff belonging
   to a coal merchant near Tate Hill Pier "was found dead in the roadway … its throat was
   torn away, and its belly was slit open as if with a savage claw."

4. **Animals will not approach the seat.** During the sea-captain's funeral on 10 August —
   the same morning Swales is found — *Mina Murray's Journal*, lines 415–437: a dog
   refuses to come to its master at the seat, "barking and howling … in a sort of fury,"
   and when thrown onto the tombstone the seat is fixed to, "the moment it touched the
   stone the poor thing became quiet and fell all into a tremble … in such a pitiable
   state of terror."

5. **The same seat is Dracula's hunting spot the very next night.** *Mina Murray's
   Journal*, 11 August, lines 512–532: Mina finds Lucy sleepwalking "on our favourite
   seat," and

   > "it seemed to me as though something dark stood behind the seat where the white figure
   > shone, and bent over it … I could see a white face and red, gleaming eyes."

6. **The seat stands on unhallowed ground — and Van Helsing says Dracula went there.**
   The seat is not just any spot in the churchyard. On Mina's first visit, Swales points to
   "a stone at our feet which had been laid down as a slab, on which the seat was rested"
   (*Mina Murray's Journal*, lines 176–177): the grave of George Canon, who, Swales says,
   "committed suicide" (lines 182–192). Lucy draws the conclusion at once — "I must go on
   sitting over the grave of a suicide" (lines 207–209). It is the same stone the terrified
   dog is thrown onto the morning Swales is found: "the tombstone on which the seat is
   fixed" (line 426). Weeks later, explaining the Count's powers, Van Helsing names the
   place: he can do as he will "when he have his earth-home, his coffin-home, his
   hell-home, the place unhallowed, **as we saw when he went to the grave of the suicide at
   Whitby**" (*Mina Harker's Journal*, lines 820–823).

So: Dracula lands in that churchyard on the 8th; an old man is found dead of fright on
the seat there on the 10th, with animals in terror of the spot; Lucy is attacked on that
same seat on the 11th; and Van Helsing later states outright that Dracula went to the
suicide's grave the seat rests on. That last link means the attribution is **not** mere
spatial coincidence — an in-world account places Dracula at the exact spot. But no document
states "Dracula killed Swales", and the one that places him there is about Lucy, not about
Swales. That remaining gap is exactly why the lenient gold makes him optional.

Note also how thin the footprint is: Swales appears in **one** of the 46 documents —
*Mina Murray's Journal* — with no inquest, no certificate and no newspaper item. The
doctor's opinion reaches us only as hearsay inside that journal ("as the doctor said"),
and the "look of fear and horror" is what "the men said", not the doctor.

### 5.3 Mrs. Westenra — victim 11

**(i) The death.** *Memorandum left by Lucy Westenra*, night of 17 September, lines 41–58,
in Lucy's own hand:

> "there was a crash at the window, and a lot of broken glass was hurled on the floor …
> in the aperture of the broken panes there was the head of a great, gaunt grey wolf.
> Mother cried out in a fright … there was a strange and horrible gurgling in her throat;
> then she fell over--as if struck with lightning … dear mother's poor body, which seemed
> to grow cold already--for her dear heart had ceased to beat."

**(ii) Why she was killable by fright.** *Mina Murray's Journal*, 15 August, lines 709–713:

> "her doctor told her that within a few months, at most, she must die, for her heart is
> weakening. At any time, even now, a sudden shock would be almost sure to kill her."

**(iii) That the wolf was Dracula's.** The proof is a newspaper interview with the
zookeeper Thomas Bilder in *The Pall Mall Gazette*, 18 September:

1. **A man matching Dracula visits the wolf enclosure.** Lines 75–80:

   > "close at hand was only one man, a tall, thin chap, with a 'ook nose and a pointed
   > beard, with a few white hairs runnin' through it. He had a 'ard, cold look and red
   > eyes."

   and line 84, "he smiled a kind of insolent smile, with a mouth full of white, sharp
   teeth"; line 103, "not exactly in the business, but I 'ave made pets of several."

   Compare Harker's description of the Count, *Jonathan Harker's Journal*, lines 640–650:
   "a strong--a very strong--aquiline, with high bridge of the thin nose … peculiarly
   sharp white teeth; these protruded over the lips."

2. **The wolf escapes that night.** *The Pall Mall Gazette*, lines 113–115: "when I kem
   opposite to old Bersicker's cage I see the rails broken and twisted about and the cage
   empty."

3. **The forensic link — the wolf comes back with the window in its head.**
   *The Pall Mall Gazette*, lines 201–203:

   > "Here's his head all cut and full of broken glass. 'E's been a-gettin' over some
   > bloomin' wall or other."

   Bilder guesses a wall topped with bottles. But Lucy's memorandum has the wolf coming
   through a window in a shower of glass (*Memorandum left by Lucy Westenra*, line 41).
   The escaped zoo wolf and the window wolf are the same animal.

4. **Dracula commands wolves** throughout *Jonathan Harker's Journal*: he stills a ring of
   them with a gesture (lines 484–490), calls a pack to kill the mother at the castle
   (lines 1736–1743), and names them "the children of the night" (line 669).

The chain is therefore: *Dracula frees the zoo wolf → the wolf smashes Lucy's window →
the shock stops Mrs. Westenra's diseased heart.* Note the second consequence: in her
fright she "clutched the wreath of flowers … and tore it away from me" (*Memorandum left
by Lucy Westenra*, lines 46–47), stripping Lucy of the garlic that had been protecting
her.

### 5.4 Lucy Westenra — victim 12

**(i) The attack and its signature.** Attacked on the churchyard seat, 11 August (*Mina
Murray's Journal*, lines 512–532, quoted above). Mina finds two punctures and
misattributes them to her own safety-pin — *Mina Murray's Journal*, lines 596–601: "there
are two little red points like pin-pricks, and on the band of her nightdress was a drop of
blood." They do not heal — lines 732–737: "the tiny wounds seem not to have healed. They
are still open, and, if anything, larger than before."

**(ii) Repeated feeding, and the medical impossibility.** Lucy records nightly "scratching
or flapping at the window" and waking "horribly weak" (*Lucy Westenra's Diary*, lines
19–24). She receives **four transfusions** — Arthur, Seward, Van Helsing, Quincey — and
Quincey states the arithmetic that rules out ordinary illness, *Dr. Seward's Diary*, lines
1433–1437:

> "that poor pretty creature that we all love has had put into her veins within that time
> the blood of four strong men. Man alive, her whole body wouldn't hold it. … What took
> it out?"

**(iii) The protection works, and is removed twice.** Van Helsing's garlic gives her four
quiet nights (*Lucy Westenra's Diary*, lines 48–62 and 67–89). It is undone first by her
mother, in innocence — *Dr. Seward's Diary*, lines 1045–1051: "There were a lot of those
horrible, strong-smelling flowers about everywhere … so I took them all away and opened a
bit of the window"; and finally on 17 September when the wolf shatters the window and the
wreath is torn off (*Memorandum left by Lucy Westenra*, lines 41–47).

**(iv) The death.** *Dr. Seward's Diary*, 20 September, lines 1645–1648:

> "And then Lucy's breathing became stertorous again, and all at once it ceased.
> 'It is all over,' said Van Helsing. 'She is dead!'"

Van Helsing immediately signals that this is not an ending — *Dr. Seward's Diary*, line
1671: "Not so; alas! not so. It is only the beginning!"

### 5.5 Renfield — victim 13

**(i) He is Dracula's instrument first.** Renfield's fits track the Count's movements. He
announces the arrival — *Dr. Seward's Diary*, lines 197–198: "I don't want to talk to you:
you don't count now; **the Master is at hand**"; he escapes to the Carfax chapel door and
pleads, lines 282–285: "I am here to do Your bidding, Master. I am Your slave, and You
will reward me." He attacks the men carrying the earth-boxes (*Report from Dr. Patrick
Hennessey to Dr. Seward*, lines 40–54), shouting "I'll frustrate them! … I'll fight for my
Lord and Master!"

**(ii) He lets Dracula in — the invitation that dooms Mina.** *Dr. Seward's Diary*, line
4415: "before I knew what I was doing, I found myself opening the sash and saying to Him:
**'Come in, Lord and Master!'**"

**(iii) He turns on Dracula to protect Mina, and is killed for it.** *Dr. Seward's Diary*,
lines 4451–4461:

> "it made me mad to know that He had been taking the life out of her. … So when He came
> to-night I was ready for Him. … I held tight; and I thought I was going to win … till I
> saw His eyes. They burned into me, and my strength became like water. He slipped through
> it, and when I tried to cling to Him, **He raised me up and flung me down**."

**(iv) The injuries rule out self-harm.** The attendant's puzzle, *Dr. Seward's Diary*,
lines 4232–4243: "his back is broken … I can't understand the two things. If his back was
broke, he couldn't beat his head; and if his face was like that before the fall out of
bed, there would be marks of it." Medical finding, lines 4271–4272: "the real injury was a
depressed fracture of the skull, extending right up through the motor area."

**(v) The death.** *Dr. Seward's Diary*, lines 4643–4644, Arthur reporting: "except that
the poor fellow is dead." Dated 3 October, in England, while Dracula is still in London —
inside the window.

---

## 6. Exclusions — deaths that must **not** be counted

### 6.1 Before the window: the castle child and its mother

The child, *Jonathan Harker's Journal*, lines 1479–1484: "she pointed to the bag which he
had thrown upon the floor, and which moved as though there were some living thing within
it … there was a gasp and a low wail, as of a half-smothered child." The mother, lines
1728–1745: "'Monster, give me my child!' … a pack of them poured, like a pent-up dam …
There was no cry from the woman, and the howling of the wolves was but short."

Both die in Transylvania in **May–June**, before the 6 July sailing. *(These two are the
difference between an earlier gold of 15 and the current 13.)*

### 6.2 After the window: Skinsky and Quincey Morris

**Petrof Skinsky** — killed by Dracula, but at Galatz after he has fled England.
*Jonathan Harker's Journal*, lines 3910–3912: "the body of Skinsky had been found inside
the wall of the churchyard of St. Peter, and that the throat had been torn open as if by
some wild animal." Mina's memorandum makes the attribution explicit — *Mina Harker's
Memorandum*, lines 94–95: "he blotted out his traces, as he thought, **by murdering his
agent**."

**Quincey Morris** — doubly excluded: outside the window *and* not killed by Dracula. He
is cut down by the Szgany while forcing the ring around the cart (*Mina Harker's Journal*,
line 1728), and dies of the wound: line 1779, "his hand pressed to his side; the blood
still gushed through his fingers"; lines 1799–1800, "with a smile and in silence, he died,
a gallant gentleman." The closing *Note*, line 6, fixes the date.

### 6.3 Inside the window, but natural deaths

**Mr. Hawkins** — *Letter, Mina Harker to Lucy Westenra* (18 September), line 9: "Such a
sad blow has befallen us. Mr. Hawkins has died very suddenly." He is elderly and
gout-ridden, dies in Exeter far from Dracula, and the text records no wound, no blood
loss, no vampiric signature; Mina and Jonathan treat it purely as bereavement.

**Arthur's father (Lord Godalming)** — a long decline tracked across documents: "his
father is not very well" (*Mina Murray's Journal*, lines 266–268), "Mr. Holmwood has been
suddenly called to Ring to see his father, who has been taken seriously ill" (same
journal, lines 282–284), and the wire "Am summoned to see my father, who is worse"
(*Telegram, Arthur Holmwood to Seward*, line 5). Seward records both deaths in one breath
— *Dr. Seward's Diary*, lines 1511–1512: "Lucy's mother and Arthur's father, and now…."
Renfield confirms the succession, lines 3726–3728: "Lord Godalming, I had the honour of
seconding your father at the Windham; I grieve to know, by your holding the title, that he
is no more."

---

## 7. Decoys — things that look like deaths but are not

**The "bloofer lady" children survive.** Un-dead Lucy preys on children at Hampstead, but
*The Westminster Gazette*, 25 September (doc 33), lines 32–36 reports wounds "of not much
importance individually," and its *Extra Special* second edition of the same day (doc 34),
lines 14–17, has a missing child "when partially restored" telling the same story. None dies. (*Dr. Seward's Diary*, lines 2198–2200, references the same events.)

**Mina survives.** Dracula forces her to drink his blood on the night Renfield dies
(*Dr. Seward's Diary*, lines 4507–4514), but she is saved and is alive in the closing
*Note*.

**Un-dead Lucy and the three sisters are releases, not kills.** Van Helsing destroys beings
already dead — *Dr. Van Helsing's Memorandum*, lines 95–98: "hardly had my knife severed
the head of each, before the whole body began to melt away and crumble in to its native
dust."

---

## 8. Why the question is hard

Getting 13 requires chaining five distinct operations, each defeating a different
strategy:

1. **Arithmetic over a muster line** — "five hands … two mates, cook, and myself" → 9. The
   number is never stated; it must be computed. *Defeats retrieval:* there is no passage
   saying "nine".
2. **Narrative state-tracking** — confirming that all nine of that muster die, across
   eight log entries. *Defeats summarization:* lossy views of the log produce 6, 12, or 20.
3. **Indirect causal attribution** — Swales (fright), Mrs. Westenra (wolf → heart), the
   captain (terror/duty), the mate (suicide) are kills only under "directly or indirectly",
   and the Dracula link runs through a landing site and a zookeeper's interview.
   *Defeats lexical scanning:* no "kill/murder" cue is anywhere near these deaths.
4. **Two-sided temporal windowing** — ordering events against the 6 July sailing and the
   October flight. *Defeats naïve aggregation:* it is what excludes four real Dracula kills.
5. **Kill-vs-not disambiguation** — natural deaths, survivors, and un-dead releases.

And the evidence is **dispersed by design**: the answer touches a ship's log, a newspaper
cutting, a second newspaper cutting (the zoo), one woman's private journal, a dying
woman's memorandum, and a doctor's phonograph diary — six documents of five types, with the
two halves of a single victim's chain (Mrs. Westenra) sitting in different documents
written by different people for different reasons.

## 9. Scoring

Answers are graded by an LLM judge run **twice**, giving two 0/1 metrics
(`score` returns the pair):

- **strict** — against the 13-victim gold.
- **lenient** — against the 12-victim gold (Swales omitted), combined in code as
  `lenient = max(strict, alt)`, so lenient is always a superset of strict.

Both calls use the *same* prompt; only the gold differs. The prompt states one rule —
every number in the gold must be quotable from the model's answer, never supplied by the
grader — and carries six worked examples on a deliberately unrelated scenario, so the
grader cannot copy an example's verdict onto the real roster. This replaced an earlier
rule-list prompt that was caught passing an answer while asserting "it explicitly names the
crew as nine" of a text in which the word never appears.
