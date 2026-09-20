"""Split Project Gutenberg's Dracula (pg345.txt) into its in-world documents.

Dracula is an epistolary novel: nothing but journals, diaries, letters, telegrams,
newspaper cuttings, a ship's log, memoranda, and one closing note. This script
decompiles the novel back into that document set:

- The five running journals/diaries (Jonathan Harker, Mina Murray, Mina Harker,
  Lucy Westenra, Dr. Seward) are each ONE document — all their sections in book
  order. Mina Murray vs Mina Harker follow the novel's own titling and stay two
  documents (linking them is part of the reading task).
- Every letter, telegram, report, memorandum, newspaper cutting, the log of the
  Demeter, the phonograph message spoken by Van Helsing, Van Helsing's undelivered
  portmanteau note, and the closing NOTE is its own document.
- Chapter headings are book apparatus, not in-world documents — dropped. Stoker's
  editorial preface ("How these papers have been placed in sequence...") is saved
  aside as editorial_note.txt, not a corpus document.

Header grammar (verified against the actual file):
- CAPS headers, possibly indented, possibly with a "--_continued_" suffix and/or a
  trailing period: JONATHAN HARKER'S JOURNAL / DR. SEWARD'S DIARY--_continued_. /
  DR. SEWARD'S PHONOGRAPH DIARY, SPOKEN BY VAN HELSING / CUTTING FROM "THE
  DAILYGRAPH," 8 AUGUST / LOG OF THE "DEMETER." / NOTE
- Italic one-liners: _Dr. Seward's Diary._ / _Letter, Mina Harker to Lucy
  Westenra._ / _Telegram, ..._ / _Memorandum..._ / _"The Westminster Gazette," 25
  September._ — sometimes spanning 2-3 lines (the Billington / Sister Agatha /
  Hennessey letter headers). Italic lines WITHOUT a source keyword (e.g.
  _Whitby_, _The "Bloofer Lady."_, "_17 August._ datelines) are content, not
  headers.

Output: docs/NN_slug.txt (book order) + manifest.json.
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

HERE = Path(__file__).parent
RAW = HERE / "raw" / "pg345.txt"
OUT = HERE / "docs"

# Keywords that make an italic _..._ line a source header (not content).
ITALIC_KEYWORDS = re.compile(
    r"\b(Letter|Telegram|Report|Memorandum|Journal|Diary|Gazette|Dailygraph|Note)\b"
)
CHAPTER_RE = re.compile(r"^CHAPTER [IVXLC]+$")
CONTINUED_RE = re.compile(r"--_?continued_?\.?$", re.IGNORECASE)

# The one NESTED document in the novel, and the line at which its container resumes.
#
# The log of the "Demeter" is printed *inside* the Dailygraph cutting: the
# correspondent writes his lead-in, says he will "send you a rescript", transcribes
# the log, and then RESUMES IN HIS OWN VOICE with the inquest, the funeral and the
# dog. Both are genuine in-world artifacts written by different men months apart, so
# both are documents -- but the correspondent's closing belongs to his article, not
# to the log. Scanning alone cannot find that seam: it is marked only by a "* * *"
# rule and a change of voice, with no header. So it is declared here.
#
# This matters beyond tidiness. The corpse at the wheel is an anonymous "dead seaman"
# in the front half; it is the log in his pocket -- "myself (captain)" plus "I shall
# tie my hands to the wheel ... and along with them ... that which He--It!--dare not
# touch" -- that identifies him. The correspondent's closing IS that inference. Filing
# it under the log hides a cross-document hop the corpus exists to require.
#
#   {nested title: (container title, first line of the container's resumption)}
NESTED_RESUME = {
    # NB the key is the NORMALIZED title. The header ends `DEMETER.”` -- the period
    # is followed by a curly quote, so `rstrip("_. ")` never reaches it and the title
    # keeps it. Getting this key wrong makes the rejoin silently do nothing, which is
    # why main() asserts every key below was actually matched.
    'Log Of The "Demeter."': (
        'Cutting From "The Dailygraph," 8 August',
        "Of course the verdict was an open one",
    )
}

# The five running sources that group into one document each (normalized titles).
GROUPED_SOURCES = {
    "Jonathan Harker's Journal",
    "Mina Murray's Journal",
    "Mina Harker's Journal",
    "Lucy Westenra's Diary",
    "Dr. Seward's Diary",
}


def normalize_title(header: str) -> str:
    """Canonical title from a raw header: drop italics markers, --continued,
    trailing period; normalize curly quotes; title-case the CAPS headers."""
    t = header.strip().lstrip("_").strip()
    t = t.rstrip("_. ").strip()  # first: "--continued._" -> "--continued"
    t = CONTINUED_RE.sub("", t).strip()
    t = t.rstrip("_. ").strip()
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    if t == t.upper():  # CAPS header -> readable form
        lowered = t.title()
        # Fix title() artifacts on possessives and known words.
        lowered = re.sub(r"'S\b", "'s", lowered)
        lowered = lowered.replace("'s", "'s")
        lowered = re.sub(r"\bDr\b", "Dr", lowered)
        t = lowered
    # Collapse internal whitespace (multi-line italic headers join with spaces).
    return re.sub(r"\s+", " ", t)


def classify(title: str) -> str:
    tl = title.lower()
    if "phonograph" in tl:
        return "phonograph_message"
    if "journal" in tl or "diary" in tl:
        return "journal"
    if tl.startswith("letter"):
        return "letter"
    if tl.startswith("telegram"):
        return "telegram"
    if tl.startswith("report"):
        return "report"
    if "memorandum" in tl:
        return "memorandum"
    if tl.startswith("cutting") or "gazette" in tl or "dailygraph" in tl:
        return "newspaper_cutting"
    if tl.startswith("log of"):
        return "ships_log"
    if tl == "note" or tl.startswith("note left"):
        return "note"
    raise ValueError(f"Unclassifiable header title: {title!r}")


def is_caps_header(stripped: str) -> bool:
    # Strip the "--_continued_" tail BEFORE the all-caps test: it is lowercase,
    # so testing the raw line would reject "JONATHAN HARKER'S JOURNAL--_continued_"
    # (which is exactly the bug this replaces — 7 such breaks were swallowed).
    if not stripped:
        return False
    base = CONTINUED_RE.sub("", stripped).rstrip(".").strip()
    if not base or base != base.upper():
        return False
    if CHAPTER_RE.match(base):
        return False
    return bool(
        re.search(r"\b(JOURNAL|DIARY)\b", base)
        or base.startswith("LOG OF THE")
        or base.startswith("CUTTING FROM")
        or base == "NOTE"
    )


def main() -> None:
    text = RAW.read_text(encoding="utf-8")
    lines = text.split("\n")

    # Content window: after the *** START marker, up to the publisher ads / *** END.
    start = next(i for i, l in enumerate(lines) if l.startswith("*** START OF"))
    end = next(i for i, l in enumerate(lines) if l.startswith("*** END OF"))
    for i in range(start, end):
        if lines[i].startswith("DETECTIVE STORIES BY"):
            end = i
            break
    for i in range(start, end):
        if lines[i].strip() == "THE END":  # book apparatus, not in-world
            end = i
            break
    lines = lines[start + 1 : end]

    # Stoker's editorial preface: the paragraph before the "DRACULA" half-title.
    body_start = next(i for i, l in enumerate(lines) if l.strip() == "CHAPTER I")
    front = "\n".join(lines[:body_start])
    m = re.search(r"How these papers.*?made them\.", front, re.DOTALL)
    (HERE / "editorial_note.txt").write_text(
        (m.group(0) if m else "").strip() + "\n", encoding="utf-8"
    )
    lines = lines[body_start:]

    # --- scan into (header, body-lines) sections -----------------------------
    sections: list[tuple[str, list[str]]] = []
    current_header: str | None = None
    current: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if CHAPTER_RE.match(stripped):
            i += 1
            continue
        header: str | None = None
        consumed = 1
        if is_caps_header(stripped):
            # A caps header may carry a "--_continued_" tail on the same line.
            header = stripped
        elif (
            stripped.startswith("_")
            # NB: search with underscores blanked — "_" is a \w char, so \bLetter\b
            # would never match the "_Letter" at the start of a header line.
            and ITALIC_KEYWORDS.search(stripped.replace("_", " "))
            # "._--" is the entry-dateline signature (_11 August, 3 a. m._--Diary
            # again...) — an entry whose prose happens to contain a keyword.
            and "._--" not in stripped
        ):
            # Indentation does NOT disqualify a header: the SECOND
            # "_The Westminster Gazette, 25 September._" -- the "_Extra Special._"
            # edition carrying THE HAMPSTEAD HORROR -- is indented 15 spaces and is a
            # genuinely separate edition, published later the same day. Block-quoted
            # datelines inside a journal (the 28 October telegram in Mina Harker's
            # journal) are excluded by the "._--" test above, not by indentation.
            # Italic header; may span multiple lines until one ends with "_".
            # NB terminal test strips trailing periods first: this edition
            # closes some headers "._" and others "_." (e.g. _Letter, Lucy
            # Westenra to Mina Murray_.).
            parts = [stripped]
            while not parts[-1].rstrip(". ").endswith("_") and consumed < 4:
                parts.append(lines[i + consumed].strip())
                consumed += 1
            joined = " ".join(parts)
            # Only a real header if the italics actually closed and it is
            # header-sized (letter headers wrap to at most ~3 short lines).
            if joined.rstrip(". ").endswith("_") and len(joined) <= 160:
                header = joined
        if header is not None:
            if current_header is not None:
                sections.append((current_header, current))
            current_header, current = header, []
            i += consumed
            continue
        if current_header is None:
            if stripped:  # nothing before the first header should carry text
                raise ValueError(f"Orphan text before first header: {stripped!r}")
        else:
            current.append(lines[i])
        i += 1
    sections.append((current_header, current))

    # --- return each nested document's trailing container prose to its container --
    rejoined: set[str] = set()
    for idx, (header, body) in enumerate(sections):
        spec = NESTED_RESUME.get(normalize_title(header))
        if spec is None:
            continue
        parent_title, marker = spec
        cut = next(
            (i for i, l in enumerate(body) if l.strip().startswith(marker)), None
        )
        if cut is None:
            raise ValueError(f"{header!r}: container resumption {marker!r} not found")
        # Walk back over the blank lines and the "* * *" rule that introduce the
        # resumption so they travel with it, and the log ends on its own last entry.
        while cut > 0 and (not body[cut - 1].strip() or set(body[cut - 1].strip()) <= {"*", " "}):
            cut -= 1
        if idx == 0 or normalize_title(sections[idx - 1][0]) != parent_title:
            raise ValueError(f"{header!r}: expected to be nested inside {parent_title!r}")
        sections[idx - 1][1].extend(body[cut:])
        sections[idx] = (header, body[:cut])
        rejoined.add(normalize_title(header))
    missed = set(NESTED_RESUME) - rejoined
    if missed:  # a stale key would silently reproduce the un-rejoined split
        raise ValueError(f"NESTED_RESUME keys never matched a section: {sorted(missed)}")

    # --- assemble documents ---------------------------------------------------
    docs: list[dict] = []
    grouped: dict[str, dict] = {}
    for header, body in sections:
        title = normalize_title(header)
        body_text = "\n".join(body).strip("\n")
        section_text = header.strip() + "\n\n" + body_text + "\n"
        if title in GROUPED_SOURCES:
            if title not in grouped:
                grouped[title] = {"title": title, "type": "journal", "parts": []}
                docs.append(grouped[title])
            grouped[title]["parts"].append(section_text)
        else:
            docs.append({"title": title, "type": classify(title), "parts": [section_text]})

    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.txt"):
        old.unlink()
    manifest = []
    for n, doc in enumerate(docs, 1):
        content = "\n\n".join(doc["parts"])
        slug = unicodedata.normalize("NFKD", doc["title"].lower())
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")[:60]
        fname = f"{n:02d}_{slug}.txt"
        (OUT / fname).write_text(content + "\n", encoding="utf-8")
        manifest.append(
            {
                "doc": n,
                "file": fname,
                "title": doc["title"],
                "type": doc["type"],
                "sections": len(doc["parts"]),
                "words": len(content.split()),
            }
        )
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = sum(m["words"] for m in manifest)
    print(f"{len(manifest)} documents, {total:,} words total")
    for m in manifest:
        print(f"  {m['doc']:>2}  {m['type']:<18} {m['words']:>7,}w  {m['title']}")


if __name__ == "__main__":
    main()
