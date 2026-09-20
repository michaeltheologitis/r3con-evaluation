# Loong — changes from the upstream benchmark

Every way our harness's Loong differs from upstream
([github.com/mozerwang/loong](https://github.com/mozerwang/loong)), for reporting
in the paper. **Only changes are listed**; everything else faithfully mirrors
upstream's `src/utils/prompt.py:get_content`.

## 1. Bug fix — level-4 legal verdict leak (the only behavioral change)

The level-4 legal **"match each judgment document to its verdict"** task (113 of
1,600 instances) gives the model the case documents plus a separate shuffled list
of verdicts to match. The verdict (`result`) must be withheld from each document or
the answer is handed over in the input. Upstream's `get_content` tries to withhold
it but checks the wrong field:

```python
if doc_level == 4 and ('阅读以上判决文书，我将给你若干份判决结果：' in item['instruction']):
    content = legal_js[doc_name]["content"]                                   # verdict withheld
else:
    content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]    # verdict INCLUDED
```

The marker only ever appears in `item['question']`, **never** `item['instruction']`
(verified: 0/179 L4-legal instructions vs 113/179 questions). So the guard never
fires, every document is emitted as `content + result`, and the verdict leaks. The
leak is total — the gold verdict text appears verbatim in **1688/1688**
document–verdict pairs — so the task is solvable by string-matching, not legal
reasoning. StructRAG (ICLR 2025) vendored this code unchanged, so its published
Loong numbers inherit the leak.

**Fix:** we check `item['question']`. The verdict is withheld for exactly the 113
verdict-matching instances; all other legal items — including the 66 L4
"match-to-category" items — keep `content + result`, faithful to upstream.
→ `loader._resolve_doc` (legal branch), `loader._LEGAL_VERDICT_MARKER`.

*Residual (inherent, not fixable faithfully):* the fix withholds the structured
`result` field, but in **28 of ~900** verdict-matching doc-pairs (~3%) the judgment
**body** (`content`) also narrates its own disposition, so the verdict is partially
recoverable there. This is a data property — upstream's own content-only branch
would leave it too — and removing it would mean rewriting the documents, which we
don't. Still a vast improvement over the 100%-leak the unfixed code produced.

We verified `get_task` end-to-end on all 113 verdict-matching tasks: instruction
and question are returned verbatim from the (upstream-identical) data; docs are
content-only (0 resolution bugs); each doc's gold-pointed verdict is present among
the question's candidate list and consistent with the data (0 gold inconsistencies);
and the positional `《判决文书N》` titles line up with the gold's references.

## 2. Scope — full benchmark (EN + ZH)

We load all 1,600 instances (was: English-only, 695). `get_task_ids(languages=…)`
selects; default = both. A coverage choice — no instance is altered. Domains: paper
(EN, 400), financial (EN 295 + ZH 405), legal (ZH-only, 500).

## 3. Doc-pool extraction — filename encoding (no content change)

Upstream's macOS-made `doc.zip` stores the ZH financial filenames as UTF-8 bytes
without the zip UTF-8 flag, so `zipfile` mis-decodes them as CP437 → mojibake and
the financial glob can't find them. We recover the real names on extract
(`cp437 → utf-8` when the flag is unset). Document **contents** and ASCII (EN)
filenames are untouched. → `download_docs._recovered_name` / `_extract_utf8`.

## 4. Pre-existing harness deviations (unchanged here, noted for completeness)

- **Doc order:** upstream optionally shuffles + length-truncates the doc string per
  prompt; we return docs in `doc` order (deterministic, reproducible). Legal's
  positional title `《判决文书{idx+1}》` is keyed to this order — matching upstream and
  the gold answers.
- **Glob tie-break:** upstream takes `glob(...)[0]` in arbitrary order; we `sorted()`
  first (matters only on a rare multi-match).
- **Judge:** unchanged and language-agnostic (`gpt-5.4-mini`, multilingual; gold
  JSON-encoded with `ensure_ascii=False`) — grades ZH as-is, no prompt change.
