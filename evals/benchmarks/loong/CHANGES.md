# Loong — changes from upstream

Every way this harness's Loong differs from
[upstream](https://github.com/mozerwang/loong). Only differences are listed; everything
else mirrors `src/utils/prompt.py:get_content`.

## 1. Bug fix: the level-4 legal verdict leak

The level-4 legal "match each judgment document to its verdict" task (113 of 1,600
instances) gives the model case documents plus a shuffled list of verdicts to match. The
verdict must be withheld from each document or the answer is handed over in the input.
Upstream tries to withhold it but checks the wrong field:

```python
if doc_level == 4 and ('阅读以上判决文书，我将给你若干份判决结果：' in item['instruction']):
    content = legal_js[doc_name]["content"]                                   # withheld
else:
    content = legal_js[doc_name]["content"] + legal_js[doc_name]["result"]    # leaked
```

That marker only ever appears in `item['question']`, never `item['instruction']` (0/179
L4-legal instructions vs 113/179 questions), so the guard never fires and the gold verdict
appears verbatim in 1688/1688 document–verdict pairs. The task is solvable by string
matching. StructRAG vendored this code unchanged, so its published Loong numbers inherit
the leak.

We check `item['question']` instead. The verdict is withheld for exactly the 113
verdict-matching instances; every other legal item keeps `content + result`, as upstream.

In ~28 of ~900 verdict-matching pairs the judgment body narrates its own disposition, so
the verdict stays partially recoverable there. That is a property of the data — upstream's
own content-only branch would leave it too — and fixing it would mean rewriting documents.

## 2. Scope

All 1,600 instances load (upstream tooling is English-only, 695).
`get_task_ids(languages=…)` selects; default is both.

## 3. Doc-pool filename encoding

Upstream's macOS-made `doc.zip` stores the Chinese filenames as UTF-8 without the zip
UTF-8 flag, so `zipfile` mis-decodes them as CP437 and the financial glob cannot find
them. We recover the real names on extract. Document contents are untouched.

## 4. Pre-existing harness differences

- **Doc order.** Upstream optionally shuffles and length-truncates the doc string per
  prompt; we return docs in `doc` order. Legal's positional `《判决文书N》` titles and the
  gold answers are keyed to that order.
- **Glob tie-break.** Upstream takes `glob(...)[0]` in arbitrary order; we sort first.
- **Judge.** `gpt-5.4-mini`, multilingual, grading ZH as-is with no prompt change.
