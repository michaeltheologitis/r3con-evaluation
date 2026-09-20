"""Download CorpusQA's question file into the (gitignored) data dir.

CorpusQA vendors NOTHING in-repo — the wired ``1m`` tier is a single huge
``1m_4domains.jsonl`` (≈ 1 GB) that must be fetched on demand. This module is the
**single source of truth for acquiring** it; it deliberately knows nothing about the
benchmark's row shape (that's :mod:`evals.benchmarks.corpusqa.loader`, which builds a
small offset index over whatever this downloads).

Run it explicitly (the recommended, visible way) — the cluster pre-fetches so the
runner's parallel child processes don't all block on a cold ~1 GB download:

    python -m evals.benchmarks.corpusqa.download_data --set 1m
    python -m evals.benchmarks.corpusqa.download_data --set 1m --force

It is also invoked lazily on the first ``corpusqa.get_task_ids()`` / ``get_task()``
(so the baselines "just work"), but a 1 GB fetch surprising you mid-run is no fun —
pre-fetch up front.

**One mirror — Hugging Face** ``Tongyi-Zhiwen/CorpusQA``. CorpusQA is deliberately
**1m-only** — that's the tier we measure on. 128k/4m/10m exist upstream but are NOT
wired (128k is too small to be interesting; 4m/10m fit no window and aren't on HF).
See :data:`SETS`. Pure stdlib (urllib).

``ensure()`` is concurrency-safe (an exclusive per-tier file lock + atomic
publish), so the runner's parallel children won't corrupt a file by racing the
lazy download. The OFFSET INDEX over the file is built separately by the loader
(:func:`evals.benchmarks.corpusqa.loader._tier_index`); ``main()`` warms it after
download so a pre-fetch leaves both the data and its index ready.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path

from evals.settings import settings

# The ONLY wired tier — ``1m`` (a 1m instance is ~1M tokens of documents), hosted on
# Hugging Face. CorpusQA is deliberately 1m-only: that's the tier we measure on, so the
# loader always returns it and there is no tier to choose. (128k/4m/10m exist upstream
# but are NOT wired.) The loader re-exports this as ``corpusqa.SETS``.
SETS = frozenset({"1m"})

# The sole mirror — Hugging Face hosts the 1m tier.
_HF = "https://huggingface.co/datasets/Tongyi-Zhiwen/CorpusQA/resolve/main/{tier}_4domains.jsonl"


def tier_file(tier: str) -> Path:
    """The downloaded question file for one tier: ``CORPUSQA_DIR/{tier}_4domains.jsonl``."""
    return settings.CORPUSQA_DIR / f"{tier}_4domains.jsonl"


def is_present(tier: str) -> bool:
    """Whether the tier's question file has already been downloaded."""
    return tier_file(tier).exists()


def ensure(tier: str, *, force: bool = False) -> Path:
    """Download the tier's question file from Hugging Face if needed; return its path.

    No-op when the file is already present (unless ``force``). This is the single
    source of truth for acquiring the data — the loader's lazy path calls straight
    through to it.

    **Concurrency-safe.** The runner fans out many child processes whose first
    ``get_task()`` would otherwise race here. An exclusive per-tier file lock
    serializes the work (exactly one process downloads, the rest block then see the
    file present and return), and the download is staged through a temp path and
    moved into place with ``os.replace`` (atomic) — so an interrupted/concurrent run
    never leaves a partial ``{tier}_4domains.jsonl`` a later run would mistake for
    complete.
    """
    if tier not in SETS:
        raise ValueError(f"Unknown tier {tier!r}. Wired tiers: {sorted(SETS)}.")
    target = tier_file(tier)
    if target.exists() and not force:
        return target

    settings.CORPUSQA_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = settings.CORPUSQA_DIR / f".{tier}.lock"
    with open(lock_path, "w") as lock_file:
        _acquire_lock(lock_file)
        # Double-checked under the lock: a peer may have finished while we blocked.
        if target.exists() and not force:
            return target
        _download(tier, target)
    return target


def _acquire_lock(handle) -> None:
    """Best-effort exclusive cross-process lock (POSIX ``flock``) on an open file.

    Released when the handle is closed. Where ``fcntl`` is unavailable (non-POSIX)
    this is a no-op — pre-fetch the data (see the module docstring) to avoid races
    there. Blocks until the lock is held.
    """
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(handle, fcntl.LOCK_EX)


def _download(tier: str, target: Path) -> None:
    """Fetch the tier file from Hugging Face, atomically publishing it.

    Downloaded to a temp file and only ``os.replace``-d into place on success, so a
    failed/partial fetch never publishes. Must be called while holding the lock
    acquired in :func:`ensure`.
    """
    url = _HF.format(tier=tier)
    fd, tmp = tempfile.mkstemp(dir=settings.CORPUSQA_DIR, suffix=".jsonl.part")
    os.close(fd)
    try:
        print(f"[corpusqa] downloading {tier} tier from {url}")
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, target)  # atomic publish of the completed file
        print(f"[corpusqa] {tier} tier ready at {target}")
    except Exception as e:  # noqa: BLE001
        if os.path.exists(tmp):
            os.remove(tmp)
        raise RuntimeError(
            f"Failed to download CorpusQA {tier!r} tier from Hugging Face ({url.split('?')[0]})"
        ) from e


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download the CorpusQA 1m question file from Hugging Face (and warm "
                    "its offset index) into the (gitignored) data dir."
    )
    parser.add_argument(
        "--set", dest="sets", action="append", choices=sorted(SETS), required=True,
        help="Context-length tier to fetch — only `1m` is wired (`--set 1m`).",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the tier file is already present.",
    )
    args = parser.parse_args()
    # Imported lazily: the loader imports THIS module, so importing it at top would
    # be a cycle. Warming the offset index here means a pre-fetch leaves both the
    # data file AND its index ready, so the runner's children never build it cold.
    from evals.benchmarks.corpusqa import loader

    for tier in args.sets:
        if is_present(tier) and not args.force:
            print(f"[corpusqa] {tier} tier already present at {tier_file(tier)} (use --force to refresh)")
        else:
            ensure(tier, force=args.force)
        if args.force:
            loader.rebuild_index(tier)
        loader._tier_index(tier)  # build + load the offset index (no-op if present)
        print(f"[corpusqa] {tier} tier index ready ({len(loader._tier_index(tier))} instances)")


if __name__ == "__main__":
    main()
