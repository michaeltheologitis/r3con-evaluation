"""Download + extract Loong's document pool into the (gitignored) data dir.

Loong vendors only its *questions* (``data/loong.jsonl``, committed); the
*documents* are a separate ~34 MB ``doc.zip`` in the upstream Alibaba OSS
bucket. This script fetches and unzips it to
``evals/benchmarks/loong/data/doc/`` (financial / paper / legal subdirs),
reproducing the ``doc/<type>/...`` layout upstream's ``--doc_path ./doc``
expects. That tree (~110 MB extracted) is **gitignored** — reconstructible on
demand, not vendored.

Run it explicitly (the recommended, visible way):

    python -m evals.benchmarks.loong.download_docs          # fetch once
    python -m evals.benchmarks.loong.download_docs --force  # re-fetch

It's also invoked lazily on the first ``loong.get_documents(...)`` call (so the
baselines "just work"), but running it up front makes the data step explicit and
keeps a 34 MB network fetch from surprising you mid-run. ``ensure()`` is
concurrency-safe (an exclusive file lock + atomic publish), so the runner's
parallel child processes won't corrupt the pool by racing the lazy download —
but for big parallel runs, **pre-fetching once up front is still recommended** so
workers don't all block on a cold download.

Note the URL is **https**: the bucket's plain-http endpoint intermittently times
out on connect; https is reliable. Pure stdlib (urllib + zipfile) — no extra dep.

The zip was created on macOS and stores its Chinese filenames (the ``zh`` financial
reports) as UTF-8 bytes WITHOUT the zip UTF-8 flag, which ``zipfile`` mis-decodes as
CP437 → mojibake. :func:`_extract_utf8` recovers the real names on extract (see
:func:`_recovered_name`); ASCII names (en financial / paper / the legal json) are
untouched.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from evals.settings import settings

# Upstream OSS bucket. https (NOT http — the http endpoint times out on connect).
DOC_URL = "https://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"


def doc_dir() -> Path:
    """The extracted document pool: ``LOONG_DIR/doc/`` (financial/paper/legal)."""
    return settings.LOONG_DIR / "doc"


def is_present() -> bool:
    """Whether the pool has already been downloaded + extracted."""
    return doc_dir().is_dir()


def ensure(force: bool = False) -> Path:
    """Download + extract the document pool if needed; return the pool dir.

    No-op when the pool is already present (unless ``force``). The downloaded
    ``doc.zip`` is kept (gitignored) so a re-extract doesn't re-download. This is
    the single source of truth for acquiring the docs — the loader's lazy path
    calls straight through to it.

    **Concurrency-safe.** The runner fans out many child processes whose first
    ``get_documents()`` would otherwise race here — two workers writing the same
    ``doc.zip``, or one extracting a zip another is still downloading. An exclusive
    cross-process file lock serializes the work: exactly one process downloads +
    extracts, the rest block then see the pool present and return. Download and
    extract are staged through temp paths and moved into place with ``os.replace``
    (atomic), so an interrupted run never leaves a partial ``doc.zip`` / ``doc/``
    that a later run would mistake for complete.
    """
    target = doc_dir()
    if target.is_dir() and not force:
        return target

    settings.LOONG_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = settings.LOONG_DIR / ".doc.lock"
    with open(lock_path, "w") as lock_file:
        _acquire_lock(lock_file)
        # Double-checked under the lock: a peer may have finished while we blocked.
        if target.is_dir() and not force:
            return target
        _download_and_extract(force=force)
    return target


def _acquire_lock(handle) -> None:
    """Best-effort exclusive cross-process lock (POSIX ``flock``) on an open file.

    Released when the handle is closed. Where ``fcntl`` is unavailable (non-POSIX)
    this is a no-op — pre-fetch the pool (see the module docstring) to avoid races
    there. Blocks until the lock is held.
    """
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(handle, fcntl.LOCK_EX)


def _recovered_name(info: zipfile.ZipInfo) -> str:
    """The entry's true UTF-8 filename.

    Loong's ``doc.zip`` stores Chinese filenames (the ``zh`` financial reports) as
    UTF-8 bytes WITHOUT setting the zip "filename is UTF-8" flag (bit 0x800), so
    ``zipfile`` decodes them as CP437 → mojibake (e.g.
    ``report_…-σ╣│σ«ëΘô╢Φíî-…``). When the flag is unset we round-trip
    ``cp437 → utf-8`` to recover the real name (``…-平安银行-…``). ASCII names
    (en financial / paper / the legal json) round-trip to themselves, so this is a
    no-op for them. A name that can't be recovered is left untouched rather than
    failing the whole extract.
    """
    name = info.filename
    if not (info.flag_bits & 0x800):  # 0x800 = "filename is UTF-8"; unset → cp437-decoded
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    return name


def _extract_utf8(zf: zipfile.ZipFile, dest: Path) -> None:
    """Like ``zf.extractall(dest)`` but recovers mis-decoded non-ASCII filenames
    (:func:`_recovered_name`), skips the macOS ``__MACOSX`` resource-fork entries,
    and guards against path traversal (zip-slip)."""
    dest_resolved = dest.resolve()
    for info in zf.infolist():
        name = _recovered_name(info)
        if name == "__MACOSX" or name.startswith("__MACOSX/"):
            continue
        target = dest / name
        if not target.resolve().is_relative_to(dest_resolved):
            continue  # refuse to write outside dest
        if info.is_dir() or name.endswith("/"):
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)


def _download_and_extract(*, force: bool) -> None:
    """Fetch (if missing) + extract the pool, each staged through a temp path and
    published with an atomic ``os.replace`` — so an interrupted/concurrent run
    never leaves a half-written ``doc.zip`` or half-extracted ``doc/`` behind.

    Must be called while holding the lock acquired in :func:`ensure`.
    """
    zip_path = settings.LOONG_DIR / "doc.zip"
    if force or not zip_path.exists():
        print(f"[loong] downloading document pool (~34 MB) from {DOC_URL}")
        fd, tmp_zip = tempfile.mkstemp(dir=settings.LOONG_DIR, suffix=".zip.part")
        os.close(fd)
        try:
            urllib.request.urlretrieve(DOC_URL, tmp_zip)
            os.replace(tmp_zip, zip_path)  # atomic publish of the completed zip
        finally:
            if os.path.exists(tmp_zip):
                os.remove(tmp_zip)

    print(f"[loong] extracting {zip_path.name} → {settings.LOONG_DIR}")
    tmp_extract = Path(tempfile.mkdtemp(dir=settings.LOONG_DIR, prefix=".doc_extract_"))
    try:
        with zipfile.ZipFile(zip_path) as zf:
            _extract_utf8(zf, tmp_extract)
        dest = doc_dir()
        if dest.exists():  # only reachable on a force=True refresh
            shutil.rmtree(dest)
        os.replace(tmp_extract / "doc", dest)  # atomic publish of the doc/ tree
    finally:
        shutil.rmtree(tmp_extract, ignore_errors=True)
    print(f"[loong] document pool ready at {doc_dir()}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Download + extract Loong's ~34 MB document pool into the "
                    "(gitignored) data dir."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download + re-extract even if the pool is already present.",
    )
    args = parser.parse_args()
    if is_present() and not args.force:
        print(f"[loong] document pool already present at {doc_dir()} (use --force to refresh)")
        return
    ensure(force=args.force)


if __name__ == "__main__":
    main()
