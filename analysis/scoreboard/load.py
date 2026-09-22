"""Read ``logs/`` into one tidy table — one row per inference.

Two sources, one schema (the ``source`` column tells them apart):

- **baseline** — ``logs/{benchmark}/{baseline}/``. Each manifest carries its own ``usage``
  (tokens) and ``config``; a sibling ``score.json`` holds the judge's verdict.
- **method** — ``logs/r3con/<run-folder>/``, one row per downstream reasoner
  (``grounded-llm`` / ``grounded-codeact``). Tokens are summed from each stage's
  ``result.json`` ``totals.tokens``: the shared summaries + proposer + extractor stages plus
  that reasoner's own inference = the full cost of producing that answer.

Nothing here calls a model or re-judges an answer; scores are read from disk as they were
written. :mod:`scoreboard.report` aggregates the result.
"""
from __future__ import annotations

import json
import re
from functools import cache
from pathlib import Path

import pandas as pd

from scoreboard import paths

COLUMNS = [
    "source", "method", "benchmark", "model", "variant", "summary_rounds", "config",
    "task_id", "status", "error_type", "capped", "score", "scored", "scored_with",
    "input_tokens", "output_tokens", "total_tokens", "usd", "created", "log_dir",
]

# Directory names under `logs/` that are not a benchmark. R3Con's runs sit at `logs/r3con/`
# beside the per-benchmark directories, and scanning it as a benchmark would invent one
# baseline row per run folder. Skipped by name, in one place.
NON_BENCHMARK_DIRS = frozenset({"r3con"})

_STAGES = ("summaries", "proposer", "extractor")   # shared by a run's downstream reasoners
# Inference strategies are DISCOVERED per run-folder (each is an `inference/<strategy>/` dir),
# not hardcoded — so ablation strategies (e.g. `codeact_ablation_nostruct`, codeact re-run
# over the summaries with no structured parse) are picked up automatically, exactly like the
# method board's `discover`. Today: llm, codeact, codeact_ablation_nostruct.

# Exception-class matcher for a traceback blob — the method records per-strategy failures as
# a raw `error.txt` (no structured `error_type`), so we recover the class the way the method
# board does (grounded.eval.results.parse_error_type). The suffix set deliberately includes
# non-`*Error` names (Timeout/Interrupt/Exit) so a litellm Timeout isn't mislabeled.
_ERR_RE = re.compile(r"^([A-Za-z_][\w.]*(?:Error|Exception|Exceeded|Timeout|Interrupt|Exit))\s*:")


def _parse_error_type(text: str) -> str | None:
    """Exception class from a ``traceback.format_exc()`` blob — the last matching
    ``Module.Class:`` line (the actually-raised exception). Mirrors
    ``grounded.eval.results.parse_error_type``."""
    last = None
    for line in (text or "").splitlines():
        m = _ERR_RE.match(line.strip())
        if m:
            last = m.group(1).split(".")[-1]
    return last or ("ContextWindowExceededError" if "ContextWindowExceeded" in (text or "") else None)


def _hit_step_cap(manifest: dict) -> bool:
    """True if an agentic baseline EXHAUSTED its step/iteration budget — a forced,
    non-genuine answer that the ⁺all view re-scores as worst. Two signals in the manifest
    ``trace``: codeact (smolagents) ``state == "max_steps_error"``; rlm ``n_iterations >=
    max_iterations``. Mirrors ``evals.baselines._common.hit_step_cap``. False for every non-agentic baseline (they write no ``trace``)."""
    trace = manifest.get("trace") or {}
    if trace.get("state") == "max_steps_error":
        return True
    ni, mi = trace.get("n_iterations"), trace.get("max_iterations")
    return ni is not None and mi is not None and ni >= mi

# USD per 1M tokens, (input, output). Rates from https://openrouter.ai (edit + re-run to
# reprice). Matched by canonical-slug prefix, so e.g. claude-haiku-4-5-20251001 → haiku.
# The Qwen sizes are priced on ONE provider's endpoint — DeepInfra — so a size comparison reflects
# the models, not a mix of providers/quantizations (OpenRouter's per-provider list for 35B-A3B runs
# $0.08–$0.24 in; its default "Alibaba" route is $0.163/$1.30). Checked 2026-09-14 via
# https://openrouter.ai/api/v1/models/<id>/endpoints.
PRICE = {
    "qwen3-5-35b-a3b": (0.14, 1.00),         # Qwen3.5-35B-A3B (the served vLLM model) — DeepInfra fp8
    "qwen3-5-9b": (0.10, 0.15),              # Qwen3.5-9B (our method, size study) — DeepInfra bf16
    "qwen3-5-4b": (0.05, 0.07),              # Qwen3.5-4B (our method, size study) — NOT on OpenRouter; owner-supplied
    "rl-memoryagent-14b": (0.20, 0.20),      # MemAgent's trained model (BytedTsinghua-SIA/RL-MemoryAgent-14B, Qwen2.5-14B base)
    "claude-opus-4-8": (5.00, 25.00),        # Claude Opus 4.8 (claude-code)
    "claude-sonnet-5": (2.00, 10.00),        # Claude Sonnet 5 (claude-code, newest frontier ref)
    "claude-sonnet-4-6": (3.00, 15.00),      # Claude Sonnet 4.6 (claude-code, a second frontier ref)
    "claude-haiku-4-5": (1.00, 5.00),        # Claude Haiku 4.5 (claude-code's helper model)
    "text-embedding-3-small": (0.02, 0.00),  # OpenAI embedder (linearrag etc.) — input-only
}


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def canonical_model(model: str | None) -> str:
    """Provider-independent slug: ``hosted_vllm/Qwen/Qwen3.5-35B-A3B`` → ``qwen3-5-35b-a3b``."""
    tail = (model or "").split("/")[-1]
    return re.sub(r"[^a-z0-9]+", "-", tail.lower()).strip("-")


def _rate(model: str) -> tuple[float, float] | None:
    """(input, output) USD/1M for a canonical model slug, or None if unpriced."""
    return next((r for key, r in PRICE.items() if model.startswith(key)), None)


def _usd(inp: int, out: int, model: str) -> float | None:
    rate = _rate(model)
    return None if rate is None else inp / 1e6 * rate[0] + out / 1e6 * rate[1]


def _is_embedder(model: str) -> bool:
    """An embedding model. Its tokens are a retrieval cost — they price into USD but are NOT
    part of the in/out token comparison (which is the language model's input/output)."""
    return "embedding" in model


def _baseline_cost(usage: dict | None) -> tuple[int, int, float | None]:
    """(input, output, usd) from a manifest ``usage`` block. input/output are the LANGUAGE
    MODEL tokens — top-level ``prompt_tokens`` / ``completion_tokens`` per model, the FULL
    input with cache counted as normal (nested ``*_tokens_details`` ignored). An EMBEDDER's
    tokens (text-embedding-*) price into usd but are NOT counted in input/output. usd sums
    per model (claude-code's opus + haiku price separately); None if any model is unpriced."""
    inp = out = 0
    usd, all_priced = 0.0, True
    for model, b in ((usage or {}).get("total") or {}).items():
        slug = canonical_model(model)
        p, c = int(b.get("prompt_tokens", 0) or 0), int(b.get("completion_tokens", 0) or 0)
        u = _usd(p, c, slug)
        if u is None:
            all_priced = False
        else:
            usd += u
        if not _is_embedder(slug):              # embedder tokens count toward USD only
            inp += p
            out += c
    return inp, out, (usd if all_priced else None)


def _stage_pc(result: dict | None) -> tuple[int, int]:
    """(prompt, completion) tokens from one method stage's ``result.json``."""
    t = ((result or {}).get("totals") or {}).get("tokens") or {}
    return int(t.get("prompt", 0) or 0), int(t.get("completion", 0) or 0)


def _config_key(cfg: dict) -> str:
    """The run identity: every config knob (model canonicalized) except the group dims.
    Generic, so any per-baseline knob (effort, search_method, index versions, …) keeps two
    distinct runs in separate rows."""
    return "|".join(f"{k}={canonical_model(cfg[k]) if k == 'model' else cfg[k]}"
                    for k in sorted(cfg) if k not in ("benchmark", "baseline"))


def _baseline_variant(cfg: dict) -> str:
    """Readable run knob for a baseline: the sampling preset name, or claude-code's effort."""
    return cfg.get("config_name") or (f"effort={cfg['effort']}" if cfg.get("effort") else "default")


def load_baselines(root: Path | None = None) -> pd.DataFrame:
    """One row per baseline inference and per recorded error."""
    root = root or paths.logs_dir()
    if not root.is_dir():
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for bench_dir in sorted(p for p in root.iterdir()
                            if p.is_dir() and p.name not in NON_BENCHMARK_DIRS):
        benchmark = bench_dir.name.lower()
        for baseline_dir in sorted(p for p in bench_dir.iterdir() if p.is_dir()):
            baseline = baseline_dir.name
            for mp in baseline_dir.rglob("manifest.json"):
                m = _read_json(mp)
                if m is None:
                    continue
                cfg = m.get("config") or {}
                score = _read_json(mp.with_name("score.json")) or {}
                inp, out, usd = _baseline_cost(m.get("usage"))
                rows.append({
                    "source": "baseline", "method": baseline, "benchmark": benchmark,
                    "model": canonical_model(cfg.get("model")), "variant": _baseline_variant(cfg),
                    "summary_rounds": None, "config": _config_key(cfg), "task_id": m.get("task_id"),
                    "status": "ok", "error_type": None, "capped": _hit_step_cap(m),
                    "score": score.get("score"),
                    "scored": "score" in score, "scored_with": score.get("scored_with"),
                    "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                    "usd": usd, "created": m.get("created"), "log_dir": str(mp.parent),
                })
            for ep in baseline_dir.rglob("error.json"):
                e = _read_json(ep)
                if e is None:
                    continue
                cfg = e.get("config") or {}
                rows.append({
                    "source": "baseline", "method": baseline, "benchmark": benchmark,
                    "model": canonical_model(cfg.get("model")), "variant": _baseline_variant(cfg),
                    "summary_rounds": None, "config": _config_key(cfg), "task_id": e.get("task_id"),
                    "status": "error", "error_type": e.get("error_type"), "capped": False,
                    "score": None, "scored": False, "scored_with": None,
                    "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usd": 0.0,
                    "created": e.get("created"), "log_dir": str(ep.parent),
                })
    return pd.DataFrame(rows, columns=COLUMNS)


def load_method(root: Path | None = None) -> pd.DataFrame:
    """One row per (method run × strategy). Tokens = shared stages + that strategy's inference."""
    root = root or paths.method_logs()
    if not root.is_dir():
        return pd.DataFrame(columns=COLUMNS)
    rows = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _read_json(run_dir / "manifest.json")
        if m is None:
            continue
        cfg = m.get("config") or {}
        model = canonical_model(cfg.get("model"))
        shared_p = shared_c = 0
        for s in _STAGES:                              # summaries+proposer+extractor, shared
            p, c = _stage_pc(_read_json(run_dir / s / "result.json"))
            shared_p += p
            shared_c += c
        common = {
            "source": "method", "benchmark": (m.get("benchmark") or "").lower(), "model": model,
            "variant": cfg.get("sampling_preset") or "default",
            "summary_rounds": cfg.get("summary_rounds"), "task_id": m.get("task_id"),
            "created": m.get("created"),
        }
        inf_root = run_dir / "inference"
        strat_dirs = sorted(p for p in inf_root.iterdir() if p.is_dir()) if inf_root.is_dir() else []
        for sdir in strat_dirs:
            strategy = sdir.name
            result = _read_json(sdir / "result.json")
            score = _read_json(sdir / "score.json")
            row = {**common, "method": f"grounded-{strategy}",
                   "config": f"{_config_key(cfg)}|strategy={strategy}", "log_dir": str(sdir)}
            if result is None and score is None:
                # No answer for this strategy: a recorded failure (error.txt) or simply not run.
                # The method writes a per-strategy `error.txt` (a traceback blob, no structured
                # type) — recover the class so the ⁺all view can fold it (corpusqa: every error;
                # loong: ctx-window only). Mirrors the method board's `_read_strategy_result`.
                err_txt = sdir / "error.txt"
                if not err_txt.is_file():
                    continue                               # strategy not run for this config
                rows.append({**row, "status": "error",
                             "error_type": _parse_error_type(err_txt.read_text(errors="replace")),
                             "capped": False, "score": None, "scored": False, "scored_with": None,
                             "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "usd": 0.0})
                continue
            ip, ic = _stage_pc(result)
            inp, out = shared_p + ip, shared_c + ic
            rows.append({**row, "status": "ok", "error_type": None, "capped": False,
                         "score": (score or {}).get("score"),
                         "scored": bool(score and "score" in score),
                         "scored_with": (score or {}).get("scored_with"),
                         "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
                         "usd": _usd(inp, out, model)})
    return pd.DataFrame(rows, columns=COLUMNS)


def load_all(baselines_root: Path | None = None, method_root: Path | None = None) -> pd.DataFrame:
    """Baselines + method in one table, deduped to the latest run per (method, config, task)."""
    df = pd.concat([load_baselines(baselines_root), load_method(method_root)], ignore_index=True)
    if df.empty:
        return df
    key = ["source", "method", "benchmark", "config", "task_id"]
    return (df.sort_values("created", na_position="first", kind="stable")
              .drop_duplicates(subset=key, keep="last").reset_index(drop=True))


# ── Method artifact sizes: how much the just-in-time pipeline compresses the task ────────────
# Token counts come from the SERVED MODEL'S OWN TOKENIZER — exact, not estimated. `tokenizer.json`
# is fetched once from the HF hub and cached locally (~/.cache/huggingface), so every later call is
# a local file read. This is an ordinary package dependency, not a call into a model: tokenizing is
# deterministic and offline, so the analysis stays reproducible and cheap.
TOKENIZER_REPO = "Qwen/Qwen3.5-35B-A3B"        # the served vLLM model (== report.PRIMARY_MODEL)

# The task's INPUT documents are counted exactly too, just not here — the benchmark task files are
# far too large to ship, so each task's document length in these same tokens is vendored as
# `input_tokens` in data/*_meta.json (verified to reproduce every manifest's `context_chars`).

ARTIFACT_COLUMNS = [
    "benchmark", "task_id", "model", "variant", "summary_rounds", "n_docs", "context_chars",
    "n_rounds", "n_summaries", "summary_chars", "parse_chars", "schema_chars",
    "summary_tokens", "parse_tokens", "schema_tokens", "run_dir",
]


@cache
def tokenizer():
    """The served model's tokenizer (cached per process). Imported lazily so the rest of the
    package works without the tokenizer deps installed."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    return Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))


def count_tokens_batch(texts: list[str]) -> list[int]:
    """EXACT token counts for many texts (batched — the Rust tokenizer runs ~13 MB/s this way,
    far faster than one call per text). Empty strings cost nothing and return 0."""
    idx = [i for i, t in enumerate(texts) if t]
    out = [0] * len(texts)
    if idx:
        enc = tokenizer().encode_batch([texts[i] for i in idx], add_special_tokens=False)
        for i, e in zip(idx, enc):
            out[i] = len(e.ids)
    return out


def load_method_artifacts(root: Path | None = None) -> pd.DataFrame:
    """One row per method RUN FOLDER (not per strategy — summaries/proposer/extractor are shared
    by all of a run's strategies): the size of each just-in-time artifact next to the task's input.

    - ``context_chars`` — the input documents' size, straight from the manifest.
    - ``summary_chars`` — the FINAL round's per-doc summaries, concatenated (``rounds[-1]``).
      A ``summary_rounds=0`` run has no rounds, hence 0 — that's the "w/o summaries" ablation.
    - ``parse_chars`` — the extractor's merged structured ``parsed`` object as COMPACT JSON
      (no whitespace, non-ASCII kept). This measures the structured data itself: the inference
      prompt pretty-prints the same object, but that indentation is about a third of those
      tokens and says more about the serializer than about the representation.
    - ``schema_chars`` — the proposer's generated Pydantic ``schema_code``.

    Both char AND token columns are EXACT — the token counts come from the served model's own
    tokenizer (:func:`count_tokens_batch`), applied in one batched pass at the end.
    """
    root = root or paths.method_logs()
    if not root.is_dir():
        return pd.DataFrame(columns=ARTIFACT_COLUMNS)
    rows, texts = [], []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        m = _read_json(run_dir / "manifest.json")
        if m is None:
            continue
        cfg = m.get("config") or {}
        rounds = ((_read_json(run_dir / "summaries" / "result.json") or {}).get("rounds")) or []
        final = (rounds[-1].get("summaries") or []) if rounds else []
        summary_text = "".join(s for s in final if isinstance(s, str))
        parsed = (_read_json(run_dir / "extractor" / "result.json") or {}).get("parsed")
        # COMPACT: whitespace is not part of the representation (see the docstring)
        parse_text = "" if parsed is None else json.dumps(parsed, ensure_ascii=False,
                                                          separators=(",", ":"))
        schema_text = (_read_json(run_dir / "proposer" / "result.json") or {}).get("schema_code") or ""
        texts.append((summary_text, parse_text, schema_text))
        rows.append({
            "benchmark": (m.get("benchmark") or "").lower(), "task_id": m.get("task_id"),
            "model": canonical_model(cfg.get("model")), "variant": cfg.get("sampling_preset") or "default",
            "summary_rounds": cfg.get("summary_rounds"), "n_docs": m.get("n_docs"),
            "context_chars": m.get("context_chars"),
            "n_rounds": len(rounds), "n_summaries": len(final),
            "summary_chars": len(summary_text), "parse_chars": len(parse_text),
            "schema_chars": len(schema_text), "run_dir": str(run_dir),
        })
    # One batched tokenizer pass over every artifact (fast; per-text calls would be ~50x slower).
    flat = count_tokens_batch([t for triple in texts for t in triple])
    for i, row in enumerate(rows):
        row["summary_tokens"], row["parse_tokens"], row["schema_tokens"] = flat[3 * i:3 * i + 3]
    return pd.DataFrame(rows, columns=ARTIFACT_COLUMNS)
