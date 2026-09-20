"""RAPTOR baseline pipeline — connects the vendored recursive-summary-tree retriever
to the harness.

RAPTOR (Sarthi et al., ICLR 2024; arXiv 2401.18059) is a retrieval method: chunk the
document into ~100-token leaves, embed them, then RECURSIVELY cluster (UMAP + GMM) and
LLM-summarize each cluster into the next tree level — a tree whose leaves are raw chunks
and whose higher levels are progressively more abstract summaries. At query time it
"collapses" the tree (all levels → one pool) and retrieves the top nodes by embedding
similarity (up to a token budget), then answers over that context.

Per task: pool the instance's document bundle into one text → ``add_documents`` builds
the tree (via our injected litellm/OpenAI seams) → ``answer_question`` retrieves + answers.
We pose the task through RAPTOR's own front door (``RetrievalAugmentation``), so the method
is unchanged — only the model backends are seamed (PROVENANCE).

SIMPLE NO-REUSE logging (the readagent/rlm layout, the maintainer's call): each
task run gets its OWN folder ``logs/{benchmark}/raptor/{run_tag}/`` holding EVERYTHING — a
readable ``tree.json`` (the built tree's structure + summaries), ``manifest.json`` (with the
**TOTAL** cost — every build summary + every embedding + the QA answer, in one number),
``calls.json``, and (at score time) ``score.json``. Nothing is content-addressed, no tree is
reused; every run rebuilds its tree from scratch (the maintainer's benchmarks have no
doc-set overlap, so reuse buys nothing). The runner DOES resume (skips tasks already done
for the config).

SUPPORTED_BENCHMARKS = {loong, corpusqa, longhealth, dracula} — per-instance multi-doc bundles (pooled into one
text, then RAPTOR's own chunker + tree build).

**Offline-embed / online-build split** (``--phase {all,embed,build}``): ``embed`` runs phase 1 — the
OpenAI leaf embeddings (the heavy ~82% burst), NO completion LLM — and checkpoints them
(``embed.json`` + ``leaves.pkl``); ``build`` runs phase 2 against the served LLM (cluster + summarize
+ QA). Split so the rate-limit-prone OpenAI embedding burst runs separately from (and can be
concurrency-capped independently of) the vLLM summary phase. ``--phase all`` (default) does both in
one pass, byte-compatible with the pre-split manifest. Deviation ledger:
evals/baselines/raptor/PROVENANCE.md
"""
from __future__ import annotations

import contextlib
import json
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARKS: frozenset[str] = frozenset({"loong", "corpusqa", "dracula"})

# Run-config version — bump on a change that alters a run's output (tree params, composition).
# v2: CJK-aware leaf chunking (D9) — RAPTOR's ASCII-only `split_text` left Chinese prose as
#     oversized leaves → tiny-cluster UMAP `n_neighbors ≤ 1` crash (~50% of Loong ZH `legal`).
# v3: reasoning summaries (D10) — the summary length control moves from the max_tokens cap to a
#     prompt hint + a 32768 budget, so RAPTOR runs on the served (thinking) model. Changes every
#     summary → every tree, so a new run version.
# v4: tree hyperparameters re-scaled for ≫6K-token corpora (D12) — paper-default chunk=100 yields
#     ~10K leaves on a 1M-token CorpusQA instance (13× past RAPTOR's validated ~78K-tok/~780-leaf
#     ceiling) → ~2,300 reasoning summaries/task = intractable. Re-scaled below to bring the leaf
#     count back into RAPTOR's regime; also drops the summary generation cap (llm.py). New tree → new version.
_RUN_VERSION = "v4"

# v4 tree hyperparameters (D12), re-scaled to the corpus (RAPTOR's defaults assumed ~6K-token docs);
# tied to _RUN_VERSION so v4 outputs never mix with v3. Rationale + the principled argument: PROVENANCE D12.
_CHUNK_TOKENS = 2000          # leaf size (was 100): ~500 leaves @1M → ~115 summaries/task (the cost driver)
_RECLUSTER_THRESHOLD = 40000  # cluster-split ceiling (was 3500): lets the natural GMM clusters stand —
                              #   recursion fires only on genuinely huge, safe-to-recluster clusters (no crash)
_SUMMARY_LENGTH = 400         # summary prompt-hint tokens (was 100): ~28× compression of the bigger clusters
_RETRIEVAL_TOP_K = 20         # collapse-tree nodes retrieved (was 10)
_RETRIEVAL_MAX_TOKENS = 16000 # retrieved-context budget (was 3500): sized to the bigger nodes + reasoning reader

# RAPTOR's PUBLISHED hyperparameters (the values the authors chose), selected by `--paper-hparams`
# (run_config["paper_hparams"] is True). Right for docs in RAPTOR's validated regime (≤~78K tok, e.g.
# LongHealth), where the D12 large-corpus re-scale degenerates — chunk=2000 yields too few leaves
# (~5-12) to cluster, so no summary tree forms and retrieval covers ~everything (RAPTOR ≈ direct-llm).
# Do NOT use on the 1M-token corpora: chunk=100 → ~10,000 leaves → thousands of summaries/task (exactly
# the blowup D12 exists to prevent). Folded into the run config (hashed), so paper vs D12 runs never mix.
_PAPER_HPARAMS = {
    "chunk_tokens": 100, "recluster_threshold": 3500, "summary_length": 100,
    "retrieval_top_k": 10, "retrieval_max_tokens": 3500,
}
_D12_HPARAMS = {
    "chunk_tokens": _CHUNK_TOKENS, "recluster_threshold": _RECLUSTER_THRESHOLD,
    "summary_length": _SUMMARY_LENGTH, "retrieval_top_k": _RETRIEVAL_TOP_K,
    "retrieval_max_tokens": _RETRIEVAL_MAX_TOKENS,
}


def _hparams(run_config: dict) -> dict:
    """The 5 tree/retrieval hyperparameters for this run: RAPTOR's PAPER defaults when
    `--paper-hparams` was passed (``run_config["paper_hparams"]`` is True), else the D12 large-corpus
    re-scale (the default — byte-identical to before when the key is absent)."""
    return _PAPER_HPARAMS if run_config.get("paper_hparams") else _D12_HPARAMS

# The built tree, dumped here (readable) inside the run folder for deep-dives.
_TREE_FILE = "tree.json"

# Live progress for a long-running task — RAPTOR writes tree.json + manifest only at the END, and a
# single task can fire hundreds–thousands of summary calls over minutes, so this tiny file is
# updated as the task moves through its stages (and per cluster summarized) so a mid-flight task's
# folder shows where it is + how far. See ``_Progress``.
_PROGRESS_FILE = "progress.json"

# Phase-1 (embed) checkpoint — the OFFLINE leaf embeddings (the heavy ~82% of all node embeds) that
# phase-2 (build) consumes, split so the OpenAI embedding burst can run separately (and rate-limited)
# from the vLLM summary phase. ``embed.json`` is the SMALL marker (task_id + config + query + the
# embed usage) that resumption + the cleaner key on (mirrors linearrag's ``retrieval.json``);
# ``leaves.pkl`` is the big leaf-Node data kept alongside (mirrors linearrag's ``index/``). Analysis
# / scoring IGNORE both (they key on manifest/error), so a ``--phase all`` run (which never writes
# them) is unaffected, and its manifest is byte-compatible with the pre-split one.
_EMBED_FILE = "embed.json"
_LEAVES_FILE = "leaves.pkl"

# The summary length is a PROMPT hint (`_SUMMARY_LENGTH`), not the generation cap: on a reasoning model
# the cap is eaten by reasoning_content → empty summaries (D10), so the summarize call now sends NO
# max_tokens at all (llm.py) and lets the model reason then write a complete summary. v4 (D12) sets the
# hint to `_SUMMARY_LENGTH` and the recluster threshold to `_RECLUSTER_THRESHOLD` — large enough that the
# natural GMM clusters are never force-split, which keeps the tree well-formed AND removes the
# tiny-cluster UMAP `n_neighbors ≤ 1` crash. See llm.py / PROVENANCE D10+D12.


class _Progress:
    """Writes/updates a tiny live ``progress.json`` in the run folder so a long RAPTOR task's stage
    + counts are visible WHILE it runs (debug long batches; estimate remaining work). The dominant
    cost is the per-cluster summaries, so ``summary_done()`` ticks one per summary (called from the
    tree builder's worker threads — D8 parallel summaries — hence the lock). ``stage(name, **extra)``
    marks a phase (pooling → building_tree[n_leaves] → retrieving → done[n_tree_nodes]). Written
    atomically (tmp + replace) so a reader never sees a half-written file."""

    def __init__(self, path: Path, task_id: str, n_docs: int):
        self._path = Path(path)
        self._lock = threading.Lock()
        self._state: dict[str, Any] = {
            "task_id": str(task_id), "n_docs": n_docs, "stage": "starting",
            "n_leaves": None, "summaries_done": 0, "n_tree_nodes": None,
            "started_at": round(time.time(), 1),
        }
        self._flush()  # single-threaded at construction

    def stage(self, name: str, **extra: Any) -> None:
        with self._lock:
            self._state["stage"] = name
            self._state.update(extra)
            self._flush()

    def summary_done(self) -> None:
        with self._lock:
            self._state["summaries_done"] += 1
            self._flush()

    def _flush(self) -> None:
        # The caller holds ``self._lock`` (except ``__init__``, single-threaded at construction).
        now = time.time()
        self._state["updated_at"] = round(now, 1)
        self._state["elapsed_s"] = round(now - self._state["started_at"], 1)
        tmp = self._path.with_name(self._path.name + ".tmp")
        tmp.write_text(json.dumps(self._state, ensure_ascii=False, indent=2))
        tmp.replace(self._path)  # atomic — readers see a complete file


def _benchmark_name(benchmark) -> str:
    return benchmark.__name__.rsplit(".", 1)[-1]


def _build_query(benchmark, task_id: str) -> str:
    """The question handed to RAPTOR's ``answer_question`` — the SAME components every
    other baseline poses (the documents are the tree, dropped from the question).

    - **loong**: ``instruction`` (the whole task when ``question`` is empty, e.g. paper
      Chain-of-Reasoning) else ``instruction\\n\\nquestion``.
    - **corpusqa**: ``question\\n\\n{output-requirements}`` — the output-requirements block (with
      the ``The answer is:`` contract the judge extracts) MUST reach the model.
    """
    name = _benchmark_name(benchmark)
    if name == "loong":
        instruction, question, _docs = benchmark.get_task(task_id)
        return instruction if not question.strip() else f"{instruction}\n\n{question}"
    if name == "corpusqa":
        instruction, question, _docs = benchmark.get_task(task_id)
        return f"{question}\n\n{instruction}"
    if name == "dracula":
        # The bare question; the 45-doc corpus is pooled into the summary tree
        # (the D12 scale applies as for loong/corpusqa — similar token profile).
        question, _docs = benchmark.get_task(task_id)
        return question
    raise ValueError(f"raptor has no query assembly for benchmark {name!r}")


def _pool_documents(documents: list[str]) -> str:
    """RAPTOR's ``add_documents`` takes ONE text; Loong is a per-instance multi-doc bundle.
    Pool the docs into a single blob (they already carry their Loong titles) so RAPTOR's
    own ~100-token chunker + tree build runs unmodified over the whole bundle (the
    per-document-then-pool convention the other multi-doc baselines use)."""
    return "\n\n".join(documents)


@contextlib.contextmanager
def _lenient_tiktoken():
    """D11 — let tiktoken encode special-token strings (e.g. ``<|endoftext|>`` / ``<|endofprompt|>``)
    that appear in DOCUMENT text as normal tokens instead of raising. RAPTOR counts tokens with
    ``tokenizer.encode(text)`` in three places — the chunker, the clustering recluster-threshold
    (``cluster_utils``), and the retrieval token budget (``tree_retriever``) — none of which pass
    ``disallowed_special``, so tiktoken raises ``ValueError`` on any doc that literally contains such a
    marker (some Loong ``paper`` docs do). We default ``disallowed_special=()`` for the task's
    duration. Faithful: the doc genuinely contains that text, so counting it as ordinary tokens (vs
    crashing) is the correct reading — it shifts chunk boundaries trivially, never the method. One
    task per child process, and the patch wraps the build's ThreadPool (applied/removed on the main
    thread around it), so there's no toggling while worker threads run."""
    import tiktoken
    orig = tiktoken.Encoding.encode

    def encode(self, text, *args, **kwargs):  # noqa: ANN001
        kwargs.setdefault("disallowed_special", ())
        return orig(self, text, *args, **kwargs)

    tiktoken.Encoding.encode = encode
    try:
        yield
    finally:
        tiktoken.Encoding.encode = orig


def _tree_summary(tree: Any) -> dict[str, Any]:
    """A readable JSON view of the built tree (for deep-dives): per-node text + children,
    plus the layer structure. The node embeddings are NOT included (large, not useful to
    read)."""
    nodes = {
        str(idx): {"text": node.text, "children": sorted(node.children)}
        for idx, node in tree.all_nodes.items()
    }
    layers = {
        str(layer): [n.index for n in node_list]
        for layer, node_list in tree.layer_to_nodes.items()
    }
    return {
        "num_layers": tree.num_layers,
        "n_nodes": len(tree.all_nodes),
        "n_leaf_nodes": len(tree.leaf_nodes),
        "n_root_nodes": len(tree.root_nodes),
        "layer_to_node_indices": layers,
        "nodes": nodes,
    }


@contextlib.contextmanager
def _summary_pool_size(n: int | None):
    """Scope a per-task cap/raise on RAPTOR's cluster-summary ThreadPool (D8). ``construct_tree``
    creates an UNCAPPED ``ThreadPoolExecutor()`` → Python's default ``min(32, cpu_count+4)``; when
    ``n`` is given (``--summary-workers``) we monkeypatch the vendored module's ``ThreadPoolExecutor``
    name to inject ``max_workers=n`` for the duration of one tree build (each task is its own
    subprocess, so the patch is process-local + scoped to the synchronous ``construct_tree`` call).
    ``n is None`` → NO patch → the vendored default is **byte-for-byte unchanged** (the no-flag
    contract). Output-identical either way — the pool SIZE never changes the summaries or the
    deterministic main-thread node-index assignment; it's a pure throughput knob (like
    ``--embed-workers``)."""
    if n is None:
        yield
        return
    import functools
    from evals.baselines.raptor.upstream import cluster_tree_builder as _ctb
    orig = _ctb.ThreadPoolExecutor
    _ctb.ThreadPoolExecutor = functools.partial(orig, max_workers=n)
    try:
        yield
    finally:
        _ctb.ThreadPoolExecutor = orig


def _connector_tree_builder(tree_builder_config, progress: Any = None, embed_workers: int | None = None,
                            summary_workers: int | None = None):
    """A ``ClusterTreeBuilder`` carrying the connector's faithful tree-build deviations.

    **D8 — parallel cluster summaries (throughput only).** RAPTOR's ``construct_tree`` already
    supports multithreading (the authors wrote it), but ``build_from_text`` calls it WITHOUT the
    flag → it falls back to the sequential default, so every per-cluster summary runs one-at-a-time
    (the dominant per-task cost). We flip RAPTOR's own flag on. **Output-identical**: node indices
    are assigned in deterministic main-thread order (``next_node_index`` incremented in the submit
    loop, passed by value), the ``new_level_nodes`` writes are already lock-guarded upstream, and the
    summaries are independent → the tree is byte-identical, just built in parallel rounds. Our
    summary + embedding seams are thread-safe (they lock their usage accumulators).

    **D9 — CJK-aware leaf chunking.** RAPTOR's ``utils.split_text`` splits only on ASCII
    ``.!?\\n,;:``; Chinese prose (full-width ``。！？，；：、``) is never split → oversized leaves →
    a 2–4-node cluster exceeds the 3500-token recluster threshold → recursion on a tiny set → UMAP
    ``n_neighbors ≤ 1`` crash (observed ~50% of Loong ZH ``legal``). We override ``build_from_text``
    — mirroring upstream VERBATIM except the splitter — to chunk with ``chunker.split_text`` (a strict
    superset: ASCII text is byte-identical, so EN trees are unchanged). See chunker.py / PROVENANCE D9.

    NEITHER is a method change: the clustering/retrieval/QA all run unmodified; D8 is RAPTOR's own
    flag and D9 produces the ~100-token leaves RAPTOR intends.

    The two halves of ``build_from_text`` are split into ``build_leaves`` (chunk + embed the leaves —
    the OpenAI-heavy ~82% burst) and ``build_tree_from_leaves`` (cluster + LLM-summarize → tree), so
    the ``--phase embed`` / ``--phase build`` split can run them separately; ``build_from_text``
    composes the two, byte-identically to before. ``embed_workers`` optionally caps the leaf-embed
    concurrency (rate-limit the OpenAI burst) — ``None`` keeps RAPTOR's default ThreadPool.
    ``summary_workers`` (``--summary-workers``) likewise caps/RAISES the per-task cluster-SUMMARY
    ThreadPool (D8) — ``None`` keeps RAPTOR's ``min(32, cpu+4)`` default unchanged; output-identical."""
    import copy

    from evals.baselines.raptor.chunker import split_text as _cjk_split_text
    from evals.baselines.raptor.upstream.cluster_tree_builder import ClusterTreeBuilder
    from evals.baselines.raptor.upstream.tree_structures import Tree

    class _ConnectorClusterTreeBuilder(ClusterTreeBuilder):
        def construct_tree(self, current_level_nodes, all_tree_nodes, layer_to_nodes,
                           use_multithreading: bool = True):
            # --summary-workers (D8): cap/raise the cluster-summary ThreadPool; None → vendored default.
            with _summary_pool_size(summary_workers):
                return super().construct_tree(current_level_nodes, all_tree_nodes, layer_to_nodes,
                                              use_multithreading=use_multithreading)

        def multithreaded_create_leaf_nodes(self, chunks):
            # Upstream multithreaded_create_leaf_nodes VERBATIM except an OPTIONAL bounded worker
            # count (``--embed-workers``) so the OpenAI leaf-embed burst can be rate-limited. None →
            # upstream's default (unbounded) ThreadPool.
            if embed_workers is None:
                return super().multithreaded_create_leaf_nodes(chunks)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=embed_workers) as executor:
                future_nodes = {executor.submit(self.create_node, i, t): i
                                for i, t in enumerate(chunks)}
                leaf_nodes = {}
                for future in as_completed(future_nodes):
                    index, node = future.result()
                    leaf_nodes[index] = node
            return leaf_nodes

        def build_leaves(self, text, stage: str = "building_tree"):
            # PHASE 1: chunk (D9 CJK splitter) + embed the leaf nodes (multithreaded). The leaf
            # embeddings are the dominant ~82% of all node embeds → the rate-limit-prone burst.
            chunks = _cjk_split_text(text, self.tokenizer, self.max_tokens)
            if progress is not None:
                progress.stage(stage, n_leaves=len(chunks))
            return self.multithreaded_create_leaf_nodes(chunks)

        def build_tree_from_leaves(self, leaf_nodes):
            # PHASE 2: VERBATIM upstream build_from_text tail — cluster + LLM-summarize the given
            # leaf nodes into the tree (D8 multithreaded summaries). No leaf embedding here.
            layer_to_nodes = {0: list(leaf_nodes.values())}
            all_nodes = copy.deepcopy(leaf_nodes)
            root_nodes = self.construct_tree(all_nodes, all_nodes, layer_to_nodes)
            return Tree(all_nodes, root_nodes, leaf_nodes, self.num_layers, layer_to_nodes)

        def build_from_text(self, text, use_multithreading: bool = True):
            # --phase all: the two halves composed — byte-identical to the pre-split single pass.
            return self.build_tree_from_leaves(self.build_leaves(text, stage="building_tree"))

    return _ConnectorClusterTreeBuilder(tree_builder_config)


def embed_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    *,
    progress: Any = None,
    embed_workers: int | None = None,
) -> dict[str, Any]:
    """PHASE 1 (offline, **NO completion LLM / NO vLLM**): pool docs → chunk → embed the leaves.
    Returns the checkpoint — the leaf nodes (with their embeddings), the query, and the **embedding
    usage** — which is everything phase-2 needs. Only OpenAI embeddings; the leaves are ~82% of all
    node embeds, so this is the heavy, rate-limit-prone burst. The ``--phase all`` path calls this
    then ``build_one`` in-memory; ``--phase embed`` saves the checkpoint (``save_embed``) so a
    separate ``--phase build`` can finish later. ``embed_workers`` optionally caps the leaf-embed
    concurrency (rate-limit the burst)."""
    logging.getLogger().setLevel(logging.WARNING)  # RAPTOR's modules basicConfig(INFO) at import.
    from evals.baselines.raptor.embedding import RaptorEmbeddingModel
    # Build the tree-builder config DIRECTLY (not via RetrievalAugmentationConfig, which constructs a
    # default GPT3TurboQAModel → an OpenAI client we don't want in the LLM-free embed phase). The
    # leaf chunking params (max_tokens=100, the CJK splitter) are identical either way.
    from evals.baselines.raptor.upstream.cluster_tree_builder import ClusterTreeConfig

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    documents = benchmark.get_documents(task_id)
    if progress is None:
        progress = _Progress(run_dir / _PROGRESS_FILE, task_id, len(documents))
    progress.stage("embedding")
    pooled = _pool_documents(documents)
    query = _build_query(benchmark, task_id)

    embedder = RaptorEmbeddingModel(run_config["embedding_model"])  # OpenAI via env key
    # cluster_embedding_model "EMB" matches what RetrievalAugmentationConfig(embedding_model=…) uses,
    # so the leaf embeddings land under the SAME key the phase-2 clustering reads.
    hp = _hparams(run_config)  # PAPER defaults with --paper-hparams, else the D12 re-scale
    tb_config = ClusterTreeConfig(embedding_models={"EMB": embedder}, cluster_embedding_model="EMB",
                                  max_tokens=hp["chunk_tokens"])  # leaf chunk size (drives the leaf count)
    builder = _connector_tree_builder(tb_config, progress, embed_workers=embed_workers)
    with _lenient_tiktoken():  # D11: docs may literally contain <|endoftext|> &c. — count, don't crash
        leaf_nodes = builder.build_leaves(pooled, stage="embedding")
    progress.stage("embedded", n_leaves=len(leaf_nodes))
    return {
        "query": query,
        "n_docs": len(documents),
        "n_leaves": len(leaf_nodes),
        "pooled_chars": len(pooled),
        "leaf_nodes": leaf_nodes,            # the big data (→ leaves.pkl); NOT in embed.json
        "embed_usage": embedder.usage,       # the LLM-free cost, carried into the TOTAL in build_one
        "_progress": progress,               # in-memory only (run_one shares it; not serialized)
    }


def build_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    summary_workers: int | None = None,
) -> dict[str, Any]:
    """PHASE 2 (online, the vLLM summaries + QA): reconstruct the tree from phase-1's leaves
    (cluster + LLM-summarize, D8/D10), collapse-tree retrieve, answer. ``checkpoint`` is the
    ``embed_one`` output (in-memory for ``--phase all``, or loaded from disk for ``--phase build``).
    Returns the harness record — ``raw_answer`` + the **TOTAL** ``usage`` (phase-1 leaf embeddings +
    these summaries + summary-node embeddings + QA) + ``calls_full`` + the ``trace``. The manifest is
    byte-compatible with the pre-split ``run_one``."""
    logging.getLogger().setLevel(logging.WARNING)
    from evals.baselines.raptor.embedding import RaptorEmbeddingModel
    from evals.baselines.raptor.llm import RaptorQAModel, RaptorSummarizationModel
    # Front door from its submodule (the package __init__ pulls faiss at import; RA uses the tree
    # retriever, not faiss — see PROVENANCE).
    from evals.baselines.raptor.upstream.RetrievalAugmentation import (
        RetrievalAugmentation,
        RetrievalAugmentationConfig,
    )
    from evals.baselines.raptor.upstream.tree_retriever import TreeRetriever

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    query = checkpoint["query"]
    leaf_nodes = checkpoint["leaf_nodes"]
    progress = checkpoint.get("_progress") or _Progress(
        run_dir / _PROGRESS_FILE, task_id, checkpoint["n_docs"])

    seed = run_config.get("seed")
    completion_params = run_config.get("completion_params")
    summarizer = RaptorSummarizationModel(litellm_kwargs, seed=seed, completion_params=completion_params)
    summarizer._progress = progress  # live per-summary tick during the (long) tree build
    qa = RaptorQAModel(litellm_kwargs, seed=seed, completion_params=completion_params)
    embedder = RaptorEmbeddingModel(run_config["embedding_model"])  # ONLY the summary-node embeds here

    # Build the tree-builder config DIRECTLY so the D12 clustering knobs reach the vendored builder:
    # RetrievalAugmentationConfig exposes tb_summarization_length but NOT clustering_params /
    # reduction_dimension, so we pass a full ClusterTreeConfig. embedding_model= still flows to the
    # retriever (tr_*); the tree builder uses tb_config (cluster_embedding_model "EMB" matches the
    # phase-1 leaf embeddings). max_tokens is a no-op here (build reconstructs from leaves.pkl, no
    # re-chunk) but kept for config consistency.
    from evals.baselines.raptor.upstream.cluster_tree_builder import ClusterTreeConfig
    hp = _hparams(run_config)  # PAPER defaults with --paper-hparams, else the D12 re-scale
    tb_config = ClusterTreeConfig(
        embedding_models={"EMB": embedder}, cluster_embedding_model="EMB",
        summarization_model=summarizer,
        max_tokens=hp["chunk_tokens"], summarization_length=hp["summary_length"],
        clustering_params={"max_length_in_cluster": hp["recluster_threshold"]},
    )
    config = RetrievalAugmentationConfig(
        tree_builder_config=tb_config, qa_model=qa, embedding_model=embedder,
    )
    ra = RetrievalAugmentation(config=config)
    builder = _connector_tree_builder(config.tree_builder_config, progress, summary_workers=summary_workers)
    progress.stage("building_tree", n_leaves=len(leaf_nodes))
    # Reconstruct the tree from the checkpointed leaves (cluster + summarize), then wire the retriever
    # exactly as `add_documents` would have (it builds the tree AND creates the TreeRetriever), and
    # answer. D11: the clustering recluster-threshold + the retrieval token budget both tiktoken-encode
    # node text, so wrap them too (a leaf can contain <|endoftext|> &c.).
    with _lenient_tiktoken():
        ra.tree = builder.build_tree_from_leaves(leaf_nodes)
        ra.retriever = TreeRetriever(ra.tree_retriever_config, ra.tree)
        # D12: retrieval scaled to the bigger leaves + the reasoning reader (was RAPTOR's top_k=10 /
        # max_tokens=3500); still collapse_tree, still a selective subset (~20 of ~600 nodes).
        progress.stage("retrieving")
        raw_answer, layer_information = ra.answer_question(
            query, top_k=hp["retrieval_top_k"], max_tokens=hp["retrieval_max_tokens"],
            return_layer_information=True)

    tree_summary = _tree_summary(ra.tree)
    progress.stage("done", n_tree_nodes=tree_summary["n_nodes"], num_layers=tree_summary["num_layers"])
    (run_dir / _TREE_FILE).write_text(json.dumps(tree_summary, ensure_ascii=False, indent=2))

    # TOTAL = phase-1 leaf embeddings (from the checkpoint) + summaries + summary-node embeddings + QA.
    usage = _merge_usage(checkpoint.get("embed_usage") or {}, summarizer.usage, qa.usage, embedder.usage)
    return {
        "raw_answer": raw_answer,
        "usage": usage,
        # Full request/response of every summary + QA call (embeddings are vectors — counted
        # in usage but not dumped here) → calls.json.
        "calls_full": summarizer.full_calls + qa.full_calls,
        "trace": {
            "query": query,
            "n_docs": checkpoint["n_docs"],
            "pooled_chars": checkpoint["pooled_chars"],
            "num_tree_nodes": tree_summary["n_nodes"],
            "num_layers": tree_summary["num_layers"],
            "n_leaf_nodes": tree_summary["n_leaf_nodes"],
            # Which tree nodes the collapse-tree retrieval picked (index + layer) — the
            # retrieved context the QA answered over.
            "retrieved_layer_information": layer_information,
            "n_retrieved": len(layer_information) if layer_information else 0,
        },
    }


def run_one(
    benchmark,
    task_id: str,
    run_config: dict[str, Any],
    run_dir: Path,
    litellm_kwargs: dict[str, Any],
    *,
    embed_workers: int | None = None,
    summary_workers: int | None = None,
) -> dict[str, Any]:
    """``--phase all``: phase-1 (embed) + phase-2 (build) in one pass, sharing one progress and the
    in-memory checkpoint (no checkpoint file). Produces the IDENTICAL record to before the split, so
    ``all`` is fully backward compatible."""
    checkpoint = embed_one(benchmark, task_id, run_config, run_dir, embed_workers=embed_workers)
    return build_one(benchmark, task_id, run_config, run_dir, litellm_kwargs, checkpoint,
                     summary_workers=summary_workers)


def save_embed(
    run_dir: Path, task_id: str, run_config: dict[str, Any], checkpoint: dict[str, Any],
) -> None:
    """Persist the phase-1 checkpoint: the SMALL ``embed.json`` marker (``task_id`` + ``config`` for
    phase-2 resumption matching, like manifest/error, + query/usage) and the big ``leaves.pkl`` (the
    leaf Node dict with embeddings). The ``_progress`` is in-memory only and never serialized."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    embed_usage = checkpoint.get("embed_usage") or {}
    # Big data → leaves.pkl: the leaf nodes (vectors) AND the full embed usage (incl. its per-call
    # list, ~1 entry per leaf) — both only needed by phase-2 build. Written FIRST.
    with open(run_dir / _LEAVES_FILE, "wb") as f:
        pickle.dump({"leaf_nodes": checkpoint["leaf_nodes"], "embed_usage": embed_usage}, f)
    # Small marker → embed.json: what the resumption scan + cleaner key on (task_id + config) plus the
    # query/sizes + the embed-usage TOTAL (the per-model rollup, for at-a-glance cost; the bulky
    # per-call list lives in leaves.pkl). Written LAST, so embed.json present ⇒ leaves.pkl complete.
    meta = {
        "task_id": str(task_id), "config": run_config,
        "query": checkpoint["query"], "n_docs": checkpoint["n_docs"],
        "n_leaves": checkpoint["n_leaves"], "pooled_chars": checkpoint["pooled_chars"],
        "embed_usage_total": embed_usage.get("total", {}),
    }
    (run_dir / _EMBED_FILE).write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def load_embed(run_dir: Path) -> dict[str, Any]:
    """Load the phase-1 checkpoint (``embed.json`` marker + ``leaves.pkl`` leaf nodes + embed usage)
    so ``build_one`` can finish the task in a later ``--phase build`` pass."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / _EMBED_FILE).read_text())
    with open(run_dir / _LEAVES_FILE, "rb") as f:
        data = pickle.load(f)
    return {**meta, "leaf_nodes": data["leaf_nodes"], "embed_usage": data["embed_usage"]}


def _merge_usage(*usages: dict[str, Any]) -> dict[str, Any]:
    """Merge ``{total, calls}`` envelopes (per-model numeric sum + concatenated call list)."""
    from evals.llm.usage import _merge_numeric
    total: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []
    for usage in usages:
        for model, bucket in (usage.get("total") or {}).items():
            _merge_numeric(total.setdefault(model, {}), bucket)
        calls.extend(usage.get("calls") or [])
    return {"total": total, "calls": calls}
