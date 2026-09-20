"""R3Con — just-in-time reasoning over a collection of documents.

Two halves, deliberately separate:

- ``pipeline`` — the method itself: four stages (summaries -> schema proposal ->
  extraction -> reasoning) over a collection of documents, each assumed to fit in the
  model's context. ``pipeline.gr_answer(task=..., documents=..., config=...)`` is the
  whole entry point; it is benchmark-agnostic and never imports a benchmark.
- ``harness`` — what it takes to run that over a benchmark: the three adapters
  (loong / corpusqa / dracula) over the repo's own ``evals.benchmarks`` surface, plus
  the subprocess launcher.

Run it with ``scripts/r3con/<benchmark>/run.py``. Each task lands in its own folder under
``logs/r3con/`` — the answer, and every intermediate the method built to get there.
Nothing here grades anything: the artifacts are for reading, not scoring.
"""
