"""What it takes to run the method over a benchmark.

Everything here is about *driving* the pipeline (which lives in ``evals.r3con.pipeline``
and never imports a benchmark), not about judging what comes out:

- ``loong`` / ``corpusqa`` / ``dracula`` — one adapter per benchmark, each a thin
  ``(task, documents)`` view over the repo's own ``evals.benchmarks`` surface, plus the
  per-task body that writes the run folder. The three stay decoupled and never import
  each other.
- ``runner`` — the subprocess launcher: one child per task, for true concurrency and
  crash isolation. The child script is a parameter, so all three reuse it.

The dependency direction is one-way: the harness imports the pipeline, never the reverse.
"""
