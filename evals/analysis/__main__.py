"""Top-level CLI: `python -m evals.analysis [--benchmark X] [--baseline Y]
[--no-score] [--rescore]`.

Groups every inference on disk by config, computes + caches each one's
``score.json`` (``--no-score`` skips computing; ``--rescore`` forces it), and
prints a plain-text metric + cost summary. See ``evals.analysis.aggregate`` for
the underlying functions when you want to script richer reports.
"""
from evals.analysis.aggregate import main

if __name__ == "__main__":
    main()
