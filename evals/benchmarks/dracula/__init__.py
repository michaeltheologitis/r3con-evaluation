"""Dracula showcase mini-benchmark — the standard benchmark surface.

Registered in ``evals.baselines._common.BENCHMARK_IMPORT_MAP`` and supported by
every baseline here except ``claude_code``. The corpus + questions are vendored
in-package (shipped via package-data).

**One deviation from the shared surface:** ``score`` / ``score_batch`` return the
PAIR ``(strict, lenient)`` rather than a single metric — dracula grades every
answer twice (Mr. Swales required vs optional; see ``judge.py``). ``score_details``
keeps the standard ScoreResult shape, so a consumer that knows nothing about this
benchmark still reads one metric: its ``score`` is the strict one, and the pair
rides in ``parsed`` (decode with ``read_verdicts``).
"""
from evals.benchmarks.dracula.judge import (  # noqa: F401
    GRADER_MODEL,
    SCORER,
    encode_verdicts,
    read_verdicts,
    score,
    score_batch,
    score_details,
)
from evals.benchmarks.dracula.loader import (  # noqa: F401
    STARTER_FILTER,
    get_documents,
    get_task,
    get_task_answer,
    get_task_ids,
    get_task_metadata,
)
