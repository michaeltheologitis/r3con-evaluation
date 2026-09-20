"""Cheap per-iteration analysis surface over the inference manifests.

Globs `logs/{benchmark}/{baseline}/inferences/*/manifest.json`, groups them by
`config`, reads each inference's cached `score.json` for the metric, and reports
headline accuracy + a per-metadata-axis breakdown + cost (inference vs
construction, the latter deduped across shared indices). The expensive deep-dive
(read each failure, attribute it, write a narrative — see CLAUDE.md's "After every
meaningful measurement") happens on top of this output.
"""
from .aggregate import (
    aggregate_group,
    attach_scores,
    construction_usage,
    cost_per_model_usd,
    find_groups,
    sum_usage,
)
from .score import ensure_scores, read_score

__all__ = [
    "find_groups",
    "attach_scores",
    "aggregate_group",
    "construction_usage",
    "sum_usage",
    "cost_per_model_usd",
    "ensure_scores",
    "read_score",
]
