"""Experiment orchestration, result records, the scale sweep, and external validation."""

from __future__ import annotations

from scale_aware_compression.experiments.runner import (
    ExperimentError,
    ExperimentRecord,
    ExperimentRunner,
    ExperimentTracker,
    get_git_commit,
    make_experiment_id,
    utc_timestamp,
)
from scale_aware_compression.experiments.scale_sweep import (
    SweepCell,
    SweepPlan,
    build_sweep_plan,
    executable_cells,
    find_comparison_pairs,
    run_sweep,
    scale_trend,
)
from scale_aware_compression.experiments.validation import (
    ValidationOutcome,
    assess_transfer,
    interpolate_expected_gain,
    run_validation,
)

__all__ = [
    "ExperimentError",
    "ExperimentRecord",
    "ExperimentRunner",
    "ExperimentTracker",
    "SweepCell",
    "SweepPlan",
    "ValidationOutcome",
    "assess_transfer",
    "build_sweep_plan",
    "executable_cells",
    "find_comparison_pairs",
    "get_git_commit",
    "interpolate_expected_gain",
    "make_experiment_id",
    "run_sweep",
    "run_validation",
    "scale_trend",
    "utc_timestamp",
]
