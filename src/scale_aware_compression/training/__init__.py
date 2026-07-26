"""Training loop, recovery fine-tuning, and callbacks.

Everything here may run on GPU: training is not a deployment measurement. The CPU-only policy
applies to :mod:`scale_aware_compression.benchmarking` and to final quality evaluation.
"""

from __future__ import annotations

from scale_aware_compression.training.callbacks import (
    FakeQuantisationCallback,
    GradualPruningCallback,
    LoggingCallback,
    TrainerCallback,
)
from scale_aware_compression.training.recovery import (
    RecoveryBudget,
    RecoveryError,
    build_recovery_callbacks,
    matched_budget,
    recovery_step_budget,
    run_recovery,
)
from scale_aware_compression.training.trainer import (
    Trainer,
    TrainingError,
    TrainingResult,
    TrainingState,
)

__all__ = [
    "FakeQuantisationCallback",
    "GradualPruningCallback",
    "LoggingCallback",
    "RecoveryBudget",
    "RecoveryError",
    "Trainer",
    "TrainerCallback",
    "TrainingError",
    "TrainingResult",
    "TrainingState",
    "build_recovery_callbacks",
    "matched_budget",
    "recovery_step_budget",
    "run_recovery",
]
