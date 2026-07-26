"""Post-compression recovery fine-tuning.

Thin orchestration over :class:`~scale_aware_compression.training.trainer.Trainer`: build the
loader, install the right callbacks, run, and report the exact optimisation budget consumed.

The budget accounting is the point. :func:`recovery_step_budget` is what the runner uses to
give the sequential and joint arms the same number of optimiser steps, so that a measured joint
gain reflects the compression method rather than a longer training run.

Status: placeholder for the run path; the budget helpers are implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import ExperimentConfig, RecoveryConfig
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

    from scale_aware_compression.compression.masks import MaskSet
    from scale_aware_compression.training.callbacks import TrainerCallback
    from scale_aware_compression.training.trainer import TrainingResult

LOGGER = get_logger(__name__)


class RecoveryError(RuntimeError):
    """Raised when recovery cannot run as configured."""


@dataclass(frozen=True, slots=True)
class RecoveryBudget:
    """The optimisation budget one arm is allowed.

    Both arms are constructed from the same budget so their costs are directly comparable.
    """

    optimiser_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    sequence_length: int

    @property
    def effective_batch_size(self) -> int:
        """Sequences per optimiser step."""
        return self.batch_size * self.gradient_accumulation_steps

    @property
    def total_tokens(self) -> int:
        """Training tokens the budget corresponds to."""
        return self.optimiser_steps * self.effective_batch_size * self.sequence_length

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "optimiser_steps": self.optimiser_steps,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "sequence_length": self.sequence_length,
            "total_tokens": self.total_tokens,
        }


def recovery_step_budget(
    config: RecoveryConfig,
    *,
    steps_per_epoch: int,
    sequence_length: int,
) -> RecoveryBudget:
    """Resolve the configured recovery budget into a concrete step count.

    Args:
        config: Recovery section of the compression config.
        steps_per_epoch: Optimiser steps in one pass over the training loader; must be
            positive.
        sequence_length: Tokens per sequence.

    Returns:
        The resolved budget. ``config.max_steps`` takes precedence over ``config.epochs``,
        because matching budgets across arms is easier in steps than in epochs.

    Raises:
        RecoveryError: If ``steps_per_epoch`` is not positive, or the config yields no steps.
    """
    if steps_per_epoch <= 0:
        raise RecoveryError(f"steps_per_epoch must be > 0, got {steps_per_epoch}")

    steps = config.max_steps if config.max_steps is not None else config.epochs * steps_per_epoch
    if steps <= 0:
        raise RecoveryError(
            "Recovery budget resolved to 0 steps. Set compression.recovery.max_steps or "
            "epochs, or disable recovery with compression.recovery.enabled=false."
        )
    return RecoveryBudget(
        optimiser_steps=steps,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        sequence_length=sequence_length,
    )


def matched_budget(budget: RecoveryBudget, other: RecoveryBudget) -> bool:
    """Whether two arms were given the same optimisation budget.

    Args:
        budget: One arm's budget.
        other: The other arm's budget.

    Returns:
        ``True`` when the optimiser steps and token counts both match. A ``False`` here means
        any joint gain computed from the two runs is confounded with training cost.
    """
    return (
        budget.optimiser_steps == other.optimiser_steps
        and budget.total_tokens == other.total_tokens
    )


def build_recovery_callbacks(
    config: ExperimentConfig,
    mask_set: MaskSet | None = None,
) -> list[TrainerCallback]:
    """Assemble the callbacks for a recovery run.

    Args:
        config: The full experiment config.
        mask_set: Masks to maintain during training, when the arm prunes.

    Returns:
        The callbacks to pass to the trainer.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(recovery): always include LoggingCallback(config.compression.recovery
    # .log_every_steps). Add GradualPruningCallback when mask_set is not None and the schedule
    # is gradual. Add FakeQuantisationCallback only for the joint arm -- adding it to the
    # sequential arm would turn it into a third method rather than the baseline.
    raise NotImplementedError(
        "build_recovery_callbacks is not implemented yet; see the TODO in training/recovery.py"
    )


def run_recovery(
    model: nn.Module,
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    mask_set: MaskSet | None = None,
    budget: RecoveryBudget | None = None,
) -> TrainingResult:
    """Fine-tune a compressed model to recover quality. May run on GPU.

    Args:
        model: The compressed model.
        config: The full experiment config.
        tokenizer: Tokeniser for the training loader.
        mask_set: Masks to maintain, when the arm prunes.
        budget: Explicit budget, used by the runner to match the two arms. When ``None`` the
            budget is derived from ``config.compression.recovery``.

    Returns:
        The training result, including the exact optimiser-step count consumed.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(recovery): build the training loader from config.data, resolve the budget via
    # recovery_step_budget() unless one was passed, construct the Trainer with
    # build_recovery_callbacks(), and run. Log the resolved budget at INFO before starting:
    # a budget mismatch discovered after a multi-hour run is expensive.
    raise NotImplementedError(
        "run_recovery is not implemented yet; see the TODO in training/recovery.py"
    )
