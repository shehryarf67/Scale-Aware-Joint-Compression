"""Trainer callbacks.

The pruning schedule and the fake-quantisation toggle are both implemented as callbacks, so
that the sequential and joint arms differ only in *which* callbacks they install and not in
their training loops. That is what makes the two arms comparable at the code level, not just
in intent.

Status: the base class and logging callback are implemented; the compression callbacks are
placeholders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn

    from scale_aware_compression.compression.masks import MaskSet
    from scale_aware_compression.training.trainer import TrainingState

LOGGER = get_logger(__name__)


class TrainerCallback:
    """No-op callback base class. Override only the hooks you need."""

    def on_train_begin(self, model: nn.Module, state: TrainingState) -> None:
        """Called once before the first step."""

    def on_step_end(self, model: nn.Module, state: TrainingState) -> None:
        """Called after every optimiser step, before the next forward pass.

        This is the only point at which the weights are consistent with the optimiser state,
        so mask updates and mask re-application both belong here.
        """

    def on_epoch_end(self, model: nn.Module, state: TrainingState) -> None:
        """Called at the end of each pass over the loader."""

    def on_train_end(self, model: nn.Module, state: TrainingState) -> None:
        """Called once after the last step."""

    def report(self) -> dict[str, Any]:
        """Return anything this callback wants recorded in the run record."""
        return {}


class LoggingCallback(TrainerCallback):
    """Logs loss and learning rate every ``log_every_steps`` steps."""

    def __init__(self, log_every_steps: int = 20) -> None:
        """Initialise the callback.

        Args:
            log_every_steps: Steps between log lines; must be positive.

        Raises:
            ValueError: If ``log_every_steps`` is not positive.
        """
        if log_every_steps <= 0:
            raise ValueError(f"log_every_steps must be > 0, got {log_every_steps}")
        self.log_every_steps = log_every_steps
        self._logged_steps = 0

    def on_step_end(self, model: nn.Module, state: TrainingState) -> None:
        """Log progress at the configured interval."""
        if state.step % self.log_every_steps != 0:
            return
        self._logged_steps += 1
        LOGGER.info(
            "step %d/%d (%.0f%%) loss=%.4f lr=%.2e tokens=%d",
            state.step,
            state.total_steps,
            100 * state.progress,
            state.last_loss,
            state.learning_rate,
            state.tokens_processed,
        )

    def report(self) -> dict[str, Any]:
        """Return how many log lines were emitted."""
        return {"logged_steps": self._logged_steps}


class GradualPruningCallback(TrainerCallback):
    """Ramps sparsity during training and keeps pruned positions at zero.

    Used by both the sequential arm (during recovery, when the schedule is gradual) and the
    joint arm (during joint fine-tuning). Identical schedule logic in both, from
    :mod:`scale_aware_compression.compression.schedules`.
    """

    def __init__(self, mask_set: MaskSet, config: Any) -> None:
        """Initialise the callback.

        Args:
            mask_set: Masks to maintain.
            config: The compression config section holding the schedule.
        """
        self.mask_set = mask_set
        self.config = config
        self.mask_updates = 0
        self.current_sparsity = 0.0

    def on_step_end(self, model: nn.Module, state: TrainingState) -> None:
        """Update masks on schedule and re-apply them every step.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(callbacks): on each step,
        #   1. target = sparsity_at_step(state.step, ...)
        #   2. if is_mask_update_step(...) and state.step < mask_freeze_step(...):
        #          rebuild masks at `target` and increment self.mask_updates
        #   3. apply_masks(model, self.mask_set) unconditionally -- the optimiser will have
        #      moved masked weights off zero even when the mask itself did not change.
        raise NotImplementedError(
            "GradualPruningCallback.on_step_end is not implemented yet; see the TODO in "
            "training/callbacks.py"
        )

    def report(self) -> dict[str, Any]:
        """Return the realised schedule for the run record."""
        return {
            "mask_updates": self.mask_updates,
            "final_sparsity": self.current_sparsity,
        }


class FakeQuantisationCallback(TrainerCallback):
    """Toggles fake quantisation on after a warmup, for the joint arm.

    A short dense warmup before quantisation is switched on stabilises the early steps, where
    the combined pruning and rounding perturbation is largest.
    """

    def __init__(self, warmup_steps: int = 0) -> None:
        """Initialise the callback.

        Args:
            warmup_steps: Dense steps before fake quantisation is enabled.

        Raises:
            ValueError: If ``warmup_steps`` is negative.
        """
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        self.warmup_steps = warmup_steps
        self.enabled_at_step: int | None = None

    def on_step_end(self, model: nn.Module, state: TrainingState) -> None:
        """Enable fake quantisation once the warmup has elapsed.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(callbacks): when state.step >= self.warmup_steps and self.enabled_at_step is
        # None, enable every fake-quantisation node in the model, record the step, and log it
        # at INFO. Recording the step matters: a run where quantisation never switched on is
        # otherwise indistinguishable from a successful joint run in the results table.
        raise NotImplementedError(
            "FakeQuantisationCallback.on_step_end is not implemented yet; see the TODO in "
            "training/callbacks.py"
        )

    def report(self) -> dict[str, Any]:
        """Return when fake quantisation was enabled."""
        return {
            "fake_quantisation_warmup_steps": self.warmup_steps,
            "fake_quantisation_enabled_at_step": self.enabled_at_step,
        }
