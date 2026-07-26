"""Minimal training loop used by recovery and joint fine-tuning.

A hand-written loop rather than ``transformers.Trainer``, for one reason: both the sequential
and joint arms need to run callbacks at exact optimiser-step boundaries (rebuild masks, toggle
fake quantisation) and to report an exact optimiser-step count. Matching that count across the
two arms is what keeps a measured joint gain from being an extra-training artefact, so the step
accounting has to be ours.

May run on GPU. Nothing here is a deployment measurement.

Status: placeholder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import Device, ExperimentConfig
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from torch.utils.data import DataLoader

    from scale_aware_compression.training.callbacks import TrainerCallback

LOGGER = get_logger(__name__)


class TrainingError(RuntimeError):
    """Raised when training cannot start or fails irrecoverably."""


@dataclass(slots=True)
class TrainingState:
    """Mutable state passed to callbacks at each step boundary."""

    step: int = 0
    epoch: int = 0
    total_steps: int = 0
    tokens_processed: int = 0
    last_loss: float = float("nan")
    learning_rate: float = 0.0
    should_stop: bool = False

    @property
    def progress(self) -> float:
        """Fraction of the planned schedule completed, in ``[0, 1]``."""
        return self.step / self.total_steps if self.total_steps > 0 else 0.0


@dataclass(slots=True)
class TrainingResult:
    """What a training run consumed and achieved."""

    optimiser_steps: int
    tokens_processed: int
    final_loss: float
    duration_seconds: float
    device: str
    loss_history: list[float] = field(default_factory=list)
    early_stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "optimiser_steps": self.optimiser_steps,
            "tokens_processed": self.tokens_processed,
            "final_loss": self.final_loss,
            "duration_seconds": self.duration_seconds,
            "device": self.device,
            "early_stopped": self.early_stopped,
            "loss_history": self.loss_history,
        }


@dataclass(slots=True)
class Trainer:
    """Runs a causal-language-modelling fine-tune with step-accurate callbacks.

    Attributes:
        config: The full experiment config; the trainer reads the runtime and data sections
            as well as the compression one.
        callbacks: Callbacks invoked at step and epoch boundaries.
        device: Device to train on. May be CUDA.
    """

    config: ExperimentConfig
    callbacks: Sequence[TrainerCallback] = field(default_factory=tuple)
    device: Device = Device.AUTO

    def build_optimiser(self, model: nn.Module, learning_rate: float, weight_decay: float) -> Any:
        """Construct the optimiser.

        Args:
            model: The model to optimise.
            learning_rate: Peak learning rate.
            weight_decay: Decoupled weight decay.

        Returns:
            The optimiser.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(training): AdamW over parameters with requires_grad, excluding biases and
        # LayerNorm weights from weight decay. Note for the pruning arms: weight decay applied
        # to masked positions still moves them, so masks must be re-applied after every step
        # regardless of the decay setting.
        raise NotImplementedError(
            "Trainer.build_optimiser is not implemented yet; see the TODO in training/trainer.py"
        )

    def build_scheduler(self, optimiser: Any, total_steps: int, warmup_ratio: float) -> Any:
        """Construct the learning-rate schedule.

        Args:
            optimiser: The optimiser to schedule.
            total_steps: Total optimiser steps.
            warmup_ratio: Fraction of steps spent warming up.

        Returns:
            The scheduler.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(training): linear warmup then cosine decay. Both arms must use the identical
        # schedule shape, or the joint arm's advantage could come from a better LR curve.
        raise NotImplementedError(
            "Trainer.build_scheduler is not implemented yet; see the TODO in training/trainer.py"
        )

    def train(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        *,
        max_steps: int | None = None,
        epochs: int = 1,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.0,
        warmup_ratio: float = 0.03,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ) -> TrainingResult:
        """Run the training loop.

        Args:
            model: The model to fine-tune.
            dataloader: Training batches.
            max_steps: Hard step cap; overrides ``epochs`` when set.
            epochs: Passes over the loader when ``max_steps`` is ``None``.
            learning_rate: Peak learning rate.
            weight_decay: Decoupled weight decay.
            warmup_ratio: Fraction of steps spent warming up.
            gradient_accumulation_steps: Micro-batches per optimiser step.
            max_grad_norm: Gradient-norm clip threshold.

        Returns:
            The training result, including the exact optimiser-step count.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(training): standard accumulate / clip / step / schedule loop with
        # on_step_end callbacks fired *after* the optimiser step and *before* the next
        # forward pass -- that is the only point at which a mask update is consistent.
        # Increment TrainingState.step once per optimiser step, not once per micro-batch: the
        # cross-arm budget comparison depends on this being unambiguous.
        raise NotImplementedError(
            "Trainer.train is not implemented yet; see the TODO in training/trainer.py"
        )

    def evaluate(self, model: nn.Module, dataloader: DataLoader) -> dict[str, float]:
        """Compute validation loss during training.

        Exploratory only: final reported quality numbers come from
        :mod:`scale_aware_compression.evaluation` on CPU.

        Args:
            model: The model to evaluate.
            dataloader: Validation batches.

        Returns:
            Mapping with at least ``loss`` and ``perplexity``.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(training): mean token-weighted loss under torch.inference_mode(), restoring the
        # model's previous train/eval mode afterwards.
        raise NotImplementedError(
            "Trainer.evaluate is not implemented yet; see the TODO in training/trainer.py"
        )
