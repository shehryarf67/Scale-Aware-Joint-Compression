"""Joint arm: pruning-aware quantisation.

Stages, explicitly::

    dense model
        -> fake quantisation preparation
        -> gradual pruning during optimisation
        -> recovery / joint fine-tuning
        -> final conversion

The difference from :mod:`sequential` is *when* the model learns about quantisation. Here fake
quantisation is inserted before any weight is pruned, so:

* the pruning criterion ranks weights as the quantisation grid will actually represent them, and
  a weight whose quantised value rounds to zero is no longer worth keeping;
* a single optimisation run compensates for both perturbations together, instead of recovering
  from pruning and then absorbing an unrecovered quantisation error.

The cost is a longer, more fragile training run. Whether that cost pays off — and whether it
pays off *more at larger scale* — is the research question.

Fairness requirement: ``config.compression.joint.match_sequential_budget`` exists because the
joint arm gets one training run covering both perturbations while the sequential arm gets one
covering only pruning. If the joint run is simply longer, any measured joint gain is confounded
with extra training. The runner must equalise optimiser steps, and
:meth:`JointCompressor.report_statistics` reports the step count so a mismatch is visible.

Status: placeholder. Stage methods raise :class:`NotImplementedError`;
:meth:`JointCompressor.report_statistics` is implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import Compressor
from scale_aware_compression.compression.schedules import mask_freeze_step, sparsity_at_step
from scale_aware_compression.constants import JOINT_STAGES, CompressionMethod, CompressionStage
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import measure_sparsity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

    from scale_aware_compression.compression.masks import MaskSet

LOGGER = get_logger(__name__)


class JointCompressor(Compressor):
    """Gradual pruning and fake quantisation optimised together in a single run."""

    method = CompressionMethod.JOINT
    pipeline_stages = JOINT_STAGES
    apply_stage = CompressionStage.GRADUAL_PRUNING
    recover_stage = CompressionStage.JOINTLY_FINE_TUNED

    def __init__(self, config: Any) -> None:
        """Initialise the arm with empty mask and observer state."""
        super().__init__(config)
        self.mask_set: MaskSet | None = None
        self.module_names: list[str] = []
        self.optimiser_steps: int = 0
        self.mask_updates: int = 0
        self.is_converted: bool = False

    def prepare(self, model: nn.Module) -> nn.Module:
        """Insert fake quantisation before any pruning happens.

        This is the stage that makes the arm "pruning-aware quantisation" rather than
        "quantisation after pruning".

        Args:
            model: The dense model.

        Returns:
            The model with fake-quantisation nodes installed.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(joint): select compressible modules once and share the selection with both the
        # pruning and quantisation paths -- they must operate on the same module set, or the
        # arms are not comparable. Then insert fake-quantisation nodes on the weights, with a
        # straight-through estimator when config.compression.joint.straight_through_estimator
        # is set, so gradients flow through the rounding operation.
        # Calibrate observers on the dense weights here, using the same calibration set as
        # every other arm.
        raise NotImplementedError(
            "JointCompressor.prepare is not implemented yet; see the TODO in compression/joint.py"
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """Install masks at the initial sparsity and arm the gradual schedule.

        Unlike the sequential arm, this stage does not reach the target sparsity. The ramp runs
        inside :meth:`recover`, interleaved with optimisation, which is what lets the two
        perturbations be traded off against each other.

        Args:
            model: The fake-quantisation-prepared model.

        Returns:
            The model with initial masks installed.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(joint): build masks at pruning.initial_sparsity from the *fake-quantised*
        # weights -- ranking the FP32 shadow weights instead would discard the whole point of
        # this arm -- then register mask hooks so the optimiser cannot refill pruned positions.
        raise NotImplementedError(
            "JointCompressor.apply is not implemented yet; see the TODO in compression/joint.py"
        )

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Joint fine-tuning: ramp sparsity while training through fake quantisation.

        May run on GPU. This is the expensive stage, and the one whose cost has to be reported
        next to the quality benefit.

        Args:
            model: The masked, fake-quantised model.
            tokenizer: Tokeniser for the training data loader.

        Returns:
            The jointly fine-tuned model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(joint): run one optimisation loop that, per step:
        #   1. computes the current target via sparsity_at_step()
        #   2. on mask-update steps before mask_freeze_step(), rebuilds masks from the
        #      fake-quantised weights and re-applies them
        #   3. takes an optimiser step with fake quantisation active (after
        #      quantisation_warmup_steps dense steps, if configured)
        #   4. re-applies masks so momentum cannot refill pruned positions
        # Increment self.optimiser_steps and self.mask_updates, and record them via
        # record_stage: total_optimiser_steps is compared against the sequential arm, and a
        # mismatch invalidates the joint gain.
        raise NotImplementedError(
            "JointCompressor.recover is not implemented yet; see the TODO in compression/joint.py"
        )

    def convert(self, model: nn.Module) -> nn.Module:
        """Final conversion: fold masks in and make the quantisation real.

        Args:
            model: The jointly fine-tuned model.

        Returns:
            The CPU-deployable artefact.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(joint): remove mask hooks, fold masks into the weights, then convert fake
        # quantisation to real low-precision storage using the same code path as
        # Quantiser.convert -- the two arms must produce byte-identical artefact *formats*, or
        # the size and latency comparison measures the serialisation rather than the method.
        # Set self.is_converted = True and verify both the realised sparsity and the on-disk
        # size against their targets.
        raise NotImplementedError(
            "JointCompressor.convert is not implemented yet; see the TODO in compression/joint.py"
        )

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Report the joint schedule, its training cost, and the realised compression.

        Args:
            model: The model to measure. When ``None``, only configuration-derived fields are
                returned.

        Returns:
            A serialisable mapping. ``optimiser_steps`` and ``match_sequential_budget`` are
            what make the fairness of a joint-gain comparison auditable.
        """
        compression = self.compression_config
        joint = compression.joint
        pruning = compression.pruning
        total_steps = joint.joint_max_steps or pruning.schedule_end_step

        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "pipeline": [stage.value for stage in JOINT_STAGES],
            "fake_quantisation": joint.fake_quantisation,
            "straight_through_estimator": joint.straight_through_estimator,
            "mask_update_interval": joint.mask_update_interval,
            "quantisation_warmup_steps": joint.quantisation_warmup_steps,
            "freeze_masks_after_ratio": joint.freeze_masks_after_ratio,
            "mask_freeze_step": mask_freeze_step(
                total_steps=max(total_steps, 1),
                freeze_after_ratio=joint.freeze_masks_after_ratio,
            ),
            "match_sequential_budget": joint.match_sequential_budget,
            "optimiser_steps": self.optimiser_steps,
            "mask_updates": self.mask_updates,
            "num_compressible_modules": len(self.module_names),
            "is_converted": self.is_converted,
            "final_scheduled_sparsity": sparsity_at_step(
                pruning.schedule_end_step,
                schedule=pruning.schedule,
                final_sparsity=pruning.sparsity,
                initial_sparsity=pruning.initial_sparsity,
                start_step=pruning.schedule_start_step,
                end_step=pruning.schedule_end_step,
            ),
        }
        if self.mask_set is not None:
            statistics["masks"] = self.mask_set.report()
        if model is not None:
            statistics.update(
                {f"measured_{key}": value for key, value in measure_sparsity(model).items()}
            )
        return statistics
