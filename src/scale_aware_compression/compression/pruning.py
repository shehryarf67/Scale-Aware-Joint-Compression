"""Pruning-only arm.

Pipeline::

    dense model -> prune -> recovery

Status: placeholder. Stage methods raise :class:`NotImplementedError` with a pointer to what
needs writing; :meth:`Pruner.report_statistics` is implemented, since it is what tells you
whether the eventual implementation actually did anything.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import Compressor
from scale_aware_compression.compression.schedules import sparsity_at_step
from scale_aware_compression.constants import CompressionMethod, CompressionStage
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import measure_sparsity

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn
    from transformers import PreTrainedTokenizerBase

    from scale_aware_compression.compression.masks import MaskSet

LOGGER = get_logger(__name__)


class SaliencyError(ValueError):
    """Raised when saliency inputs are shaped inconsistently."""


def activation_weighted_saliency(
    weight: torch.Tensor,
    column_norms: torch.Tensor,
) -> torch.Tensor:
    """Score weights by ``S_ij = |W_ij| * ||X_j||_2``, the plan's §3.3 criterion.

    A weight is unimportant only if it is *both* small and multiplies a low-energy input. Plain
    magnitude ignores the second factor and will happily keep a large weight sitting on an input
    channel that the calibration data never excites, while pruning a smaller weight that carries
    real signal.

    Which weights are passed in is what separates the arms, and it is the whole of decision
    **D3**: the joint arm passes fake-quantised weights so the mask is chosen against the grid it
    will actually live on (§3.7, §3.8), while sequential P->Q passes dense weights because at that
    point no quantiser exists yet. The function itself is identical for both -- the arms differ in
    the pipeline, not the criterion.

    Args:
        weight: Weight of shape ``(out_features, in_features)``.
        column_norms: Per-input-column norms of shape ``(in_features,)``, from
            :meth:`~scale_aware_compression.compression.activations.ActivationStatistics.column_norms`.

    Returns:
        Non-negative scores with the same shape as ``weight``.

    Raises:
        SaliencyError: If the weight is not 2-D or the norms do not match ``in_features``.
    """
    if weight.ndim != 2:
        raise SaliencyError(
            f"expected a 2-D (out_features, in_features) weight, got shape {tuple(weight.shape)}"
        )
    if column_norms.ndim != 1 or column_norms.shape[0] != weight.shape[1]:
        raise SaliencyError(
            f"column_norms must have shape ({weight.shape[1]},) to match in_features, got "
            f"{tuple(column_norms.shape)}"
        )
    return weight.abs() * column_norms.to(weight.dtype).unsqueeze(0)


def keep_benefit_saliency(
    dense_weight: torch.Tensor,
    quantised_weight: torch.Tensor,
    column_norms: torch.Tensor,
) -> torch.Tensor:
    """Score weights by how much is gained by keeping them *in quantised form* rather than pruning.

    .. code-block:: text

        B_ij = ||X_j||^2 * [  W_ij^2  -  (W_ij - Q(W_ij))^2  ]
                              ^^^^^^^     ^^^^^^^^^^^^^^^^^^
                              error if    error if kept but
                              pruned      quantised

    Under a diagonal approximation to the layer Hessian, the first term is the output error caused by
    zeroing the weight and the second is the error caused by keeping it on the grid. Their difference
    is what keeping it actually buys.

    It is **not** a tunable blend -- there is no free coefficient to fit -- so it does not fall under
    the standing prohibition on ``alpha*|w| + beta*|w - Q(w)|`` scores in
    ``method_definition.md#mask-scoring``. Both terms are error estimates in the same units.

    **Measured worse than plain activation-weighted magnitude, and off by default.** Layer-objective
    joint gain on six real Pythia-160M layers falls from +1.12% to -16.15% at W4. Retained as a
    declared ablation, because the reason it fails is informative:

    For round-to-nearest symmetric quantisation the score is bounded below by zero. If
    ``|W| < s/2`` then ``Q(W) = 0``, both error terms equal ``W^2``, and ``B = 0`` exactly;
    otherwise ``|W - Q(W)| <= s/2 <= |W|`` so ``B >= 0``. And for any weight comfortably above the
    step size, ``(W - Q(W))^2`` is roughly uniform in ``[0, (s/2)^2]`` and nearly independent of
    ``W``, which leaves

    .. code-block:: text

        B_ij ~= ||X_j||^2 * W_ij^2 - (a near-constant)

    a monotone transform of activation-weighted magnitude. So the criterion mostly *reproduces* the
    magnitude ranking, and where it deviates it prefers weights that happen to sit near a grid
    point -- a property of the current scale rather than a statement about importance. That is why
    it costs quality instead of buying it.

    The sequential arm cannot use this at all: it has no quantiser at mask time. That asymmetry *is*
    the pipeline difference under study, not an unfair advantage.

    Args:
        dense_weight: The original weight, shape ``(out_features, in_features)``.
        quantised_weight: The same weight after fake quantisation, same shape.
        column_norms: Per-input-column activation norms, shape ``(in_features,)``.

    Returns:
        Non-negative scores of the same shape, zero wherever the weight quantises to zero.

    Raises:
        SaliencyError: If the shapes are inconsistent.
    """
    if dense_weight.shape != quantised_weight.shape:
        raise SaliencyError(
            f"dense and quantised weights must match: {tuple(dense_weight.shape)} vs "
            f"{tuple(quantised_weight.shape)}"
        )
    if dense_weight.ndim != 2:
        raise SaliencyError(
            f"expected a 2-D (out_features, in_features) weight, got {tuple(dense_weight.shape)}"
        )
    if column_norms.ndim != 1 or column_norms.shape[0] != dense_weight.shape[1]:
        raise SaliencyError(
            f"column_norms must have shape ({dense_weight.shape[1]},) to match in_features, got "
            f"{tuple(column_norms.shape)}"
        )

    energy = (column_norms.to(dense_weight.dtype) ** 2).unsqueeze(0)
    pruned_error = dense_weight**2
    kept_error = (dense_weight - quantised_weight) ** 2
    return energy * (pruned_error - kept_error)


class Pruner(Compressor):
    """Magnitude-based pruning followed by recovery fine-tuning.

    The pruning criterion, granularity, and schedule all come from
    ``config.compression.pruning``. Embeddings and the output head are excluded by default;
    see :func:`~scale_aware_compression.models.adapters.select_compressible_modules` for why.
    """

    method = CompressionMethod.PRUNING
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.PRUNED,
        CompressionStage.RECOVERED,
    )
    apply_stage = CompressionStage.PRUNED
    recover_stage = CompressionStage.RECOVERED

    def __init__(self, config: Any) -> None:
        """Initialise the arm with an empty mask set."""
        super().__init__(config)
        self.mask_set: MaskSet | None = None
        self.module_names: list[str] = []

    def prepare(self, model: nn.Module) -> nn.Module:
        """Select compressible modules and record the pre-pruning parameter count.

        Args:
            model: The dense model.

        Returns:
            The model, unchanged.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(pruning): call select_compressible_modules() with
        # config.compression.pruning.target_modules / exclude_patterns, store the result on
        # self.module_names, and log how many modules and parameters are in scope. An empty
        # selection must raise: it would otherwise produce a "pruned" model identical to
        # dense, which looks like an excellent compression result.
        raise NotImplementedError(
            "Pruner.prepare is not implemented yet; see the TODO in compression/pruning.py"
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """Build masks at the target sparsity and zero the masked weights.

        Args:
            model: The prepared model.

        Returns:
            The pruned model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(pruning): for a one-shot schedule, call build_masks() at the full target and
        # apply_masks() once. For a gradual schedule this stage only installs the masks at
        # initial_sparsity; the ramp itself belongs to recover(), which owns the optimiser
        # loop. Store the mask set on self.mask_set so report_statistics can describe it.
        raise NotImplementedError(
            "Pruner.apply is not implemented yet; see the TODO in compression/pruning.py"
        )

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Fine-tune the pruned model, ramping sparsity if the schedule is gradual.

        May run on GPU.

        Args:
            model: The pruned model.
            tokenizer: Tokeniser for the recovery data loader.

        Returns:
            The recovered model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(pruning): drive training.recovery.run_recovery() with a callback that, on each
        # mask-update step, calls sparsity_at_step() for the current target, rebuilds masks,
        # and re-applies them. Record optimiser_steps via record_stage: it is the
        # training-cost figure the joint arm gets compared against, so it must be exact.
        raise NotImplementedError(
            "Pruner.recover is not implemented yet; see the TODO in compression/pruning.py"
        )

    def convert(self, model: nn.Module) -> nn.Module:
        """Fold masks into the weights to produce the deployable artefact.

        Args:
            model: The recovered model.

        Returns:
            The converted model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(pruning): call fold_masks_into_weights(), remove hooks, and assert the measured
        # sparsity is within a small tolerance of the target before returning.
        raise NotImplementedError(
            "Pruner.convert is not implemented yet; see the TODO in compression/pruning.py"
        )

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Report the target sparsity, the schedule, and the sparsity actually measured.

        Args:
            model: The model to measure. When ``None``, only configuration-derived fields are
                returned.

        Returns:
            A serialisable mapping. ``measured_sparsity_percentage`` next to
            ``target_sparsity`` is the check that pruning was really applied rather than only
            configured.
        """
        pruning = self.compression_config.pruning
        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "pruning_method": pruning.method,
            "granularity": pruning.granularity.value,
            "schedule": pruning.schedule.value,
            "schedule_start_step": pruning.schedule_start_step,
            "schedule_end_step": pruning.schedule_end_step,
            "global_ranking": pruning.global_ranking,
            "num_compressible_modules": len(self.module_names),
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
