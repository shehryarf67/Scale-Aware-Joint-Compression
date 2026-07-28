"""Compression arms: the interface, the four methods, and their shared machinery.

The dense baseline has no compressor: it is the unmodified model, so there is nothing to apply.
:func:`get_compressor` returns ``None`` for it, and callers evaluate the model directly.
"""

from __future__ import annotations

from scale_aware_compression.compression.activations import (
    ActivationCaptureError,
    ActivationStatistics,
    LinearActivationCapture,
)
from scale_aware_compression.compression.arms import (
    JointArm,
    LayerwiseArm,
    PruningArm,
    QuantisationArm,
    SequentialArm,
    SequentialQPArm,
    plan_from_config,
)
from scale_aware_compression.compression.base import (
    CompressionError,
    CompressionResult,
    Compressor,
    StageRecord,
)
from scale_aware_compression.compression.joint import JointCompressor
from scale_aware_compression.compression.layerwise import (
    LayerPlan,
    LayerResult,
    LayerwiseError,
    LayerwiseReport,
    assert_matched_plans,
    compress_layer,
    compress_model_layerwise,
)
from scale_aware_compression.compression.masks import (
    MaskError,
    MaskSet,
    MaskStatistics,
    build_mask_from_scores,
    realised_sparsity,
)
from scale_aware_compression.compression.packed import (
    convert_model_to_packed,
    pack_linear,
    packed_linear_class,
    verify_packing,
)
from scale_aware_compression.compression.pruning import Pruner, activation_weighted_saliency
from scale_aware_compression.compression.quantisation import (
    QuantisationError,
    QuantisedWeight,
    Quantiser,
    compute_symmetric_scales,
    effective_bits_per_weight,
    fake_quantise,
    pack_low_bit,
    quantise_weight,
    unpack_low_bit,
)
from scale_aware_compression.compression.reconstruct import (
    ReconstructionError,
    ReconstructionResult,
    reconstruct,
    reconstruction_loss,
    solve_masked_rows,
    sweep_reconstruct,
)
from scale_aware_compression.compression.schedules import (
    ScheduleError,
    is_mask_update_step,
    mask_freeze_step,
    schedule_values,
    sparsity_at_step,
)
from scale_aware_compression.compression.sequential import SequentialCompressor
from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import CompressionMethod

__all__ = [
    "COMPRESSOR_REGISTRY",
    "ActivationCaptureError",
    "ActivationStatistics",
    "CompressionError",
    "CompressionResult",
    "Compressor",
    "JointArm",
    "JointCompressor",
    "LayerPlan",
    "LayerResult",
    "LayerwiseArm",
    "LayerwiseError",
    "LayerwiseReport",
    "LinearActivationCapture",
    "MaskError",
    "MaskSet",
    "MaskStatistics",
    "Pruner",
    "PruningArm",
    "QuantisationArm",
    "QuantisationError",
    "QuantisedWeight",
    "Quantiser",
    "ReconstructionError",
    "ReconstructionResult",
    "ScheduleError",
    "SequentialArm",
    "SequentialCompressor",
    "SequentialQPArm",
    "StageRecord",
    "activation_weighted_saliency",
    "assert_matched_plans",
    "build_mask_from_scores",
    "compress_layer",
    "compress_model_layerwise",
    "compute_symmetric_scales",
    "convert_model_to_packed",
    "effective_bits_per_weight",
    "fake_quantise",
    "get_compressor",
    "is_mask_update_step",
    "mask_freeze_step",
    "pack_linear",
    "pack_low_bit",
    "packed_linear_class",
    "plan_from_config",
    "quantise_weight",
    "realised_sparsity",
    "reconstruct",
    "reconstruction_loss",
    "schedule_values",
    "solve_masked_rows",
    "sparsity_at_step",
    "sweep_reconstruct",
    "unpack_low_bit",
    "verify_packing",
]

COMPRESSOR_REGISTRY: dict[CompressionMethod, type[Compressor]] = {
    CompressionMethod.PRUNING: PruningArm,
    CompressionMethod.QUANTISATION: QuantisationArm,
    CompressionMethod.SEQUENTIAL: SequentialArm,
    CompressionMethod.SEQUENTIAL_QP: SequentialQPArm,
    CompressionMethod.JOINT: JointArm,
}
"""Maps each compression method to its implementation. ``DENSE`` is absent by design.

These are the layerwise arms from :mod:`~scale_aware_compression.compression.arms`. The older
``Pruner`` / ``Quantiser`` / ``SequentialCompressor`` / ``JointCompressor`` classes remain importable
but are **not** registered: they were built for the superseded full-model fine-tuning design, and
their stage methods still raise. Keeping them exported avoids breaking imports while making it
impossible to run one by accident.
"""


def get_compressor(config: ExperimentConfig) -> Compressor | None:
    """Instantiate the compressor for a configuration's method.

    Args:
        config: A validated experiment config.

    Returns:
        The compressor for ``config.compression.method``, or ``None`` for the dense baseline.

    Raises:
        CompressionError: If the method has no registered implementation.
    """
    method = config.compression.method
    if method is CompressionMethod.DENSE:
        return None
    compressor_class = COMPRESSOR_REGISTRY.get(method)
    if compressor_class is None:
        raise CompressionError(
            f"No compressor registered for method {method.value!r}. Registered: "
            f"{sorted(item.value for item in COMPRESSOR_REGISTRY)}"
        )
    return compressor_class(config)
