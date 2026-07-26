"""Compression arms: the interface, the four methods, and their shared machinery.

The dense baseline has no compressor: it is the unmodified model, so there is nothing to apply.
:func:`get_compressor` returns ``None`` for it, and callers evaluate the model directly.
"""

from __future__ import annotations

from scale_aware_compression.compression.base import (
    CompressionError,
    CompressionResult,
    Compressor,
    StageRecord,
)
from scale_aware_compression.compression.joint import JointCompressor
from scale_aware_compression.compression.masks import MaskSet, MaskStatistics
from scale_aware_compression.compression.pruning import Pruner
from scale_aware_compression.compression.quantisation import Quantiser
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
    "CompressionError",
    "CompressionResult",
    "Compressor",
    "JointCompressor",
    "MaskSet",
    "MaskStatistics",
    "Pruner",
    "Quantiser",
    "ScheduleError",
    "SequentialCompressor",
    "StageRecord",
    "get_compressor",
    "is_mask_update_step",
    "mask_freeze_step",
    "schedule_values",
    "sparsity_at_step",
]

COMPRESSOR_REGISTRY: dict[CompressionMethod, type[Compressor]] = {
    CompressionMethod.PRUNING: Pruner,
    CompressionMethod.QUANTISATION: Quantiser,
    CompressionMethod.SEQUENTIAL: SequentialCompressor,
    CompressionMethod.JOINT: JointCompressor,
}
"""Maps each compression method to its implementation. ``DENSE`` is absent by design."""


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
