"""Small, testable metric functions.

Split by what they measure: :mod:`compression` for size and sparsity, :mod:`efficiency` for
latency and memory, :mod:`joint_gain` for the joint-versus-sequential comparison that the
research question turns on. None of these modules import torch at module level.
"""

from __future__ import annotations

from scale_aware_compression.metrics.compression import (
    checkpoint_size_bytes,
    checkpoint_size_mib,
    compression_ratio,
    count_parameters,
    count_zero_parameters,
    count_zeros,
    effective_compression_ratio,
    measure_sparsity,
    size_reduction_percentage,
    sparsity_fraction,
    sparsity_percentage,
    theoretical_size_bytes,
)
from scale_aware_compression.metrics.efficiency import (
    latency_reduction_percentage,
    memory_reduction_percentage,
    sparsity_realisation,
    speedup,
    theoretical_speedup_from_sparsity,
    throughput_gain,
    training_cost_overhead,
)
from scale_aware_compression.metrics.joint_gain import (
    JointGainSummary,
    accuracy_retention,
    joint_gain,
    joint_gain_from_quality_loss,
    joint_gain_summary,
    perplexity_increase_percentage,
    perplexity_retention,
    relative_joint_gain,
)

__all__ = [
    "JointGainSummary",
    "accuracy_retention",
    "checkpoint_size_bytes",
    "checkpoint_size_mib",
    "compression_ratio",
    "count_parameters",
    "count_zero_parameters",
    "count_zeros",
    "effective_compression_ratio",
    "joint_gain",
    "joint_gain_from_quality_loss",
    "joint_gain_summary",
    "latency_reduction_percentage",
    "measure_sparsity",
    "memory_reduction_percentage",
    "perplexity_increase_percentage",
    "perplexity_retention",
    "relative_joint_gain",
    "size_reduction_percentage",
    "sparsity_fraction",
    "sparsity_percentage",
    "sparsity_realisation",
    "speedup",
    "theoretical_size_bytes",
    "theoretical_speedup_from_sparsity",
    "throughput_gain",
    "training_cost_overhead",
]
