"""Efficiency metrics: speedup, latency and memory reduction, sparsity realisation.

The last of these is the point of the CPU-only policy. A configuration can report 70%
sparsity and a 4x theoretical compression ratio while delivering no wall-clock improvement,
because unstructured zeros do not remove work from a dense CPU kernel.
:func:`sparsity_realisation` puts that gap in the results table instead of leaving it to a
footnote.
"""

from __future__ import annotations

from scale_aware_compression.constants import EPSILON
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


def speedup(baseline_latency: float, compressed_latency: float) -> float:
    """Wall-clock speedup of the compressed model over the dense baseline.

    Args:
        baseline_latency: Dense-baseline latency; must be positive. Units are arbitrary but
            must match ``compressed_latency``.
        compressed_latency: Compressed-model latency; must be positive.

    Returns:
        ``baseline / compressed``. Greater than 1 means faster than baseline.

    Raises:
        ValueError: If either latency is not positive.
    """
    if baseline_latency <= 0:
        raise ValueError(f"baseline_latency must be > 0, got {baseline_latency}")
    if compressed_latency <= 0:
        raise ValueError(f"compressed_latency must be > 0, got {compressed_latency}")
    return baseline_latency / compressed_latency


def latency_reduction_percentage(baseline_latency: float, compressed_latency: float) -> float:
    """Percentage of baseline latency removed.

    Args:
        baseline_latency: Dense-baseline latency; must be positive.
        compressed_latency: Compressed-model latency; must be positive.

    Returns:
        ``100 * (1 - compressed / baseline)``. Negative when compression made it slower,
        which is a common and reportable outcome for unstructured sparsity on CPU.

    Raises:
        ValueError: If either latency is not positive.
    """
    return 100.0 * (1.0 - 1.0 / speedup(baseline_latency, compressed_latency))


def throughput_gain(baseline_throughput: float, compressed_throughput: float) -> float:
    """Ratio of compressed to baseline throughput.

    Args:
        baseline_throughput: Dense-baseline tokens per second; must be positive.
        compressed_throughput: Compressed-model tokens per second; must be positive.

    Returns:
        ``compressed / baseline``. Greater than 1 means more tokens per second.

    Raises:
        ValueError: If either throughput is not positive.
    """
    if baseline_throughput <= 0:
        raise ValueError(f"baseline_throughput must be > 0, got {baseline_throughput}")
    if compressed_throughput <= 0:
        raise ValueError(f"compressed_throughput must be > 0, got {compressed_throughput}")
    return compressed_throughput / baseline_throughput


def memory_reduction_percentage(baseline_memory: float, compressed_memory: float) -> float:
    """Percentage of baseline peak memory removed.

    Args:
        baseline_memory: Dense-baseline peak process memory; must be positive.
        compressed_memory: Compressed-model peak process memory; must be positive.

    Returns:
        ``100 * (1 - compressed / baseline)``.

    Raises:
        ValueError: If either measurement is not positive.
    """
    if baseline_memory <= 0:
        raise ValueError(f"baseline_memory must be > 0, got {baseline_memory}")
    if compressed_memory <= 0:
        raise ValueError(f"compressed_memory must be > 0, got {compressed_memory}")
    return 100.0 * (1.0 - compressed_memory / baseline_memory)


def sparsity_realisation(
    measured_speedup: float,
    theoretical_speedup: float,
) -> float:
    """Fraction of the theoretical speedup that the CPU actually delivered.

    Args:
        measured_speedup: Speedup measured by the CPU benchmark.
        theoretical_speedup: Speedup implied by the compression budget, e.g.
            ``1 / (1 - sparsity)`` for a perfectly sparsity-exploiting kernel.

    Returns:
        ``(measured - 1) / (theoretical - 1)``, so 1.0 means the theoretical benefit was
        fully realised and 0.0 means none of it was. Values can be negative when the
        compressed model is slower than baseline. Returns 0.0 when no speedup was
        theoretically available.

    Raises:
        ValueError: If ``theoretical_speedup`` is less than 1.
    """
    if theoretical_speedup < 1.0:
        raise ValueError(f"theoretical_speedup must be >= 1, got {theoretical_speedup}")
    available = theoretical_speedup - 1.0
    if available < EPSILON:
        return 0.0
    return (measured_speedup - 1.0) / available


def theoretical_speedup_from_sparsity(sparsity: float) -> float:
    """Upper bound on speedup from removing a fraction of the multiply-accumulates.

    This is an optimistic bound: it assumes a kernel that skips every zero at no overhead,
    which no dense CPU GEMM does.

    Args:
        sparsity: Fraction of weights removed, in ``[0, 1)``.

    Returns:
        ``1 / (1 - sparsity)``.

    Raises:
        ValueError: If ``sparsity`` is outside ``[0, 1)``.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must lie in [0, 1), got {sparsity}")
    return 1.0 / (1.0 - sparsity)


def training_cost_overhead(joint_cost: float, sequential_cost: float) -> float:
    """Extra optimisation cost the joint pipeline required, as a ratio.

    Needed to answer the secondary question of how much additional training joint
    optimisation demands. Cost may be measured in optimiser steps, GPU-seconds, or tokens
    processed, as long as both arguments use the same unit.

    Args:
        joint_cost: Cost of the joint arm; must be non-negative.
        sequential_cost: Cost of the sequential arm; must be positive.

    Returns:
        ``joint / sequential``. 1.0 means the budgets were matched.

    Raises:
        ValueError: If ``sequential_cost`` is not positive or ``joint_cost`` is negative.
    """
    if sequential_cost <= 0:
        raise ValueError(f"sequential_cost must be > 0, got {sequential_cost}")
    if joint_cost < 0:
        raise ValueError(f"joint_cost must be >= 0, got {joint_cost}")
    return joint_cost / sequential_cost
