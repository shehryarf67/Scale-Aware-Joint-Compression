"""Latency sample statistics.

Pure functions over a list of per-run durations. Kept separate from the runner so the
statistics can be tested without executing a model, and so that a saved list of raw samples
can be re-summarised later without re-running the benchmark.

Percentiles use linear interpolation between order statistics, matching
``numpy.percentile``'s default, but without requiring NumPy.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

MS_PER_SECOND = 1000.0


@dataclass(frozen=True, slots=True)
class LatencyStatistics:
    """Summary of a set of measured latencies, in milliseconds."""

    num_runs: int
    mean_ms: float
    median_ms: float
    std_ms: float
    p25_ms: float
    p75_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    total_seconds: float

    @property
    def iqr_ms(self) -> float:
        """Interquartile range, which §4.7 requires reported alongside the median.

        Preferred to the standard deviation for latency because the distribution is bounded
        below and has a long right tail: one scheduler preemption inflates the std and moves the
        mean, and leaves the IQR alone. A median with an IQR describes the typical run; a mean
        with a std describes a distribution these samples do not have.
        """
        return self.p75_ms - self.p25_ms

    @property
    def coefficient_of_variation(self) -> float:
        """Standard deviation as a fraction of the mean.

        A high value means the measurement environment was noisy and the run should be
        repeated rather than reported.
        """
        return self.std_ms / self.mean_ms if self.mean_ms > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "num_runs": self.num_runs,
            "latency_mean_ms": self.mean_ms,
            "latency_median_ms": self.median_ms,
            "latency_std_ms": self.std_ms,
            "latency_p25_ms": self.p25_ms,
            "latency_p75_ms": self.p75_ms,
            "latency_iqr_ms": self.iqr_ms,
            "latency_p95_ms": self.p95_ms,
            "latency_p99_ms": self.p99_ms,
            "latency_min_ms": self.min_ms,
            "latency_max_ms": self.max_ms,
            "latency_cv": self.coefficient_of_variation,
            "measured_total_seconds": self.total_seconds,
        }


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Linear-interpolated percentile of a sample list.

    Args:
        samples: Non-empty sequence of measurements. Need not be sorted.
        fraction: Percentile position in ``[0, 1]``, e.g. ``0.95`` for p95.

    Returns:
        The interpolated percentile.

    Raises:
        ValueError: If ``samples`` is empty or ``fraction`` is outside ``[0, 1]``.
    """
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must lie in [0, 1], got {fraction}")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight


def summarise_latencies(samples_seconds: Sequence[float]) -> LatencyStatistics:
    """Summarise per-run durations measured in seconds.

    Args:
        samples_seconds: One duration per measured run, in seconds. At least two are
            required so that a standard deviation and a p95 mean something.

    Returns:
        The summary, with all latency fields converted to milliseconds.

    Raises:
        ValueError: If fewer than two samples are given, or any sample is negative.
    """
    if len(samples_seconds) < 2:
        raise ValueError(
            f"summarise_latencies needs at least 2 samples for a std and p95, got "
            f"{len(samples_seconds)}"
        )
    if any(sample < 0 for sample in samples_seconds):
        raise ValueError("latency samples must be non-negative")

    milliseconds = [sample * MS_PER_SECOND for sample in samples_seconds]
    statistics_ = LatencyStatistics(
        num_runs=len(milliseconds),
        mean_ms=statistics.fmean(milliseconds),
        median_ms=statistics.median(milliseconds),
        # Sample standard deviation: these are repeated measurements of one configuration,
        # not a full population.
        std_ms=statistics.stdev(milliseconds),
        p25_ms=percentile(milliseconds, 0.25),
        p75_ms=percentile(milliseconds, 0.75),
        p95_ms=percentile(milliseconds, 0.95),
        p99_ms=percentile(milliseconds, 0.99),
        min_ms=min(milliseconds),
        max_ms=max(milliseconds),
        total_seconds=sum(samples_seconds),
    )
    if statistics_.coefficient_of_variation > 0.15:
        LOGGER.warning(
            "Latency spread is high (CV=%.1f%%, median=%.2f ms). Check for background load "
            "before reporting this measurement.",
            100 * statistics_.coefficient_of_variation,
            statistics_.median_ms,
        )
    return statistics_
