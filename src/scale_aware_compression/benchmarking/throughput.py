"""Throughput derived from measured latency and the shape of the benchmark workload.

Throughput is not measured independently: it is computed from the same timings the latency
statistics come from, so the two can never disagree. Which quantity is meaningful depends on
the workload:

* prefill / single forward pass -> ``batch_size * sequence_length`` tokens per call
* autoregressive decoding -> ``batch_size * generated_tokens`` tokens per call

Both are reported, and :func:`throughput_from_latency` picks the right denominator from the
benchmark configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ThroughputStatistics:
    """Throughput of a benchmarked configuration."""

    tokens_per_second: float
    samples_per_second: float
    tokens_per_call: int
    workload: str
    """``"forward"`` for a single forward pass, ``"generate"`` for decoding."""

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "throughput_tokens_per_s": self.tokens_per_second,
            "throughput_samples_per_s": self.samples_per_second,
            "tokens_per_call": self.tokens_per_call,
            "workload": self.workload,
        }


def tokens_per_second(token_count: int, seconds: float) -> float:
    """Tokens processed per second.

    Args:
        token_count: Tokens processed in the measured interval; must be positive.
        seconds: Duration of the interval; must be positive.

    Returns:
        Tokens per second.

    Raises:
        ValueError: If either argument is not positive.
    """
    if token_count <= 0:
        raise ValueError(f"token_count must be > 0, got {token_count}")
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    return token_count / seconds


def samples_per_second(sample_count: int, seconds: float) -> float:
    """Sequences processed per second.

    Args:
        sample_count: Sequences processed in the measured interval; must be positive.
        seconds: Duration of the interval; must be positive.

    Returns:
        Sequences per second.

    Raises:
        ValueError: If either argument is not positive.
    """
    if sample_count <= 0:
        raise ValueError(f"sample_count must be > 0, got {sample_count}")
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    return sample_count / seconds


def throughput_from_latency(
    *,
    latency_seconds: float,
    batch_size: int,
    sequence_length: int,
    generated_tokens: int = 0,
) -> ThroughputStatistics:
    """Derive throughput from a single representative latency.

    Pass the *median* latency rather than the mean: the median is the reported central
    measure in this study's protocol, so the throughput figure stays consistent with it.

    Args:
        latency_seconds: Time for one benchmark call; must be positive.
        batch_size: Sequences per call; must be positive.
        sequence_length: Tokens per sequence in the forward pass; must be positive.
        generated_tokens: When positive, the workload is decoding and this many new tokens
            per sequence are counted instead of the prompt length.

    Returns:
        The derived throughput statistics.

    Raises:
        ValueError: If any size or duration is not positive.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    if generated_tokens < 0:
        raise ValueError(f"generated_tokens must be >= 0, got {generated_tokens}")

    if generated_tokens > 0:
        workload = "generate"
        tokens_per_call = batch_size * generated_tokens
    else:
        workload = "forward"
        tokens_per_call = batch_size * sequence_length

    return ThroughputStatistics(
        tokens_per_second=tokens_per_second(tokens_per_call, latency_seconds),
        samples_per_second=samples_per_second(batch_size, latency_seconds),
        tokens_per_call=tokens_per_call,
        workload=workload,
    )
