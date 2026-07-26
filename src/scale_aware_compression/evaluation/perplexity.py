"""Perplexity evaluation.

Perplexity is the primary quality metric for this study: it needs no task-specific head, it is
defined identically for every model in the sweep, and it is sensitive enough to register the
small degradations that separate a joint arm from a sequential one at moderate compression.

Final reported perplexity is computed on CPU, per the evaluation policy.

Status: the pure aggregation functions are implemented; the model-evaluation path is a
placeholder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import EvaluationConfig
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from torch.utils.data import DataLoader

LOGGER = get_logger(__name__)

MAX_REPRESENTABLE_PERPLEXITY = 1e30
"""Cap for reporting. A perplexity above this means the model is broken, not merely worse, and
``inf`` in a CSV column breaks downstream aggregation."""


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """Perplexity together with the token count it was computed over."""

    perplexity: float
    total_nll: float
    total_tokens: int
    num_sequences: int
    sequence_length: int
    device: str
    dataset_fingerprint: str | None = None
    """Fingerprint of the evaluation token stream. Two perplexities with different
    fingerprints were not measured on the same data and must not be compared."""

    @property
    def bits_per_token(self) -> float:
        """Cross-entropy in bits, an alternative scale for the same quantity."""
        return (
            (self.total_nll / self.total_tokens) / math.log(2)
            if self.total_tokens
            else float("nan")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "perplexity": self.perplexity,
            "total_nll": self.total_nll,
            "total_tokens": self.total_tokens,
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "bits_per_token": self.bits_per_token,
            "evaluation_device": self.device,
            "dataset_fingerprint": self.dataset_fingerprint,
        }


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Convert a summed negative log-likelihood into perplexity.

    Summing NLL over tokens and exponentiating once at the end is not the same as averaging
    per-batch perplexities: the latter over-weights short batches. This function exists so the
    correct aggregation is used everywhere.

    Args:
        total_nll: Sum of per-token negative log-likelihoods, in nats.
        total_tokens: Number of tokens the sum covers; must be positive.

    Returns:
        ``exp(total_nll / total_tokens)``, capped at :data:`MAX_REPRESENTABLE_PERPLEXITY`.

    Raises:
        ValueError: If ``total_tokens`` is not positive or ``total_nll`` is negative.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
    if total_nll < 0:
        raise ValueError(f"total_nll must be >= 0, got {total_nll}")
    mean_nll = total_nll / total_tokens
    if mean_nll > math.log(MAX_REPRESENTABLE_PERPLEXITY):
        LOGGER.warning(
            "Mean NLL of %.2f nats exceeds the reporting cap; the model has likely collapsed.",
            mean_nll,
        )
        return MAX_REPRESENTABLE_PERPLEXITY
    return math.exp(mean_nll)


def aggregate_nll(batch_nlls: list[float], batch_token_counts: list[int]) -> tuple[float, int]:
    """Sum per-batch NLL and token counts.

    Args:
        batch_nlls: Summed NLL per batch, in nats.
        batch_token_counts: Token count per batch.

    Returns:
        ``(total_nll, total_tokens)``.

    Raises:
        ValueError: If the two lists have different lengths, or either is empty.
    """
    if len(batch_nlls) != len(batch_token_counts):
        raise ValueError(
            f"Got {len(batch_nlls)} NLL values but {len(batch_token_counts)} token counts"
        )
    if not batch_nlls:
        raise ValueError("aggregate_nll requires at least one batch")
    return sum(batch_nlls), sum(batch_token_counts)


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    config: EvaluationConfig,
    *,
    dataset_fingerprint: str | None = None,
) -> PerplexityResult:
    """Evaluate perplexity over a data loader.

    Args:
        model: The model to evaluate, on CPU for a reported number.
        dataloader: Fixed-length evaluation batches.
        config: Evaluation section of an experiment config.
        dataset_fingerprint: Fingerprint of the evaluation stream, stored in the result.

    Returns:
        The perplexity result.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(evaluation): under torch.inference_mode(), for each batch compute the shifted
    # cross-entropy with reduction='sum' and accumulate NLL and the token count
    # (batch_size * (sequence_length - 1) per batch, since the first token has no target).
    # Then call perplexity_from_nll() once. Do not average per-batch perplexities.
    # Warn if config.device is not CPU: a reported quality number must come from CPU.
    # When config.stride is set, use a sliding window with only the non-overlapping suffix
    # contributing to the loss, and document the stride in the result -- perplexities computed
    # at different strides are not comparable.
    raise NotImplementedError(
        "compute_perplexity is not implemented yet; see the TODO in evaluation/perplexity.py"
    )
