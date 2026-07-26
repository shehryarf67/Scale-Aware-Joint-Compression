"""Dense-versus-compressed prediction agreement.

Perplexity can hide a behavioural change: a compressed model can match the dense model's
average loss while making different predictions on individual tokens. Agreement measures that
directly, and it is more sensitive than perplexity at the mild compression budgets where the
joint and sequential arms are closest.

Status: the pure comparison functions are implemented; the model-evaluation path is a
placeholder.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import EvaluationConfig
from scale_aware_compression.constants import EPSILON
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from torch.utils.data import DataLoader

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AgreementResult:
    """How closely a compressed model reproduces the dense model's predictions."""

    top1_agreement: float
    """Fraction of positions where both models' argmax token matches, in ``[0, 1]``."""
    top5_agreement: float
    mean_kl_divergence: float
    """Mean KL(dense || compressed) over next-token distributions, in nats."""
    num_positions: int
    device: str

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "top1_agreement": self.top1_agreement,
            "top5_agreement": self.top5_agreement,
            "mean_kl_divergence": self.mean_kl_divergence,
            "agreement_positions": self.num_positions,
            "agreement_device": self.device,
        }


def agreement_rate(reference: Sequence[int], candidate: Sequence[int]) -> float:
    """Fraction of positions where two prediction sequences match.

    Args:
        reference: Dense-model predictions.
        candidate: Compressed-model predictions.

    Returns:
        Agreement in ``[0, 1]``.

    Raises:
        ValueError: If the sequences differ in length or are empty.
    """
    if len(reference) != len(candidate):
        raise ValueError(
            f"Prediction sequences differ in length: {len(reference)} vs {len(candidate)}"
        )
    if not reference:
        raise ValueError("agreement_rate requires at least one position")
    matches = sum(1 for left, right in zip(reference, candidate, strict=True) if left == right)
    return matches / len(reference)


def top_k_agreement_rate(
    reference_top_k: Sequence[Sequence[int]],
    candidate_predictions: Sequence[int],
) -> float:
    """Fraction of positions where the compressed prediction is in the dense model's top-k.

    A gentler criterion than exact match: a compressed model that consistently picks the dense
    model's second choice has changed behaviour far less than one picking an unrelated token.

    Args:
        reference_top_k: Per position, the dense model's top-k token ids.
        candidate_predictions: The compressed model's argmax per position.

    Returns:
        Agreement in ``[0, 1]``.

    Raises:
        ValueError: If the lengths differ or the inputs are empty.
    """
    if len(reference_top_k) != len(candidate_predictions):
        raise ValueError(
            f"Got {len(reference_top_k)} top-k lists but {len(candidate_predictions)} predictions"
        )
    if not reference_top_k:
        raise ValueError("top_k_agreement_rate requires at least one position")
    matches = sum(
        1
        for candidates, prediction in zip(reference_top_k, candidate_predictions, strict=True)
        if prediction in candidates
    )
    return matches / len(candidate_predictions)


def kl_divergence(reference: Sequence[float], candidate: Sequence[float]) -> float:
    """KL(reference || candidate) between two discrete distributions, in nats.

    Args:
        reference: Dense-model probabilities. Must sum to approximately 1.
        candidate: Compressed-model probabilities. Must sum to approximately 1.

    Returns:
        The divergence in nats, always non-negative.

    Raises:
        ValueError: If lengths differ, a probability is negative, or either distribution does
            not sum to 1 within tolerance.
    """
    if len(reference) != len(candidate):
        raise ValueError(f"Distributions differ in length: {len(reference)} vs {len(candidate)}")
    if not reference:
        raise ValueError("kl_divergence requires a non-empty distribution")
    for name, distribution in (("reference", reference), ("candidate", candidate)):
        if any(value < 0 for value in distribution):
            raise ValueError(f"{name} distribution has a negative probability")
        if abs(sum(distribution) - 1.0) > 1e-4:
            raise ValueError(f"{name} distribution sums to {sum(distribution):.6f}, not 1")

    total = 0.0
    for probability_reference, probability_candidate in zip(reference, candidate, strict=True):
        if probability_reference <= EPSILON:
            continue
        # Clamp rather than divide by zero: a compressed model assigning exactly zero to a
        # token the dense model likes should register as a large but finite divergence.
        total += probability_reference * math.log(
            probability_reference / max(probability_candidate, EPSILON)
        )
    return total


def compute_agreement(
    dense_model: nn.Module,
    compressed_model: nn.Module,
    dataloader: DataLoader,
    config: EvaluationConfig,
) -> AgreementResult:
    """Compare a compressed model's predictions against the dense baseline.

    Args:
        dense_model: The dense FP32 reference.
        compressed_model: The compressed model.
        dataloader: Evaluation batches; the same batches both models see elsewhere.
        config: Evaluation section of an experiment config.

    Returns:
        The agreement result.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(evaluation): under torch.inference_mode(), run both models over the same batches,
    # accumulate top-1 and top-5 agreement and mean KL(dense || compressed) over
    # config.agreement_samples positions.
    # Memory note: at 1B+ parameters, both models resident in FP32 on CPU may not fit. Prefer
    # running the dense model first and caching its argmax and top-5 ids (not the full logits:
    # vocab x positions in FP32 is far larger than either model).
    raise NotImplementedError(
        "compute_agreement is not implemented yet; see the TODO in evaluation/agreement.py"
    )
