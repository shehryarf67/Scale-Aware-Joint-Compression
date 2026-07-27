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
from scale_aware_compression.evaluation.common import EvaluationError, check_evaluation_device
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

    Both models are run over the same batches, one batch at a time, and only the reduced
    statistics are kept. Caching the dense model's full logits would be far larger than either
    model: vocabulary x positions in FP32 runs to gigabytes for a few hundred sequences.

    Args:
        dense_model: The dense FP32 reference.
        compressed_model: The compressed model.
        dataloader: Evaluation batches; the same batches both models see elsewhere.
        config: Evaluation section of an experiment config.

    Returns:
        The agreement result.

    Raises:
        EvaluationError: If either model returns no logits, or the loader is empty.
    """
    import torch
    import torch.nn.functional as functional

    check_evaluation_device(config)

    device = next(compressed_model.parameters()).device
    modes = (dense_model.training, compressed_model.training)
    dense_model.eval()
    compressed_model.eval()

    budget = max(1, config.agreement_samples)
    top1_matches = 0
    top5_matches = 0
    kl_total = 0.0
    positions = 0

    try:
        with torch.inference_mode():
            for batch in dataloader:
                if positions >= budget:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                dense_logits = _logits(dense_model, input_ids, attention_mask, "dense_model")
                other_logits = _logits(
                    compressed_model, input_ids, attention_mask, "compressed_model"
                )

                # Drop the final position: it predicts nothing inside this window.
                dense_logits = dense_logits[:, :-1, :].float().reshape(-1, dense_logits.size(-1))
                other_logits = other_logits[:, :-1, :].float().reshape(-1, other_logits.size(-1))

                remaining = budget - positions
                if dense_logits.shape[0] > remaining:
                    dense_logits = dense_logits[:remaining]
                    other_logits = other_logits[:remaining]

                dense_top5 = dense_logits.topk(min(5, dense_logits.size(-1)), dim=-1).indices
                dense_argmax = dense_top5[:, 0]
                other_argmax = other_logits.argmax(dim=-1)

                top1_matches += int((dense_argmax == other_argmax).sum().item())
                top5_matches += int(
                    (dense_top5 == other_argmax.unsqueeze(-1)).any(dim=-1).sum().item()
                )

                # KL(dense || compressed), summed over the vocabulary and over positions.
                # log_softmax rather than log(softmax) for numerical stability.
                dense_log_probabilities = functional.log_softmax(dense_logits, dim=-1)
                other_log_probabilities = functional.log_softmax(other_logits, dim=-1)
                kl_total += float(
                    functional.kl_div(
                        other_log_probabilities,
                        dense_log_probabilities,
                        log_target=True,
                        reduction="sum",
                    ).item()
                )
                positions += int(dense_logits.shape[0])
    finally:
        dense_model.train(modes[0])
        compressed_model.train(modes[1])

    if positions == 0:
        raise EvaluationError("Agreement evaluation saw no positions; the loader was empty")

    result = AgreementResult(
        top1_agreement=top1_matches / positions,
        top5_agreement=top5_matches / positions,
        mean_kl_divergence=kl_total / positions,
        num_positions=positions,
        device=str(device),
    )
    LOGGER.info(
        "Agreement over %d positions: top-1 %.4f, top-5 %.4f, mean KL %.5f nats",
        result.num_positions,
        result.top1_agreement,
        result.top5_agreement,
        result.mean_kl_divergence,
    )
    return result


def _logits(model: nn.Module, input_ids: Any, attention_mask: Any, label: str) -> Any:
    """Run a model and return its logits, with a clear error if it has none."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise EvaluationError(f"{label} returned no `logits`; agreement needs a causal LM head.")
    return logits
