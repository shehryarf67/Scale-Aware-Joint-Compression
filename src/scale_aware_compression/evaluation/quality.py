"""Quality evaluation: assembles perplexity, agreement, and generation into one report.

This is the module the runner calls. It also owns the retention calculation against the dense
baseline, which is the quantity the joint-versus-sequential comparison is computed on: absolute
perplexity is not comparable across model sizes, but retention relative to each model's own
dense baseline is.

Status: the report container and retention assembly are implemented; the evaluation entry point
is a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import EvaluationConfig, ExperimentConfig
from scale_aware_compression.constants import Device
from scale_aware_compression.evaluation.agreement import AgreementResult
from scale_aware_compression.evaluation.generation import GenerationReport
from scale_aware_compression.evaluation.perplexity import PerplexityResult
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.joint_gain import (
    perplexity_increase_percentage,
    perplexity_retention,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

PRIMARY_QUALITY_METRIC = "perplexity_retention"
"""The metric joint gain is computed on by default. Higher is better, so a positive joint gain
always means the joint arm won."""


@dataclass(slots=True)
class QualityReport:
    """Everything measured about one model's quality."""

    perplexity: PerplexityResult | None = None
    agreement: AgreementResult | None = None
    generation: GenerationReport | None = None
    retention: dict[str, float] = field(default_factory=dict)
    device: str = Device.CPU.value

    @property
    def primary_score(self) -> float | None:
        """The score joint gain is computed from, or ``None`` if not available.

        Retention against the model's own dense baseline when one was supplied; otherwise
        ``None``, because raw perplexity is not comparable across model scales and using it as
        the primary score would make the scale trend meaningless.
        """
        return self.retention.get(PRIMARY_QUALITY_METRIC)

    def to_dict(self) -> dict[str, Any]:
        """Return a nested, serialisable mapping."""
        payload: dict[str, Any] = {
            "evaluation_device": self.device,
            "primary_quality_metric": PRIMARY_QUALITY_METRIC,
            "primary_score": self.primary_score,
            "retention": self.retention,
        }
        if self.perplexity is not None:
            payload["perplexity"] = self.perplexity.to_dict()
        if self.agreement is not None:
            payload["agreement"] = self.agreement.to_dict()
        if self.generation is not None:
            payload["generation"] = self.generation.to_dict()
        return payload

    def summary_line(self) -> str:
        """One-line summary for logs."""
        parts: list[str] = []
        if self.perplexity is not None:
            parts.append(f"ppl={self.perplexity.perplexity:.3f}")
        if PRIMARY_QUALITY_METRIC in self.retention:
            parts.append(f"retention={self.retention[PRIMARY_QUALITY_METRIC]:.2f}%")
        if self.agreement is not None:
            parts.append(f"top1_agreement={self.agreement.top1_agreement:.3f}")
        return " ".join(parts) if parts else "no quality metrics recorded"


def compute_retention(
    compressed: PerplexityResult,
    dense: PerplexityResult,
) -> dict[str, float]:
    """Compute quality retention against the dense baseline.

    Args:
        compressed: Perplexity of the compressed model.
        dense: Perplexity of that same model's dense FP32 baseline.

    Returns:
        Mapping with ``perplexity_retention`` (higher is better) and
        ``perplexity_increase_percentage`` (lower is better). Both are reported because joint
        gain is defined over each convention.

    Raises:
        ValueError: If either perplexity is not positive, or the two were measured on different
            evaluation data.
    """
    if (
        compressed.dataset_fingerprint is not None
        and dense.dataset_fingerprint is not None
        and compressed.dataset_fingerprint != dense.dataset_fingerprint
    ):
        raise ValueError(
            "Refusing to compute retention across different evaluation data "
            f"(compressed fingerprint {compressed.dataset_fingerprint}, dense "
            f"{dense.dataset_fingerprint}). Re-run both arms on the same evaluation split."
        )
    return {
        "perplexity_retention": perplexity_retention(dense.perplexity, compressed.perplexity),
        "perplexity_increase_percentage": perplexity_increase_percentage(
            dense.perplexity, compressed.perplexity
        ),
    }


def evaluate_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: ExperimentConfig,
    *,
    dense_reference: PerplexityResult | None = None,
    dense_model: nn.Module | None = None,
) -> QualityReport:
    """Run the configured quality evaluation on CPU.

    Args:
        model: The model to evaluate.
        tokenizer: Matching tokeniser.
        config: The full experiment config.
        dense_reference: Dense-baseline perplexity, needed for retention. Loaded from the dense
            run's record rather than recomputed.
        dense_model: The dense model, needed only for agreement.

    Returns:
        The assembled quality report.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(evaluation): build the evaluation loader via data.loaders
    # .build_evaluation_dataloader(), then run the metrics named in config.evaluation.metrics:
    #   'perplexity' -> compute_perplexity()
    #   'agreement'  -> compute_agreement(), requires dense_model
    #   'generation' -> generate_samples()
    # Fill `retention` via compute_retention() when dense_reference is given, and warn loudly
    # when it is not: without it there is no primary score and the row cannot contribute to a
    # joint-gain comparison.
    # Assert config.evaluation.device is CPU for a reported number.
    raise NotImplementedError(
        "evaluate_model is not implemented yet; see the TODO in evaluation/quality.py"
    )


def check_evaluation_device(config: EvaluationConfig) -> None:
    """Warn when a reported quality number would be produced off CPU.

    Args:
        config: Evaluation section of an experiment config.
    """
    if config.device is not Device.CPU:
        LOGGER.warning(
            "evaluation.device=%s. Exploratory evaluation on GPU is fine, but any number "
            "reported in the write-up must be produced on CPU.",
            config.device.value,
        )
