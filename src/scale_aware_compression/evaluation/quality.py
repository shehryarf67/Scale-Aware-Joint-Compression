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

from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import CompressionMethod, Device
from scale_aware_compression.evaluation.agreement import AgreementResult, compute_agreement
from scale_aware_compression.evaluation.common import check_evaluation_device
from scale_aware_compression.evaluation.generation import GenerationReport, generate_samples
from scale_aware_compression.evaluation.perplexity import PerplexityResult, compute_perplexity
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
    dataloader: Any = None,
    dataset_summary: Any = None,
) -> QualityReport:
    """Run the configured quality evaluation on CPU.

    Args:
        model: The model to evaluate.
        tokenizer: Matching tokeniser.
        config: The full experiment config.
        dense_reference: Dense-baseline perplexity, needed for retention. Loaded from the dense
            run's record rather than recomputed, so both arms are normalised against exactly the
            same number. Ignored for the dense arm, which is its own reference.
        dense_model: The dense model, needed only for agreement. When absent, the agreement
            metric is skipped rather than failing the run.
        dataloader: Pre-built evaluation loader. Built from ``config.data`` when omitted.
        dataset_summary: Summary matching ``dataloader``, used for the fingerprint.

    Returns:
        The assembled quality report.

    Raises:
        EvaluationError: If a requested metric cannot be computed.
    """
    from scale_aware_compression.data.loaders import build_evaluation_dataloader

    check_evaluation_device(config.evaluation)
    metrics = [metric.lower() for metric in config.evaluation.metrics]
    report = QualityReport(device=config.evaluation.device.value)

    needs_loader = any(metric in {"perplexity", "agreement"} for metric in metrics)
    if needs_loader and dataloader is None:
        dataloader, dataset_summary = build_evaluation_dataloader(
            config.data,
            tokenizer,
            batch_size=config.evaluation.batch_size,
            max_samples=config.evaluation.max_samples,
        )
    fingerprint = getattr(dataset_summary, "fingerprint", None)

    if "perplexity" in metrics:
        report.perplexity = compute_perplexity(
            model, dataloader, config.evaluation, dataset_fingerprint=fingerprint
        )

    if "agreement" in metrics:
        if dense_model is None:
            # Expected for the dense arm, which has nothing to compare against. Info, not a
            # warning: warning on every dense run would train the reader to ignore warnings.
            LOGGER.info("Skipping agreement: no dense reference model was supplied")
        else:
            report.agreement = compute_agreement(dense_model, model, dataloader, config.evaluation)

    if "generation" in metrics:
        report.generation = generate_samples(model, tokenizer, config.evaluation)

    # Retention. The dense arm is its own reference, which makes its retention exactly 100% and
    # keeps the column populated for every row rather than only the compressed ones.
    is_dense = config.compression.method is CompressionMethod.DENSE
    reference = dense_reference
    if reference is None and is_dense:
        reference = report.perplexity

    if report.perplexity is not None and reference is not None:
        report.retention = compute_retention(report.perplexity, reference)
    elif report.perplexity is not None and not is_dense:
        LOGGER.warning(
            "No dense reference for %s/%s, so this run has no quality retention and cannot "
            "contribute to a joint-gain comparison. Run the dense baseline for this model first.",
            config.model.name,
            config.compression.method.value,
        )

    LOGGER.info("Quality: %s", report.summary_line())
    return report
