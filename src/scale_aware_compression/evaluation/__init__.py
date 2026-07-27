"""Quality evaluation: perplexity, dense-vs-compressed agreement, and generation.

Final reported quality numbers are produced on CPU. Evaluating on GPU during development is
fine and :func:`check_evaluation_device` warns rather than fails, but the write-up's numbers
come from CPU runs.
"""

from __future__ import annotations

from scale_aware_compression.evaluation.agreement import (
    AgreementResult,
    agreement_rate,
    compute_agreement,
    kl_divergence,
    top_k_agreement_rate,
)
from scale_aware_compression.evaluation.common import EvaluationError, check_evaluation_device
from scale_aware_compression.evaluation.generation import (
    DEFAULT_PROMPTS,
    GenerationReport,
    GenerationSample,
    distinct_token_ratio,
    generate_samples,
    repetition_rate,
)
from scale_aware_compression.evaluation.perplexity import (
    PerplexityResult,
    aggregate_nll,
    compute_perplexity,
    perplexity_from_nll,
)
from scale_aware_compression.evaluation.quality import (
    PRIMARY_QUALITY_METRIC,
    QualityReport,
    compute_retention,
    evaluate_model,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "PRIMARY_QUALITY_METRIC",
    "AgreementResult",
    "EvaluationError",
    "GenerationReport",
    "GenerationSample",
    "PerplexityResult",
    "QualityReport",
    "aggregate_nll",
    "agreement_rate",
    "check_evaluation_device",
    "compute_agreement",
    "compute_perplexity",
    "compute_retention",
    "distinct_token_ratio",
    "evaluate_model",
    "generate_samples",
    "kl_divergence",
    "perplexity_from_nll",
    "repetition_rate",
    "top_k_agreement_rate",
]
