"""Joint gain: the central dependent variable of the study.

Definition
----------
Joint gain is the quality of the joint pipeline minus the quality of the sequential
pipeline, at a matched compression budget::

    joint_gain = joint_quality_score - sequential_quality_score

A positive value means joint optimisation was worth its extra cost. Two conventions have to
coexist, because the two headline metrics point in opposite directions:

* **higher-is-better** scores (accuracy, retention): use :func:`joint_gain` with the
  default, or explicitly ``higher_is_better=True``.
* **lower-is-better** losses (perplexity, quality loss relative to dense): use
  :func:`joint_gain_from_quality_loss`, or :func:`joint_gain` with
  ``higher_is_better=False``.

Both return a positive number when joint is better, so the sign of a reported gain always
carries the same meaning regardless of which metric produced it.

Comparisons are only meaningful when the two arms share a model, a compression budget, and
a seed; :func:`joint_gain_summary` records the budget so mismatched pairs are visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scale_aware_compression.constants import EPSILON
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


def joint_gain(
    joint_score: float,
    sequential_score: float,
    *,
    higher_is_better: bool = True,
) -> float:
    """Quality advantage of the joint pipeline over the sequential pipeline.

    Args:
        joint_score: Quality score of the joint arm.
        sequential_score: Quality score of the sequential arm.
        higher_is_better: ``True`` for scores where larger is better (accuracy, retention);
            ``False`` for losses where smaller is better (perplexity).

    Returns:
        A signed gain, positive whenever the joint arm is better under the stated
        convention.
    """
    if higher_is_better:
        return joint_score - sequential_score
    return sequential_score - joint_score


def joint_gain_from_quality_loss(joint_loss: float, sequential_loss: float) -> float:
    """Joint gain expressed from lower-is-better quality losses.

    Args:
        joint_loss: Quality degradation of the joint arm relative to dense, e.g. perplexity
            increase or accuracy points lost.
        sequential_loss: The same quantity for the sequential arm.

    Returns:
        ``sequential_loss - joint_loss``: positive when joint lost less quality.
    """
    return joint_gain(joint_loss, sequential_loss, higher_is_better=False)


def relative_joint_gain(
    joint_score: float,
    sequential_score: float,
    *,
    higher_is_better: bool = True,
) -> float:
    """Joint gain as a percentage of the sequential arm's score.

    Absolute gains are not comparable across model sizes when the underlying scores differ
    by an order of magnitude, which they do across a 160M-to-1.4B sweep. The relative form
    is what the scale trend should be fitted on.

    Args:
        joint_score: Quality score of the joint arm.
        sequential_score: Quality score of the sequential arm; must be non-zero.
        higher_is_better: See :func:`joint_gain`.

    Returns:
        Gain as a percentage of ``abs(sequential_score)``.

    Raises:
        ValueError: If ``sequential_score`` is effectively zero.
    """
    if abs(sequential_score) < EPSILON:
        raise ValueError("sequential_score is zero; a relative gain is undefined")
    absolute = joint_gain(joint_score, sequential_score, higher_is_better=higher_is_better)
    return 100.0 * absolute / abs(sequential_score)


def accuracy_retention(compressed_accuracy: float, dense_accuracy: float) -> float:
    """Fraction of the dense model's accuracy that survives compression.

    Args:
        compressed_accuracy: Accuracy of the compressed model.
        dense_accuracy: Accuracy of the dense FP32 baseline; must be positive.

    Returns:
        ``100 * compressed / dense``. 100 means no loss; above 100 means compression helped,
        which does happen at mild sparsity and is worth reporting rather than clipping.

    Raises:
        ValueError: If ``dense_accuracy`` is not positive or accuracies are negative.
    """
    if dense_accuracy <= 0:
        raise ValueError(f"dense_accuracy must be > 0, got {dense_accuracy}")
    if compressed_accuracy < 0:
        raise ValueError(f"compressed_accuracy must be >= 0, got {compressed_accuracy}")
    return 100.0 * compressed_accuracy / dense_accuracy


def perplexity_retention(dense_perplexity: float, compressed_perplexity: float) -> float:
    """Quality retention derived from perplexity, on a higher-is-better scale.

    Perplexity is lower-is-better, so the ratio is inverted to keep every ``*_retention``
    metric in the results table pointing the same way.

    Args:
        dense_perplexity: Perplexity of the dense FP32 baseline; must be positive.
        compressed_perplexity: Perplexity of the compressed model; must be positive.

    Returns:
        ``100 * dense / compressed``. 100 means perplexity was preserved exactly; below 100
        means the compressed model is worse.

    Raises:
        ValueError: If either perplexity is not positive.
    """
    if dense_perplexity <= 0:
        raise ValueError(f"dense_perplexity must be > 0, got {dense_perplexity}")
    if compressed_perplexity <= 0:
        raise ValueError(f"compressed_perplexity must be > 0, got {compressed_perplexity}")
    return 100.0 * dense_perplexity / compressed_perplexity


def perplexity_increase_percentage(dense_perplexity: float, compressed_perplexity: float) -> float:
    """Percentage increase in perplexity caused by compression (lower is better).

    This is the natural "quality loss" input to :func:`joint_gain_from_quality_loss`.

    Args:
        dense_perplexity: Perplexity of the dense FP32 baseline; must be positive.
        compressed_perplexity: Perplexity of the compressed model; must be positive.

    Returns:
        ``100 * (compressed / dense - 1)``. Zero means unchanged; negative means improved.

    Raises:
        ValueError: If either perplexity is not positive.
    """
    if dense_perplexity <= 0:
        raise ValueError(f"dense_perplexity must be > 0, got {dense_perplexity}")
    if compressed_perplexity <= 0:
        raise ValueError(f"compressed_perplexity must be > 0, got {compressed_perplexity}")
    return 100.0 * (compressed_perplexity / dense_perplexity - 1.0)


@dataclass(frozen=True, slots=True)
class JointGainSummary:
    """One joint-versus-sequential comparison at a single model scale and budget."""

    model_name: str
    size_label: str
    parameter_count: int
    budget_label: str
    metric_name: str
    joint_score: float
    sequential_score: float
    absolute_gain: float
    relative_gain_percentage: float
    higher_is_better: bool
    seed: int | None = None

    @property
    def joint_is_better(self) -> bool:
        """Whether the joint arm won this comparison."""
        return self.absolute_gain > 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping for CSV export."""
        return {
            "model_name": self.model_name,
            "size_label": self.size_label,
            "parameter_count": self.parameter_count,
            "budget_label": self.budget_label,
            "metric_name": self.metric_name,
            "joint_score": self.joint_score,
            "sequential_score": self.sequential_score,
            "absolute_gain": self.absolute_gain,
            "relative_gain_percentage": self.relative_gain_percentage,
            "higher_is_better": self.higher_is_better,
            "joint_is_better": self.joint_is_better,
            "seed": self.seed,
        }


def joint_gain_summary(
    *,
    model_name: str,
    size_label: str,
    parameter_count: int,
    budget_label: str,
    metric_name: str,
    joint_score: float,
    sequential_score: float,
    higher_is_better: bool = True,
    seed: int | None = None,
) -> JointGainSummary:
    """Build a :class:`JointGainSummary` from a matched pair of scores.

    Args:
        model_name: Registry short name shared by both arms.
        size_label: Scale label used on the plot axis.
        parameter_count: Parameter count, the x axis of the scale trend.
        budget_label: Compression budget both arms were run at. Comparing across budgets is
            a measurement error, so it is recorded explicitly.
        metric_name: Which quality metric produced the scores.
        joint_score: Score for the joint arm.
        sequential_score: Score for the sequential arm.
        higher_is_better: Direction of ``metric_name``.
        seed: Seed shared by both runs, when applicable.

    Returns:
        The populated summary.
    """
    absolute = joint_gain(joint_score, sequential_score, higher_is_better=higher_is_better)
    try:
        relative = relative_joint_gain(
            joint_score, sequential_score, higher_is_better=higher_is_better
        )
    except ValueError:
        LOGGER.warning(
            "Sequential score for %s at budget %s is zero; relative gain reported as nan",
            model_name,
            budget_label,
        )
        relative = float("nan")
    return JointGainSummary(
        model_name=model_name,
        size_label=size_label,
        parameter_count=parameter_count,
        budget_label=budget_label,
        metric_name=metric_name,
        joint_score=joint_score,
        sequential_score=sequential_score,
        absolute_gain=absolute,
        relative_gain_percentage=relative,
        higher_is_better=higher_is_better,
        seed=seed,
    )
