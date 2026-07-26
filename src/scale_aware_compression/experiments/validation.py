"""External validation on a different model family.

The Pythia sweep can establish that joint gain trends with scale *within* one family trained on
one corpus with one recipe. It cannot establish that the trend is a property of transformer
compression rather than of Pythia. Running the same arms on Qwen2.5-0.5B tests that.

What transfers and what does not:

* **Does not transfer:** absolute perplexity. Different tokeniser, different vocabulary,
  different training data. Comparing Qwen's perplexity to Pythia's is meaningless.
* **Does transfer:** the *sign* and rough magnitude of joint gain, since it is defined relative
  to each model's own dense baseline and its own sequential arm.

So the validation is a directional check at one scale, not a fifth sweep point. Qwen2.5-0.5B
sits between pythia-410m and pythia-1b in size, which is what makes the comparison legible.

Status: the interpolation and comparison helpers are implemented; execution is a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.experiments.runner import ExperimentRecord, ExperimentTracker
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.models.registry import get_model_spec, scale_sweep_models

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ValidationOutcome:
    """Whether the external model behaved as the Pythia trend predicts."""

    validation_model: str
    validation_parameter_count: int
    validation_joint_gain: float
    expected_joint_gain: float
    """Gain interpolated from the Pythia trend at the validation model's size."""
    bracketing_models: tuple[str, str]
    sign_agrees: bool
    relative_error: float

    @property
    def transfers(self) -> bool:
        """Whether the trend can be said to transfer.

        Requires the sign to agree and the magnitude to be within a factor of two. The magnitude
        tolerance is deliberately loose: with different data and tokenisers, agreement in
        direction is the claim being tested, not agreement in value.
        """
        return self.sign_agrees and self.relative_error < 1.0

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "validation_model": self.validation_model,
            "validation_parameter_count": self.validation_parameter_count,
            "validation_joint_gain": self.validation_joint_gain,
            "expected_joint_gain": self.expected_joint_gain,
            "bracketing_models": list(self.bracketing_models),
            "sign_agrees": self.sign_agrees,
            "relative_error": self.relative_error,
            "transfers": self.transfers,
        }


def interpolate_expected_gain(
    trend: list[tuple[int, float]],
    parameter_count: int,
) -> tuple[float, tuple[int, int]]:
    """Interpolate the Pythia joint-gain trend at an arbitrary model size.

    Args:
        trend: ``(parameter_count, joint_gain)`` points from the Pythia sweep. At least two.
        parameter_count: Size to interpolate at.

    Returns:
        The interpolated gain and the two parameter counts that bracket it. Outside the trend's
        range the nearest endpoint is returned and both bracket values are that endpoint, since
        extrapolating a scale trend from three or four points is not defensible.

    Raises:
        ValueError: If fewer than two trend points are given.
    """
    if len(trend) < 2:
        raise ValueError(f"Interpolation needs at least 2 trend points, got {len(trend)}")
    points = sorted(trend)

    if parameter_count <= points[0][0]:
        LOGGER.warning(
            "Validation model (%d parameters) is smaller than every sweep point; clamping to "
            "the smallest rather than extrapolating.",
            parameter_count,
        )
        return points[0][1], (points[0][0], points[0][0])
    if parameter_count >= points[-1][0]:
        LOGGER.warning(
            "Validation model (%d parameters) is larger than every sweep point; clamping to "
            "the largest rather than extrapolating.",
            parameter_count,
        )
        return points[-1][1], (points[-1][0], points[-1][0])

    for (lower_size, lower_gain), (upper_size, upper_gain) in zip(points, points[1:], strict=False):
        if lower_size <= parameter_count <= upper_size:
            span = upper_size - lower_size
            weight = (parameter_count - lower_size) / span if span else 0.0
            return lower_gain + weight * (upper_gain - lower_gain), (lower_size, upper_size)

    raise ValueError(  # pragma: no cover - unreachable given the clamps above
        f"Could not bracket {parameter_count} within the trend"
    )


def assess_transfer(
    *,
    validation_model: str,
    validation_joint_gain: float,
    pythia_trend: list[tuple[int, float]],
) -> ValidationOutcome:
    """Compare a measured external gain against the Pythia trend's prediction.

    Args:
        validation_model: Registry short name of the external model.
        validation_joint_gain: Joint gain measured on it.
        pythia_trend: ``(parameter_count, joint_gain)`` points from the Pythia sweep.

    Returns:
        The assessment.

    Raises:
        ValueError: If the trend has fewer than two points.
        UnknownModelError: If ``validation_model`` is not registered.
    """
    spec = get_model_spec(validation_model)
    expected, bracket = interpolate_expected_gain(pythia_trend, spec.parameter_count)

    sign_agrees = (validation_joint_gain >= 0) == (expected >= 0)
    denominator = max(abs(expected), 1e-9)
    relative_error = abs(validation_joint_gain - expected) / denominator

    bracket_names = tuple(
        next(
            (
                name
                for name in scale_sweep_models(include_optional=True)
                if get_model_spec(name).parameter_count == size
            ),
            str(size),
        )
        for size in bracket
    )
    outcome = ValidationOutcome(
        validation_model=spec.short_name,
        validation_parameter_count=spec.parameter_count,
        validation_joint_gain=validation_joint_gain,
        expected_joint_gain=expected,
        bracketing_models=(bracket_names[0], bracket_names[1]),
        sign_agrees=sign_agrees,
        relative_error=relative_error,
    )
    LOGGER.info(
        "External validation on %s: measured gain %.4f vs %.4f expected from the Pythia trend "
        "(bracketed by %s); transfers=%s",
        outcome.validation_model,
        outcome.validation_joint_gain,
        outcome.expected_joint_gain,
        " / ".join(outcome.bracketing_models),
        outcome.transfers,
    )
    return outcome


def run_validation(
    config: ExperimentConfig,
    *,
    tracker: ExperimentTracker | None = None,
) -> list[ExperimentRecord]:
    """Run the compression arms on the external validation model.

    Args:
        config: An experiment config whose model is the validation model, normally
            ``configs/experiments/qwen_validation.yaml``.
        tracker: Where records go. Defaults to ``<output_dir>/metrics``.

    Returns:
        The completed records.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(validation): build a sweep plan restricted to the validation model, using the *same*
    # budgets, seeds, and arms as the Pythia sweep, then run it via scale_sweep.run_sweep().
    # Warn if config.model.name is a scale-sweep model: running the sweep family here would not
    # be external validation.
    # Qwen2.5 has tied embeddings; check that the pruning and quantisation exclude patterns
    # actually exclude 'lm_head' for this family, or the tied input embedding gets compressed
    # as a side effect and the arms stop being comparable to Pythia's.
    raise NotImplementedError(
        "run_validation is not implemented yet; see the TODO in experiments/validation.py"
    )
