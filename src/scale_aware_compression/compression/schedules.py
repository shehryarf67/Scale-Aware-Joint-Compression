"""Sparsity schedules.

These are pure functions of the step index, implemented and tested here because they are
shared by the sequential and joint arms and a mismatch between them would invalidate the
comparison: if the joint arm reaches its target sparsity earlier, it also gets more recovery
steps at full sparsity, and the measured joint gain would partly be a schedule artefact.

The cubic schedule is the standard gradual-pruning ramp (Zhu & Gupta, 2017):

    s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / n)^3

It removes weights quickly at first, then slows down, which empirically recovers better than
a linear ramp at the same final sparsity.
"""

from __future__ import annotations

from scale_aware_compression.constants import PruningScheduleName
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


class ScheduleError(ValueError):
    """Raised when a schedule is misconfigured."""


def sparsity_at_step(
    step: int,
    *,
    schedule: PruningScheduleName | str,
    final_sparsity: float,
    initial_sparsity: float = 0.0,
    start_step: int = 0,
    end_step: int = 1000,
) -> float:
    """Target sparsity at a given optimisation step.

    Args:
        step: Current optimiser step, zero-based.
        schedule: Which ramp to use.
        final_sparsity: Sparsity to reach at ``end_step``, in ``[0, 1)``.
        initial_sparsity: Sparsity at ``start_step``, in ``[0, 1)``.
        start_step: Step at which the ramp begins. Before it, ``initial_sparsity`` holds.
        end_step: Step at which ``final_sparsity`` is reached. After it, the value holds.

    Returns:
        Target sparsity for this step.

    Raises:
        ScheduleError: If the sparsity bounds or the step window are invalid.
    """
    name = (
        PruningScheduleName(schedule) if not isinstance(schedule, PruningScheduleName) else schedule
    )

    if not 0.0 <= initial_sparsity < 1.0:
        raise ScheduleError(f"initial_sparsity must lie in [0, 1), got {initial_sparsity}")
    if not 0.0 <= final_sparsity < 1.0:
        raise ScheduleError(f"final_sparsity must lie in [0, 1), got {final_sparsity}")
    if initial_sparsity > final_sparsity:
        raise ScheduleError(
            f"initial_sparsity ({initial_sparsity}) must not exceed final_sparsity "
            f"({final_sparsity})"
        )
    if step < 0:
        raise ScheduleError(f"step must be >= 0, got {step}")
    if end_step < start_step:
        raise ScheduleError(f"end_step ({end_step}) must be >= start_step ({start_step})")

    if name in {PruningScheduleName.ONE_SHOT, PruningScheduleName.CONSTANT}:
        # One-shot pruning applies the full target immediately; constant holds it throughout.
        return final_sparsity
    if step <= start_step:
        return initial_sparsity
    if step >= end_step:
        return final_sparsity

    span = end_step - start_step
    progress = (step - start_step) / span
    if name is PruningScheduleName.LINEAR:
        return initial_sparsity + (final_sparsity - initial_sparsity) * progress
    if name is PruningScheduleName.CUBIC:
        return final_sparsity + (initial_sparsity - final_sparsity) * (1.0 - progress) ** 3

    raise ScheduleError(f"Unhandled schedule {name!r}")  # pragma: no cover - enum is exhaustive


def schedule_values(
    *,
    schedule: PruningScheduleName | str,
    final_sparsity: float,
    initial_sparsity: float = 0.0,
    start_step: int = 0,
    end_step: int = 1000,
    num_points: int = 11,
) -> list[tuple[int, float]]:
    """Sample a schedule at evenly spaced steps, for logging and plots.

    Args:
        schedule: Which ramp to sample.
        final_sparsity: Final sparsity target.
        initial_sparsity: Starting sparsity.
        start_step: First step of the ramp.
        end_step: Step at which the target is reached.
        num_points: How many samples to take, at least 2.

    Returns:
        ``(step, sparsity)`` pairs spanning ``[start_step, end_step]``.

    Raises:
        ScheduleError: If ``num_points`` is less than 2, or the schedule is invalid.
    """
    if num_points < 2:
        raise ScheduleError(f"num_points must be >= 2, got {num_points}")
    span = end_step - start_step
    steps = [start_step + round(span * index / (num_points - 1)) for index in range(num_points)]
    return [
        (
            step,
            sparsity_at_step(
                step,
                schedule=schedule,
                final_sparsity=final_sparsity,
                initial_sparsity=initial_sparsity,
                start_step=start_step,
                end_step=end_step,
            ),
        )
        for step in steps
    ]


def is_mask_update_step(
    step: int,
    *,
    frequency: int,
    start_step: int = 0,
    end_step: int = 1000,
) -> bool:
    """Whether the pruning mask should be recomputed at this step.

    Masks are updated periodically rather than every step, because re-ranking every weight is
    expensive and near-identical between adjacent steps.

    Args:
        step: Current optimiser step.
        frequency: Steps between updates; must be positive.
        start_step: First step at which updates may happen.
        end_step: Last step at which updates may happen. The end step always updates, so the
            final mask matches the final target exactly.

    Returns:
        ``True`` if the mask should be recomputed.

    Raises:
        ScheduleError: If ``frequency`` is not positive.
    """
    if frequency <= 0:
        raise ScheduleError(f"frequency must be > 0, got {frequency}")
    if step < start_step or step > end_step:
        return False
    if step == end_step:
        return True
    return (step - start_step) % frequency == 0


def mask_freeze_step(*, total_steps: int, freeze_after_ratio: float) -> int:
    """Step after which masks stop changing during joint optimisation.

    The tail of joint training runs at a fixed mask so the final phase is pure recovery. Without
    it, the last mask update lands with too few steps left to recover from, and the joint arm is
    penalised for a scheduling detail rather than measured on its merits.

    Args:
        total_steps: Total optimisation steps in the joint stage; must be positive.
        freeze_after_ratio: Fraction of training after which masks freeze, in ``[0, 1]``.

    Returns:
        The step index at which mask updates stop.

    Raises:
        ScheduleError: If ``total_steps`` is not positive or the ratio is out of range.
    """
    if total_steps <= 0:
        raise ScheduleError(f"total_steps must be > 0, got {total_steps}")
    if not 0.0 <= freeze_after_ratio <= 1.0:
        raise ScheduleError(f"freeze_after_ratio must lie in [0, 1], got {freeze_after_ratio}")
    return int(total_steps * freeze_after_ratio)
