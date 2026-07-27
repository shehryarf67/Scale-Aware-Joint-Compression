"""Shared leaf utilities for the evaluation subpackage.

:mod:`quality` imports :mod:`perplexity`, :mod:`agreement`, and :mod:`generation`, and all four
need the error type and the device-policy check. Keeping those here avoids an import cycle.
"""

from __future__ import annotations

from scale_aware_compression.config import EvaluationConfig
from scale_aware_compression.constants import Device
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


class EvaluationError(RuntimeError):
    """Raised when a quality metric cannot be computed."""


def check_evaluation_device(config: EvaluationConfig) -> None:
    """Warn when a reported quality number would be produced off CPU.

    A warning rather than an error: evaluating on GPU while iterating is normal and useful. Only
    the numbers that reach the write-up have to come from CPU.

    Args:
        config: Evaluation section of an experiment config.
    """
    if config.device is not Device.CPU:
        LOGGER.warning(
            "evaluation.device=%s. Exploratory evaluation on GPU is fine, but any number "
            "reported in the write-up must be produced on CPU.",
            config.device.value,
        )
