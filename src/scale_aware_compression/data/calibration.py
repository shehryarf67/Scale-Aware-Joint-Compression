"""Calibration set construction for quantisation.

The calibration set determines the quantisation scales, so it directly affects quality. Two
requirements follow, and both are properties of this module rather than of the quantiser:

1. **Identical across arms.** The quantisation-only, sequential, and joint arms must calibrate
   on the same sequences in the same order. A different calibration draw would show up as a
   quality difference and be misread as a compression-method effect.
2. **Disjoint from evaluation.** Calibration is drawn from the training split. Calibrating on
   evaluation text leaks the test set into the quantisation parameters.

Both are enforced by deriving the sample indices from a fixed seed
(``data.calibration_seed``) rather than from the run seed, so varying the run seed to get
error bars does not also change the calibration set.

Status: placeholder for the loading path; :func:`select_calibration_indices` is implemented.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class CalibrationError(RuntimeError):
    """Raised when a calibration set cannot be built as configured."""


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Description of the calibration set used by a run."""

    num_samples: int
    sequence_length: int
    split: str
    seed: int
    indices_fingerprint: str
    token_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "calibration_num_samples": self.num_samples,
            "calibration_sequence_length": self.sequence_length,
            "calibration_split": self.split,
            "calibration_seed": self.seed,
            "calibration_indices_fingerprint": self.indices_fingerprint,
            "calibration_token_fingerprint": self.token_fingerprint,
        }


def select_calibration_indices(
    population_size: int,
    num_samples: int,
    *,
    seed: int,
) -> list[int]:
    """Choose which sequences form the calibration set.

    Deterministic in ``(population_size, num_samples, seed)`` and independent of the run seed,
    so every arm at a given scale calibrates on the same sequences.

    Args:
        population_size: Number of available sequences in the calibration split.
        num_samples: How many to draw; must not exceed ``population_size``.
        seed: Fixed calibration seed from ``data.calibration_seed``.

    Returns:
        Sorted, unique indices into the calibration split.

    Raises:
        CalibrationError: If the population is too small or the counts are invalid.
    """
    if population_size <= 0:
        raise CalibrationError(f"population_size must be > 0, got {population_size}")
    if num_samples <= 0:
        raise CalibrationError(f"num_samples must be > 0, got {num_samples}")
    if num_samples > population_size:
        raise CalibrationError(
            f"Cannot draw {num_samples} calibration samples from a split of "
            f"{population_size} sequences. Lower data.calibration_samples or use a larger "
            "split."
        )
    generator = random.Random(seed)
    return sorted(generator.sample(range(population_size), num_samples))


def load_calibration_set(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, CalibrationSummary]:
    """Build the calibration loader shared by every quantising arm.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.

    Returns:
        The loader and a summary describing exactly which sequences it holds.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(calibration): build the calibration split via
    # data.loaders.build_language_modelling_dataset(), select indices with
    # select_calibration_indices(config.calibration_samples, seed=config.calibration_seed),
    # subset the dataset, and return an unshuffled loader.
    # Assert the calibration split differs from config.eval_split, or that the selected
    # indices are disjoint from the evaluation indices when they share a split. Raise
    # CalibrationError rather than warning: silent evaluation leakage would invalidate every
    # quantised result in the study.
    raise NotImplementedError(
        "load_calibration_set is not implemented yet; see the TODO in data/calibration.py"
    )


def cache_calibration_set(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | None = None,
) -> CalibrationSummary:
    """Materialise the calibration set to ``data/calibration`` for reuse.

    Caching it once means every arm and every seed provably reads the same bytes, rather than
    relying on the selection being reproducible.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        output_dir: Destination directory. Defaults to ``data/calibration``.

    Returns:
        The summary of what was written.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(calibration): write token ids as a .npy plus a metadata.json holding the summary,
    # under data/calibration/<dataset>_<split>_<tokenizer>_<n>_<seed>/. Reuse an existing
    # directory when the metadata matches instead of rebuilding.
    raise NotImplementedError(
        "cache_calibration_set is not implemented yet; see the TODO in data/calibration.py"
    )
