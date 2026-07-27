"""Calibration set construction for quantisation and layerwise reconstruction.

The calibration set determines the quantisation scales and the reconstruction targets, so it
directly affects quality. Two requirements follow, and both are properties of this module rather
than of the compressors:

1. **Identical across arms.** The quantisation-only, sequential, and joint arms must calibrate on
   the same sequences in the same order. A different calibration draw would show up as a quality
   difference and be misread as a compression-method effect.
2. **Disjoint from evaluation.** Calibration is drawn from the training split. Calibrating on
   evaluation text leaks the test set into the quantisation parameters.

Both are enforced rather than documented. Sample indices derive from a fixed
``data.calibration_seed`` rather than the run seed, so varying the run seed to get error bars does
not also change the calibration set; and :func:`load_calibration_set` guarantees disjointness by
construction when the calibration and evaluation splits coincide.

A **held-out reconstruction subset** is also provided. Layerwise reconstruction fits weights to
minimise error on the calibration activations, so calibration loss alone cannot tell you whether
the method generalised or merely memorised those few hundred sequences.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
from scale_aware_compression.constants import DEFAULT_DATA_DIR
from scale_aware_compression.data.errors import CalibrationError
from scale_aware_compression.data.loaders import (
    TokenBlockDataset,
    build_dataloader,
    build_language_modelling_dataset,
)
from scale_aware_compression.data.preprocessing import fingerprint_token_ids
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import DataLoader
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

DEFAULT_HELDOUT_FRACTION = 0.2
"""Share of the drawn sequences reserved for the overfitting check."""

__all__ = [
    "CalibrationError",
    "CalibrationSet",
    "CalibrationSummary",
    "cache_calibration_set",
    "load_calibration_set",
    "select_calibration_indices",
]


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Description of the calibration set used by a run."""

    num_samples: int
    sequence_length: int
    split: str
    seed: int
    indices_fingerprint: str
    token_fingerprint: str | None = None
    num_heldout_samples: int = 0
    heldout_indices_fingerprint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "calibration_num_samples": self.num_samples,
            "calibration_sequence_length": self.sequence_length,
            "calibration_split": self.split,
            "calibration_seed": self.seed,
            "calibration_indices_fingerprint": self.indices_fingerprint,
            "calibration_token_fingerprint": self.token_fingerprint,
            "calibration_num_heldout_samples": self.num_heldout_samples,
            "calibration_heldout_indices_fingerprint": self.heldout_indices_fingerprint,
        }


@dataclass(slots=True)
class CalibrationSet:
    """The calibration sequences, the held-out check sequences, and their provenance."""

    loader: DataLoader
    dataset: TokenBlockDataset
    indices: list[int]
    summary: CalibrationSummary
    heldout_loader: DataLoader | None = None
    heldout_dataset: TokenBlockDataset | None = None
    heldout_indices: list[int] | None = None

    def __len__(self) -> int:
        return len(self.indices)


def select_calibration_indices(
    population_size: int,
    num_samples: int,
    *,
    seed: int,
    exclude_below: int = 0,
) -> list[int]:
    """Choose which sequences form the calibration set.

    Deterministic in ``(population_size, num_samples, seed, exclude_below)`` and independent of
    the run seed, so every arm at a given scale calibrates on the same sequences.

    Args:
        population_size: Number of available sequences in the calibration split.
        num_samples: How many to draw; must not exceed the eligible population.
        seed: Fixed calibration seed from ``data.calibration_seed``.
        exclude_below: Skip indices below this value. Used to keep the draw clear of an
            evaluation prefix when both share a split.

    Returns:
        Sorted, unique indices into the calibration split.

    Raises:
        CalibrationError: If the population is too small or the counts are invalid.
    """
    if population_size <= 0:
        raise CalibrationError(f"population_size must be > 0, got {population_size}")
    if num_samples <= 0:
        raise CalibrationError(f"num_samples must be > 0, got {num_samples}")
    if exclude_below < 0:
        raise CalibrationError(f"exclude_below must be >= 0, got {exclude_below}")

    eligible = range(exclude_below, population_size)
    if num_samples > len(eligible):
        raise CalibrationError(
            f"Cannot draw {num_samples} calibration samples from {len(eligible)} eligible "
            f"sequences (population {population_size}, first {exclude_below} reserved for "
            "evaluation). Lower data.calibration_samples or use a larger split."
        )
    generator = random.Random(seed)
    return sorted(generator.sample(eligible, num_samples))


def load_calibration_set(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    num_samples: int | None = None,
    batch_size: int | None = None,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
    cache_root: Any = None,
) -> CalibrationSet:
    """Build the calibration loader shared by every arm that quantises or reconstructs.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        num_samples: Overrides ``config.calibration_samples``.
        batch_size: Overrides ``config.batch_size``.
        heldout_fraction: Share of drawn sequences reserved for the overfitting check. Set to
            0.0 to skip the held-out set.
        cache_root: Cache root for prepared tokens.

    Returns:
        The calibration set, with a summary describing exactly which sequences it holds.

    Raises:
        CalibrationError: If the split is too small, or the calibration and evaluation sets
            cannot be made disjoint.
    """
    if not 0.0 <= heldout_fraction < 1.0:
        raise CalibrationError(f"heldout_fraction must lie in [0, 1), got {heldout_fraction}")

    requested = num_samples if num_samples is not None else config.calibration_samples
    dataset, _ = build_language_modelling_dataset(
        config, tokenizer, config.calibration_split, cache_root=cache_root
    )

    # Disjointness. Evaluation always takes a deterministic prefix of its split, so when the two
    # splits coincide it is enough to draw calibration from beyond that prefix. Doing it by
    # construction is safer than checking afterwards and hoping the check is remembered.
    exclude_below = 0
    if config.calibration_split == config.eval_split:
        exclude_below = config.max_eval_samples or 0
        LOGGER.warning(
            "data.calibration_split == data.eval_split (%s). Reserving the first %d sequences "
            "for evaluation and drawing calibration only from beyond them, so the two sets "
            "cannot overlap.",
            config.eval_split,
            exclude_below,
        )

    total_to_draw = requested
    if heldout_fraction > 0.0:
        total_to_draw = requested + max(1, round(requested * heldout_fraction))

    drawn = select_calibration_indices(
        len(dataset),
        total_to_draw,
        seed=config.calibration_seed,
        exclude_below=exclude_below,
    )

    # Split the draw deterministically, using the same fixed seed. Interleaving rather than
    # slicing would correlate the held-out set with position in the corpus.
    shuffler = random.Random(config.calibration_seed + 1)
    shuffled = list(drawn)
    shuffler.shuffle(shuffled)
    calibration_indices = sorted(shuffled[:requested])
    heldout_indices = sorted(shuffled[requested:])

    calibration_dataset = dataset.subset(calibration_indices)
    loader = build_dataloader(calibration_dataset, config, batch_size=batch_size, shuffle=False)

    heldout_dataset = None
    heldout_loader = None
    if heldout_indices:
        heldout_dataset = dataset.subset(heldout_indices)
        heldout_loader = build_dataloader(
            heldout_dataset, config, batch_size=batch_size, shuffle=False
        )

    summary = CalibrationSummary(
        num_samples=len(calibration_indices),
        sequence_length=config.sequence_length,
        split=config.calibration_split,
        seed=config.calibration_seed,
        indices_fingerprint=_fingerprint_indices(calibration_indices),
        token_fingerprint=fingerprint_token_ids(calibration_dataset.token_ids()),
        num_heldout_samples=len(heldout_indices),
        heldout_indices_fingerprint=(
            _fingerprint_indices(heldout_indices) if heldout_indices else None
        ),
    )
    LOGGER.info(
        "Calibration set: %d sequences from %s (seed %d, fingerprint %s), plus %d held-out",
        summary.num_samples,
        summary.split,
        summary.seed,
        summary.indices_fingerprint,
        summary.num_heldout_samples,
    )
    return CalibrationSet(
        loader=loader,
        dataset=calibration_dataset,
        indices=calibration_indices,
        summary=summary,
        heldout_loader=heldout_loader,
        heldout_dataset=heldout_dataset,
        heldout_indices=heldout_indices or None,
    )


def cache_calibration_set(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str | Path | None = None,
    *,
    num_samples: int | None = None,
    cache_root: Any = None,
) -> CalibrationSummary:
    """Materialise the calibration set to disk for reuse.

    Caching it once means every arm and every seed provably reads the same bytes, rather than
    relying on the selection being reproducible.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        output_dir: Destination directory. Defaults to ``<project>/data/calibration``.
        num_samples: Overrides ``config.calibration_samples``.
        cache_root: Cache root for prepared tokens.

    Returns:
        The summary of what was written.

    Raises:
        CalibrationError: If the calibration set cannot be built.
    """
    calibration = load_calibration_set(
        config, tokenizer, num_samples=num_samples, cache_root=cache_root
    )
    summary = calibration.summary

    base = Path(output_dir) if output_dir is not None else DEFAULT_DATA_DIR / "calibration"
    name = (
        f"{config.dataset}_{config.calibration_split}_{summary.num_samples}"
        f"_len{config.sequence_length}_seed{config.calibration_seed}"
    )
    directory = base / "".join(
        character if character.isalnum() or character in "._-" else "-" for character in name
    )
    directory.mkdir(parents=True, exist_ok=True)

    payload = {
        **summary.to_dict(),
        "indices": calibration.indices,
        "heldout_indices": calibration.heldout_indices or [],
        "dataset": config.dataset,
        "subset": config.subset,
        "text_column": config.text_column,
    }
    (directory / "metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (directory / "calibration_tokens.json").write_text(
        json.dumps(calibration.dataset.token_ids()), encoding="utf-8"
    )
    if calibration.heldout_dataset is not None:
        (directory / "heldout_tokens.json").write_text(
            json.dumps(calibration.heldout_dataset.token_ids()), encoding="utf-8"
        )

    LOGGER.info("Cached calibration set to %s", directory)
    return summary


def _fingerprint_indices(indices: list[int]) -> str:
    """Short stable hash of an index list, for cross-arm comparison."""
    return fingerprint_token_ids(indices)
