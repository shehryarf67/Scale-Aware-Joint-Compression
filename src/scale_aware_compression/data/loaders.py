"""Dataset loading and data loader construction.

Status: placeholder. Signatures and the fairness constraints they must respect are recorded
here so the implementation has a specification.

The controlling constraint: every arm at a given model scale must see the *same* evaluation
tokens, the same calibration set, and the same recovery data in the same order. Otherwise the
joint-versus-sequential difference includes a data difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch.utils.data import DataLoader, Dataset
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class DataError(RuntimeError):
    """Raised when a dataset cannot be loaded or is unusable as configured."""


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """What was actually loaded, for the run record."""

    dataset: str
    subset: str | None
    split: str
    num_examples: int
    num_tokens: int
    sequence_length: int
    num_sequences: int
    fingerprint: str
    """Stable hash of the token stream. Two runs with different fingerprints were not
    evaluated on the same data and must not be compared."""

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "dataset": self.dataset,
            "subset": self.subset,
            "split": self.split,
            "num_examples": self.num_examples,
            "num_tokens": self.num_tokens,
            "sequence_length": self.sequence_length,
            "num_sequences": self.num_sequences,
            "fingerprint": self.fingerprint,
        }


def load_raw_dataset(config: DataConfig, split: str) -> Any:
    """Load one split of the configured corpus.

    Args:
        config: Data section of an experiment config.
        split: Split name, e.g. ``"validation"``.

    Returns:
        The raw ``datasets.Dataset``.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(data): call datasets.load_dataset(config.dataset, config.subset, split=split,
    # cache_dir=config.cache_dir). Wrap failures in DataError with the dataset name, since a
    # missing subset name is the usual cause. Do not download at import time -- only here.
    raise NotImplementedError(
        "load_raw_dataset is not implemented yet; see the TODO in data/loaders.py"
    )


def build_language_modelling_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
) -> tuple[Dataset, DatasetSummary]:
    """Tokenise and chunk a split into fixed-length language-modelling sequences.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.

    Returns:
        The chunked dataset and its summary.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(data): concatenate the text column, tokenise once, then chunk into
    # config.sequence_length blocks, dropping the final partial block so every sequence has
    # the same length -- a ragged final block changes the perplexity denominator between runs.
    # Compute the fingerprint from the token ids (e.g. sha256 of the first and last 4096 ids
    # plus the total count) so cross-arm comparability is checkable.
    #
    # Note on tokenisers: Pythia and Qwen have different vocabularies, so token counts and
    # perplexities are not comparable between the two families. Only the *retention* ratios
    # transfer, which is why the Qwen run is validation rather than a sweep point.
    raise NotImplementedError(
        "build_language_modelling_dataset is not implemented yet; see the TODO in "
        "data/loaders.py"
    )


def build_dataloader(
    dataset: Dataset,
    config: DataConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """Wrap a dataset in a deterministic data loader.

    Args:
        dataset: A chunked language-modelling dataset.
        config: Data section of an experiment config.
        batch_size: Overrides ``config.batch_size``.
        shuffle: Whether to shuffle. Evaluation never shuffles.
        seed: Seed for the shuffle generator, required when ``shuffle`` is true.

    Returns:
        The data loader.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(data): construct torch.utils.data.DataLoader with worker_init_fn=seed_worker and an
    # explicit torch.Generator seeded from `seed`, so recovery and joint training see the same
    # batch order. drop_last=True during training keeps step counts identical across arms.
    raise NotImplementedError(
        "build_dataloader is not implemented yet; see the TODO in data/loaders.py"
    )


def build_evaluation_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[DataLoader, DatasetSummary]:
    """Build the evaluation loader used for every arm at a given scale.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.

    Returns:
        The loader and its dataset summary.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(data): build the eval split, truncate to config.max_eval_samples, and return an
    # unshuffled loader. Log the fingerprint at INFO: it is the field to check first when two
    # arms produce suspiciously different perplexities.
    raise NotImplementedError(
        "build_evaluation_dataloader is not implemented yet; see the TODO in data/loaders.py"
    )
