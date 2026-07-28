"""Dataset loading and data loader construction.

The controlling constraint: every arm at a given model scale must see the *same* evaluation
tokens, the same calibration set, and the same recovery data in the same order. Otherwise the
joint-versus-sequential difference includes a data difference.

That is enforced by construction rather than by convention. The token stream is prepared once,
cached, and fingerprinted; :class:`TokenBlockDataset` is a plain indexed view over that cached
stream; and evaluation loaders never shuffle. Two runs that disagree on the fingerprint were not
evaluated on the same data, and :func:`~scale_aware_compression.evaluation.quality.compute_retention`
refuses to compare them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
from scale_aware_compression.data.errors import DataError
from scale_aware_compression.data.preprocessing import (
    chunk_sequence,
    fingerprint_token_ids,
    load_prepared_tokens,
    prepare_dataset,
)
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

__all__ = [
    "DataError",
    "DatasetSummary",
    "TokenBlockDataset",
    "build_dataloader",
    "build_evaluation_dataloader",
    "build_language_modelling_dataset",
    "collate_token_blocks",
    "load_raw_dataset",
]


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
        DataError: If ``datasets`` is missing, or the split cannot be loaded. A wrong or absent
            subset name is the usual cause, so the message names both.
    """
    try:
        import datasets
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise DataError(
            "The `datasets` package is required to load a corpus. Install the project "
            "dependencies with `pip install -e .`."
        ) from error

    LOGGER.info(
        "Loading %s/%s split=%s", config.dataset, config.subset or "(no subset)", split
    )
    try:
        loaded = datasets.load_dataset(
            config.dataset,
            config.subset,
            split=split,
            cache_dir=str(config.cache_dir) if config.cache_dir else None,
        )
    except Exception as error:
        hint = ""
        if "/" not in config.dataset:
            # The most likely cause, and the message datasets gives for it is opaque: it complains
            # about an internal `hf://` URI rather than about the name that was passed in.
            hint = (
                f" Hint: {config.dataset!r} is not a namespaced Hub repository id. `datasets` 5.x "
                f"rejects bare canonical aliases -- try a 'namespace/name' form such as "
                f"'Salesforce/{config.dataset}'."
            )
        raise DataError(
            f"Could not load {config.dataset!r} (subset={config.subset!r}, split={split!r}): "
            f"{error}{hint}"
        ) from error
    return loaded


class TokenBlockDataset:
    """Fixed-length windows over a cached token stream.

    Implements the ``torch.utils.data.Dataset`` protocol without subclassing it, so importing
    this module does not require torch. Tensors are built lazily in :meth:`__getitem__`.
    """

    def __init__(self, blocks: list[list[int]]) -> None:
        """Store the blocks.

        Args:
            blocks: Equal-length token windows.

        Raises:
            DataError: If ``blocks`` is empty or the windows differ in length.
        """
        if not blocks:
            raise DataError("TokenBlockDataset requires at least one block")
        lengths = {len(block) for block in blocks}
        if len(lengths) != 1:
            raise DataError(f"All blocks must be the same length, got lengths {sorted(lengths)}")
        self.blocks = blocks
        self.sequence_length = lengths.pop()

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        import torch

        return {"input_ids": torch.tensor(self.blocks[index], dtype=torch.long)}

    def subset(self, indices: list[int]) -> TokenBlockDataset:
        """Return a new dataset holding only the given block indices, in the order given.

        Args:
            indices: Block indices to keep.

        Returns:
            The subset.

        Raises:
            DataError: If an index is out of range.
        """
        out_of_range = [index for index in indices if not 0 <= index < len(self.blocks)]
        if out_of_range:
            raise DataError(
                f"Block indices out of range for a dataset of {len(self.blocks)}: {out_of_range[:5]}"
            )
        return TokenBlockDataset([self.blocks[index] for index in indices])

    def token_ids(self) -> list[int]:
        """Return the flattened token stream these blocks cover."""
        return [token for block in self.blocks for token in block]


def collate_token_blocks(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack token blocks into a batch.

    ``labels`` is set equal to ``input_ids``; the shift between prediction and target is applied
    at loss time so that both the model's own loss and our perplexity computation agree on it.

    Args:
        batch: Items from :class:`TokenBlockDataset`.

    Returns:
        Mapping with ``input_ids``, ``attention_mask``, and ``labels``, each ``(B, L)``.
    """
    import torch

    input_ids = torch.stack([item["input_ids"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


def build_language_modelling_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    *,
    max_sequences: int | None = None,
    cache_root: Any = None,
) -> tuple[TokenBlockDataset, DatasetSummary]:
    """Tokenise and chunk a split into fixed-length language-modelling sequences.

    Preparation is cached, so the second call for a given (dataset, subset, split, tokeniser,
    sequence length) reads tokens from disk instead of re-tokenising.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.
        max_sequences: Keep only the first N windows. Truncation is from the front, never
            random, so the evaluation set is a deterministic prefix.
        cache_root: Cache root passed through to
            :func:`~scale_aware_compression.data.preprocessing.prepare_dataset`.

    Returns:
        The chunked dataset and its summary.

    Raises:
        DataError: If the split cannot be prepared or yields no complete window.
    """
    metadata = prepare_dataset(config, tokenizer, split, root=cache_root)
    token_ids = load_prepared_tokens(config, tokenizer, split, root=cache_root)

    # Recomputed rather than trusted from metadata: it is the cheap check that the cached token
    # file and the cached metadata still describe each other.
    fingerprint = fingerprint_token_ids(token_ids)
    if fingerprint != metadata["fingerprint"]:
        raise DataError(
            f"Cached tokens for {config.dataset}/{split} do not match their metadata "
            f"(fingerprint {fingerprint} vs {metadata['fingerprint']}). Delete the cache "
            "directory and re-prepare."
        )

    blocks = chunk_sequence(token_ids, config.sequence_length)
    if max_sequences is not None and max_sequences < len(blocks):
        blocks = blocks[:max_sequences]

    dataset = TokenBlockDataset(blocks)
    summary = DatasetSummary(
        dataset=config.dataset,
        subset=config.subset,
        split=split,
        num_examples=int(metadata["num_documents"]),
        num_tokens=len(blocks) * config.sequence_length,
        sequence_length=config.sequence_length,
        num_sequences=len(blocks),
        # Fingerprint the windows actually used, not the whole prepared stream: a run truncated
        # to 128 windows did not see the same data as one truncated to 512.
        fingerprint=fingerprint_token_ids(dataset.token_ids()),
    )
    LOGGER.info(
        "Built %s/%s: %d sequences of %d tokens (fingerprint %s)",
        config.dataset,
        split,
        summary.num_sequences,
        summary.sequence_length,
        summary.fingerprint,
    )
    return dataset, summary


def build_dataloader(
    dataset: TokenBlockDataset,
    config: DataConfig,
    *,
    batch_size: int | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> DataLoader:
    """Wrap a dataset in a deterministic data loader.

    Args:
        dataset: A chunked language-modelling dataset.
        config: Data section of an experiment config.
        batch_size: Overrides ``config.batch_size``.
        shuffle: Whether to shuffle. Evaluation never shuffles.
        seed: Seed for the shuffle generator. Required when ``shuffle`` is true, so that a
            shuffled order is always reproducible.
        drop_last: Drop a final partial batch. Use during training to keep step counts identical
            across arms.

    Returns:
        The data loader.

    Raises:
        DataError: If ``shuffle`` is requested without a seed, or torch is unavailable.
    """
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise DataError("PyTorch is required to build a data loader") from error

    if shuffle and seed is None:
        raise DataError(
            "build_dataloader(shuffle=True) requires an explicit seed; an unseeded shuffle would "
            "make the batch order irreproducible and the arms incomparable."
        )

    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))  # type: ignore[arg-type]

    from scale_aware_compression.seed import seed_worker

    return DataLoader(
        dataset,  # type: ignore[arg-type]
        batch_size=batch_size or config.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=config.num_workers,
        collate_fn=collate_token_blocks,
        generator=generator,
        worker_init_fn=seed_worker if config.num_workers > 0 else None,
    )


def build_evaluation_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int | None = None,
    max_samples: int | None = None,
    cache_root: Any = None,
) -> tuple[DataLoader, DatasetSummary]:
    """Build the evaluation loader used for every arm at a given scale.

    Never shuffles and never drops a batch, so every arm sees the identical token stream in the
    identical order.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        batch_size: Overrides ``config.batch_size``.
        max_samples: Overrides ``config.max_eval_samples``.
        cache_root: Cache root for prepared tokens.

    Returns:
        The loader and its dataset summary.

    Raises:
        DataError: If the evaluation split cannot be prepared.
    """
    limit = max_samples if max_samples is not None else config.max_eval_samples
    dataset, summary = build_language_modelling_dataset(
        config,
        tokenizer,
        config.eval_split,
        max_sequences=limit,
        cache_root=cache_root,
    )
    loader = build_dataloader(dataset, config, batch_size=batch_size, shuffle=False)
    LOGGER.info(
        "Evaluation loader: %d sequences, fingerprint %s -- check this matches the dense "
        "baseline before comparing arms",
        summary.num_sequences,
        summary.fingerprint,
    )
    return loader, summary
