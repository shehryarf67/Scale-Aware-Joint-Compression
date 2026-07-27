"""Tokenisation, chunking, and corpus fingerprinting.

Status: placeholder for the tokenisation paths; :func:`fingerprint_token_ids` and
:func:`chunk_sequence` are implemented, since they are pure functions and the fingerprint is
what makes cross-arm comparability checkable.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
from scale_aware_compression.constants import DEFAULT_DATA_DIR
from scale_aware_compression.data.errors import DataError
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

_FINGERPRINT_WINDOW = 4096
"""Ids hashed from each end of the stream. Hashing the whole stream is unnecessary: a
difference in evaluation data always shows up in the length or the boundaries."""


def fingerprint_token_ids(token_ids: Sequence[int], *, window: int = _FINGERPRINT_WINDOW) -> str:
    """Stable short hash of a token stream.

    Recorded per run so two records can be checked for having been evaluated on identical
    data. Comparing a joint arm against a sequential arm with different fingerprints is a
    measurement error, not a result.

    Args:
        token_ids: The token stream.
        window: How many ids to hash from each end.

    Returns:
        A 16-character hex digest.

    Raises:
        ValueError: If ``window`` is not positive.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    digest = hashlib.sha256()
    digest.update(str(len(token_ids)).encode("utf-8"))
    head = token_ids[:window]
    tail = token_ids[-window:] if len(token_ids) > window else ()
    for chunk in (head, tail):
        digest.update(b"|")
        digest.update(",".join(str(value) for value in chunk).encode("utf-8"))
    return digest.hexdigest()[:16]


def chunk_sequence(
    token_ids: Sequence[int],
    sequence_length: int,
    *,
    drop_last: bool = True,
) -> list[list[int]]:
    """Split a token stream into fixed-length blocks.

    Args:
        token_ids: The token stream.
        sequence_length: Block length; must be positive.
        drop_last: Drop a final short block. Kept true for evaluation, because a ragged block
            changes the per-token perplexity denominator between runs.

    Returns:
        The blocks, in order.

    Raises:
        ValueError: If ``sequence_length`` is not positive.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be > 0, got {sequence_length}")
    blocks = [
        list(token_ids[start : start + sequence_length])
        for start in range(0, len(token_ids), sequence_length)
    ]
    if drop_last and blocks and len(blocks[-1]) < sequence_length:
        blocks.pop()
    return blocks


DOCUMENT_SEPARATOR = "\n\n"
"""Joined between documents before tokenisation.

WikiText ships as many short lines rather than whole articles, so joining with the EOS token
would insert thousands of spurious document boundaries into what is really continuous prose.
A blank line is what the corpus itself uses, and it is what the standard perplexity recipes for
this dataset do.
"""


def tokenise_corpus(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = False,
    separator: str = DOCUMENT_SEPARATOR,
) -> list[int]:
    """Tokenise a corpus into one continuous id stream.

    The whole corpus is joined and tokenised in a single call rather than document by document.
    That keeps the result exactly reproducible: batching the documents would leave the tokeniser
    with different context at each batch boundary, and byte-pair merges near those boundaries
    could differ between runs with different batch sizes.

    Args:
        texts: Documents to concatenate. Empty and whitespace-only entries are dropped, which
            matters for WikiText: roughly a third of its lines are blank.
        tokenizer: Tokeniser for the model under test.
        add_special_tokens: Whether to insert BOS/EOS. Off by default: for continuous
            language-modelling evaluation, special tokens inserted at arbitrary chunk
            boundaries change perplexity without changing the underlying text.
        separator: String joined between documents.

    Returns:
        The concatenated token ids.

    Raises:
        DataError: If the corpus is empty once blank documents are removed.
    """
    kept = [text for text in texts if text and text.strip()]
    if not kept:
        raise DataError("Corpus is empty after dropping blank documents")

    joined = separator.join(kept)
    # The fast tokeniser warns when a sequence exceeds the model's positional limit. That limit
    # is irrelevant here: the stream is chunked into windows immediately afterwards, and nothing
    # is fed to the model at this length.
    encoded = tokenizer(joined, add_special_tokens=add_special_tokens)
    token_ids: list[int] = list(encoded["input_ids"])

    LOGGER.info(
        "Tokenised %d document(s) (%d dropped as blank) into %d tokens",
        len(kept),
        len(texts) - len(kept),
        len(token_ids),
    )
    return token_ids


def processed_cache_dir(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    *,
    root: Path | None = None,
) -> Path:
    """Return the cache directory for one prepared split.

    The tokeniser name is part of the key. Pythia and Qwen produce different token streams from
    identical text, so a cache shared between them would silently serve the wrong tokens.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.
        root: Cache root. Defaults to ``<project>/data/processed``.

    Returns:
        The directory path. Not created.
    """
    base = Path(root) if root is not None else DEFAULT_DATA_DIR / "processed"
    parts = [
        _slug(config.dataset),
        _slug(config.subset) if config.subset else "default",
        _slug(split),
        _slug(_tokenizer_key(tokenizer)),
        f"len{config.sequence_length}",
    ]
    return base / "_".join(parts)


def prepare_dataset(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    *,
    root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Tokenise, chunk, and cache one split.

    The cache is content-addressed by dataset, subset, split, tokeniser, and sequence length, so
    a rerun with any of those changed builds a new entry rather than serving stale tokens.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.
        root: Cache root. Defaults to ``<project>/data/processed``.
        force: Rebuild even when a matching cache exists.

    Returns:
        Metadata for the prepared split: token and block counts, the fingerprint, and the paths
        written.

    Raises:
        DataError: If the split cannot be loaded or yields no complete window.
    """
    from scale_aware_compression.data.loaders import load_raw_dataset

    directory = processed_cache_dir(config, tokenizer, split, root=root)
    metadata_path = directory / "metadata.json"
    tokens_path = directory / "tokens.json"

    if metadata_path.is_file() and not force:
        cached: dict[str, Any] = json.loads(metadata_path.read_text(encoding="utf-8"))
        LOGGER.info(
            "Reusing cached %s split: %d blocks, fingerprint %s",
            split,
            cached.get("num_blocks"),
            cached.get("fingerprint"),
        )
        return cached

    dataset = load_raw_dataset(config, split)
    try:
        texts = list(dataset[config.text_column])
    except (KeyError, TypeError) as error:
        raise DataError(
            f"Column {config.text_column!r} not found in {config.dataset}/{split}. "
            f"Available: {getattr(dataset, 'column_names', 'unknown')}"
        ) from error

    token_ids = tokenise_corpus(texts, tokenizer)
    blocks = chunk_sequence(token_ids, config.sequence_length)
    if not blocks:
        raise DataError(
            f"{config.dataset}/{split} yielded {len(token_ids)} tokens, which is fewer than one "
            f"window of {config.sequence_length}. Use a shorter sequence_length or a larger split."
        )

    metadata = {
        "dataset": config.dataset,
        "subset": config.subset,
        "split": split,
        "text_column": config.text_column,
        "tokenizer": _tokenizer_key(tokenizer),
        "sequence_length": config.sequence_length,
        "num_documents": len(texts),
        "num_tokens": len(token_ids),
        "num_blocks": len(blocks),
        "fingerprint": fingerprint_token_ids(token_ids),
        "tokens_path": tokens_path.as_posix(),
    }

    directory.mkdir(parents=True, exist_ok=True)
    # Plain JSON rather than .npy: it needs no NumPy to read, diffs legibly when something looks
    # wrong, and these are at most a few million small integers.
    tokens_path.write_text(json.dumps(token_ids), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    LOGGER.info(
        "Prepared %s/%s: %d tokens -> %d blocks of %d (fingerprint %s)",
        config.dataset,
        split,
        len(token_ids),
        len(blocks),
        config.sequence_length,
        metadata["fingerprint"],
    )
    return metadata


def load_prepared_tokens(
    config: DataConfig,
    tokenizer: PreTrainedTokenizerBase,
    split: str,
    *,
    root: Path | None = None,
) -> list[int]:
    """Read a previously prepared token stream from the cache.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.
        root: Cache root. Defaults to ``<project>/data/processed``.

    Returns:
        The cached token ids.

    Raises:
        DataError: If no cache entry exists for this combination.
    """
    directory = processed_cache_dir(config, tokenizer, split, root=root)
    tokens_path = directory / "tokens.json"
    if not tokens_path.is_file():
        raise DataError(
            f"No prepared tokens at {tokens_path}. Run prepare_dataset() (or "
            "scripts/prepare_data.py) first."
        )
    loaded: list[int] = json.loads(tokens_path.read_text(encoding="utf-8"))
    return loaded


def _tokenizer_key(tokenizer: PreTrainedTokenizerBase) -> str:
    """Return a short, stable identifier for a tokeniser."""
    for attribute in ("name_or_path", "_name_or_path"):
        value = getattr(tokenizer, attribute, None)
        if value:
            return str(value)
    return type(tokenizer).__name__


def _slug(value: str) -> str:
    """Reduce a string to a filename-safe slug."""
    return "".join(character if character.isalnum() else "-" for character in str(value)).strip("-")
