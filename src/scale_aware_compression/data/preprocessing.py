"""Tokenisation, chunking, and corpus fingerprinting.

Status: placeholder for the tokenisation paths; :func:`fingerprint_token_ids` and
:func:`chunk_sequence` are implemented, since they are pure functions and the fingerprint is
what makes cross-arm comparability checkable.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import DataConfig
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


def tokenise_corpus(
    texts: Sequence[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    add_special_tokens: bool = False,
) -> list[int]:
    """Tokenise a corpus into one continuous id stream.

    Args:
        texts: Documents to concatenate.
        tokenizer: Tokeniser for the model under test.
        add_special_tokens: Whether to insert BOS/EOS. Off by default: for continuous
            language-modelling evaluation, special tokens inserted at arbitrary chunk
            boundaries change perplexity without changing the underlying text.

    Returns:
        The concatenated token ids.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(preprocessing): batch-tokenise with the fast tokeniser and flatten. Join documents
    # with the eos token rather than raw concatenation, so document boundaries are marked;
    # do it identically for every arm.
    raise NotImplementedError(
        "tokenise_corpus is not implemented yet; see the TODO in data/preprocessing.py"
    )


def prepare_dataset(config: DataConfig, tokenizer: PreTrainedTokenizerBase, split: str) -> Any:
    """Tokenise, chunk, and cache one split to ``data/processed``.

    Args:
        config: Data section of an experiment config.
        tokenizer: Tokeniser for the model under test.
        split: Split name.

    Returns:
        The prepared dataset.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(preprocessing): cache under data/processed/<dataset>_<subset>_<split>_
    # <tokenizer-name>_<sequence_length>/, keyed by tokeniser because Pythia and Qwen produce
    # different streams from identical text. Write the fingerprint alongside as metadata.json.
    raise NotImplementedError(
        "prepare_dataset is not implemented yet; see the TODO in data/preprocessing.py"
    )
