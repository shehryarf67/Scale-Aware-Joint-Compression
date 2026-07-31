"""Perplexity evaluation.

Perplexity is the primary quality metric for this study: it needs no task-specific head, it is
defined identically for every model in the sweep, and it is sensitive enough to register the
small degradations that separate a joint arm from a sequential one at moderate compression.

Final reported perplexity is computed on CPU, per the evaluation policy.

Status: the pure aggregation functions are implemented; the model-evaluation path is a
placeholder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import EvaluationConfig
from scale_aware_compression.evaluation.common import EvaluationError, check_evaluation_device
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from torch.utils.data import DataLoader

LOGGER = get_logger(__name__)

MAX_REPRESENTABLE_PERPLEXITY = 1e30
"""Cap for reporting. A perplexity above this means the model is broken, not merely worse, and
``inf`` in a CSV column breaks downstream aggregation."""


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """Perplexity together with the token count it was computed over."""

    perplexity: float
    total_nll: float
    total_tokens: int
    num_sequences: int
    sequence_length: int
    device: str
    dataset_fingerprint: str | None = None
    """Fingerprint of the evaluation token stream. Two perplexities with different
    fingerprints were not measured on the same data and must not be compared."""
    window_nll: list[float] = field(default_factory=list)
    """Summed NLL for each evaluation window, in loader order.

    The unit of resampling for the paired block bootstrap A1 §5.1 requires. Neighbouring tokens are
    dependent, so a token-level bootstrap understates the interval; complete windows are the block.
    Per **sequence** rather than per batch, so the block structure is invariant to batch size and a
    160M run at batch 4 is comparable with a 410M run at batch 2.

    Two arms must be bootstrapped over the **same resampled window indices**, which is what makes the
    comparison paired -- hence loader order matters and the loader never shuffles.
    """
    window_tokens: list[int] = field(default_factory=list)
    """Predicted-token count for each window, parallel to :attr:`window_nll`."""

    @property
    def bits_per_token(self) -> float:
        """Cross-entropy in bits, an alternative scale for the same quantity."""
        return (
            (self.total_nll / self.total_tokens) / math.log(2)
            if self.total_tokens
            else float("nan")
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "perplexity": self.perplexity,
            "total_nll": self.total_nll,
            "total_tokens": self.total_tokens,
            "num_sequences": self.num_sequences,
            "sequence_length": self.sequence_length,
            "bits_per_token": self.bits_per_token,
            "evaluation_device": self.device,
            "dataset_fingerprint": self.dataset_fingerprint,
            "window_nll": list(self.window_nll),
            "window_tokens": list(self.window_tokens),
        }


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Convert a summed negative log-likelihood into perplexity.

    Summing NLL over tokens and exponentiating once at the end is not the same as averaging
    per-batch perplexities: the latter over-weights short batches. This function exists so the
    correct aggregation is used everywhere.

    Args:
        total_nll: Sum of per-token negative log-likelihoods, in nats.
        total_tokens: Number of tokens the sum covers; must be positive.

    Returns:
        ``exp(total_nll / total_tokens)``, capped at :data:`MAX_REPRESENTABLE_PERPLEXITY`.

    Raises:
        ValueError: If ``total_tokens`` is not positive or ``total_nll`` is negative.
    """
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be > 0, got {total_tokens}")
    if total_nll < 0:
        raise ValueError(f"total_nll must be >= 0, got {total_nll}")
    mean_nll = total_nll / total_tokens
    if mean_nll > math.log(MAX_REPRESENTABLE_PERPLEXITY):
        LOGGER.warning(
            "Mean NLL of %.2f nats exceeds the reporting cap; the model has likely collapsed.",
            mean_nll,
        )
        return MAX_REPRESENTABLE_PERPLEXITY
    return math.exp(mean_nll)


def aggregate_nll(batch_nlls: list[float], batch_token_counts: list[int]) -> tuple[float, int]:
    """Sum per-batch NLL and token counts.

    Args:
        batch_nlls: Summed NLL per batch, in nats.
        batch_token_counts: Token count per batch.

    Returns:
        ``(total_nll, total_tokens)``.

    Raises:
        ValueError: If the two lists have different lengths, or either is empty.
    """
    if len(batch_nlls) != len(batch_token_counts):
        raise ValueError(
            f"Got {len(batch_nlls)} NLL values but {len(batch_token_counts)} token counts"
        )
    if not batch_nlls:
        raise ValueError("aggregate_nll requires at least one batch")
    return sum(batch_nlls), sum(batch_token_counts)


def compute_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    config: EvaluationConfig,
    *,
    dataset_fingerprint: str | None = None,
    progress_every: int = 20,
) -> PerplexityResult:
    """Evaluate perplexity over a data loader.

    NLL is summed over every predicted token and exponentiated once at the end. Averaging
    per-batch perplexities instead would over-weight short batches and give a different, wrong
    answer.

    Args:
        model: The model to evaluate. On CPU for a number that will be reported.
        dataloader: Fixed-length evaluation batches, unshuffled.
        config: Evaluation section of an experiment config.
        dataset_fingerprint: Fingerprint of the evaluation stream, stored in the result so two
            perplexities can be checked for having been measured on the same data.
        progress_every: Log progress every N batches.

    Returns:
        The perplexity result.

    Raises:
        EvaluationError: If the loader is empty, or the model produces no usable logits.
        NotImplementedError: If ``config.stride`` is set. Sliding-window perplexity needs
            overlapping windows built at chunking time, which the current data pipeline does not
            produce; a stride silently ignored here would make results incomparable with
            published numbers that use one.
    """
    import torch
    import torch.nn.functional as functional

    if config.stride is not None:
        raise NotImplementedError(
            f"evaluation.stride={config.stride} is not supported yet. Only non-overlapping "
            "windows (stride: null) are implemented. Sliding-window perplexity requires "
            "overlapping windows from data/preprocessing.py; implement it there rather than "
            "approximating it here, since strided and non-strided perplexities are not "
            "comparable."
        )

    check_evaluation_device(config)

    device = _model_device(model)
    was_training = model.training
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    num_sequences = 0
    sequence_length = 0
    num_batches = 0
    window_nll: list[float] = []
    window_tokens: list[int] = []

    try:
        with torch.inference_mode():
            for index, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                if input_ids.ndim != 2 or input_ids.shape[1] < 2:
                    raise EvaluationError(
                        f"Evaluation batches must be (batch, sequence>=2); got "
                        f"{tuple(input_ids.shape)}. A window of one token has no target to "
                        "predict."
                    )

                outputs = model(
                    input_ids=input_ids, attention_mask=batch["attention_mask"].to(device)
                )
                logits = getattr(outputs, "logits", None)
                if logits is None:
                    raise EvaluationError(
                        f"{type(model).__name__} returned no `logits`; perplexity needs a causal "
                        "language-modelling head."
                    )

                # The final position predicts nothing inside this window, and the first token has
                # no predecessor, so a window of L tokens contributes L-1 predictions.
                shift_logits = logits[:, :-1, :].float()
                shift_labels = input_ids[:, 1:]
                # `reduction="none"` then summed per row, rather than a single summed scalar, so each
                # evaluation window's NLL is retained. A1 §5.1 requires a paired block bootstrap that
                # resamples COMPLETE WINDOWS -- neighbouring tokens are dependent, so a token-level
                # bootstrap understates the interval. Aggregates alone make that analysis impossible,
                # and discovering it after the ~38 h confirmatory stage would mean re-running all of it.
                #
                # Per SEQUENCE, not per batch, deliberately: batch size differs between configs (4 at
                # 160M, 2 at 410M for memory), and a per-batch block structure would not be comparable
                # across scales. Per-sequence blocks are invariant to batching.
                token_nll = functional.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    reduction="none",
                )
                per_window = token_nll.reshape(shift_labels.shape).sum(dim=1)

                window_nll.extend(float(value) for value in per_window)
                window_tokens.extend([int(shift_labels.shape[1])] * int(shift_labels.shape[0]))
                total_nll += float(per_window.sum().item())
                total_tokens += int(shift_labels.numel())
                num_sequences += int(input_ids.shape[0])
                sequence_length = int(input_ids.shape[1])
                num_batches += 1

                if progress_every and index and index % progress_every == 0:
                    running = perplexity_from_nll(total_nll, total_tokens)
                    LOGGER.debug("  batch %d: running perplexity %.3f", index, running)
    finally:
        if was_training:
            model.train()

    if num_batches == 0:
        raise EvaluationError("Evaluation loader produced no batches")

    result = PerplexityResult(
        perplexity=perplexity_from_nll(total_nll, total_tokens),
        total_nll=total_nll,
        total_tokens=total_tokens,
        num_sequences=num_sequences,
        sequence_length=sequence_length,
        device=str(device),
        dataset_fingerprint=dataset_fingerprint,
        window_nll=window_nll,
        window_tokens=window_tokens,
    )
    # The blocks must reconstruct the aggregate, or a bootstrap over them estimates the uncertainty of
    # a different quantity from the one reported. Checked rather than assumed, because the two are now
    # computed by different code paths.
    #
    # The tolerance is relative 1e-5 and cannot be tighter. `total_nll` accumulates one **float32**
    # tensor reduction per batch, while `block_total` sums the extracted per-window values in float64,
    # so the two differ by float32 association error that grows with batch size. Deliberately NOT fixed
    # by recomputing `total_nll` from the blocks: that would be marginally more accurate but would
    # change every reported perplexity, and the 18 records already written today would need ~3 hours of
    # recompute for a ~1e-7 relative difference -- smaller than the thread-configuration sensitivity
    # already documented in F-23.
    #
    # This tolerance still catches what the guard is for: a wrong reshape, a dropped batch, or
    # double-counted windows are off by orders of magnitude, not by float32 noise.
    block_total = sum(window_nll)
    if abs(block_total - total_nll) > max(1e-4, 1e-5 * abs(total_nll)):
        raise EvaluationError(
            f"per-window NLL sums to {block_total!r} but the aggregate is {total_nll!r}. The block "
            "decomposition does not reconstruct the reported perplexity, so a paired block bootstrap "
            "over these windows would not describe the reported number."
        )
    if sum(window_tokens) != total_tokens:
        raise EvaluationError(
            f"per-window tokens sum to {sum(window_tokens)} but the aggregate is {total_tokens}"
        )
    LOGGER.info(
        "Perplexity %.4f over %d tokens (%d sequences of %d) on %s",
        result.perplexity,
        result.total_tokens,
        result.num_sequences,
        result.sequence_length,
        result.device,
    )
    return result


def _model_device(model: nn.Module) -> Any:
    """Return the device a model's parameters live on, defaulting to CPU."""
    import torch

    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.device("cpu")
