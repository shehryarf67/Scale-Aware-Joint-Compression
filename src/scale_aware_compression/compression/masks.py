"""Pruning mask storage and bookkeeping.

Masks are kept separate from the weights during optimisation, because both arms need to
re-apply them after every optimiser step: an optimiser with momentum or weight decay will
otherwise reintroduce non-zero values into pruned positions, and the model's measured sparsity
will drift below its target without anything failing.

:class:`MaskSet` and the counting helpers are implemented, since they are pure bookkeeping and
worth testing. The tensor-level mask construction is a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.constants import PruningGranularity
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import sparsity_percentage

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class MaskStatistics:
    """Element counts for one mask, or for a whole :class:`MaskSet`."""

    total_elements: int
    pruned_elements: int

    @property
    def kept_elements(self) -> int:
        """Elements the mask keeps."""
        return self.total_elements - self.pruned_elements

    @property
    def sparsity_percentage(self) -> float:
        """Percentage of elements the mask removes."""
        if self.total_elements == 0:
            return 0.0
        return sparsity_percentage(self.total_elements, self.pruned_elements)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "total_elements": self.total_elements,
            "pruned_elements": self.pruned_elements,
            "kept_elements": self.kept_elements,
            "sparsity_percentage": self.sparsity_percentage,
        }


@dataclass(slots=True)
class MaskSet:
    """Binary masks for every compressible weight tensor in a model.

    Attributes:
        masks: Module name to boolean mask tensor, ``True`` meaning "keep".
        granularity: Sparsity pattern the masks follow.
        target_sparsity: Sparsity the masks were built for, kept so the realised value can be
            checked against the intent.
        counts: Per-module element counts, maintained alongside the tensors so reporting does
            not have to re-scan them.
    """

    masks: dict[str, torch.Tensor] = field(default_factory=dict)
    granularity: PruningGranularity = PruningGranularity.UNSTRUCTURED
    target_sparsity: float = 0.0
    counts: dict[str, MaskStatistics] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.masks)

    def __contains__(self, name: str) -> bool:
        return name in self.masks

    @property
    def module_names(self) -> list[str]:
        """Names of the masked modules, in insertion order."""
        return list(self.masks)

    def add(self, name: str, mask: torch.Tensor, statistics: MaskStatistics) -> None:
        """Register a mask and its counts.

        Args:
            name: Module name the mask belongs to.
            mask: Boolean tensor, ``True`` meaning keep.
            statistics: Element counts for this mask.
        """
        self.masks[name] = mask
        self.counts[name] = statistics

    def total_statistics(self) -> MaskStatistics:
        """Aggregate counts across every registered mask.

        Returns:
            Combined statistics. All-zero when the set is empty.
        """
        return MaskStatistics(
            total_elements=sum(item.total_elements for item in self.counts.values()),
            pruned_elements=sum(item.pruned_elements for item in self.counts.values()),
        )

    def report(self) -> dict[str, Any]:
        """Summarise the mask set for the run record.

        Returns:
            Mapping with the aggregate counts, the target sparsity, and per-module sparsity.
        """
        total = self.total_statistics()
        return {
            "granularity": self.granularity.value,
            "target_sparsity": self.target_sparsity,
            "num_masked_modules": len(self.masks),
            **total.to_dict(),
            "per_module_sparsity_percentage": {
                name: item.sparsity_percentage for name, item in self.counts.items()
            },
        }


def build_masks(
    weights: dict[str, torch.Tensor],
    *,
    sparsity: float,
    granularity: PruningGranularity = PruningGranularity.UNSTRUCTURED,
    global_ranking: bool = False,
) -> MaskSet:
    """Build pruning masks for a set of weight tensors.

    Args:
        weights: Module name to weight tensor.
        sparsity: Target fraction to prune, in ``[0, 1)``.
        granularity: Sparsity pattern to produce.
        global_ranking: Rank magnitudes across all tensors at once rather than per tensor.
            Global ranking usually gives better quality at the same overall sparsity but
            leaves individual layers at very different sparsities, which changes what the
            latency measurement means.

    Returns:
        The constructed mask set.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(masks): implement per-granularity mask construction.
    #   unstructured    : threshold |w| at the sparsity-th quantile, per tensor or globally
    #   2:4 / 4:8       : reshape the last dim into groups of 4/8 and keep the top-k by
    #                     magnitude within each group; this is the only pattern with CPU
    #                     kernel support, so it is the one that can show real latency gains
    #   channel         : rank output channels by L2 norm and drop whole rows
    # For GPT-NeoX, remember that attention.query_key_value fuses q/k/v: a channel pattern
    # must be applied consistently across all three slices or the heads break.
    raise NotImplementedError(
        "build_masks is not implemented yet; see the TODO in compression/masks.py"
    )


def apply_masks(model: nn.Module, mask_set: MaskSet) -> None:
    """Zero the masked-out weights in place.

    Must be called after every optimiser step during gradual pruning, not just once, or
    momentum and weight decay will refill pruned positions.

    Args:
        model: The model whose weights to mask.
        mask_set: Masks to apply, keyed by module name.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(masks): under torch.no_grad(), multiply each module's weight by its mask. Warn on
    # names present in mask_set but absent from the model, which means the mask set is stale.
    raise NotImplementedError(
        "apply_masks is not implemented yet; see the TODO in compression/masks.py"
    )


def register_mask_hooks(model: nn.Module, mask_set: MaskSet) -> list[Any]:
    """Attach gradient hooks that keep pruned positions at zero during training.

    Args:
        model: The model to instrument.
        mask_set: Masks to enforce.

    Returns:
        The registered hook handles, so the caller can remove them before conversion.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(masks): register a per-parameter hook zeroing masked gradient entries. Prefer this
    # over post-step re-masking: it stops the optimiser accumulating momentum in pruned
    # positions, which otherwise produces a large weight update the moment a mask changes.
    raise NotImplementedError(
        "register_mask_hooks is not implemented yet; see the TODO in compression/masks.py"
    )


def fold_masks_into_weights(model: nn.Module, mask_set: MaskSet) -> nn.Module:
    """Bake masks into the weights and detach the mask bookkeeping.

    Called during conversion, so the deployed artefact carries no mask tensors: keeping them
    would add one byte per weight to the checkpoint and distort the measured compression
    ratio.

    Args:
        model: The model to finalise.
        mask_set: Masks to fold in.

    Returns:
        The model with masks applied and hooks removed.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(masks): apply masks one final time, remove hooks, delete any *_mask buffers, and
    # verify the measured sparsity matches mask_set.target_sparsity before returning.
    raise NotImplementedError(
        "fold_masks_into_weights is not implemented yet; see the TODO in compression/masks.py"
    )
