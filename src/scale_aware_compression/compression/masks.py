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

from scale_aware_compression.constants import MaskComparisonGroup, PruningGranularity
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


class MaskError(ValueError):
    """Raised when a mask cannot be built at the requested sparsity or pattern."""


_SEMI_STRUCTURED_GROUPS: dict[PruningGranularity, tuple[int, int]] = {
    PruningGranularity.SEMI_STRUCTURED_2_4: (4, 2),
    PruningGranularity.SEMI_STRUCTURED_4_8: (8, 4),
}
"""Granularity to ``(group_size, kept_per_group)``. Both patterns are 50% sparse by construction."""


def build_mask_from_scores(
    scores: torch.Tensor,
    *,
    sparsity: float,
    granularity: PruningGranularity = PruningGranularity.UNSTRUCTURED,
    comparison_group: MaskComparisonGroup = MaskComparisonGroup.OUTPUT,
) -> torch.Tensor:
    """Build a boolean keep-mask for one weight tensor from its saliency scores.

    The realised sparsity is **exact**, not approximate: the number of pruned entries is computed
    up front and that many are removed. A quantile threshold would miss the target whenever scores
    tie -- and under activation-weighted saliency a dead input column makes an entire column of
    scores exactly zero, so ties are common rather than pathological.

    Args:
        scores: Non-negative saliency, same shape as the weight. Larger means more important.
        sparsity: Fraction to prune, in ``[0, 1)``.
        granularity: Sparsity pattern. Unstructured ranks freely; the semi-structured patterns rank
            within fixed groups along the last dimension, which is what admits a sparse kernel if
            the backend provides one.
        comparison_group: Which weights compete for survival. Applies to unstructured only -- the
            semi-structured patterns define their own groups. Defaults to
            :data:`~scale_aware_compression.constants.MaskComparisonGroup.OUTPUT`, which is
            per-output-channel, on measured grounds: ranking across the whole tensor let
            activation-weighted saliency delete entire low-energy input columns and cost 6.7x
            perplexity on Pythia-160M at 50% sparsity.

    Returns:
        Boolean tensor, ``True`` meaning keep.

    Raises:
        MaskError: If sparsity is out of range, the pattern is unsupported, or a
            semi-structured pattern does not divide the last dimension.
    """
    import torch

    if not 0.0 <= sparsity < 1.0:
        raise MaskError(f"sparsity must be in [0, 1), got {sparsity}")
    if granularity is PruningGranularity.STRUCTURED_CHANNEL:
        raise MaskError(
            "channel-structured pruning is optional future work per plan §3.10: it changes tensor "
            "shapes and needs a different runtime. Use unstructured or a semi-structured pattern."
        )

    if granularity in _SEMI_STRUCTURED_GROUPS:
        group_size, kept = _SEMI_STRUCTURED_GROUPS[granularity]
        implied = 1.0 - kept / group_size
        if abs(sparsity - implied) > 1e-9:
            raise MaskError(
                f"granularity {granularity.value!r} implies sparsity {implied}, got {sparsity}"
            )
        if scores.shape[-1] % group_size != 0:
            raise MaskError(
                f"granularity {granularity.value!r} needs the last dimension divisible by "
                f"{group_size}, got {scores.shape[-1]}"
            )
        grouped = scores.reshape(-1, group_size)
        mask = torch.zeros_like(grouped, dtype=torch.bool)
        keep_indices = grouped.topk(kept, dim=1).indices
        mask.scatter_(1, keep_indices, True)
        return mask.reshape(scores.shape)

    if granularity is not PruningGranularity.UNSTRUCTURED:
        raise MaskError(f"unsupported granularity {granularity!r}")

    if comparison_group is MaskComparisonGroup.OUTPUT:
        if scores.ndim != 2:
            raise MaskError(
                f"per-output comparison needs a 2-D (out_features, in_features) score tensor, got "
                f"{tuple(scores.shape)}"
            )
        per_row = scores.shape[1]
        num_kept = per_row - round(per_row * sparsity)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        if num_kept:
            mask.scatter_(1, scores.topk(num_kept, dim=1).indices, True)
        return mask

    if comparison_group is not MaskComparisonGroup.TENSOR:
        raise MaskError(f"unsupported comparison group {comparison_group!r}")

    flat = scores.reshape(-1)
    num_pruned = round(flat.numel() * sparsity)
    num_kept = flat.numel() - num_pruned
    mask = torch.zeros_like(flat, dtype=torch.bool)
    if num_kept:
        mask[flat.topk(num_kept).indices] = True
    return mask.reshape(scores.shape)


def realised_sparsity(mask: torch.Tensor) -> float:
    """Fraction of entries a mask prunes.

    Reported next to the target everywhere, per the project rule that measured and requested
    values always appear together.

    Args:
        mask: Boolean keep-mask.

    Returns:
        Pruned fraction in ``[0, 1]``; ``0.0`` for an empty mask.
    """
    total = mask.numel()
    if total == 0:
        return 0.0
    return float((~mask).sum()) / total


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
    # TODO(masks): this is the *model-level* driver and belongs to Phase 6, not Phase 5. The
    # per-tensor work is already done in build_mask_from_scores(); what is missing is the plumbing
    # around it, and the signature has to change first:
    #
    #   * saliency is activation-weighted magnitude (plan §3.3), so this needs the per-layer
    #     column norms from compression.activations, not just the weights. Ranking on |w| alone is
    #     the superseded design.
    #   * the joint arm scores on QUANTISED weights (§3.8, decision D3), so the caller must be
    #     able to pass in already-fake-quantised tensors. That is the whole difference between the
    #     arms and it cannot be bolted on afterwards.
    #   * global_ranking has to concatenate scores across layers before the top-k, which changes
    #     what a per-layer latency measurement means -- hence it is not the default.
    #
    # Implement it as part of the layerwise driver so there is exactly one call path, then have
    # this delegate per tensor to build_mask_from_scores.
    raise NotImplementedError(
        "build_masks is not implemented yet; see the TODO in compression/masks.py. The per-tensor "
        "primitive build_mask_from_scores is implemented and tested."
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
