"""Quantisation-only arm.

Pipeline::

    dense model -> calibration -> quantisation -> conversion

Post-training quantisation, so no recovery fine-tuning by default: the point of this arm is
to isolate the effect of reduced precision alone, as a reference for the sequential and joint
arms.

Status: placeholder. Stage methods raise :class:`NotImplementedError`;
:meth:`Quantiser.report_statistics` is implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import Compressor
from scale_aware_compression.constants import (
    BITS_PER_BYTE,
    EPSILON,
    FP32_BITS,
    CompressionMethod,
    CompressionStage,
    QuantisationGranularity,
)
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import count_parameters, theoretical_size_bytes

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

PACKABLE_BITS: tuple[int, ...] = (2, 4, 8)
"""Bit widths this module can pack losslessly.

Only widths that divide a byte are supported. A 3-bit code would straddle byte boundaries, and an
unpacking bug there is exactly the kind of silent corruption §4.8 asks to be verified rather than
assumed. W8 and W4 are the two the study screens (§3.9), so nothing needed is missing.
"""


class QuantisationError(ValueError):
    """Raised when a quantisation request is unsatisfiable or a round-trip is inconsistent."""


def _symmetric_qmax(bits: int) -> int:
    """Return the largest representable code magnitude for a symmetric ``bits``-wide grid.

    Args:
        bits: Bit width.

    Returns:
        ``2^(bits-1) - 1``. The grid is ``[-qmax, +qmax]``, which is symmetric about zero and
        therefore holds ``2*qmax + 1 = 2^bits - 1`` distinct values -- one fewer than the full
        two's-complement range, in exchange for zero being an exact grid point.

    Raises:
        QuantisationError: If ``bits`` is below 2.
    """
    if bits < 2:
        raise QuantisationError(f"bits must be >= 2, got {bits}")
    return (1 << (bits - 1)) - 1


def _grouped_view(
    weight: torch.Tensor,
    granularity: QuantisationGranularity,
    group_size: int,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    """Reshape a weight into ``(num_groups, group_elements)`` so one scale covers each row.

    Collapsing all three granularities onto one shape means the scale computation, the rounding,
    and the round-trip check are written once rather than three times.

    Args:
        weight: Weight tensor of shape ``(out_features, in_features)``.
        granularity: Scope a single scale covers.
        group_size: Elements per group, used only for ``PER_GROUP``.

    Returns:
        The reshaped view, and the shape the resulting per-group scales should take.

    Raises:
        QuantisationError: If the weight is not 2-D, or the group size does not divide
            ``in_features``.
    """
    if weight.ndim != 2:
        raise QuantisationError(
            f"expected a 2-D (out_features, in_features) weight, got shape {tuple(weight.shape)}"
        )
    out_features, in_features = weight.shape

    if granularity is QuantisationGranularity.PER_TENSOR:
        return weight.reshape(1, -1), (1,)
    if granularity is QuantisationGranularity.PER_CHANNEL:
        return weight.reshape(out_features, in_features), (out_features, 1)
    if granularity is QuantisationGranularity.PER_GROUP:
        if group_size <= 0:
            raise QuantisationError(f"group_size must be > 0, got {group_size}")
        if in_features % group_size != 0:
            raise QuantisationError(
                f"group_size {group_size} does not divide in_features {in_features}; a ragged "
                "final group would get a scale fitted to fewer weights than every other group"
            )
        num_groups = in_features // group_size
        return weight.reshape(out_features * num_groups, group_size), (
            out_features * num_groups,
            1,
        )
    raise QuantisationError(f"unsupported granularity {granularity!r}")


@dataclass(slots=True)
class QuantisedWeight:
    """A weight tensor held as integer codes plus the scales needed to reconstruct it.

    This is the *real* quantised representation, as distinct from fake quantisation: the codes are
    integers and the storage cost is genuinely ``bits`` per weight plus scale overhead. Only this
    representation may back a size or latency measurement.

    Attributes:
        codes: Integer codes in ``[-qmax, +qmax]``, stored as ``int8`` and shaped like the weight.
        scales: Positive per-group scales, broadcastable against the grouped weight view.
        bits: Bit width the codes occupy once packed.
        granularity: Scope each scale covers.
        group_size: Elements per group; meaningful only for ``PER_GROUP``.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    bits: int
    granularity: QuantisationGranularity
    group_size: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the weight this represents."""
        return tuple(self.codes.shape)

    def dequantise(self) -> torch.Tensor:
        """Reconstruct the floating-point weight from codes and scales.

        Returns:
            A float tensor of the original shape. Equal to what fake quantisation produces, which
            is what makes the fake and real paths checkable against each other.
        """
        grouped, _ = _grouped_view(
            self.codes.to(self.scales.dtype), self.granularity, self.group_size
        )
        return (grouped * self.scales).reshape(self.codes.shape)

    def distinct_values_per_group(self) -> int:
        """Return the largest number of distinct codes any single group uses.

        §4.8 requires confirming the bit width is real rather than silently dequantised. A group
        holding more than ``2^bits`` distinct values proves it is not.

        Returns:
            Maximum distinct code count across groups.
        """
        grouped, _ = _grouped_view(self.codes, self.granularity, self.group_size)
        return max(int(row.unique().numel()) for row in grouped)

    def report(self) -> dict[str, Any]:
        """Summarise the representation for the run record."""
        return {
            "bits": self.bits,
            "granularity": self.granularity.value,
            "group_size": self.group_size,
            "shape": list(self.shape),
            "num_scales": int(self.scales.numel()),
            "max_distinct_values_per_group": self.distinct_values_per_group(),
            "effective_bits_per_weight": effective_bits_per_weight(self),
        }


def compute_symmetric_scales(
    weight: torch.Tensor,
    *,
    bits: int,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
) -> torch.Tensor:
    """Fit symmetric quantisation scales to a weight tensor.

    Args:
        weight: Weight of shape ``(out_features, in_features)``.
        bits: Target bit width.
        granularity: Scope a single scale covers.
        group_size: Elements per group, for ``PER_GROUP``.

    Returns:
        Positive scales, shaped to broadcast against the grouped view of the weight.

    Raises:
        QuantisationError: If the shape or granularity is unsupported.
    """
    import torch

    qmax = _symmetric_qmax(bits)
    grouped, scale_shape = _grouped_view(weight, granularity, group_size)
    amax = grouped.abs().amax(dim=1)
    # An all-zero group has no scale; leave it at 1.0 so it round-trips to zero rather than NaN.
    scales = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    return scales.reshape(scale_shape)


DEFAULT_CLIP_RATIOS: tuple[float, ...] = tuple(round(0.50 + 0.05 * step, 2) for step in range(11))
"""Clipping ratios searched by :func:`search_clipping_scales`: 0.50, 0.55, ... 1.00.

``1.00`` is plain max-abs, so the search can never do worse than not searching.
"""


def search_clipping_scales(
    weight: torch.Tensor,
    gram: torch.Tensor,
    *,
    bits: int,
    mask: torch.Tensor | None = None,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
    candidates: Sequence[float] = DEFAULT_CLIP_RATIOS,
) -> torch.Tensor:
    """Fit scales that minimise layer-output reconstruction error rather than matching ``max|W|``.

    Max-abs scaling spends the whole grid covering the largest weight in a row, so a single outlier
    coarsens every other weight in it. Clipping the outlier costs a large error on one weight and
    saves a small error on many, which is usually a net win -- and it is *measurably* a win here:
    across six real Pythia-160M layers this reduces the layer objective by 12.8% at W4 and 38.5% at
    W3, while correctly selecting no clipping at all (``alpha = 1.00``, 0.0% change) at W8, where
    quantisation error is already negligible.

    It also repairs a hole in the joint method. Max-abs scales are effectively blind to pruning,
    because saliency removes the smallest weights and each row's maximum survives -- measured at 0.2%
    of channels changing on refit. The optimal clipping ratio, in contrast, depends on the whole
    surviving distribution, so re-estimating it after a mask change genuinely moves the grid. That is
    what makes §3.8's "re-estimate scales after mask changes" a live mechanism rather than a formality.

    The search is exact per output channel. The objective
    ``sum_o (w_o - w_hat_o)^T H (w_o - w_hat_o)`` decomposes over output channels, so each row's best
    ratio can be chosen independently without any greedy approximation.

    **Both arms must use this.** A joint arm given a better quantiser than the sequential arm would
    show a "joint gain" that was really a quantiser difference (§3.11).

    Args:
        weight: Dense weight of shape ``(out_features, in_features)``.
        gram: ``H = X^T X`` for this layer's inputs.
        bits: Target bit width.
        mask: Keep-mask to fit the scales under. Passing the current mask is what makes the result
            mask-dependent; ``None`` fits against the dense tensor.
        granularity: Quantisation granularity. Under ``PER_GROUP`` one ratio is chosen per output
            channel and shared by that row's groups, because the objective does not decompose over
            groups -- ``H`` couples columns across group boundaries.
        group_size: Elements per group.
        candidates: Clipping ratios to try.

    Returns:
        Scales shaped for the grouped view, exactly as :func:`compute_symmetric_scales` returns.

    Raises:
        QuantisationError: If shapes are inconsistent or ``candidates`` is empty.
    """
    import torch

    if not candidates:
        raise QuantisationError("candidates must contain at least one clipping ratio")
    if gram.shape[0] != weight.shape[1]:
        raise QuantisationError(
            f"gram width {gram.shape[0]} does not match in_features {weight.shape[1]}"
        )

    keep = torch.ones_like(weight) if mask is None else mask.to(weight.dtype)
    target = weight * keep
    gram32 = gram.to(torch.float32)

    best_losses: torch.Tensor | None = None
    best_scales: torch.Tensor | None = None

    for ratio in candidates:
        scales = compute_symmetric_scales(
            target, bits=bits, granularity=granularity, group_size=group_size
        ) * float(ratio)
        scales = scales.clamp_min(EPSILON)
        candidate = (
            fake_quantise(
                target, bits=bits, granularity=granularity, group_size=group_size, scales=scales
            )
            * keep
        )
        delta = (weight - candidate).to(torch.float32)
        # Per-output-channel loss, so each row picks its own ratio. Exact, not greedy.
        losses = ((delta @ gram32) * delta).sum(dim=1)

        if best_losses is None:
            best_losses, best_scales = losses, scales
            continue
        improved = losses < best_losses
        best_losses = torch.where(improved, losses, best_losses)
        # Broadcast the per-row choice back onto whatever shape the scales take.
        rows = improved.reshape(-1, 1) if best_scales.shape[0] == improved.shape[0] else None
        if rows is not None:
            best_scales = torch.where(rows, scales, best_scales)
        else:
            per_group = best_scales.shape[0] // improved.shape[0]
            expanded = improved.repeat_interleave(per_group).reshape(-1, 1)
            best_scales = torch.where(expanded, scales, best_scales)

    assert best_scales is not None  # noqa: S101 - candidates is non-empty, checked above
    return best_scales


def quantise_weight(
    weight: torch.Tensor,
    *,
    bits: int,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
    scales: torch.Tensor | None = None,
) -> QuantisedWeight:
    """Quantise a weight onto a symmetric integer grid.

    Args:
        weight: Weight of shape ``(out_features, in_features)``.
        bits: Target bit width.
        granularity: Scope a single scale covers.
        group_size: Elements per group, for ``PER_GROUP``.
        scales: Pre-fitted scales. Supplied by the joint arm, which re-estimates scales on the
            surviving weights after each mask change (§3.8) rather than refitting to the full
            tensor, whose pruned entries would drag the scale down.

    Returns:
        The integer representation.

    Raises:
        QuantisationError: If the shape, bit width, or granularity is unsupported.
    """
    import torch

    qmax = _symmetric_qmax(bits)
    if scales is None:
        scales = compute_symmetric_scales(
            weight, bits=bits, granularity=granularity, group_size=group_size
        )
    grouped, _ = _grouped_view(weight, granularity, group_size)
    codes = torch.round(grouped / scales).clamp_(-qmax, qmax)
    return QuantisedWeight(
        codes=codes.reshape(weight.shape).to(torch.int8),
        scales=scales,
        bits=bits,
        granularity=granularity,
        group_size=group_size,
    )


def fake_quantise(
    weight: torch.Tensor,
    *,
    bits: int,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
    scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Snap a weight onto the quantisation grid while keeping float storage.

    This is what the layerwise solver and the joint saliency work with: the *values* are
    quantised, so the reconstruction objective and the mask ranking both see the grid, but the
    tensor stays float and differentiable-shaped. It is **not** smaller or faster, and must never
    back a size or latency measurement -- that is what :meth:`QuantisedWeight.dequantise` after a
    real conversion is for.

    Args:
        weight: Weight of shape ``(out_features, in_features)``.
        bits: Target bit width.
        granularity: Scope a single scale covers.
        group_size: Elements per group, for ``PER_GROUP``.
        scales: Pre-fitted scales; fitted from ``weight`` when omitted.

    Returns:
        A float tensor of the same shape and dtype, holding only grid values.
    """
    quantised = quantise_weight(
        weight, bits=bits, granularity=granularity, group_size=group_size, scales=scales
    )
    return quantised.dequantise().to(weight.dtype)


def pack_low_bit(codes: torch.Tensor, *, bits: int) -> torch.Tensor:
    """Pack signed integer codes into a dense ``uint8`` buffer.

    Codes are offset by ``qmax`` into an unsigned range before packing, so a 4-bit code occupies
    exactly one nibble and no sign extension is needed on the way out.

    Args:
        codes: Integer codes in ``[-qmax, +qmax]``, any shape.
        bits: Bit width; must be in :data:`PACKABLE_BITS`.

    Returns:
        1-D ``uint8`` tensor of ``ceil(numel * bits / 8)`` bytes.

    Raises:
        QuantisationError: If ``bits`` is not packable, or a code is out of range.
    """
    import torch

    if bits not in PACKABLE_BITS:
        raise QuantisationError(
            f"cannot pack {bits}-bit codes; supported widths are {list(PACKABLE_BITS)}"
        )
    qmax = _symmetric_qmax(bits)
    flat = codes.reshape(-1).to(torch.int64)
    if flat.numel() and (int(flat.min()) < -qmax or int(flat.max()) > qmax):
        raise QuantisationError(
            f"codes outside the representable range [-{qmax}, {qmax}] for {bits}-bit packing"
        )

    per_byte = BITS_PER_BYTE // bits
    shifted = (flat + qmax).to(torch.uint8)
    padding = (-shifted.numel()) % per_byte
    if padding:
        shifted = torch.cat(
            [shifted, torch.zeros(padding, dtype=torch.uint8, device=shifted.device)]
        )

    if bits == BITS_PER_BYTE:
        return shifted
    lanes = shifted.reshape(-1, per_byte)
    packed = torch.zeros(lanes.shape[0], dtype=torch.uint8, device=shifted.device)
    for lane in range(per_byte):
        packed |= lanes[:, lane] << (lane * bits)
    return packed


def unpack_low_bit(packed: torch.Tensor, *, bits: int, numel: int) -> torch.Tensor:
    """Reverse :func:`pack_low_bit`.

    Args:
        packed: The ``uint8`` buffer produced by :func:`pack_low_bit`.
        bits: Bit width used to pack.
        numel: Number of codes to recover, needed because packing may have padded the last byte.

    Returns:
        1-D ``int8`` tensor of ``numel`` signed codes.

    Raises:
        QuantisationError: If ``bits`` is not packable, or the buffer is too short for ``numel``.
    """
    import torch

    if bits not in PACKABLE_BITS:
        raise QuantisationError(
            f"cannot unpack {bits}-bit codes; supported widths are {list(PACKABLE_BITS)}"
        )
    qmax = _symmetric_qmax(bits)
    per_byte = BITS_PER_BYTE // bits
    required_bytes = -(-numel // per_byte)
    if packed.numel() < required_bytes:
        raise QuantisationError(
            f"buffer holds {packed.numel()} bytes, need {required_bytes} for {numel} "
            f"{bits}-bit codes"
        )

    if bits == BITS_PER_BYTE:
        recovered = packed[:numel].to(torch.int64)
    else:
        mask = (1 << bits) - 1
        lanes = [(packed >> (lane * bits)) & mask for lane in range(per_byte)]
        recovered = torch.stack(lanes, dim=1).reshape(-1)[:numel].to(torch.int64)
    return (recovered - qmax).to(torch.int8)


def effective_bits_per_weight(quantised: QuantisedWeight) -> float:
    """Bits per weight including scale overhead.

    §4.5 defines effective bits as including scale and zero-point overhead "where possible".
    Reporting the nominal bit width alone overstates the compression, and the gap widens as the
    group size shrinks -- a per-group scheme with a small group can cost meaningfully more than
    its nominal width.

    Args:
        quantised: The representation to measure.

    Returns:
        Bits per weight, counting packed codes plus fp32 scales.
    """
    num_weights = 1
    for dimension in quantised.shape:
        num_weights *= dimension
    if num_weights == 0:
        return 0.0
    code_bits = num_weights * quantised.bits
    scale_bits = int(quantised.scales.numel()) * FP32_BITS
    return (code_bits + scale_bits) / num_weights


class Quantiser(Compressor):
    """Post-training weight quantisation with calibration.

    Two representations are involved and the distinction matters for every number this study
    reports:

    * **fake quantisation** -- weights are rounded to the quantisation grid but stored as
      FP32. Differentiable, so it is what training uses, but it is *not* smaller or faster.
    * **real quantisation** -- weights stored at the target bit width with integer kernels.
      Produced by :meth:`convert`, and the only representation the CPU benchmark may measure.
    """

    method = CompressionMethod.QUANTISATION
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.CALIBRATED,
        CompressionStage.QUANTISED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.QUANTISED
    recover_stage = CompressionStage.RECOVERED

    def __init__(self, config: Any) -> None:
        """Initialise the arm with empty calibration state."""
        super().__init__(config)
        self.module_names: list[str] = []
        self.calibration_batches: int = 0
        self.is_converted: bool = False

    def prepare(self, model: nn.Module) -> nn.Module:
        """Select modules to quantise and insert observers.

        Args:
            model: The dense model.

        Returns:
            The model with observers attached.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): select modules via select_compressible_modules() with
        # config.compression.quantisation.exclude_patterns, then attach observers
        # (torch.ao.quantization.MinMaxObserver / PerChannelMinMaxObserver, chosen from
        # config.observer and config.granularity). Keep the observer set on self so
        # report_statistics can list which modules were actually instrumented.
        raise NotImplementedError(
            "Quantiser.prepare is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def calibrate(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Run calibration data through the model to populate the observers.

        May run on GPU. Uses the *same* calibration set as the sequential and joint arms:
        different calibration data would be an uncontrolled variable in the comparison.

        Args:
            model: The prepared model with observers attached.
            tokenizer: Tokeniser for building the calibration batches.

        Returns:
            The calibrated model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): draw config.compression.quantisation.calibration_samples
        # sequences from data.calibration.load_calibration_set() under torch.inference_mode()
        # and run forward passes to fill the observers. Record the sample count and the
        # calibration set's fingerprint in the stage record, so a rerun with different
        # calibration data is detectable in the results table.
        raise NotImplementedError(
            "Quantiser.calibrate is not implemented yet; see the TODO in "
            "compression/quantisation.py"
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """Compute quantisation parameters and apply fake quantisation.

        Args:
            model: The calibrated model.

        Returns:
            The fake-quantised model, still FP32 in storage.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): derive scale and zero point from each observer, then round
        # weights onto the grid in place while keeping FP32 storage. For bits < 8, implement
        # the grid directly rather than relying on torch.ao, which only supports 8-bit
        # integer paths on CPU.
        raise NotImplementedError(
            "Quantiser.apply is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Optional quantisation-aware fine-tuning.

        Disabled by default for this arm: leaving it off keeps "quantisation only" a
        post-training method, which is how it is normally deployed. Enable it via
        ``compression.recovery.enabled`` only if the sequential arm's recovery budget is
        matched, or the comparison stops being about compression method.

        Args:
            model: The fake-quantised model.
            tokenizer: Tokeniser for the recovery data loader.

        Returns:
            The fine-tuned model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): if enabled, run training.recovery.run_recovery() with a
        # straight-through estimator on the fake-quantisation nodes.
        raise NotImplementedError(
            "Quantiser.recover is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def convert(self, model: nn.Module) -> nn.Module:
        """Convert fake quantisation into real low-precision storage.

        This is the stage that makes the size and latency measurements meaningful. Skipping it
        yields a model that is numerically quantised but still FP32 on disk and in memory.

        Args:
            model: The fake-quantised model.

        Returns:
            A CPU-deployable, really-quantised model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): for 8-bit, use torch.ao.quantization.convert with the backend
        # from config.backend. On the pinned torch the only supported engine is 'onednn'; the
        # familiar 'x86' / 'fbgemm' / 'qnnpack' names raise. For sub-8-bit, pack weights into
        # int8 storage with the scales kept alongside and swap in a custom linear module.
        # Set self.is_converted = True, and verify the checkpoint shrank as expected: a
        # conversion that silently no-ops is the failure mode this whole arm is exposed to.
        raise NotImplementedError(
            "Quantiser.convert is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Report the bit width, scheme, calibration size, and theoretical size.

        Args:
            model: The model to measure. When ``None``, only configuration-derived fields are
                returned.

        Returns:
            A serialisable mapping including ``is_converted``, which distinguishes a genuinely
            quantised artefact from a fake-quantised FP32 one.
        """
        quantisation = self.compression_config.quantisation
        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "bits": quantisation.bits,
            "scheme": quantisation.scheme.value,
            "granularity": quantisation.granularity.value,
            "group_size": quantisation.group_size,
            "backend": quantisation.backend,
            "observer": quantisation.observer,
            "calibration_samples": quantisation.calibration_samples,
            "calibration_batches_seen": self.calibration_batches,
            "quantise_activations": quantisation.quantise_activations,
            "activation_bits": quantisation.activation_bits
            if quantisation.quantise_activations
            else None,
            "num_quantised_modules": len(self.module_names),
            "is_converted": self.is_converted,
        }
        if model is not None:
            parameters = count_parameters(model)
            statistics["measured_total_parameters"] = parameters
            statistics["theoretical_size_bytes"] = theoretical_size_bytes(
                parameters, quantisation.bits
            )
            statistics["theoretical_size_bytes_fp32"] = theoretical_size_bytes(
                parameters, FP32_BITS
            )
        return statistics
