"""Real low-bit weight storage: the artefact that gets measured.

Everything upstream of this module works in *fake* quantisation -- weights snapped to the grid but
still held as FP32. That representation is correct numerically and useless for measurement: it is
exactly the same size as the dense model. §4.8 requires confirming that "quantized weights use the
intended bit-width rather than being silently dequantized to full precision", and this module is
what makes that confirmation possible.

:class:`PackedLinear` replaces an ``nn.Linear`` and stores

* integer codes packed at the target bit width, in a ``uint8`` buffer
* one fp32 scale per quantisation group
* nothing else -- **no mask buffer**

Dropping the mask matters for honesty. A boolean mask costs a byte per weight, which at 4 bits would
be *twice* the size of the weights it describes; keeping one would make a "13x compressed" model
larger than the dense one on disk. Sparsity survives as exact zeros among the codes instead, which
is why realised sparsity is re-verified after conversion rather than trusted.

**One artefact format for every arm.** All five arms convert through this same class, so a size or
quality comparison between them cannot be a comparison of storage formats (§3.11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.quantisation import (
    PACKABLE_BITS,
    QuantisationError,
    pack_low_bit,
    quantise_weight,
    unpack_low_bit,
)
from scale_aware_compression.constants import BITS_PER_BYTE, FP32_BITS, QuantisationGranularity
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)

_GRANULARITY_ORDER: tuple[QuantisationGranularity, ...] = (
    QuantisationGranularity.PER_TENSOR,
    QuantisationGranularity.PER_CHANNEL,
    QuantisationGranularity.PER_GROUP,
)
"""Fixed order for encoding granularity as an integer in the scheme buffer.

Append only. Reordering these would silently reinterpret every checkpoint already written.
"""


def _build_packed_linear() -> type:
    """Define ``PackedLinear`` against ``nn.Module``, lazily.

    The package contract is that ``import scale_aware_compression`` must not pull in torch, and a
    class statement inheriting from ``nn.Module`` would break that at import time. Building the
    class on first use keeps the contract without making callers think about it.
    """
    import torch
    from torch import nn

    class _PackedLinear(nn.Module):
        """A linear layer whose weights are stored at ``bits`` precision.

        Attributes:
            in_features: Input width.
            out_features: Output width.
            bits: Bit width of the stored codes.
            granularity: Scope each scale covers.
            group_size: Elements per group, for per-group granularity.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            *,
            bits: int,
            granularity: QuantisationGranularity,
            group_size: int,
            packed: torch.Tensor,
            scales: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bits = bits
            self.granularity = granularity
            self.group_size = group_size
            self.register_buffer("packed", packed)
            self.register_buffer("scales", scales)
            if bias is None:
                self.bias = None
            else:
                self.register_buffer("bias", bias)
            # Scheme metadata as an int64 buffer rather than via get_extra_state(). The codes are
            # uninterpretable without it -- bit width and granularity cannot be recovered from
            # packed bytes -- but `get_extra_state` returns a plain dict, and both safetensors and
            # `save_pretrained` walk the state dict expecting every value to be a tensor. Keeping it
            # a tensor means a packed model saves through the *same* path as the dense baseline,
            # which is what makes their checkpoint sizes comparable.
            self.register_buffer("scheme", self._encode_scheme())

        def _encode_scheme(self) -> torch.Tensor:
            return torch.tensor(
                [
                    self.in_features,
                    self.out_features,
                    self.bits,
                    _GRANULARITY_ORDER.index(self.granularity),
                    self.group_size,
                ],
                dtype=torch.int64,
            )

        def _decode_scheme(self) -> None:
            values = [int(item) for item in self.scheme.tolist()]
            (
                self.in_features,
                self.out_features,
                self.bits,
                granularity_index,
                self.group_size,
            ) = values
            self.granularity = _GRANULARITY_ORDER[granularity_index]

        @property
        def num_weights(self) -> int:
            """Weight elements this layer represents."""
            return self.in_features * self.out_features

        def _load_from_state_dict(  # noqa: PLR0913 - signature fixed by nn.Module
            self,
            state_dict: dict[str, Any],
            prefix: str,
            local_metadata: dict[str, Any],
            strict: bool,
            missing_keys: list[str],
            unexpected_keys: list[str],
            error_msgs: list[str],
        ) -> None:
            """Resize the buffers to match the checkpoint before copying into them.

            A packed layer's buffer *shapes* are determined by what is being loaded: the code buffer
            length follows the bit width, and the scale count follows the granularity. The default
            loader compares shapes before it has read either, so loading a 4-bit checkpoint into a
            layer built for 8-bit fails on a size mismatch even though the data is perfectly valid.

            Reloading is not a convenience here. §4.8 requires verifying that a compressed checkpoint
            reloads independently from disk and still carries its sparsity and bit width, and Phase
            10's audit tooling does exactly that.
            """
            for name in ("packed", "scales", "bias", "scheme"):
                key = prefix + name
                incoming = state_dict.get(key)
                current = getattr(self, name, None)
                if incoming is None or current is None:
                    continue
                if incoming.shape != current.shape:
                    self._buffers[name] = torch.empty_like(incoming)
            super()._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            # The scheme buffer has now been copied in, so the Python attributes it encodes have to
            # be refreshed or dequantise() would still use the pre-load bit width.
            if prefix + "scheme" in state_dict:
                self._decode_scheme()

        def dequantise(self) -> torch.Tensor:
            """Reconstruct the FP32 weight from the packed codes and scales."""
            codes = unpack_low_bit(self.packed, bits=self.bits, numel=self.num_weights)
            codes = codes.reshape(self.out_features, self.in_features).to(self.scales.dtype)
            if self.granularity is QuantisationGranularity.PER_TENSOR:
                return codes * self.scales.reshape(1, 1)
            if self.granularity is QuantisationGranularity.PER_CHANNEL:
                return codes * self.scales.reshape(self.out_features, 1)
            num_groups = self.in_features // self.group_size
            grouped = codes.reshape(self.out_features * num_groups, self.group_size)
            return (grouped * self.scales.reshape(-1, 1)).reshape(
                self.out_features, self.in_features
            )

        def forward(self, activations: torch.Tensor) -> torch.Tensor:
            """Dequantise on the fly and apply the linear map.

            Deliberately *not* a fast path. There is no int4 CPU GEMM in the pinned torch, so this
            unpacks to FP32 and calls the dense kernel -- correct, and slower than the dense model.
            That is why decision **D1** keeps W4 out of every latency table: timing this would
            measure the unpacking, not the compression.
            """
            weight = self.dequantise().to(activations.dtype)
            return torch.nn.functional.linear(activations, weight, self.bias)

        def extra_repr(self) -> str:
            """Show the scheme in ``print(model)`` output."""
            return (
                f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits={self.bits}, granularity={self.granularity.value}"
            )

    return _PackedLinear


_PACKED_LINEAR_CLASS: type | None = None


def packed_linear_class() -> type:
    """Return the ``PackedLinear`` class, building it on first call."""
    global _PACKED_LINEAR_CLASS  # noqa: PLW0603 - one-time lazy class construction
    if _PACKED_LINEAR_CLASS is None:
        _PACKED_LINEAR_CLASS = _build_packed_linear()
    return _PACKED_LINEAR_CLASS


def pack_linear(
    layer: nn.Linear,
    *,
    bits: int,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
) -> Any:
    """Convert an ``nn.Linear`` holding fake-quantised weights into packed storage.

    The weights are expected to already sit on the grid, which is what the layerwise driver leaves
    behind. Re-quantising here is therefore a lossless re-encoding rather than a second rounding --
    and :func:`verify_packing` checks exactly that.

    Args:
        layer: The layer to convert. Not modified.
        bits: Target bit width; must be in :data:`~scale_aware_compression.compression.quantisation.PACKABLE_BITS`.
        granularity: Scope each scale covers.
        group_size: Elements per group.

    Returns:
        A ``PackedLinear`` carrying the same numerical weight.

    Raises:
        QuantisationError: If ``bits`` cannot be packed.
    """
    if bits not in PACKABLE_BITS:
        raise QuantisationError(
            f"cannot pack {bits}-bit weights; supported widths are {list(PACKABLE_BITS)}"
        )

    weight = layer.weight.detach().float()
    quantised = quantise_weight(weight, bits=bits, granularity=granularity, group_size=group_size)
    packed = pack_low_bit(quantised.codes.reshape(-1), bits=bits)

    bias = layer.bias.detach().clone() if getattr(layer, "bias", None) is not None else None
    return packed_linear_class()(
        layer.in_features,
        layer.out_features,
        bits=bits,
        granularity=granularity,
        group_size=group_size,
        packed=packed,
        scales=quantised.scales.detach().clone(),
        bias=bias,
    )


def convert_model_to_packed(
    model: nn.Module,
    module_names: list[str],
    *,
    bits: int,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
) -> dict[str, Any]:
    """Replace the named linear layers with packed equivalents, in place.

    Args:
        model: The compressed model. Modified in place.
        module_names: Layers to convert, as returned by ``select_compressible_modules``.
        bits: Target bit width.
        granularity: Scope each scale covers.
        group_size: Elements per group.

    Returns:
        Mapping with the conversion tally and the storage accounting, for the run record.

    Raises:
        QuantisationError: If a named module is not convertible.
    """
    converted = 0
    packed_bytes = 0
    scale_bytes = 0
    num_weights = 0

    for name in module_names:
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = getattr(parent, attribute)
        if not hasattr(layer, "weight"):
            raise QuantisationError(f"{name!r} has no weight to pack")

        replacement = pack_linear(layer, bits=bits, granularity=granularity, group_size=group_size)
        setattr(parent, attribute, replacement)

        converted += 1
        packed_bytes += replacement.packed.numel()
        scale_bytes += replacement.scales.numel() * (FP32_BITS // BITS_PER_BYTE)
        num_weights += replacement.num_weights

    dense_bytes = num_weights * (FP32_BITS // BITS_PER_BYTE)
    total = packed_bytes + scale_bytes
    return {
        "num_converted_modules": converted,
        "packed_weight_bytes": packed_bytes,
        "scale_bytes": scale_bytes,
        "dense_equivalent_bytes": dense_bytes,
        "weight_compression_ratio": dense_bytes / total if total else 0.0,
        "effective_bits_per_weight": (total * BITS_PER_BYTE / num_weights) if num_weights else 0.0,
    }


def verify_packing(layer: Any, reference: torch.Tensor, *, tolerance: float = 0.0) -> None:
    """Check a packed layer reproduces the weight it was built from.

    This is the §4.8 integrity check made executable. A conversion that silently dequantised, or an
    unpack that corrupted the last byte, would both leave a model that still runs and still scores --
    just on different weights than the ones the compression produced.

    Args:
        layer: The ``PackedLinear`` to check.
        reference: The fake-quantised weight it should reproduce.
        tolerance: Maximum permitted absolute deviation. Zero demands exactness, which is the
            correct expectation when the input was already on the grid.

    Raises:
        QuantisationError: If the packed layer does not reproduce ``reference``.
    """
    import torch

    recovered = layer.dequantise()
    if recovered.shape != reference.shape:
        raise QuantisationError(
            f"packed layer shape {tuple(recovered.shape)} does not match reference "
            f"{tuple(reference.shape)}"
        )
    deviation = float((recovered - reference.float()).abs().max())
    if deviation > tolerance:
        raise QuantisationError(
            f"packed layer deviates from its source weight by {deviation:.3e} (tolerance "
            f"{tolerance:.3e}). The weights were probably not on the quantisation grid before "
            "packing, which means conversion rounded a second time."
        )
    # Sparsity has to survive the round trip too: it is stored as zero codes, not as a mask.
    reference_zeros = int((reference == 0).sum())
    recovered_zeros = int((recovered == 0).sum())
    if recovered_zeros < reference_zeros:
        raise QuantisationError(
            f"packing lost sparsity: {reference_zeros} zeros before, {recovered_zeros} after"
        )
    del torch
