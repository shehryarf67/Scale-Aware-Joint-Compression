"""Size and sparsity metrics.

These functions are deliberately small and torch-free at the signature level: the ones that
inspect a model duck-type over ``parameters()`` / ``numel()`` / ``count_nonzero()``, and the
ones that turn counts into ratios take plain numbers. That keeps them unit-testable without
loading a model, which is where most measurement bugs in compression papers hide.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from scale_aware_compression.constants import BITS_PER_BYTE, BYTES_PER_MIB, FP32_BITS
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

_CHECKPOINT_SUFFIXES = (".bin", ".pt", ".pth", ".safetensors", ".gguf", ".onnx")


@runtime_checkable
class TensorLike(Protocol):
    """Minimal tensor interface these metrics need."""

    def numel(self) -> int:
        """Total number of elements."""
        ...


@runtime_checkable
class ModuleLike(Protocol):
    """Minimal module interface these metrics need."""

    def parameters(self) -> Iterable[Any]:
        """Iterate over parameter tensors."""
        ...


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------
def count_parameters(module: ModuleLike, *, trainable_only: bool = False) -> int:
    """Count parameters in a module.

    Args:
        module: Anything exposing ``parameters()`` yielding tensors with ``numel()``.
        trainable_only: Count only parameters with ``requires_grad`` true.

    Returns:
        Total element count across the selected parameters.
    """
    total = 0
    for parameter in module.parameters():
        if trainable_only and not getattr(parameter, "requires_grad", True):
            continue
        total += int(parameter.numel())
    return total


def count_zero_parameters(module: ModuleLike) -> int:
    """Count parameters that are exactly zero.

    This is the *measured* sparsity, as opposed to the target sparsity in the config. The
    two are reported side by side, because a pruning implementation that leaves masks
    unapplied produces a plausible-looking target with no measured zeros behind it.

    Args:
        module: Anything exposing ``parameters()``.

    Returns:
        Number of exactly-zero elements across all parameters.
    """
    return sum(count_zeros(parameter) for parameter in module.parameters())


def count_zeros(tensor: Any) -> int:
    """Count exactly-zero elements in a single tensor.

    Args:
        tensor: A torch tensor, a NumPy array, or anything with ``numel()`` plus either
            ``count_nonzero()`` or ``tolist()``.

    Returns:
        Number of zero elements.

    Raises:
        TypeError: If the object exposes neither route to a zero count.
    """
    numel_attribute = getattr(tensor, "numel", None)
    total = int(numel_attribute()) if callable(numel_attribute) else int(getattr(tensor, "size", 0))

    count_nonzero = getattr(tensor, "count_nonzero", None)
    if callable(count_nonzero):
        nonzero = count_nonzero()
        # torch returns a 0-d tensor; numpy-like objects may return a plain int.
        item = getattr(nonzero, "item", None)
        return total - (int(item()) if callable(item) else int(nonzero))

    to_list = getattr(tensor, "tolist", None)
    if callable(to_list):
        return sum(1 for value in _flatten(to_list()) if value == 0)

    raise TypeError(
        f"Cannot count zeros in {type(tensor).__name__}: expected count_nonzero() or tolist()"
    )


def _flatten(value: Any) -> Iterable[Any]:
    """Yield scalars from an arbitrarily nested list."""
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


# ---------------------------------------------------------------------------
# Sparsity
# ---------------------------------------------------------------------------
def sparsity_percentage(total_parameters: int, zero_parameters: int) -> float:
    """Fraction of parameters that are zero, as a percentage.

    Args:
        total_parameters: Denominator; must be positive.
        zero_parameters: Numerator; must lie in ``[0, total_parameters]``.

    Returns:
        Sparsity in the range ``[0.0, 100.0]``.

    Raises:
        ValueError: If the counts are negative, zero-denominator, or inconsistent.
    """
    if total_parameters <= 0:
        raise ValueError(f"total_parameters must be > 0, got {total_parameters}")
    if zero_parameters < 0:
        raise ValueError(f"zero_parameters must be >= 0, got {zero_parameters}")
    if zero_parameters > total_parameters:
        raise ValueError(
            f"zero_parameters ({zero_parameters}) cannot exceed total_parameters "
            f"({total_parameters})"
        )
    return 100.0 * zero_parameters / total_parameters


def sparsity_fraction(total_parameters: int, zero_parameters: int) -> float:
    """Sparsity as a fraction in ``[0, 1]`` rather than a percentage.

    Args:
        total_parameters: Denominator; must be positive.
        zero_parameters: Numerator.

    Returns:
        Sparsity in ``[0.0, 1.0]``.
    """
    return sparsity_percentage(total_parameters, zero_parameters) / 100.0


def measure_sparsity(module: ModuleLike) -> dict[str, float | int]:
    """Measure a module's realised sparsity in one pass.

    Args:
        module: Anything exposing ``parameters()``.

    Returns:
        Mapping with ``total_parameters``, ``zero_parameters``, ``nonzero_parameters``, and
        ``sparsity_percentage``.
    """
    total = count_parameters(module)
    zeros = count_zero_parameters(module)
    return {
        "total_parameters": total,
        "zero_parameters": zeros,
        "nonzero_parameters": total - zeros,
        "sparsity_percentage": sparsity_percentage(total, zeros) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Checkpoint size
# ---------------------------------------------------------------------------
def checkpoint_size_bytes(path: str | Path, *, weights_only: bool = False) -> int:
    """Total on-disk size of a checkpoint file or directory.

    Args:
        path: A checkpoint file, or a directory saved by ``save_pretrained``.
        weights_only: Count only weight files (``.safetensors``, ``.bin``, ``.pt``, ...),
            excluding tokeniser and config JSON. Use this when comparing arms, since
            tokeniser files are identical across them and only add constant noise.

    Returns:
        Size in bytes.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {target}")
    if target.is_file():
        return target.stat().st_size
    return sum(
        item.stat().st_size
        for item in target.rglob("*")
        if item.is_file() and (not weights_only or item.suffix in _CHECKPOINT_SUFFIXES)
    )


def checkpoint_size_mib(path: str | Path, *, weights_only: bool = False) -> float:
    """On-disk checkpoint size in MiB.

    Args:
        path: Checkpoint file or directory.
        weights_only: See :func:`checkpoint_size_bytes`.

    Returns:
        Size in MiB.
    """
    return checkpoint_size_bytes(path, weights_only=weights_only) / BYTES_PER_MIB


def theoretical_size_bytes(
    nonzero_parameters: int,
    bits: int = FP32_BITS,
    *,
    sparse_index_bits: int = 0,
) -> int:
    """Ideal serialised size of a weight set, ignoring container overhead.

    Reported next to the measured checkpoint size, because the gap between them is exactly
    the question of whether a sparse or low-bit format was actually used, rather than zeros
    being stored at full width.

    Args:
        nonzero_parameters: Number of weights actually kept.
        bits: Bits per stored weight.
        sparse_index_bits: Bits of index overhead per stored weight for a sparse format.
            Zero for dense storage.

    Returns:
        Size in bytes, rounded up to a whole byte.

    Raises:
        ValueError: If any argument is negative, or ``bits`` is not positive.
    """
    if nonzero_parameters < 0:
        raise ValueError(f"nonzero_parameters must be >= 0, got {nonzero_parameters}")
    if bits <= 0:
        raise ValueError(f"bits must be > 0, got {bits}")
    if sparse_index_bits < 0:
        raise ValueError(f"sparse_index_bits must be >= 0, got {sparse_index_bits}")
    total_bits = nonzero_parameters * (bits + sparse_index_bits)
    return -(-total_bits // BITS_PER_BYTE)  # ceiling division


# ---------------------------------------------------------------------------
# Compression ratio
# ---------------------------------------------------------------------------
def compression_ratio(baseline_bytes: float, compressed_bytes: float) -> float:
    """How many times smaller the compressed artefact is than the baseline.

    Args:
        baseline_bytes: Size of the dense FP32 reference; must be positive.
        compressed_bytes: Size of the compressed artefact; must be positive.

    Returns:
        ``baseline_bytes / compressed_bytes``. Greater than 1 means smaller than baseline.

    Raises:
        ValueError: If either size is not positive.
    """
    if baseline_bytes <= 0:
        raise ValueError(f"baseline_bytes must be > 0, got {baseline_bytes}")
    if compressed_bytes <= 0:
        raise ValueError(f"compressed_bytes must be > 0, got {compressed_bytes}")
    return baseline_bytes / compressed_bytes


def size_reduction_percentage(baseline_bytes: float, compressed_bytes: float) -> float:
    """Percentage of the baseline size removed.

    Args:
        baseline_bytes: Size of the dense FP32 reference; must be positive.
        compressed_bytes: Size of the compressed artefact; must be positive.

    Returns:
        ``100 * (1 - compressed / baseline)``. Negative if the artefact grew.

    Raises:
        ValueError: If either size is not positive.
    """
    ratio = compression_ratio(baseline_bytes, compressed_bytes)
    return 100.0 * (1.0 - 1.0 / ratio)


def effective_compression_ratio(
    total_parameters: int,
    zero_parameters: int,
    bits: int,
    *,
    baseline_bits: int = FP32_BITS,
) -> float:
    """Combined size benefit of sparsity and reduced precision.

    This is the *theoretical* ratio implied by a configuration, used to check that joint
    and sequential arms are being compared at a matched budget. It is not a measurement.

    Args:
        total_parameters: Parameter count before pruning.
        zero_parameters: Parameters set to zero.
        bits: Bits per surviving weight.
        baseline_bits: Bits per weight in the dense reference.

    Returns:
        Ratio of baseline bits to compressed bits, always >= 1 for valid inputs.

    Raises:
        ValueError: If counts are inconsistent or a bit width is not positive.
    """
    if bits <= 0 or baseline_bits <= 0:
        raise ValueError("bit widths must be > 0")
    fraction_kept = 1.0 - sparsity_fraction(total_parameters, zero_parameters)
    if fraction_kept <= 0.0:
        raise ValueError("Cannot compute a compression ratio when every parameter is zero")
    return baseline_bits / (bits * fraction_kept)
