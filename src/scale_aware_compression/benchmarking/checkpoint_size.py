"""Checkpoint size measurement.

The reported deployment size is the on-disk size of the weight files only. Tokeniser and
config JSON are identical across every arm of the study, so including them adds a constant
that shrinks the apparent compression ratio of the smaller models more than the larger ones.

Both the measured size and the theoretical size implied by the compression budget are
recorded. A large gap between them usually means the artefact was serialised in FP32 with
zeros still stored at full width, which is the single most common way a compression result
overstates itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scale_aware_compression.constants import BYTES_PER_MIB, FP32_BITS
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import (
    checkpoint_size_bytes,
    compression_ratio,
    theoretical_size_bytes,
)

LOGGER = get_logger(__name__)

WEIGHT_SUFFIXES: tuple[str, ...] = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


@dataclass(frozen=True, slots=True)
class CheckpointSizeReport:
    """Measured and theoretical size of one saved artefact."""

    path: str
    total_bytes: int
    weight_bytes: int
    theoretical_weight_bytes: int | None
    file_count: int
    largest_file: str | None

    @property
    def total_mib(self) -> float:
        """Total on-disk size in MiB."""
        return self.total_bytes / BYTES_PER_MIB

    @property
    def weight_mib(self) -> float:
        """Weight-file size in MiB."""
        return self.weight_bytes / BYTES_PER_MIB

    @property
    def storage_efficiency(self) -> float | None:
        """Theoretical weight size divided by measured weight size.

        1.0 means the artefact is as small as its budget allows. Well below 1.0 means
        sparsity or low precision was not realised in the serialised format.
        """
        if self.theoretical_weight_bytes is None or self.weight_bytes == 0:
            return None
        return self.theoretical_weight_bytes / self.weight_bytes

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "checkpoint_path": self.path,
            "checkpoint_total_bytes": self.total_bytes,
            "checkpoint_total_mb": self.total_mib,
            "checkpoint_weight_bytes": self.weight_bytes,
            "checkpoint_size_mb": self.weight_mib,
            "checkpoint_theoretical_weight_bytes": self.theoretical_weight_bytes,
            "checkpoint_storage_efficiency": self.storage_efficiency,
            "checkpoint_file_count": self.file_count,
            "checkpoint_largest_file": self.largest_file,
        }


def measure_checkpoint(
    path: str | Path,
    *,
    nonzero_parameters: int | None = None,
    bits: int = FP32_BITS,
    untargeted_parameters: int | None = None,
) -> CheckpointSizeReport:
    """Measure a saved checkpoint and compare it against its theoretical size.

    Args:
        path: A checkpoint file, or a directory written by ``save_pretrained``.
        nonzero_parameters: Surviving **targeted** parameter count. When given, the theoretical size
            and storage efficiency are computed.
        bits: Bits per stored weight for the theoretical size.
        untargeted_parameters: Parameters that stay FP32 *by design* -- embeddings and the LM head,
            which §2.6 excludes from compression. Counted at full precision in the theoretical size,
            because a budget that assumed they were compressed would not be achievable by this
            method and every artefact would look inefficient against it.

    Returns:
        The size report.

    Raises:
        FileNotFoundError: If the path does not exist.
    """
    target = Path(path)
    total = checkpoint_size_bytes(target)

    if target.is_file():
        weight_bytes = total if target.suffix in WEIGHT_SUFFIXES else 0
        file_count = 1
        largest = target.name
    else:
        files = [item for item in target.rglob("*") if item.is_file()]
        weight_bytes = sum(item.stat().st_size for item in files if item.suffix in WEIGHT_SUFFIXES)
        file_count = len(files)
        largest = max(files, key=lambda item: item.stat().st_size).name if files else None

    if weight_bytes == 0:
        LOGGER.warning(
            "No weight files (%s) found under %s; the reported deployment size will be 0.",
            ", ".join(WEIGHT_SUFFIXES),
            target,
        )

    theoretical = None
    if nonzero_parameters is not None:
        theoretical = theoretical_size_bytes(nonzero_parameters, bits)
        if untargeted_parameters:
            # Embeddings and the LM head are excluded from compression by design (§2.6), so they
            # stay FP32 and the achievable budget has to include them at full precision. Comparing
            # against an all-weights-at-target-bits figure understates efficiency badly -- at
            # Pythia-160M the excluded parameters are nearly half the model, which produced a
            # spurious "2.4x larger than its budget allows" warning on an artefact that was in fact
            # exactly as small as this method can make it.
            theoretical += theoretical_size_bytes(untargeted_parameters, FP32_BITS)
    report = CheckpointSizeReport(
        path=target.as_posix(),
        total_bytes=total,
        weight_bytes=weight_bytes,
        theoretical_weight_bytes=theoretical,
        file_count=file_count,
        largest_file=largest,
    )
    efficiency = report.storage_efficiency
    if efficiency is not None and efficiency < 0.5:
        LOGGER.warning(
            "%s is %.1fx larger than its budget allows (storage efficiency %.2f). The "
            "artefact was probably serialised without applying sparsity or low precision.",
            target,
            1 / efficiency,
            efficiency,
        )
    return report


def compare_to_baseline(
    baseline: CheckpointSizeReport,
    compressed: CheckpointSizeReport,
) -> dict[str, float]:
    """Compare a compressed checkpoint against the dense FP32 baseline.

    Args:
        baseline: Report for the dense artefact.
        compressed: Report for the compressed artefact.

    Returns:
        Mapping with the measured compression ratio and the sizes it was derived from.

    Raises:
        ValueError: If either weight size is zero, which makes the ratio undefined.
    """
    ratio = compression_ratio(baseline.weight_bytes, compressed.weight_bytes)
    return {
        "baseline_size_mb": baseline.weight_mib,
        "compressed_size_mb": compressed.weight_mib,
        "compression_ratio": ratio,
        "size_reduction_percentage": 100.0 * (1.0 - 1.0 / ratio),
    }
