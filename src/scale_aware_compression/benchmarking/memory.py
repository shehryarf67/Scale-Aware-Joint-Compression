"""Process memory measurement for CPU deployment.

Resident set size of the whole process is used rather than a torch allocator statistic,
because the deployment question is how much memory the served process needs. Quantised CPU
models keep packed weights outside the torch caching allocator, so an allocator-based figure
would understate them.

``psutil`` is imported lazily and its absence degrades to ``None`` measurements rather than
an exception, so the harness still runs in a minimal environment.
"""

from __future__ import annotations

import gc
import os
import platform
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from scale_aware_compression.constants import BYTES_PER_MIB
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class MemoryStatistics:
    """Memory footprint observed around a measured region."""

    baseline_mb: float | None
    peak_mb: float | None
    delta_mb: float | None
    available: bool
    """``False`` when psutil was unavailable and no measurement could be taken."""

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "memory_baseline_mb": self.baseline_mb,
            "peak_memory_mb": self.peak_mb,
            "memory_delta_mb": self.delta_mb,
            "memory_measurement_available": self.available,
        }


def _process() -> Any | None:
    """Return a psutil Process for this interpreter, or None if psutil is missing."""
    try:
        import psutil
    except ImportError:
        LOGGER.debug("psutil not installed; memory measurement unavailable")
        return None
    return psutil.Process(os.getpid())


def process_memory_mb() -> float | None:
    """Current resident set size of this process, in MiB.

    Returns:
        RSS in MiB, or ``None`` if psutil is unavailable.
    """
    process = _process()
    if process is None:
        return None
    return process.memory_info().rss / BYTES_PER_MIB


def peak_process_memory_mb() -> float | None:
    """Peak resident set size of this process, in MiB, where the OS reports one.

    Linux exposes a true high-water mark; on platforms without one this falls back to the
    current RSS, which under-reports a peak that has already been released.

    Returns:
        Peak RSS in MiB, or ``None`` if psutil is unavailable.
    """
    process = _process()
    if process is None:
        return None
    info = process.memory_info()

    # Windows exposes a true high-water mark on the memory_info tuple.
    if hasattr(info, "peak_wset"):
        return float(info.peak_wset) / BYTES_PER_MIB

    try:
        import resource
    except ImportError:  # pragma: no cover - platform dependent
        return float(info.rss) / BYTES_PER_MIB

    # ru_maxrss is kilobytes on Linux and bytes on macOS.
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    scale = 1024 if platform.system() == "Linux" else 1
    return float(max_rss * scale) / BYTES_PER_MIB


@dataclass(slots=True)
class MemoryTracker:
    """Sample process memory around a block of work.

    Sampling happens at entry, on every :meth:`sample` call, and at exit. A CPU forward pass
    is short enough that a background sampling thread would add more noise to the latency
    measurement than it removes from the memory estimate, so sampling stays explicit.

    Example:
        >>> with MemoryTracker() as tracker:
        ...     pass
        >>> tracker.result().available in (True, False)
        True
    """

    collect_garbage: bool = True
    """Run a full collection before the baseline reading, so the delta reflects this block
    rather than garbage left by the previous one."""
    _baseline_mb: float | None = field(default=None, init=False)
    _peak_mb: float | None = field(default=None, init=False)
    _available: bool = field(default=False, init=False)

    def __enter__(self) -> MemoryTracker:
        """Record the baseline reading."""
        if self.collect_garbage:
            gc.collect()
        self._baseline_mb = process_memory_mb()
        self._peak_mb = self._baseline_mb
        self._available = self._baseline_mb is not None
        return self

    def sample(self) -> float | None:
        """Take a reading and update the running peak.

        Returns:
            The reading in MiB, or ``None`` if measurement is unavailable.
        """
        current = process_memory_mb()
        if current is None:
            return None
        self._peak_mb = current if self._peak_mb is None else max(self._peak_mb, current)
        return current

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Take the final reading."""
        self.sample()

    def result(self) -> MemoryStatistics:
        """Return the tracked statistics.

        Returns:
            The baseline, peak, and delta in MiB, with ``available=False`` when psutil was
            not installed.
        """
        delta = (
            self._peak_mb - self._baseline_mb
            if self._peak_mb is not None and self._baseline_mb is not None
            else None
        )
        return MemoryStatistics(
            baseline_mb=self._baseline_mb,
            peak_mb=self._peak_mb,
            delta_mb=delta,
            available=self._available,
        )
