"""CPU deployment benchmarking: latency, throughput, memory, checkpoint size.

Every final deployment measurement in this study comes from here, and every function in this
subpackage is CPU-only by construction. Importing the subpackage runs nothing: no threads are
pinned, no torch import happens, and no timing loop starts until a runner is called
explicitly.
"""

from __future__ import annotations

from scale_aware_compression.benchmarking.checkpoint_size import (
    CheckpointSizeReport,
    compare_to_baseline,
    measure_checkpoint,
)
from scale_aware_compression.benchmarking.cpu import (
    BenchmarkCallable,
    BenchmarkError,
    BenchmarkResult,
    CpuBenchmarkRunner,
    benchmark_model,
    build_forward_callable,
)
from scale_aware_compression.benchmarking.latency import (
    LatencyStatistics,
    percentile,
    summarise_latencies,
)
from scale_aware_compression.benchmarking.memory import (
    MemoryStatistics,
    MemoryTracker,
    peak_process_memory_mb,
    process_memory_mb,
)
from scale_aware_compression.benchmarking.throughput import (
    ThroughputStatistics,
    samples_per_second,
    throughput_from_latency,
    tokens_per_second,
)

__all__ = [
    "BenchmarkCallable",
    "BenchmarkError",
    "BenchmarkResult",
    "CheckpointSizeReport",
    "CpuBenchmarkRunner",
    "LatencyStatistics",
    "MemoryStatistics",
    "MemoryTracker",
    "ThroughputStatistics",
    "benchmark_model",
    "build_forward_callable",
    "compare_to_baseline",
    "measure_checkpoint",
    "peak_process_memory_mb",
    "percentile",
    "process_memory_mb",
    "samples_per_second",
    "summarise_latencies",
    "throughput_from_latency",
    "tokens_per_second",
]
