"""CPU benchmark runner.

The runner times an arbitrary zero-argument callable. That indirection is deliberate: it
keeps the measurement protocol (pin threads, warm up, repeat, summarise) separate from what
is being measured, so the protocol is unit-testable with a dummy callable and identical
across the dense, pruned, quantised, sequential, and joint arms.

Importing this module runs nothing: it does not import torch, does not set thread counts,
and does not touch the filesystem. Threads are pinned only inside :meth:`CpuBenchmarkRunner.run`.

Protocol, from ``docs/benchmarking_protocol.md``:

* fixed PyTorch CPU thread count, recorded in the result
* fixed batch size and sequence length
* ``warmup_runs`` untimed iterations before measurement
* ``measured_runs`` timed iterations, each recorded individually
* median and p95 reported as headline numbers, mean and std alongside
* peak process memory and full hardware metadata captured with the timings
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.benchmarking.latency import LatencyStatistics, summarise_latencies
from scale_aware_compression.benchmarking.memory import MemoryStatistics, MemoryTracker
from scale_aware_compression.benchmarking.throughput import (
    ThroughputStatistics,
    throughput_from_latency,
)
from scale_aware_compression.config import BenchmarkConfig
from scale_aware_compression.constants import Device
from scale_aware_compression.hardware import (
    get_hardware_info,
    get_software_versions,
    set_cpu_threads,
)
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

BenchmarkCallable = Callable[[], Any]
"""A zero-argument callable performing exactly one unit of inference work."""


class BenchmarkError(RuntimeError):
    """Raised when a benchmark cannot be run under the required conditions."""


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Everything measured and recorded by one benchmark run."""

    label: str
    latency: LatencyStatistics
    throughput: ThroughputStatistics
    memory: MemoryStatistics
    batch_size: int
    sequence_length: int
    generated_tokens: int
    num_threads: int
    warmup_runs: int
    measured_runs: int
    device: str
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, str | None] = field(default_factory=dict)
    thread_report: dict[str, Any] = field(default_factory=dict)
    per_run_latencies_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a nested, JSON-serialisable mapping for the run record."""
        return {
            "label": self.label,
            "device": self.device,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "generated_tokens": self.generated_tokens,
            "num_threads": self.num_threads,
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
            **self.latency.to_dict(),
            **self.throughput.to_dict(),
            **self.memory.to_dict(),
            "per_run_latencies_ms": self.per_run_latencies_ms,
            "thread_report": self.thread_report,
            "hardware": self.hardware,
            "software": self.software,
        }

    def summary_line(self) -> str:
        """One-line summary for logs, using the reported central measures."""
        return (
            f"{self.label}: median={self.latency.median_ms:.2f} ms "
            f"p95={self.latency.p95_ms:.2f} ms "
            f"({self.throughput.tokens_per_second:.1f} tok/s, "
            f"{self.num_threads} threads, bs={self.batch_size}, "
            f"seq={self.sequence_length})"
        )


@dataclass(slots=True)
class CpuBenchmarkRunner:
    """Runs the CPU measurement protocol against a callable.

    Example:
        >>> config = BenchmarkConfig(warmup_runs=1, measured_runs=2, num_threads=1)
        >>> runner = CpuBenchmarkRunner(config)
        >>> result = runner.run(lambda: sum(range(1000)), label="dummy")
        >>> result.latency.num_runs
        2
    """

    config: BenchmarkConfig

    def __post_init__(self) -> None:
        """Enforce the CPU-only policy before any work is scheduled."""
        if self.config.device is not Device.CPU:
            raise BenchmarkError(
                "Deployment measurements in this study are CPU-only; "
                f"benchmark.device={self.config.device.value!r}"
            )

    def prepare(self) -> dict[str, Any]:
        """Pin CPU threads and verify the pin took effect.

        Returns:
            The thread report from :func:`set_cpu_threads`.

        Raises:
            BenchmarkError: If torch reports a different thread count than requested and
                ``config.fail_on_thread_mismatch`` is set. An unpinned thread count makes
                latencies incomparable across runs, so this fails loudly by default.
        """
        report = set_cpu_threads(self.config.num_threads, self.config.interop_threads)
        actual = report.get("torch_num_threads")
        if actual is not None and actual != self.config.num_threads:
            message = (
                f"Requested {self.config.num_threads} CPU threads but torch reports {actual}. "
                "Latencies measured under a different thread count are not comparable."
            )
            if self.config.fail_on_thread_mismatch:
                raise BenchmarkError(message)
            LOGGER.warning("%s", message)
        return report

    def warmup(self, function: BenchmarkCallable) -> None:
        """Run untimed iterations to settle caches, allocators, and kernel selection.

        Args:
            function: The callable being benchmarked.
        """
        if self.config.warmup_runs <= 0:
            LOGGER.warning("warmup_runs=0: the first measured run will include one-off costs")
            return
        LOGGER.debug("Warming up: %d runs", self.config.warmup_runs)
        for _ in range(self.config.warmup_runs):
            function()

    def measure(self, function: BenchmarkCallable) -> list[float]:
        """Time the callable ``measured_runs`` times.

        Args:
            function: The callable being benchmarked.

        Returns:
            One duration in seconds per run, in execution order.
        """
        samples: list[float] = []
        for index in range(self.config.measured_runs):
            start = time.perf_counter()
            function()
            samples.append(time.perf_counter() - start)
            if index and index % 10 == 0:
                LOGGER.debug("  run %d/%d", index, self.config.measured_runs)
        return samples

    def run(self, function: BenchmarkCallable, *, label: str = "benchmark") -> BenchmarkResult:
        """Execute the full protocol and assemble a result.

        Args:
            function: Zero-argument callable performing one unit of inference work.
            label: Name recorded with the result, e.g. ``"pythia-410m/joint"``.

        Returns:
            The assembled :class:`BenchmarkResult`.

        Raises:
            BenchmarkError: If threads cannot be pinned as required, or the callable raises.
        """
        thread_report = self.prepare()
        LOGGER.info(
            "Benchmarking %s: %d threads, bs=%d, seq=%d, %d warmup + %d measured runs",
            label,
            self.config.num_threads,
            self.config.batch_size,
            self.config.sequence_length,
            self.config.warmup_runs,
            self.config.measured_runs,
        )

        with MemoryTracker() as tracker:
            try:
                self.warmup(function)
                tracker.sample()
                samples = self.measure(function)
            except Exception as error:
                raise BenchmarkError(
                    f"Benchmark {label!r} failed while running: {error}"
                ) from error

        latency = summarise_latencies(samples)
        throughput = throughput_from_latency(
            latency_seconds=latency.median_ms / 1000.0,
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            generated_tokens=self.config.generated_tokens,
        )
        result = BenchmarkResult(
            label=label,
            latency=latency,
            throughput=throughput,
            memory=tracker.result(),
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            generated_tokens=self.config.generated_tokens,
            num_threads=self.config.num_threads,
            warmup_runs=self.config.warmup_runs,
            measured_runs=self.config.measured_runs,
            device=self.config.device.value,
            hardware=get_hardware_info(),
            software=get_software_versions(),
            thread_report=thread_report,
            per_run_latencies_ms=(
                [sample * 1000.0 for sample in samples]
                if self.config.record_per_run_latencies
                else []
            ),
        )
        LOGGER.info("%s", result.summary_line())
        return result


BENCHMARK_INPUT_SEED = 20240101
"""Seed for the synthetic benchmark input. Fixed so every arm and every run is timed on exactly
the same token tensor."""


def build_forward_callable(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: BenchmarkConfig,
) -> BenchmarkCallable:
    """Build the callable that :class:`CpuBenchmarkRunner` times for a real model.

    The input tensor is allocated once, here, so neither tokenisation nor allocation lands
    inside the timed region.

    Token *values* are synthetic and random. Latency depends on tensor shapes and on which
    kernels run, not on what the tokens mean, so real text would add a tokeniser dependency and
    a source of variation between arms without making the measurement more realistic. The shape
    is what has to be held fixed, and it is.

    Args:
        model: A model already moved to CPU and in eval mode.
        tokenizer: Tokeniser, used only for its vocabulary size and pad token.
        config: Benchmark section of the experiment config.

    Returns:
        A zero-argument callable running one forward pass, or one ``generate`` call when
        ``config.generated_tokens`` is positive.

    Raises:
        BenchmarkError: If the model cannot be prepared for benchmarking.
    """
    import torch

    if config.device is not Device.CPU:
        raise BenchmarkError(
            f"benchmark.device={config.device.value!r}; deployment measurements are CPU-only."
        )

    try:
        model.eval()
        model.to("cpu")
    except Exception as error:
        raise BenchmarkError(
            f"Could not move the model to CPU for benchmarking: {error}"
        ) from error

    vocabulary_size = int(
        getattr(getattr(model, "config", None), "vocab_size", None)
        or getattr(tokenizer, "vocab_size", 0)
        or 0
    )
    if vocabulary_size < 2:
        raise BenchmarkError(
            "Could not determine a usable vocabulary size for the synthetic benchmark input"
        )

    generator = torch.Generator().manual_seed(BENCHMARK_INPUT_SEED)
    input_ids = torch.randint(
        low=0,
        high=vocabulary_size,
        size=(config.batch_size, config.sequence_length),
        generator=generator,
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)

    if config.generated_tokens > 0:
        if not hasattr(model, "generate"):
            raise BenchmarkError(
                f"benchmark.generated_tokens={config.generated_tokens} but "
                f"{type(model).__name__} has no generate()."
            )
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or getattr(
            tokenizer, "eos_token_id", None
        )
        generated_tokens = config.generated_tokens

        def run_generate() -> Any:
            with torch.inference_mode():
                # min == max so every repetition decodes exactly the same number of tokens. An
                # early EOS would shorten later runs and pull the median down for no real reason.
                return model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    min_new_tokens=generated_tokens,
                    max_new_tokens=generated_tokens,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )

        LOGGER.debug(
            "Benchmark workload: decode %d tokens from a %dx%d prompt",
            generated_tokens,
            config.batch_size,
            config.sequence_length,
        )
        return run_generate

    def run_forward() -> Any:
        with torch.inference_mode():
            return model(input_ids=input_ids, attention_mask=attention_mask)

    LOGGER.debug(
        "Benchmark workload: single forward pass over %dx%d",
        config.batch_size,
        config.sequence_length,
    )
    return run_forward


def benchmark_model(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: BenchmarkConfig,
    *,
    label: str = "benchmark",
) -> BenchmarkResult:
    """Benchmark a loaded model end to end on CPU.

    Args:
        model: A model already moved to CPU and in eval mode.
        tokenizer: Matching tokeniser.
        config: Benchmark section of the experiment config.
        label: Name recorded with the result.

    Returns:
        The measured result.

    Raises:
        NotImplementedError: Until :func:`build_forward_callable` is implemented.
    """
    function = build_forward_callable(model, tokenizer, config)
    return CpuBenchmarkRunner(config).run(function, label=label)
