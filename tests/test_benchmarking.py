"""CPU benchmarking: latency statistics, the runner protocol, throughput, and sizes.

The runner times an arbitrary callable, so the whole measurement protocol is testable here with a
trivial function — no model, no torch. The latency statistics are checked against hand-computed
values from a known sample set.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scale_aware_compression.benchmarking.checkpoint_size import (
    CheckpointSizeReport,
    compare_to_baseline,
    measure_checkpoint,
)
from scale_aware_compression.benchmarking.cpu import (
    BenchmarkError,
    BenchmarkResult,
    CpuBenchmarkRunner,
    build_forward_callable,
)
from scale_aware_compression.benchmarking.latency import percentile, summarise_latencies
from scale_aware_compression.benchmarking.memory import MemoryTracker, process_memory_mb
from scale_aware_compression.benchmarking.throughput import (
    samples_per_second,
    throughput_from_latency,
    tokens_per_second,
)
from scale_aware_compression.config import BenchmarkConfig, ConfigError
from scale_aware_compression.constants import Device


@pytest.fixture
def fast_config() -> BenchmarkConfig:
    """A benchmark config small enough to run instantly in a test."""
    return BenchmarkConfig(
        num_threads=1,
        warmup_runs=2,
        measured_runs=5,
        batch_size=1,
        sequence_length=16,
        fail_on_thread_mismatch=False,
    )


class TestPercentile:
    def test_single_sample(self):
        assert percentile([42.0], 0.95) == 42.0

    @pytest.mark.parametrize(("fraction", "expected"), [(0.0, 1.0), (0.5, 3.0), (1.0, 5.0)])
    def test_endpoints_and_median(self, fraction: float, expected: float):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], fraction) == pytest.approx(expected)

    def test_interpolates_between_order_statistics(self):
        # position = 0.95 * 9 = 8.55 -> between 9.0 and 10.0, weight 0.55
        assert percentile([float(n) for n in range(1, 11)], 0.95) == pytest.approx(9.55)

    def test_does_not_require_sorted_input(self):
        assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == pytest.approx(3.0)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="at least one sample"):
            percentile([], 0.5)

    @pytest.mark.parametrize("fraction", [-0.1, 1.1])
    def test_out_of_range_fraction_raises(self, fraction: float):
        with pytest.raises(ValueError, match="fraction"):
            percentile([1.0, 2.0], fraction)


class TestLatencyStatistics:
    def test_all_statistics_against_known_values(self, latency_samples_seconds: list[float]):
        """10 ms .. 100 ms in 10 ms steps: mean 55, median 55, p95 95.5, sample std 30.2765."""
        statistics = summarise_latencies(latency_samples_seconds)
        assert statistics.num_runs == 10
        assert statistics.mean_ms == pytest.approx(55.0)
        assert statistics.median_ms == pytest.approx(55.0)
        assert statistics.p95_ms == pytest.approx(95.5)
        assert statistics.std_ms == pytest.approx(30.2765, abs=1e-3)
        assert statistics.min_ms == pytest.approx(10.0)
        assert statistics.max_ms == pytest.approx(100.0)
        assert statistics.total_seconds == pytest.approx(0.55)

    def test_converts_seconds_to_milliseconds(self):
        statistics = summarise_latencies([0.001, 0.001, 0.001])
        assert statistics.median_ms == pytest.approx(1.0)

    def test_identical_samples_have_zero_spread(self):
        statistics = summarise_latencies([0.05] * 10)
        assert statistics.std_ms == pytest.approx(0.0)
        assert statistics.coefficient_of_variation == pytest.approx(0.0)

    def test_coefficient_of_variation(self, latency_samples_seconds: list[float]):
        statistics = summarise_latencies(latency_samples_seconds)
        assert statistics.coefficient_of_variation == pytest.approx(30.2765 / 55.0, abs=1e-3)

    def test_p99_is_at_least_p95(self, latency_samples_seconds: list[float]):
        statistics = summarise_latencies(latency_samples_seconds)
        assert statistics.p99_ms >= statistics.p95_ms

    def test_a_single_sample_is_rejected(self):
        """One measurement supports neither a standard deviation nor a p95."""
        with pytest.raises(ValueError, match="at least 2 samples"):
            summarise_latencies([0.01])

    def test_negative_samples_are_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            summarise_latencies([0.01, -0.01])

    def test_noisy_measurement_warns(self, caplog: pytest.LogCaptureFixture):
        """A noisy run must be re-run, not reported.

        A clean joint arm measured against a noisy sequential one can manufacture a joint gain.
        """
        with caplog.at_level("WARNING"):
            summarise_latencies([0.01, 0.01, 0.01, 0.5])
        assert any("spread is high" in message for message in caplog.messages)

    def test_to_dict_is_flat_and_prefixed(self, latency_samples_seconds: list[float]):
        payload = summarise_latencies(latency_samples_seconds).to_dict()
        assert payload["latency_median_ms"] == pytest.approx(55.0)
        assert payload["latency_p95_ms"] == pytest.approx(95.5)
        assert all(not isinstance(value, dict) for value in payload.values())


class TestThroughput:
    def test_tokens_per_second(self):
        assert tokens_per_second(128, 0.5) == pytest.approx(256.0)

    def test_samples_per_second(self):
        assert samples_per_second(4, 0.5) == pytest.approx(8.0)

    @pytest.mark.parametrize(("count", "seconds"), [(0, 1.0), (10, 0.0), (-1, 1.0)])
    def test_non_positive_inputs_raise(self, count: int, seconds: float):
        with pytest.raises(ValueError):
            tokens_per_second(count, seconds)

    def test_forward_workload_counts_the_prompt(self):
        statistics = throughput_from_latency(latency_seconds=0.1, batch_size=2, sequence_length=128)
        assert statistics.workload == "forward"
        assert statistics.tokens_per_call == 256
        assert statistics.tokens_per_second == pytest.approx(2560.0)
        assert statistics.samples_per_second == pytest.approx(20.0)

    def test_generate_workload_counts_new_tokens(self):
        statistics = throughput_from_latency(
            latency_seconds=1.0, batch_size=1, sequence_length=128, generated_tokens=64
        )
        assert statistics.workload == "generate"
        assert statistics.tokens_per_call == 64
        assert statistics.tokens_per_second == pytest.approx(64.0)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError, match="batch_size"):
            throughput_from_latency(latency_seconds=0.1, batch_size=0, sequence_length=128)


class TestMemory:
    def test_current_memory_is_positive_or_unavailable(self):
        value = process_memory_mb()
        assert value is None or value > 0

    def test_tracker_reports_a_result(self):
        with MemoryTracker() as tracker:
            _ = [0] * 1000
        result = tracker.result()
        if result.available:
            assert result.baseline_mb is not None
            assert result.peak_mb is not None
            assert result.peak_mb >= result.baseline_mb
        else:
            assert result.peak_mb is None

    def test_tracker_result_serialises(self):
        with MemoryTracker() as tracker:
            pass
        payload = tracker.result().to_dict()
        assert "peak_memory_mb" in payload
        assert "memory_measurement_available" in payload


class TestCpuOnlyPolicy:
    def test_runner_rejects_a_non_cpu_device(self):
        """Constructed directly, bypassing the config validator, the runner must still refuse."""
        config = BenchmarkConfig.__new__(BenchmarkConfig)
        object.__setattr__(config, "device", Device.CUDA)
        with pytest.raises(BenchmarkError, match="CPU-only"):
            CpuBenchmarkRunner(config)

    def test_config_rejects_a_non_cpu_device(self):
        with pytest.raises(ConfigError, match="CPU-only"):
            BenchmarkConfig(device=Device.CUDA)


class TestRunnerProtocol:
    def test_runs_the_full_protocol(self, fast_config: BenchmarkConfig):
        calls = 0

        def workload() -> int:
            nonlocal calls
            calls += 1
            return calls

        result = CpuBenchmarkRunner(fast_config).run(workload, label="dummy")

        assert isinstance(result, BenchmarkResult)
        assert calls == fast_config.warmup_runs + fast_config.measured_runs
        assert result.latency.num_runs == fast_config.measured_runs
        assert result.label == "dummy"
        assert result.device == "cpu"

    def test_warmup_runs_are_untimed(self, fast_config: BenchmarkConfig):
        """Only measured_runs samples may appear, or one-off costs pollute the statistics."""
        result = CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert len(result.per_run_latencies_ms) == fast_config.measured_runs

    def test_per_run_latencies_can_be_suppressed(self, fast_config: BenchmarkConfig):
        fast_config.record_per_run_latencies = False
        result = CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert result.per_run_latencies_ms == []

    def test_zero_warmup_warns(
        self, fast_config: BenchmarkConfig, caplog: pytest.LogCaptureFixture
    ):
        fast_config.warmup_runs = 0
        with caplog.at_level("WARNING"):
            CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert any("warmup_runs=0" in message for message in caplog.messages)

    def test_records_the_pinned_thread_count(self, fast_config: BenchmarkConfig):
        result = CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert result.num_threads == fast_config.num_threads
        assert result.thread_report["requested_num_threads"] == fast_config.num_threads

    def test_captures_hardware_and_software_metadata(self, fast_config: BenchmarkConfig):
        """Without this, latencies from different machines are indistinguishable in the results."""
        result = CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert result.hardware["platform"]
        assert result.hardware["cpu_count_logical"]
        assert "python" in result.software

    def test_records_the_workload_shape(self, fast_config: BenchmarkConfig):
        result = CpuBenchmarkRunner(fast_config).run(lambda: None)
        assert result.batch_size == fast_config.batch_size
        assert result.sequence_length == fast_config.sequence_length
        assert result.throughput.tokens_per_call == (
            fast_config.batch_size * fast_config.sequence_length
        )

    def test_a_failing_workload_is_wrapped(self, fast_config: BenchmarkConfig):
        def broken() -> None:
            raise RuntimeError("kernel exploded")

        with pytest.raises(BenchmarkError, match="failed while running"):
            CpuBenchmarkRunner(fast_config).run(broken, label="broken")

    def test_result_serialises_to_a_nested_mapping(self, fast_config: BenchmarkConfig):
        payload = CpuBenchmarkRunner(fast_config).run(lambda: None).to_dict()
        for key in (
            "latency_median_ms",
            "latency_p95_ms",
            "throughput_tokens_per_s",
            "num_threads",
            "batch_size",
            "sequence_length",
            "hardware",
            "software",
        ):
            assert key in payload

    def test_summary_line_reports_median_and_p95(self, fast_config: BenchmarkConfig):
        line = CpuBenchmarkRunner(fast_config).run(lambda: None).summary_line()
        assert "median=" in line
        assert "p95=" in line
        assert "tok/s" in line

    def test_measure_returns_one_sample_per_run(self, fast_config: BenchmarkConfig):
        samples = CpuBenchmarkRunner(fast_config).measure(lambda: None)
        assert len(samples) == fast_config.measured_runs
        assert all(sample >= 0 for sample in samples)


class TestBuildForwardCallableValidation:
    """Validation only; the working paths are exercised in test_evaluation.py."""

    def test_rejects_a_non_cpu_device(self):
        config = BenchmarkConfig.__new__(BenchmarkConfig)
        object.__setattr__(config, "device", Device.CUDA)
        with pytest.raises(BenchmarkError, match="CPU-only"):
            build_forward_callable(object(), object(), config)  # type: ignore[arg-type]

    def test_reports_an_unusable_vocabulary_size(self):
        class NoVocabulary:
            config = None

            def eval(self) -> None: ...

            def to(self, _: object) -> None: ...

        with pytest.raises(BenchmarkError, match="vocabulary size"):
            build_forward_callable(NoVocabulary(), object(), BenchmarkConfig())  # type: ignore[arg-type]


class TestCheckpointMeasurement:
    def _write_checkpoint(self, directory: Path, weight_bytes: int, json_bytes: int = 200) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.safetensors").write_bytes(b"w" * weight_bytes)
        (directory / "config.json").write_bytes(b"j" * json_bytes)
        return directory

    def test_measures_weights_separately_from_total(self, tmp_path: Path):
        path = self._write_checkpoint(tmp_path / "ckpt", weight_bytes=4096, json_bytes=100)
        report = measure_checkpoint(path)
        assert isinstance(report, CheckpointSizeReport)
        assert report.weight_bytes == 4096
        assert report.total_bytes == 4196
        assert report.file_count == 2

    def test_reports_storage_efficiency_against_the_budget(self, tmp_path: Path):
        path = self._write_checkpoint(tmp_path / "ckpt", weight_bytes=1000)
        report = measure_checkpoint(path, nonzero_parameters=1000, bits=8)
        assert report.theoretical_weight_bytes == 1000
        assert report.storage_efficiency == pytest.approx(1.0)

    def test_low_storage_efficiency_warns(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        """An artefact far larger than its budget means the conversion silently no-oped."""
        path = self._write_checkpoint(tmp_path / "ckpt", weight_bytes=4000)
        with caplog.at_level("WARNING"):
            measure_checkpoint(path, nonzero_parameters=1000, bits=8)
        assert any("storage efficiency" in message for message in caplog.messages)

    def test_missing_weight_files_warn(self, tmp_path: Path, caplog: pytest.LogCaptureFixture):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "config.json").write_bytes(b"{}")
        with caplog.at_level("WARNING"):
            report = measure_checkpoint(directory)
        assert report.weight_bytes == 0
        assert any("No weight files" in message for message in caplog.messages)

    def test_measures_a_single_file(self, tmp_path: Path):
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"x" * 512)
        report = measure_checkpoint(path)
        assert report.weight_bytes == 512
        assert report.file_count == 1

    def test_no_theoretical_size_without_a_parameter_count(self, tmp_path: Path):
        path = self._write_checkpoint(tmp_path / "ckpt", weight_bytes=100)
        report = measure_checkpoint(path)
        assert report.theoretical_weight_bytes is None
        assert report.storage_efficiency is None

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            measure_checkpoint(tmp_path / "absent")

    def test_compares_against_the_dense_baseline(self, tmp_path: Path):
        dense = measure_checkpoint(self._write_checkpoint(tmp_path / "dense", 8000))
        compressed = measure_checkpoint(self._write_checkpoint(tmp_path / "compressed", 1000))
        comparison = compare_to_baseline(dense, compressed)
        assert comparison["compression_ratio"] == pytest.approx(8.0)
        assert comparison["size_reduction_percentage"] == pytest.approx(87.5)

    def test_report_serialises_flat(self, tmp_path: Path):
        payload = measure_checkpoint(
            self._write_checkpoint(tmp_path / "ckpt", 1024), nonzero_parameters=1024, bits=8
        ).to_dict()
        assert payload["checkpoint_size_mb"] == pytest.approx(1024 / (1024 * 1024))
        assert all(not isinstance(value, dict | list) for value in payload.values())


class TestNoImportTimeSideEffects:
    def test_importing_benchmarking_does_not_import_torch(self, imported_after):
        assert imported_after("scale_aware_compression.benchmarking", ["torch"]) == [], (
            "importing the benchmarking subpackage must not import torch, and must not pin "
            "threads or start timing anything"
        )

    def test_importing_benchmarking_does_not_set_thread_environment(self, environment_after_import):
        observed = environment_after_import(
            "scale_aware_compression.benchmarking.cpu",
            ["OMP_NUM_THREADS", "MKL_NUM_THREADS"],
        )
        assert observed == {"OMP_NUM_THREADS": None, "MKL_NUM_THREADS": None}, (
            "thread counts must be pinned only inside CpuBenchmarkRunner.run, never at import"
        )
