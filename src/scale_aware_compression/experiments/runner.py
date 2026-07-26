"""Experiment records, result tracking, and the single-run orchestrator.

Two things live here. :class:`ExperimentRecord` and :class:`ExperimentTracker` are implemented:
they define the result schema and persist it as JSON plus a flat CSV row. :class:`ExperimentRunner`
is a placeholder that wires config -> model -> compression -> evaluation -> benchmark together.

Result schema, per run:

* identity: experiment id, timestamp, git commit, schema version
* model: name, size label, parameter count
* compression: method, sparsity, quantisation bits, budget label, optimiser steps
* seed
* quality: perplexity, retention, agreement
* deployment (CPU): latency mean/median/p95/std, throughput, peak memory, checkpoint size
* environment: full hardware metadata and resolved software versions
* the resolved configuration itself

The environment fields are not optional decoration. CPU latencies from two different machines
are not comparable, and without the metadata a results table cannot be audited for that mistake
after the fact.
"""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import (
    RESULT_CSV_COLUMNS,
    RESULT_CSV_NAME,
    RESULT_SCHEMA_VERSION,
    CompressionMethod,
)
from scale_aware_compression.hardware import get_hardware_info, get_software_versions
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scale_aware_compression.benchmarking.cpu import BenchmarkResult
    from scale_aware_compression.compression.base import CompressionResult
    from scale_aware_compression.evaluation.quality import QualityReport

LOGGER = get_logger(__name__)


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be run or recorded."""


def utc_timestamp() -> str:
    """Current UTC time as an ISO-8601 string with second precision."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def get_git_commit(*, short: bool = False) -> str | None:
    """Resolve the current git commit, or ``None`` outside a repository.

    Args:
        short: Return the abbreviated hash.

    Returns:
        The commit hash, with ``-dirty`` appended when the working tree has uncommitted
        changes. ``None`` if git is unavailable or this is not a repository.
    """
    command = ["git", "rev-parse", "--short" if short else "HEAD"]
    if short:
        command = ["git", "rev-parse", "--short", "HEAD"]
    try:
        completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        LOGGER.debug("Could not resolve git commit: %s", error)
        return None
    if completed.returncode != 0:
        LOGGER.debug("git rev-parse failed: %s", completed.stderr.strip())
        return None
    commit = completed.stdout.strip()

    try:
        status = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return commit
    if status.returncode == 0 and status.stdout.strip():
        # A dirty tree means the recorded commit does not fully describe the code that ran.
        return f"{commit}-dirty"
    return commit


def make_experiment_id(
    *,
    model_name: str,
    method: CompressionMethod | str,
    budget_label: str,
    seed: int,
    sparsity: float,
    bits: int,
) -> str:
    """Build a deterministic, filename-safe experiment identifier.

    Deterministic rather than random, so re-running a cell overwrites its own record instead of
    accumulating near-duplicates that later get averaged together.

    Args:
        model_name: Registry short name.
        method: Compression method.
        budget_label: Compression budget label.
        seed: Run seed.
        sparsity: Target sparsity.
        bits: Target weight bit width.

    Returns:
        An identifier such as ``pythia-410m_joint_moderate_s50_b8_seed1234``.
    """
    method_value = method.value if isinstance(method, CompressionMethod) else str(method)
    safe_model = model_name.replace("/", "-").replace(" ", "-")
    return (
        f"{safe_model}_{method_value}_{budget_label}"
        f"_s{round(sparsity * 100):02d}_b{bits}_seed{seed}"
    )


@dataclass(slots=True)
class ExperimentRecord:
    """One row of the study's results, in full.

    The nested ``quality``, ``deployment``, ``compression``, ``hardware``, ``software``, and
    ``config`` mappings are written to JSON. :meth:`to_csv_row` flattens the subset in
    :data:`~scale_aware_compression.constants.RESULT_CSV_COLUMNS` for plotting and tables.
    """

    experiment_id: str
    model_name: str
    compression_method: str
    seed: int
    timestamp: str = field(default_factory=utc_timestamp)
    git_commit: str | None = field(default_factory=get_git_commit)
    schema_version: str = RESULT_SCHEMA_VERSION

    model_size_label: str = ""
    parameter_count: int = 0
    budget_label: str = ""
    sparsity: float = 0.0
    quantisation_bits: int = 32

    quality: dict[str, Any] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    compression: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    @classmethod
    def from_config(
        cls, config: ExperimentConfig, *, capture_environment: bool = True
    ) -> ExperimentRecord:
        """Start a record from a configuration, before any results exist.

        Args:
            config: The validated experiment config.
            capture_environment: Collect hardware and software metadata now. Kept switchable so
                tests can build records without touching the machine.

        Returns:
            A record with identity and configuration fields populated.
        """
        compression = config.compression
        return cls(
            experiment_id=config.experiment.id,
            model_name=config.model.name,
            compression_method=compression.method.value,
            seed=config.runtime.seed,
            model_size_label=config.model.size_label,
            budget_label=compression.budget_label,
            sparsity=compression.effective_sparsity,
            quantisation_bits=compression.effective_bits,
            hardware=get_hardware_info() if capture_environment else {},
            software=dict(get_software_versions()) if capture_environment else {},
            config=config.to_dict(),
        )

    def add_quality(self, report: QualityReport) -> None:
        """Attach a quality report.

        Args:
            report: The evaluation result.
        """
        self.quality = report.to_dict()

    def add_benchmark(self, result: BenchmarkResult) -> None:
        """Attach a CPU benchmark result.

        Args:
            result: The benchmark measurement. Its hardware metadata replaces the record's, so
                the environment stored is the one the deployment numbers were measured in.
        """
        self.deployment = result.to_dict()
        if result.hardware:
            self.hardware = result.hardware
        if result.software:
            self.software = dict(result.software)

    def add_compression(self, result: CompressionResult) -> None:
        """Attach a compression result.

        Args:
            result: The compression pipeline's output.
        """
        self.compression = result.to_dict()

    def to_dict(self) -> dict[str, Any]:
        """Return the full nested record, JSON-serialisable."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_size_label": self.model_size_label,
            "parameter_count": self.parameter_count,
            "compression_method": self.compression_method,
            "budget_label": self.budget_label,
            "sparsity": self.sparsity,
            "quantisation_bits": self.quantisation_bits,
            "seed": self.seed,
            "quality": self.quality,
            "deployment": self.deployment,
            "compression": self.compression,
            "hardware": self.hardware,
            "software": self.software,
            "config": self.config,
            "notes": self.notes,
        }

    def to_csv_row(self) -> dict[str, Any]:
        """Flatten the record into the fixed CSV column order.

        Returns:
            A mapping with exactly the keys in
            :data:`~scale_aware_compression.constants.RESULT_CSV_COLUMNS`. Missing measurements
            are ``None`` rather than absent, so every row has the same shape.
        """
        quality = self.quality
        retention = quality.get("retention", {}) if isinstance(quality, dict) else {}
        perplexity = quality.get("perplexity", {}) if isinstance(quality, dict) else {}
        agreement = quality.get("agreement", {}) if isinstance(quality, dict) else {}
        deployment = self.deployment
        compression = self.compression
        statistics = compression.get("statistics", {}) if isinstance(compression, dict) else {}

        row: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "model_name": self.model_name,
            "model_size_label": self.model_size_label,
            "parameter_count": self.parameter_count,
            "compression_method": self.compression_method,
            "sparsity": self.sparsity,
            "quantisation_bits": self.quantisation_bits,
            "seed": self.seed,
            "perplexity": perplexity.get("perplexity"),
            "accuracy": quality.get("accuracy") if isinstance(quality, dict) else None,
            "quality_retention": retention.get("perplexity_retention"),
            "top1_agreement": agreement.get("top1_agreement"),
            "latency_mean_ms": deployment.get("latency_mean_ms"),
            "latency_median_ms": deployment.get("latency_median_ms"),
            "latency_p95_ms": deployment.get("latency_p95_ms"),
            "latency_std_ms": deployment.get("latency_std_ms"),
            "throughput_tokens_per_s": deployment.get("throughput_tokens_per_s"),
            "peak_memory_mb": deployment.get("peak_memory_mb"),
            "checkpoint_size_mb": statistics.get("checkpoint_size_mb"),
            "compression_ratio": statistics.get("compression_ratio"),
            "benchmark_num_threads": deployment.get("num_threads"),
            "benchmark_batch_size": deployment.get("batch_size"),
            "benchmark_sequence_length": deployment.get("sequence_length"),
            "hardware_cpu_model": self.hardware.get("cpu_model"),
            "software_torch_version": self.software.get("torch"),
        }
        return {column: row.get(column) for column in RESULT_CSV_COLUMNS}


@dataclass(slots=True)
class ExperimentTracker:
    """Persists records as one JSON file per run plus one shared CSV.

    JSON keeps the full nested detail; the CSV is the flat view plotting and tables read. Both
    are written, because a results table that cannot be traced back to its full record is not
    auditable.
    """

    output_dir: Path
    csv_name: str = RESULT_CSV_NAME

    def __post_init__(self) -> None:
        """Normalise the output directory. No directory is created until a write happens."""
        self.output_dir = Path(self.output_dir)

    @property
    def csv_path(self) -> Path:
        """Path of the aggregated CSV."""
        return self.output_dir / self.csv_name

    def record_path(self, experiment_id: str) -> Path:
        """Path of a single run's JSON record.

        Args:
            experiment_id: The run identifier.

        Returns:
            The JSON path.
        """
        return self.output_dir / f"{experiment_id}.json"

    def exists(self, experiment_id: str) -> bool:
        """Whether a record already exists for this run.

        Used by the sweep's ``skip_existing`` so an interrupted sweep can be resumed.

        Args:
            experiment_id: The run identifier.

        Returns:
            ``True`` if the JSON record is present.
        """
        return self.record_path(experiment_id).is_file()

    def save(self, record: ExperimentRecord) -> Path:
        """Write a record to JSON and append its row to the CSV.

        Args:
            record: The record to persist.

        Returns:
            The JSON path written.

        Raises:
            ExperimentError: If the record cannot be serialised or written.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.record_path(record.experiment_id)
        try:
            destination.write_text(
                json.dumps(record.to_dict(), indent=2, sort_keys=False, default=str),
                encoding="utf-8",
            )
        except OSError as error:
            raise ExperimentError(f"Could not write record to {destination}: {error}") from error
        self.append_csv_row(record)
        LOGGER.info("Recorded %s -> %s", record.experiment_id, destination)
        return destination

    def append_csv_row(self, record: ExperimentRecord) -> Path:
        """Append one row to the aggregated CSV, writing the header if needed.

        Args:
            record: The record to flatten.

        Returns:
            The CSV path.

        Raises:
            ExperimentError: If the existing CSV has a different schema version, since
                concatenating incompatible schemas would silently misalign columns.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.csv_path
        write_header = not path.exists()

        if not write_header:
            with path.open("r", encoding="utf-8", newline="") as handle:
                existing_header = next(csv.reader(handle), [])
            if existing_header and tuple(existing_header) != RESULT_CSV_COLUMNS:
                raise ExperimentError(
                    f"{path} was written with a different result schema. Move it aside (or bump "
                    "RESULT_SCHEMA_VERSION and start a new file) before appending."
                )

        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RESULT_CSV_COLUMNS))
            if write_header:
                writer.writeheader()
            writer.writerow(record.to_csv_row())
        return path

    def load(self, experiment_id: str) -> dict[str, Any]:
        """Read one run's JSON record.

        Args:
            experiment_id: The run identifier.

        Returns:
            The parsed record.

        Raises:
            ExperimentError: If the record is missing or unparseable.
        """
        path = self.record_path(experiment_id)
        if not path.is_file():
            raise ExperimentError(f"No record found at {path}")
        try:
            parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ExperimentError(f"Could not read record {path}: {error}") from error
        return parsed

    def load_all(self) -> list[dict[str, Any]]:
        """Read every JSON record in the output directory.

        Returns:
            Records sorted by experiment id. Unreadable files are skipped with a warning rather
            than aborting an aggregation over dozens of runs.
        """
        if not self.output_dir.is_dir():
            return []
        records: list[dict[str, Any]] = []
        for path in sorted(self.output_dir.glob("*.json")):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as error:
                LOGGER.warning("Skipping unreadable record %s: %s", path, error)
        return records


@dataclass(slots=True)
class ExperimentRunner:
    """Runs one experiment end to end and records it.

    Stage order is fixed: load, compress (on GPU if configured), evaluate quality on CPU,
    benchmark on CPU, record. Every arm goes through the same sequence.

    Status: placeholder.
    """

    config: ExperimentConfig
    tracker: ExperimentTracker | None = None

    def __post_init__(self) -> None:
        """Default the tracker to ``<output_dir>/metrics``."""
        if self.tracker is None:
            self.tracker = ExperimentTracker(self.config.runtime.output_dir / "metrics")

    def run(self) -> ExperimentRecord:
        """Execute the configured experiment.

        Returns:
            The completed record.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(experiments): implement in this order.
        #   1. set_global_seed(config.runtime.seed, deterministic=config.runtime.deterministic)
        #   2. record = ExperimentRecord.from_config(config)
        #   3. loaded = models.loader.load_model_and_tokenizer(config.model);
        #      record.parameter_count = loaded.parameter_count
        #   4. compressor = compression.get_compressor(config)
        #      - None for the dense arm: evaluate the loaded model directly
        #      - otherwise compressor.run(model, tokenizer) and record.add_compression()
        #   5. compressor.save() into <output_dir>/<id>/checkpoint, then
        #      benchmarking.checkpoint_size.measure_checkpoint() for the deployment size
        #   6. move the model to CPU, then evaluation.quality.evaluate_model() with the dense
        #      run's perplexity as dense_reference -- loaded from its record, not recomputed,
        #      so the baseline is identical across arms
        #   7. benchmarking.cpu.benchmark_model() on CPU and record.add_benchmark()
        #   8. self.tracker.save(record)
        # Fail loudly if step 6 has no dense reference: the row would carry no primary score
        # and could not contribute to a joint-gain comparison.
        raise NotImplementedError(
            "ExperimentRunner.run is not implemented yet; see the TODO in experiments/runner.py"
        )

    def dry_run(self) -> dict[str, Any]:
        """Validate the configuration and report what would happen, without running it.

        Returns:
            A summary of the planned run: the resolved identity, the pipeline stages, the
            output paths, and whether a record already exists.
        """
        from scale_aware_compression.compression import COMPRESSOR_REGISTRY

        method = self.config.compression.method
        compressor_class = COMPRESSOR_REGISTRY.get(method)
        stages = (
            [stage.value for stage in compressor_class.pipeline_stages]
            if compressor_class is not None
            else ["dense"]
        )
        assert self.tracker is not None  # set in __post_init__
        return {
            "experiment_id": self.config.experiment.id,
            "description": self.config.describe(),
            "model": self.config.model.name,
            "method": method.value,
            "pipeline_stages": stages,
            "target_sparsity": self.config.compression.effective_sparsity,
            "target_bits": self.config.compression.effective_bits,
            "seed": self.config.runtime.seed,
            "output_dir": self.config.run_output_dir.as_posix(),
            "record_path": self.tracker.record_path(self.config.experiment.id).as_posix(),
            "record_exists": self.tracker.exists(self.config.experiment.id),
            "benchmark": {
                "device": self.config.benchmark.device.value,
                "num_threads": self.config.benchmark.num_threads,
                "batch_size": self.config.benchmark.batch_size,
                "sequence_length": self.config.benchmark.sequence_length,
                "warmup_runs": self.config.benchmark.warmup_runs,
                "measured_runs": self.config.benchmark.measured_runs,
            },
        }
