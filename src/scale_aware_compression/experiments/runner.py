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
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import (
    METHOD_VERSION,
    RESULT_CSV_COLUMNS,
    RESULT_CSV_NAME,
    RESULT_SCHEMA_VERSION,
    CompressionMethod,
)
from scale_aware_compression.hardware import get_hardware_info, get_software_versions, host_key
from scale_aware_compression.logging_utils import get_logger, log_stage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scale_aware_compression.benchmarking.cpu import BenchmarkResult
    from scale_aware_compression.compression.base import CompressionResult
    from scale_aware_compression.evaluation.quality import QualityReport

LOGGER = get_logger(__name__)


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be run or recorded."""


def _release_device_cache(device: str) -> None:
    """Return the caching allocator's free blocks to the driver.

    A no-op away from CUDA, and cheap on it. Called at the boundaries between the compression and
    evaluation stages, which are the two large consumers: without it the allocator holds one
    stage's peak while the next stage asks for its own, and on Windows the driver satisfies the
    shortfall from shared system memory instead of raising -- so the symptom is a run that is
    several times slower with nothing in the log, not an error.

    Args:
        device: The device the next stage will use.
    """
    if not str(device).startswith("cuda"):
        return
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


def _measured_zeros(model: Any) -> int:
    """Count exactly-zero parameters, returning 0 if the model cannot be scanned.

    Used to turn a parameter count into a *non-zero* count for the theoretical size estimate.
    Best-effort: a failure here should not lose an otherwise complete run.
    """
    from scale_aware_compression.metrics.compression import count_zero_parameters

    try:
        return count_zero_parameters(model)
    except Exception as error:  # pragma: no cover - defensive
        LOGGER.debug("Could not count zero parameters: %s", error)
        return 0


def make_experiment_id(
    *,
    model_name: str,
    method: CompressionMethod | str,
    budget_label: str,
    seed: int,
    sparsity: float,
    bits: int,
    replicate: int | None = None,
) -> str:
    """Build a deterministic, filename-safe experiment identifier.

    Deterministic rather than random, so re-running a cell overwrites its own record instead of
    accumulating near-duplicates that later get averaged together.

    ``replicate`` is what distinguishes the paired calibration draws A1 §5.1 introduced, and it must
    appear here: eight replicates of one cell are eight *different* compressed models, so an
    identifier that omitted the replicate would have each one overwrite the last and leave a single
    record where the error bar should be.

    The run ``seed`` is retained in the identifier only when no replicate is given. The seed is inert
    under this method (F-15), so it distinguishes nothing -- but records taken before A1 are named by
    it, and changing their names would orphan them.

    Args:
        model_name: Registry short name.
        method: Compression method.
        budget_label: Compression budget label.
        seed: Run seed. Ignored when ``replicate`` is given.
        sparsity: Target sparsity.
        bits: Target weight bit width.
        replicate: Calibration replicate index, or ``None`` for a single-draw run.

    Returns:
        An identifier such as ``pythia-410m_joint_moderate_s50_b8_rep3``, or
        ``..._seed1234`` when no replicate is given.
    """
    method_value = method.value if isinstance(method, CompressionMethod) else str(method)
    safe_model = model_name.replace("/", "-").replace(" ", "-")
    suffix = f"rep{replicate}" if replicate is not None else f"seed{seed}"
    return (
        f"{safe_model}_{method_value}_{budget_label}_s{round(sparsity * 100):02d}_b{bits}_{suffix}"
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
    runtime_representation: str = ""
    """What the artefact executes as: ``fp32`` or ``packed_dequantising``.

    Distinct from the storage format. §4.7 forbids comparing latencies across backends, and this is
    the field that makes such a comparison detectable after the fact -- a table mixing the two is
    mixing native FP32 kernels with a dequantise-then-matmul emulation.
    """
    experiment_group: str = ""
    """The configured ``experiment.id``, kept as a grouping label.

    Distinct from :attr:`experiment_id`, which follows §5.6's convention and encodes the model, arm,
    budget and seed so two runs cannot collide.
    """
    timestamp: str = field(default_factory=utc_timestamp)
    git_commit: str | None = field(default_factory=get_git_commit)
    schema_version: str = RESULT_SCHEMA_VERSION
    method_version: str = METHOD_VERSION
    """Which version of the algorithm produced this record.

    The git commit is too strict for a resume check -- every code change would invalidate every
    record -- and no check at all is too loose, which is how three successive joint-gain figures were
    each computed by a different algorithm before anyone noticed. This is the deliberate middle.
    """

    model_size_label: str = ""
    parameter_count: int = 0
    budget_label: str = ""
    sparsity: float = 0.0
    quantisation_bits: int = 32

    quality: dict[str, Any] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    compression: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    software: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    model_details: dict[str, Any] = field(default_factory=dict)
    seed_details: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None
    duration_seconds: float = 0.0
    status: str = "unknown"
    """``success``, ``failure``, ``running``, or ``unknown``.

    A failed run is still written, with its reason in ``notes``. A sweep with silently missing
    cells is harder to diagnose than one with recorded failures."""
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
        canonical = make_experiment_id(
            model_name=config.model.name,
            method=compression.method,
            budget_label=compression.budget_label,
            seed=config.runtime.seed,
            sparsity=compression.effective_sparsity,
            bits=compression.effective_bits,
            replicate=config.data.calibration_replicate,
        )
        # §5.6's convention, so a record is keyed by everything that makes it a distinct measurement.
        # Taking `experiment.id` alone meant two runs differing only in seed shared one identifier and
        # the second silently overwrote the first -- which also destroyed the dense baseline a
        # compressed run needed for its retention. The configured id is kept as a prefix when it adds
        # information, so a deliberately separated variant (a different evaluation window, say) still
        # gets its own record.
        label = config.experiment.id
        experiment_id = canonical if label in {"unnamed", canonical} else f"{label}__{canonical}"
        return cls(
            experiment_id=experiment_id,
            experiment_group=label,
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
            "experiment_group": self.experiment_group,
            "runtime_representation": self.runtime_representation,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "model_name": self.model_name,
            "model_size_label": self.model_size_label,
            "parameter_count": self.parameter_count,
            "compression_method": self.compression_method,
            "budget_label": self.budget_label,
            "sparsity": self.sparsity,
            "quantisation_bits": self.quantisation_bits,
            "seed": self.seed,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "quality": self.quality,
            "deployment": self.deployment,
            "compression": self.compression,
            "checkpoint": self.checkpoint,
            "checkpoint_path": self.checkpoint_path.as_posix() if self.checkpoint_path else None,
            "model": self.model_details,
            "seeding": self.seed_details,
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
        checkpoint = self.checkpoint

        row: dict[str, Any] = {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "schema_version": self.schema_version,
            "method_version": self.method_version,
            "status": self.status,
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
            "checkpoint_size_mb": checkpoint.get("checkpoint_size_mb")
            or statistics.get("checkpoint_size_mb"),
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
        """Whether *any* record file exists for this run, valid or not.

        Args:
            experiment_id: The run identifier.

        Returns:
            ``True`` if the JSON record is present.
        """
        return self.record_path(experiment_id).is_file()

    def exists_valid(self, experiment_id: str, config: ExperimentConfig) -> bool:
        """Whether a **usable** record exists, so a sweep may legitimately skip this cell.

        File presence alone is not enough, and using it as the resume condition is a way to keep a
        stale result silently. A record only stands in for running the cell if it succeeded and the
        run that produced it matches the one being asked for now:

        * ``status == "success"`` -- a crashed or half-written record must be re-run, not skipped.
        * the model **revision** matches, or the cell would be skipped on the strength of a run
          against different weights.
        * the compression **budget** matches -- sparsity and bit width.
        * the **schema version** matches, since an older schema may lack fields the analysis needs.

        Deliberately does **not** compare the git commit. Every code change would then invalidate
        every record and resumption would be useless during development. The cost of that choice is
        recorded rather than hidden: a sweep resumed across a method change mixes results, so the
        findings log carries an explicit note whenever numbers predate a method change.

        Args:
            experiment_id: The run identifier.
            config: The configuration the cell would run with now.

        Returns:
            ``True`` if the existing record can stand in for running this cell.
        """
        path = self.record_path(experiment_id)
        if not path.is_file():
            return False
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            LOGGER.warning("Record %s is unreadable (%s); will re-run", path, error)
            return False

        reasons: list[str] = []
        status = record.get("status")
        if status != "success":
            reasons.append(f"status is {status}")
        if record.get("schema_version") != RESULT_SCHEMA_VERSION:
            reasons.append(f"schema {record.get('schema_version')} is not {RESULT_SCHEMA_VERSION}")
        if record.get("method_version") != METHOD_VERSION:
            reasons.append(f"method version {record.get('method_version')} is not {METHOD_VERSION}")

        recorded_data = record.get("config", {}).get("data", {}) or {}
        for key, expected in (
            ("dataset", config.data.dataset),
            ("subset", config.data.subset),
            ("eval_split", config.data.eval_split),
            ("sequence_length", config.data.sequence_length),
            ("calibration_seed", config.data.calibration_seed),
            # The replicate index is what actually distinguishes the draws, and it must be compared
            # separately: `to_dict` serialises dataclass *fields*, so the record carries the
            # configured `calibration_seed` rather than the effective one. Comparing the effective
            # seed against a recorded raw seed would never match, and skip_existing would silently
            # stop working -- fail-safe, but it would re-run the whole grid every time.
            ("calibration_replicate", config.data.calibration_replicate),
            ("calibration_samples", config.data.calibration_samples),
        ):
            if recorded_data.get(key) != expected:
                reasons.append(f"data.{key} {recorded_data.get(key)} is not {expected}")

        # The DEVICE the quality number was produced on. GPU evaluation is ~22x faster and legitimate
        # for exploratory work -- `check_evaluation_device` warns rather than errors, because only
        # reported numbers must come from CPU. But CPU and GPU differ at ~1e-5 relative from
        # floating-point reduction order, so a grid that reused CPU records while writing GPU ones
        # would mix devices inside a single comparison. That is the unmatched-condition class of error
        # §3.11 exists to prevent, small enough here to change no conclusion and invisible without
        # this check. Recorded per run in `quality.perplexity.evaluation_device`.
        recorded_device = ((record.get("quality", {}) or {}).get("perplexity", {}) or {}).get(
            "evaluation_device"
        )
        if recorded_device is not None:
            expected_device = config.evaluation.device.value
            # `cuda` and `cuda:0` name the same device; compare only the backend.
            if str(recorded_device).split(":")[0] != str(expected_device).split(":")[0]:
                reasons.append(
                    f"quality was evaluated on {recorded_device!r} but this run evaluates on "
                    f"{expected_device!r}"
                )

        # The MACHINE the record was produced on. Once more than one host can run compression --
        # which the machine policy now permits, because compression and quality are portable in a
        # way CPU latency is not -- records from two hosts can land in the same directory. Without
        # this check `skip_existing` would reuse a record from the other machine, silently putting
        # two hosts inside one comparison. Same class as the evaluation-device gap above (B-32).
        # `host_key` is built from fields every record already carries, so existing records stay
        # valid; a record predating those fields reports "unknown" and is not invalidated.
        recorded_host = host_key(record.get("hardware") or {})
        if recorded_host != "unknown":
            current_host = host_key()
            if recorded_host != current_host:
                reasons.append(
                    f"record was produced on {recorded_host!r}, this machine is {current_host!r}"
                )

        recorded_reconstruction = (record.get("config", {}).get("compression", {}) or {}).get(
            "reconstruction", {}
        ) or {}
        reconstruction = config.compression.reconstruction
        for key, expected in (
            ("solver", reconstruction.solver.value),
            ("joint_iterations", reconstruction.joint_iterations),
            ("comparison_group", reconstruction.comparison_group.value),
            ("scale_search", reconstruction.scale_search),
            ("keep_benefit_saliency", reconstruction.keep_benefit_saliency),
        ):
            if recorded_reconstruction.get(key) != expected:
                reasons.append(
                    f"reconstruction.{key} {recorded_reconstruction.get(key)} is not {expected}"
                )

        recorded_granularity = (
            (record.get("config", {}).get("compression", {}) or {}).get("quantisation", {}) or {}
        ).get("granularity")
        if recorded_granularity != config.compression.quantisation.granularity.value:
            reasons.append("quantisation granularity differs")

        recorded = (record.get("config", {}).get("model", {}) or {}).get("revision")
        if config.model.revision and recorded != config.model.revision:
            reasons.append("model revision differs")

        recorded_sparsity = float(record.get("sparsity") or 0.0)
        if abs(recorded_sparsity - config.compression.effective_sparsity) > 1e-9:
            reasons.append(
                f"sparsity {recorded_sparsity} is not {config.compression.effective_sparsity}"
            )
        recorded_bits = int(record.get("quantisation_bits") or 32)
        if recorded_bits != config.compression.effective_bits:
            reasons.append(f"bits {recorded_bits} is not {config.compression.effective_bits}")

        if reasons:
            LOGGER.info("Re-running %s: %s", experiment_id, ", ".join(reasons))
            return False
        return True

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
        """Write one row to the aggregated CSV, **replacing** any existing row for the same run.

        Upsert rather than append. Re-running a cell overwrites its JSON record, so a plain append
        left two CSV rows for one experiment id -- and they disagree, because the second run is the
        one that counts. Anything later averaging the CSV would weight the stale row equally, which
        is the duplicate-record failure the audit is meant to reject.

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
        rows: list[dict[str, Any]] = []

        if path.exists():
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames and tuple(reader.fieldnames) != RESULT_CSV_COLUMNS:
                    raise ExperimentError(
                        f"{path} was written with a different result schema. Move it aside (or "
                        "bump RESULT_SCHEMA_VERSION and start a new file) before appending."
                    )
                rows = [row for row in reader if row.get("experiment_id") != record.experiment_id]

        rows.append(record.to_csv_row())

        # Rewritten whole rather than patched in place: one row per run keeps the file small, and a
        # partial in-place edit is much harder to reason about than a full rewrite.
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(RESULT_CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
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
    benchmark on CPU, record. Every arm goes through the same sequence, so no arm can
    accidentally acquire an advantage from a different order of operations.

    The dense arm is fully implemented. Compressed arms run as far as their compressor allows
    and will raise ``NotImplementedError`` from the compression stage until those are written.
    """

    config: ExperimentConfig
    tracker: ExperimentTracker | None = None

    def __post_init__(self) -> None:
        """Default the tracker to ``<output_dir>/metrics``."""
        if self.tracker is None:
            self.tracker = ExperimentTracker(self.config.runtime.output_dir / "metrics")

    def run(self) -> ExperimentRecord:
        """Execute the configured experiment and write its record.

        Returns:
            The completed record.

        Raises:
            ExperimentError: If a stage fails.
            NotImplementedError: From the compression stage, for arms not yet implemented.
        """
        from scale_aware_compression.benchmarking.cpu import benchmark_model
        from scale_aware_compression.compression import get_compressor
        from scale_aware_compression.evaluation.quality import evaluate_model
        from scale_aware_compression.models.loader import load_model_and_tokenizer
        from scale_aware_compression.seed import set_global_seed

        assert self.tracker is not None  # set in __post_init__
        config = self.config
        started = time.perf_counter()

        LOGGER.info("=" * 78)
        LOGGER.info("%s", config.describe())
        LOGGER.info("=" * 78)

        # 1. Seed everything before anything stochastic happens.
        seed_record = set_global_seed(
            config.runtime.seed, deterministic=config.runtime.deterministic
        )

        # 2. Start the record. Hardware and software metadata are captured here.
        record = ExperimentRecord.from_config(config)
        record.seed_details = seed_record
        record.status = "running"

        try:
            # 3. Load.
            with log_stage(LOGGER, "load model"):
                loaded = load_model_and_tokenizer(config.model)
                record.parameter_count = loaded.parameter_count
                record.model_size_label = record.model_size_label or loaded.spec.size_label
                record.model_details = loaded.describe()

            # 4. Compress. `None` means the dense arm, which is the model as loaded.
            compressor = get_compressor(config)
            model = loaded.model
            if compressor is not None:
                # Layerwise reconstruction needs calibration activations. They are drawn here, once,
                # from `data.calibration_seed` rather than the run seed, and handed to the arm --
                # so varying the run seed for error bars does not also change the calibration set,
                # and every arm at this budget sees byte-identical data (§3.11).
                self._attach_calibration(compressor, loaded)
                with log_stage(LOGGER, f"compress ({compressor.name})"):
                    compression_result = compressor.run(model, loaded.tokenizer)
                    model = compression_result.model
                    record.add_compression(compression_result)

            # 5. Deployment artefact size.
            with log_stage(LOGGER, "measure checkpoint"):
                self._measure_artefact(record, compressor, model, loaded)

            # 6. Quality, on `evaluation.device`.
            #
            # CPU is the default and is what every reported number must come from -- but this stage
            # is 86% of an exploratory cell and GPU runs it 22.5x faster for a relative difference of
            # 8.3e-06, the same magnitude as CPU thread-configuration sensitivity (F-29). The field
            # has always existed and `check_evaluation_device` warns rather than errors, precisely so
            # exploratory work can use it; nothing was reading it.
            #
            # Safe because `exists_valid` compares the recorded evaluation device (B-32), so a grid
            # cannot silently reuse CPU records while writing GPU ones. Retention and joint gain stay
            # internally consistent either way: both arms of a cell, and its dense reference, are
            # evaluated the same way.
            evaluation_device = config.evaluation.device.value
            with log_stage(LOGGER, f"evaluate quality ({evaluation_device})"):
                # Release the compression stage's cached blocks first. The caching allocator holds
                # freed memory, so at 1B the Gram temporaries stay reserved while the model is moved
                # on for evaluation -- and on Windows the driver answers the shortfall by falling
                # back to shared system memory rather than raising, which is the silent 7x
                # slowdown F-29 first measured. Costs a few milliseconds and a re-warm.
                _release_device_cache(evaluation_device)
                model.to(evaluation_device)
                report = evaluate_model(
                    model,
                    loaded.tokenizer,
                    config,
                    dense_reference=self._load_dense_reference(),
                )
                record.add_quality(report)

            # 7. Deployment measurements, on CPU -- always, whatever the evaluation device was.
            model.to("cpu")
            # Same reasoning in the other direction: the next cell's compression starts with the
            # card still holding this cell's evaluation, and it is the compression that then spills.
            _release_device_cache(evaluation_device)
            runtime = self._runtime_representation()
            record.runtime_representation = runtime
            if self._latency_is_meaningful(runtime):
                with log_stage(LOGGER, "benchmark (CPU)"):
                    benchmark = benchmark_model(
                        model,
                        loaded.tokenizer,
                        config.benchmark,
                        label=f"{config.model.name}/{config.compression.method.value}",
                    )
                    record.add_benchmark(benchmark)
            else:
                # Deliberately leaves `deployment` empty rather than filling it with a number that
                # cannot be published. An absent field is a question; a wrong field is an answer.
                record.notes = (
                    f"{record.notes} CPU latency not measured: runtime representation is "
                    f"{runtime!r}, which dequantises to FP32 on every forward pass, so a timing "
                    "would measure unpacking rather than the compression (decision D1)."
                ).strip()
                LOGGER.warning(
                    "Skipping the CPU benchmark: %s would measure dequantisation, not compression. "
                    "A native INT8 runtime path is required before W8 latency can be reported, and "
                    "W4 latency is excluded by D1 regardless.",
                    runtime,
                )

        except NotImplementedError:
            # An unimplemented arm is a known gap, not a failed experiment. Let it through
            # unrecorded so a placeholder never lands in outputs/ looking like a real result.
            raise
        except Exception as error:
            # A failed run stays in the log with its reason, rather than vanishing. A sweep with
            # silently missing cells is worse than one with recorded failures.
            record.status = "failure"
            record.notes = f"{type(error).__name__}: {error}"
            record.duration_seconds = time.perf_counter() - started
            LOGGER.exception("Run %s failed; recording the failure", record.experiment_id)
            self.tracker.save(record)
            raise ExperimentError(f"Run {record.experiment_id} failed: {error}") from error

        record.status = "success"
        record.duration_seconds = time.perf_counter() - started

        # 8. Persist.
        self.tracker.save(record)
        LOGGER.info("Completed %s in %.1fs", record.experiment_id, record.duration_seconds)
        return record

    def _attach_calibration(self, compressor: Any, loaded: Any) -> None:
        """Draw the shared calibration set and hand it to a layerwise arm.

        Only layerwise arms need this, and they are the only ones that expose
        ``set_calibration``. The older fine-tuning compressors do not, so this is a no-op for them
        rather than an error -- they raise from their own stages instead, which is where the message
        naming the module to edit lives.

        Args:
            compressor: The arm about to run.
            loaded: The loaded model bundle, for its tokeniser.
        """
        from scale_aware_compression.data.calibration import load_calibration_set

        attach = getattr(compressor, "set_calibration", None)
        if not callable(attach):
            return

        calibration = load_calibration_set(self.config.data, loaded.tokenizer)
        batches = [
            batch["input_ids"] if isinstance(batch, dict) else batch[0]
            for batch in calibration.loader
        ]
        attach(batches, fingerprint=calibration.summary.token_fingerprint)
        LOGGER.info(
            "Calibration: %d sequence(s) in %d batch(es), fingerprint %s",
            len(calibration),
            len(batches),
            calibration.summary.token_fingerprint,
        )

    def _measure_artefact(
        self,
        record: ExperimentRecord,
        compressor: Any,
        model: Any,
        loaded: Any,
    ) -> None:
        """Measure the size of the artefact that would actually be deployed.

        For a compressed arm that means saving the converted model and measuring what lands on
        disk. For the dense arm it means measuring the cached Hugging Face snapshot: re-saving a
        byte-identical copy of a multi-gigabyte checkpoint just to weigh it would waste disk for
        no extra information.

        Args:
            record: Record to update in place.
            compressor: The arm's compressor, or ``None`` for dense.
            model: The converted model.
            loaded: The :class:`LoadedModel` bundle.
        """
        from scale_aware_compression.benchmarking.checkpoint_size import measure_checkpoint

        target: Path | None = None
        if compressor is not None:
            target = self.config.run_output_dir / "checkpoint"
            compressor.save(model, target)
        else:
            target = self._cached_snapshot_path(loaded)

        if target is None:
            LOGGER.warning(
                "Could not locate an on-disk artefact for %s; checkpoint size will be absent "
                "from this record.",
                record.experiment_id,
            )
            return

        # Split the parameter count into what this method compresses and what it deliberately does
        # not. Treating the whole model as compressible makes the theoretical budget unachievable
        # and every compressed artefact look inefficient against it.
        untargeted: int | None = None
        layerwise = getattr(compressor, "report", None)
        if layerwise is not None and layerwise.targeted_parameters:
            targeted = layerwise.targeted_parameters
            targeted_nonzero = round(targeted * (1.0 - layerwise.realised_sparsity))
            untargeted = max(record.parameter_count - targeted, 0)
            nonzero = targeted_nonzero or None
        else:
            nonzero = max(record.parameter_count - _measured_zeros(model), 0) or None

        report = measure_checkpoint(
            target,
            nonzero_parameters=nonzero,
            bits=self.config.compression.effective_bits,
            untargeted_parameters=untargeted,
        )
        record.checkpoint = report.to_dict()
        record.checkpoint_path = target

    def _cached_snapshot_path(self, loaded: Any) -> Path | None:
        """Locate the cached Hub snapshot for the dense arm, without downloading."""
        try:
            from huggingface_hub import snapshot_download

            return Path(
                snapshot_download(
                    repo_id=loaded.spec.hf_id,
                    revision=loaded.revision,
                    local_files_only=True,
                )
            )
        except Exception as error:  # pragma: no cover - depends on the local cache
            LOGGER.debug("Could not resolve the cached snapshot: %s", error)
            return None

    def _runtime_representation(self) -> str:
        """What the model will actually execute as, which is not the same as how it is stored.

        Returns:
            ``"fp32"`` for dense and pruning-only arms, whose weights stay full precision and run on
            the native dense kernel; ``"packed_dequantising"`` for any quantised arm, because
            ``PackedLinear`` unpacks and dequantises to FP32 on every forward pass.
        """
        compression = self.config.compression
        if compression.effective_bits >= 32:
            return "fp32"
        return "packed_dequantising"

    def _latency_is_meaningful(self, runtime: str) -> bool:
        """Whether a CPU timing of this artefact measures the compression or the plumbing.

        Decision D1 makes PyTorch native INT8 the only latency backend, and W4 contributes quality
        and size only. But every quantised arm currently converts to ``PackedLinear``, whose forward
        unpacks integer codes, dequantises them and calls a dense FP32 matmul. Timing that measures
        the unpacking -- it is *slower* than the dense model and says nothing about either sparsity
        or precision.

        So the honest options are to build a native INT8 runtime artefact, or to record no latency.
        Until the former exists this returns ``False`` for quantised arms, and the record carries a
        note saying why rather than a number that cannot be used.

        The dense and pruning-only arms are unaffected: they stay FP32 and benchmark natively, which
        is what makes research question 4 -- does sparsity produce a real CPU speedup? -- answerable
        without a 4-bit kernel at all.

        Args:
            runtime: The value from :meth:`_runtime_representation`.

        Returns:
            ``True`` when the timing is attributable to the compression.
        """
        return runtime == "fp32"

    def _evaluation_fingerprint(self) -> str | None:
        """This run's evaluation-corpus fingerprint, or ``None`` if it cannot be resolved yet.

        Best-effort: the fingerprint is a property of the tokenised split, so resolving it needs the
        cache. A ``None`` result weakens the dense-reference check to the window comparison rather
        than failing the run.
        """
        return getattr(self, "_eval_fingerprint", None)

    def _window_mismatch(self, payload: dict[str, Any]) -> str | None:
        """Why a candidate dense record is not comparable with this run, or ``None`` if it is.

        Args:
            payload: The candidate's ``quality.perplexity`` mapping.

        Returns:
            A short reason, or ``None`` when the windows agree.
        """
        expected_length = self.config.data.sequence_length
        expected_count = self.config.evaluation.max_samples or self.config.data.max_eval_samples
        actual_length = payload.get("sequence_length")
        actual_count = payload.get("num_sequences")

        if actual_length is not None and int(actual_length) != int(expected_length):
            return f"sequence_length {actual_length} != {expected_length}"

        # The corpus itself, not just the window shape. Matching on sequence length alone would accept
        # a dense run over a *different split* that happened to use the same window -- and once final
        # results move to the test split while screening stays on validation, that is not a
        # hypothetical.
        expected_fingerprint = self._evaluation_fingerprint()
        actual_fingerprint = payload.get("dataset_fingerprint")
        if (
            expected_fingerprint
            and actual_fingerprint
            and actual_fingerprint != expected_fingerprint
        ):
            return f"dataset fingerprint {actual_fingerprint} != {expected_fingerprint}"
        # The count is a *cap*, so a dense run may legitimately have evaluated fewer sequences than
        # the cap when the split ran out. Only a genuine excess is disqualifying.
        if (
            expected_count is not None
            and actual_count is not None
            and int(actual_count) > int(expected_count)
        ):
            return f"num_sequences {actual_count} exceeds the cap {expected_count}"
        return None

    def _load_dense_reference(self) -> Any:
        """Load this model's dense-baseline perplexity from its recorded run.

        Loaded rather than recomputed, so every arm is normalised against exactly the same
        number. A dense run evaluated with a different window or sample count would otherwise
        produce a subtly different reference for each arm.

        Returns:
            The dense :class:`PerplexityResult`, or ``None`` when this *is* the dense arm or no
            dense record exists yet.
        """
        from scale_aware_compression.evaluation.perplexity import PerplexityResult

        config = self.config
        if config.compression.method is CompressionMethod.DENSE:
            return None

        assert self.tracker is not None
        for candidate in self.tracker.load_all():
            if candidate.get("compression_method") != CompressionMethod.DENSE.value:
                continue
            if candidate.get("model_name") != config.model.name:
                continue
            if candidate.get("seed") != config.runtime.seed:
                continue
            payload = candidate.get("quality", {}).get("perplexity")
            if not payload:
                continue
            # The window and the corpus must match, not just the model and seed. Retention is a ratio
            # against a dense run, and a dense run evaluated over a different window is a different
            # number: 34.77 at 64x256 against 36.97 at 493x512 on the same model. Matching on
            # model+seed alone silently normalised a pilot-window run against a screening-window
            # baseline, which is the kind of error that shows up as a plausible retention figure.
            mismatch = self._window_mismatch(payload)
            if mismatch:
                LOGGER.debug(
                    "Skipping dense record %s: %s", candidate.get("experiment_id"), mismatch
                )
                continue
            LOGGER.info(
                "Using dense reference from %s (perplexity %.4f)",
                candidate.get("experiment_id"),
                payload["perplexity"],
            )
            return PerplexityResult(
                perplexity=float(payload["perplexity"]),
                total_nll=float(payload.get("total_nll", 0.0)),
                total_tokens=int(payload.get("total_tokens", 0)),
                num_sequences=int(payload.get("num_sequences", 0)),
                sequence_length=int(payload.get("sequence_length", 0)),
                device=str(payload.get("evaluation_device", "cpu")),
                dataset_fingerprint=payload.get("dataset_fingerprint"),
            )

        LOGGER.warning(
            "No dense baseline recorded for model=%s seed=%d. This run will have no quality "
            "retention. Run scripts/run_dense_baseline.py for this model first.",
            config.model.name,
            config.runtime.seed,
        )
        return None

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
