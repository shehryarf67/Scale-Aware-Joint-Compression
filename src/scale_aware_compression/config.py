"""Typed configuration objects and the YAML loader that builds them.

A run is described entirely by one YAML document. Documents may compose other documents
through a top-level ``include:`` list, which is how ``configs/experiments/*.yaml`` pull in a
model config and a compression config without duplicating their contents::

    include:
      - ../models/pythia_160m.yaml
      - ../compression/sequential.yaml
    experiment:
      id: pilot

Includes are resolved relative to the file that declares them, merged in order, and then
overridden by the keys of the including document.

Every section is a frozen-ish dataclass that validates its own fields in ``__post_init__``,
so an invalid configuration fails at load time with a message naming the offending key
rather than part-way through a multi-hour run.
"""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml

from scale_aware_compression.constants import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    SUPPORTED_WEIGHT_BITS,
    CompressionMethod,
    Device,
    DType,
    PruningGranularity,
    PruningScheduleName,
    QuantisationGranularity,
    QuantisationScheme,
    ReconstructionSolver,
)
from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

INCLUDE_KEY = "include"
"""Top-level key holding a list of documents to merge underneath this one."""

_MAX_INCLUDE_DEPTH = 8

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration document is missing, malformed, or invalid."""


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class RunMeta:
    """Identity of an experiment, used to name output files and group records."""

    id: str = "unnamed"
    name: str = ""
    description: str = ""
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the identifier."""
        if not self.id or not self.id.strip():
            raise ConfigError("experiment.id must be a non-empty string")
        if any(character in self.id for character in ' /\\:*?"<>|'):
            raise ConfigError(
                f"experiment.id={self.id!r} must be filename-safe "
                "(no spaces, slashes, or shell metacharacters)"
            )
        if not self.name:
            self.name = self.id


@dataclass(slots=True)
class RuntimeConfig:
    """Cross-cutting execution settings: seed, threads, and where artefacts go."""

    seed: int = DEFAULT_SEED
    deterministic: bool = True
    num_threads: int | None = None
    """PyTorch CPU thread count. ``None`` leaves the torch default in place; benchmarks
    must pin it (see :class:`BenchmarkConfig`)."""
    output_dir: Path = DEFAULT_OUTPUT_DIR
    log_level: str = "INFO"
    log_to_file: bool = True

    def __post_init__(self) -> None:
        """Validate and normalise runtime settings."""
        if self.seed < 0:
            raise ConfigError(f"runtime.seed must be non-negative, got {self.seed}")
        if self.num_threads is not None and self.num_threads < 1:
            raise ConfigError(f"runtime.num_threads must be >= 1 or null, got {self.num_threads}")
        self.output_dir = Path(self.output_dir)
        self.log_level = self.log_level.upper()
        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"runtime.log_level={self.log_level!r} is not a logging level")


@dataclass(slots=True)
class ModelConfig:
    """Which checkpoint to load, at what precision, onto what device."""

    name: str = "pythia-160m"
    """Short registry name; see :mod:`scale_aware_compression.models.registry`."""
    size_label: str = ""
    """Human-readable scale label used on plot axes, e.g. ``160M``. Filled from the
    registry when left blank."""
    hf_id: str | None = None
    """Overrides the registry lookup. Normally left unset."""
    revision: str | None = None
    """Hub revision (branch, tag, or commit SHA). Pin this for reproducibility."""
    device: Device = Device.AUTO
    dtype: DType = DType.FLOAT32
    trust_remote_code: bool = False
    eval_mode: bool = True
    local_files_only: bool = False
    attn_implementation: str | None = None

    def __post_init__(self) -> None:
        """Validate the model selection."""
        if not self.name.strip():
            raise ConfigError("model.name must be a non-empty short registry name")
        self.name = self.name.strip()


@dataclass(slots=True)
class DataConfig:
    """Evaluation and calibration corpora, tokenisation window, and batching."""

    dataset: str = "Salesforce/wikitext"
    """Fully namespaced Hub repository id.

    Must be ``namespace/name``. The bare canonical alias ``wikitext`` worked under older
    ``datasets`` releases but 5.x rejects it outright, which is what the first real run of the load
    path found. A bare name fails at load time, after the model is already in memory.
    """
    subset: str | None = "wikitext-2-raw-v1"
    train_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    sequence_length: int = 512
    batch_size: int = 8
    num_workers: int = 0
    max_eval_samples: int | None = 512
    calibration_samples: int = 128
    calibration_split: str = "train"
    calibration_seed: int = DEFAULT_SEED
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate window sizes and sample counts."""
        _require_positive("data.sequence_length", self.sequence_length)
        _require_positive("data.batch_size", self.batch_size)
        _require_positive("data.calibration_samples", self.calibration_samples)
        if self.num_workers < 0:
            raise ConfigError(f"data.num_workers must be >= 0, got {self.num_workers}")
        if self.max_eval_samples is not None:
            _require_positive("data.max_eval_samples", self.max_eval_samples)
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)


@dataclass(slots=True)
class PruningConfig:
    """Sparsity target, pattern, and ramp."""

    enabled: bool = True
    method: str = "magnitude"
    sparsity: float = 0.5
    """Target fraction of prunable weights set to zero, in [0, 1)."""
    initial_sparsity: float = 0.0
    granularity: PruningGranularity = PruningGranularity.UNSTRUCTURED
    schedule: PruningScheduleName = PruningScheduleName.CUBIC
    schedule_start_step: int = 0
    schedule_end_step: int = 1000
    schedule_frequency: int = 50
    """Steps between mask updates during a gradual schedule."""
    global_ranking: bool = False
    """Rank weights across all prunable layers at once rather than per layer."""
    target_modules: list[str] = field(default_factory=lambda: ["Linear"])
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["embed", "embed_out", "lm_head", "wte", "wpe"]
    )
    """Embeddings and the output head are excluded by default: pruning them changes the
    quality/size trade-off in a way that is not comparable across tokenisers."""

    def __post_init__(self) -> None:
        """Validate sparsity levels and the schedule window."""
        _require_unit_interval("compression.pruning.sparsity", self.sparsity, upper_open=True)
        _require_unit_interval(
            "compression.pruning.initial_sparsity", self.initial_sparsity, upper_open=True
        )
        if self.initial_sparsity > self.sparsity:
            raise ConfigError(
                "compression.pruning.initial_sparsity "
                f"({self.initial_sparsity}) must not exceed sparsity ({self.sparsity})"
            )
        if self.schedule_start_step < 0:
            raise ConfigError("compression.pruning.schedule_start_step must be >= 0")
        if self.schedule_end_step < self.schedule_start_step:
            raise ConfigError(
                "compression.pruning.schedule_end_step "
                f"({self.schedule_end_step}) must be >= schedule_start_step "
                f"({self.schedule_start_step})"
            )
        _require_positive("compression.pruning.schedule_frequency", self.schedule_frequency)
        if self.granularity is PruningGranularity.SEMI_STRUCTURED_2_4 and self.sparsity != 0.5:
            raise ConfigError(
                f"compression.pruning.granularity='2:4' implies sparsity=0.5, got {self.sparsity}"
            )


@dataclass(slots=True)
class QuantisationConfig:
    """Weight (and optionally activation) precision, and how scales are estimated."""

    enabled: bool = True
    bits: int = 8
    scheme: QuantisationScheme = QuantisationScheme.SYMMETRIC
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL
    group_size: int = 128
    """Only used when ``granularity`` is ``per_group``."""
    backend: str = "onednn"
    """PyTorch quantised-kernel backend used for the CPU deployment model."""
    observer: str = "min_max"
    calibration_samples: int = 128
    quantise_activations: bool = False
    activation_bits: int = 8
    exclude_patterns: list[str] = field(
        default_factory=lambda: ["embed", "embed_out", "lm_head", "wte", "wpe"]
    )

    def __post_init__(self) -> None:
        """Validate bit widths and calibration size."""
        if self.bits not in SUPPORTED_WEIGHT_BITS:
            raise ConfigError(
                f"compression.quantisation.bits={self.bits} is not one of "
                f"{list(SUPPORTED_WEIGHT_BITS)}"
            )
        if self.quantise_activations and self.activation_bits not in SUPPORTED_WEIGHT_BITS:
            raise ConfigError(
                f"compression.quantisation.activation_bits={self.activation_bits} is not one "
                f"of {list(SUPPORTED_WEIGHT_BITS)}"
            )
        _require_positive("compression.quantisation.calibration_samples", self.calibration_samples)
        if self.granularity is QuantisationGranularity.PER_GROUP:
            _require_positive("compression.quantisation.group_size", self.group_size)


@dataclass(slots=True)
class RecoveryConfig:
    """Post-pruning recovery fine-tuning. May run on GPU."""

    enabled: bool = True
    epochs: int = 1
    max_steps: int | None = None
    """Overrides ``epochs`` when set; useful for matching optimisation budgets exactly."""
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    device: Device = Device.AUTO
    log_every_steps: int = 20
    eval_every_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate the optimisation budget."""
        if self.epochs < 0:
            raise ConfigError(f"compression.recovery.epochs must be >= 0, got {self.epochs}")
        if self.max_steps is not None:
            _require_positive("compression.recovery.max_steps", self.max_steps)
        if self.learning_rate <= 0:
            raise ConfigError("compression.recovery.learning_rate must be > 0")
        _require_unit_interval("compression.recovery.warmup_ratio", self.warmup_ratio)
        _require_positive("compression.recovery.batch_size", self.batch_size)
        _require_positive(
            "compression.recovery.gradient_accumulation_steps",
            self.gradient_accumulation_steps,
        )
        _require_positive("compression.recovery.log_every_steps", self.log_every_steps)


@dataclass(slots=True)
class JointConfig:
    """Settings unique to joint pruning-aware quantisation.

    The joint arm differs from the sequential arm in that fake quantisation is inserted
    *before* pruning begins, so the pruning criterion sees quantised weights and the
    optimiser can compensate for both perturbations at once.
    """

    fake_quantisation: bool = True
    straight_through_estimator: bool = True
    mask_update_interval: int = 50
    """Steps between re-selecting the pruning mask during joint optimisation."""
    freeze_masks_after_ratio: float = 0.8
    """Fraction of joint training after which masks stop changing, so the final phase is
    pure recovery at the target sparsity."""
    quantisation_warmup_steps: int = 0
    """Steps of dense-precision training before fake quantisation is switched on."""
    joint_epochs: int = 1
    joint_max_steps: int | None = None
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    device: Device = Device.AUTO
    match_sequential_budget: bool = True
    """When true the runner is expected to equalise optimiser steps against the sequential
    arm, so any measured joint gain is not simply extra training."""

    def __post_init__(self) -> None:
        """Validate the joint optimisation schedule."""
        _require_positive("compression.joint.mask_update_interval", self.mask_update_interval)
        _require_unit_interval(
            "compression.joint.freeze_masks_after_ratio", self.freeze_masks_after_ratio
        )
        if self.quantisation_warmup_steps < 0:
            raise ConfigError("compression.joint.quantisation_warmup_steps must be >= 0")
        if self.joint_epochs < 0:
            raise ConfigError("compression.joint.joint_epochs must be >= 0")
        if self.joint_max_steps is not None:
            _require_positive("compression.joint.joint_max_steps", self.joint_max_steps)
        if self.learning_rate <= 0:
            raise ConfigError("compression.joint.learning_rate must be > 0")
        _require_positive("compression.joint.batch_size", self.batch_size)


@dataclass(slots=True)
class ReconstructionConfig:
    """Layerwise post-training reconstruction settings (plan §3.1, gap A2).

    This is the section that governs the *actual* method. ``local_steps`` is the fairness unit
    §3.11 requires be matched between arms -- not ``recovery.max_steps``, which belongs to the
    superseded fine-tuning design and survives only for the optional ablation.
    """

    enabled: bool = True
    solver: ReconstructionSolver = ReconstructionSolver.SWEEP
    """Which minimiser runs. Must be the same for every layer and every arm in a results table:
    mixing solvers would mean different layers were optimised by different algorithms."""
    local_steps: int = 1
    """Refinement iterations per layer. **The fairness unit.** Used by the ALS solver; the sweep is
    single-pass, so its cost is fixed and matching is automatic."""
    joint_iterations: int = 4
    """Outer alternations in the joint arm (``K`` in §3.7): fake-quantise, rescore saliency under
    the quantised weights, update the mask, re-estimate scales, reconstruct."""
    damping: float = 1e-2
    """Ridge coefficient, relative to the mean Gram diagonal so one value works at every width."""
    block_size: int = 128
    """Column block width for the sweep solver. Throughput knob; does not change the result."""
    activation_order: bool = True
    """Visit high-energy columns first in the sweep. Ignored for per-group quantisation."""
    scale_search: bool = False
    """Fit scales by minimising pre-reconstruction error instead of matching ``max|W|``.

    Off because it measured worse end-to-end: it cuts naive quantisation error by 12.8% at W4 but
    turns the layer-objective joint gain from +1.12% to -0.99%. Kept as a declared ablation.
    """
    keep_benefit_saliency: bool = False
    """Score the mask by keep-versus-prune benefit instead of quantised magnitude.

    Off because it measured much worse: layer-objective joint gain falls to -16.15% at W4. Kept as
    a declared ablation. See docs/validity_threats.md.
    """
    calibration_dtype: str = "float32"
    """Accumulation dtype for ``H = XᵀX``. fp32 keeps the largest layer Hessian at 256 MiB."""

    def __post_init__(self) -> None:
        """Validate the optimisation budget."""
        if self.local_steps < 0:
            raise ConfigError(
                f"compression.reconstruction.local_steps must be >= 0, got {self.local_steps}"
            )
        _require_positive("compression.reconstruction.joint_iterations", self.joint_iterations)
        if self.damping < 0:
            raise ConfigError(
                f"compression.reconstruction.damping must be >= 0, got {self.damping}"
            )
        _require_positive("compression.reconstruction.block_size", self.block_size)
        if self.calibration_dtype not in {"float32", "float64"}:
            raise ConfigError(
                "compression.reconstruction.calibration_dtype must be 'float32' or 'float64', "
                f"got {self.calibration_dtype!r}"
            )


@dataclass(slots=True)
class CompressionConfig:
    """Which experimental arm to run, and the settings for each of its stages."""

    method: CompressionMethod = CompressionMethod.DENSE
    budget_label: str = "moderate"
    """Names the compression budget (e.g. ``moderate``, ``aggressive``) so that joint and
    sequential rows can be matched when computing joint gain."""
    pruning: PruningConfig = field(default_factory=PruningConfig)
    quantisation: QuantisationConfig = field(default_factory=QuantisationConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    joint: JointConfig = field(default_factory=JointConfig)

    def __post_init__(self) -> None:
        """Check that the enabled stages are consistent with the chosen method."""
        method = self.method
        if method is CompressionMethod.DENSE:
            return
        needs_pruning = method in {
            CompressionMethod.PRUNING,
            CompressionMethod.SEQUENTIAL,
            CompressionMethod.JOINT,
        }
        needs_quantisation = method in {
            CompressionMethod.QUANTISATION,
            CompressionMethod.SEQUENTIAL,
            CompressionMethod.JOINT,
        }
        if needs_pruning and not self.pruning.enabled:
            raise ConfigError(
                f"compression.method={method.value!r} requires compression.pruning.enabled=true"
            )
        if needs_quantisation and not self.quantisation.enabled:
            raise ConfigError(
                f"compression.method={method.value!r} requires "
                "compression.quantisation.enabled=true"
            )

    @property
    def effective_sparsity(self) -> float:
        """Target sparsity actually applied by this arm (0.0 when pruning is off)."""
        if self.method in {CompressionMethod.DENSE, CompressionMethod.QUANTISATION}:
            return 0.0
        return self.pruning.sparsity

    @property
    def effective_bits(self) -> int:
        """Weight bit width actually applied by this arm (32 when quantisation is off)."""
        if self.method in {CompressionMethod.DENSE, CompressionMethod.PRUNING}:
            return 32
        return self.quantisation.bits


@dataclass(slots=True)
class EvaluationConfig:
    """Quality evaluation. Final reported numbers must be produced on CPU."""

    device: Device = Device.CPU
    batch_size: int = 4
    sequence_length: int = 512
    stride: int | None = None
    """Sliding-window stride for perplexity. ``None`` means non-overlapping windows."""
    max_samples: int | None = 512
    metrics: list[str] = field(default_factory=lambda: ["perplexity"])
    compare_to_dense: bool = True
    agreement_samples: int = 128
    """Prompts used for dense-vs-compressed top-1 agreement."""
    generation_prompts: int = 16
    generation_max_new_tokens: int = 64

    def __post_init__(self) -> None:
        """Validate evaluation sizes."""
        _require_positive("evaluation.batch_size", self.batch_size)
        _require_positive("evaluation.sequence_length", self.sequence_length)
        if self.stride is not None:
            _require_positive("evaluation.stride", self.stride)
        if self.max_samples is not None:
            _require_positive("evaluation.max_samples", self.max_samples)
        if not self.metrics:
            raise ConfigError("evaluation.metrics must list at least one metric")


@dataclass(slots=True)
class BenchmarkConfig:
    """CPU deployment benchmark.

    ``device`` is present but constrained to CPU: a latency number measured on GPU is not
    a deployment number for this study, so the constraint is enforced rather than
    documented.
    """

    device: Device = Device.CPU
    num_threads: int = 4
    warmup_runs: int = 5
    measured_runs: int = 30
    batch_size: int = 1
    sequence_length: int = 128
    generated_tokens: int = 0
    """When > 0 the benchmark measures autoregressive decoding of this many tokens
    instead of a single forward pass."""
    interop_threads: int | None = None
    record_per_run_latencies: bool = True
    fail_on_thread_mismatch: bool = True

    def __post_init__(self) -> None:
        """Validate the benchmark protocol and enforce the CPU-only policy."""
        if self.device is not Device.CPU:
            raise ConfigError(
                "benchmark.device must be 'cpu': deployment measurements in this study are "
                f"CPU-only, got {self.device.value!r}"
            )
        _require_positive("benchmark.num_threads", self.num_threads)
        if self.warmup_runs < 0:
            raise ConfigError(f"benchmark.warmup_runs must be >= 0, got {self.warmup_runs}")
        if self.measured_runs < 2:
            raise ConfigError(
                "benchmark.measured_runs must be >= 2 so that a standard deviation and p95 "
                f"are meaningful, got {self.measured_runs}"
            )
        _require_positive("benchmark.batch_size", self.batch_size)
        _require_positive("benchmark.sequence_length", self.sequence_length)
        if self.generated_tokens < 0:
            raise ConfigError("benchmark.generated_tokens must be >= 0")
        if self.interop_threads is not None:
            _require_positive("benchmark.interop_threads", self.interop_threads)


@dataclass(slots=True)
class SweepConfig:
    """The scale sweep grid: models x arms x budgets x seeds."""

    models: list[str] = field(default_factory=list)
    methods: list[CompressionMethod] = field(default_factory=list)
    budgets: list[str] = field(default_factory=lambda: ["moderate"])
    seeds: list[int] = field(default_factory=list)
    budget_overrides: dict[str, Any] = field(default_factory=dict)
    """Per-budget override fragments, e.g.
    ``{"aggressive": {"compression": {"pruning": {"sparsity": 0.7}}}}``."""
    skip_existing: bool = True
    continue_on_error: bool = False

    def __post_init__(self) -> None:
        """Validate the grid."""
        if not self.budgets:
            raise ConfigError("sweep.budgets must list at least one budget label")
        unknown = set(self.budget_overrides) - set(self.budgets)
        if unknown:
            raise ConfigError(
                f"sweep.budget_overrides has entries for unlisted budgets: {sorted(unknown)}"
            )


# ---------------------------------------------------------------------------
# Top-level document
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class ExperimentConfig:
    """A complete, validated description of one experiment or sweep."""

    experiment: RunMeta = field(default_factory=RunMeta)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)

    @classmethod
    def from_mapping(cls, document: Mapping[str, Any]) -> ExperimentConfig:
        """Build a config from an already-merged mapping.

        Args:
            document: Mapping whose keys are section names. ``include`` is ignored here;
                use :func:`load_document` to resolve includes first.

        Returns:
            A validated :class:`ExperimentConfig`.

        Raises:
            ConfigError: If a key is unknown or a value fails validation.
        """
        payload = {key: value for key, value in document.items() if key != INCLUDE_KEY}
        return _build_dataclass(cls, payload, path="")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, YAML/JSON-serialisable dictionary of every resolved value."""
        return _to_plain(dataclasses.asdict(self))

    def save(self, path: str | Path) -> Path:
        """Write the fully resolved configuration next to a run's other artefacts.

        Args:
            path: Destination ``.yaml`` file. Parent directories are created.

        Returns:
            The path written.
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        LOGGER.debug("Wrote resolved configuration to %s", destination)
        return destination

    def describe(self) -> str:
        """One-line human-readable summary, suitable for a log header."""
        return (
            f"{self.experiment.id}: model={self.model.name} "
            f"method={self.compression.method.value} "
            f"budget={self.compression.budget_label} "
            f"sparsity={self.compression.effective_sparsity:.2f} "
            f"bits={self.compression.effective_bits} "
            f"seed={self.runtime.seed}"
        )

    @property
    def run_output_dir(self) -> Path:
        """Directory for this run's artefacts, ``<output_dir>/<experiment id>``."""
        return self.runtime.output_dir / self.experiment.id


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a single YAML file into a dictionary, without resolving includes.

    Args:
        path: Path to a ``.yaml`` / ``.yml`` file.

    Returns:
        The parsed mapping. An empty file yields an empty dict.

    Raises:
        ConfigError: If the file is missing, unparseable, or does not hold a mapping.
    """
    source = Path(path)
    if not source.is_file():
        raise ConfigError(f"Configuration file not found: {source}")
    try:
        parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise ConfigError(f"Could not parse YAML in {source}: {error}") from error
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigError(f"{source} must contain a top-level mapping, got {type(parsed).__name__}")
    return parsed


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` on top of ``base``.

    Nested mappings are merged key by key; every other type (including lists) is replaced
    wholesale, so a config can shorten a list rather than only extend it.

    Args:
        base: Lower-precedence mapping.
        override: Higher-precedence mapping.

    Returns:
        A new merged dictionary. Neither input is modified.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def load_document(
    path: str | Path, *, _depth: int = 0, _seen: frozenset[Path] = frozenset()
) -> dict[str, Any]:
    """Read a YAML config and merge everything it includes, recursively.

    Includes are resolved relative to the including file's directory, merged in list
    order, and then overridden by the including document's own keys.

    Args:
        path: Path to the entry-point YAML document.
        _depth: Internal recursion guard.
        _seen: Internal set of already-visited paths, used to detect include cycles.

    Returns:
        The merged mapping, with the ``include`` key removed.

    Raises:
        ConfigError: On a missing file, an include cycle, or excessive nesting.
    """
    source = Path(path).resolve()
    if source in _seen:
        raise ConfigError(f"Include cycle detected at {source}")
    if _depth > _MAX_INCLUDE_DEPTH:
        raise ConfigError(f"Include nesting deeper than {_MAX_INCLUDE_DEPTH} levels at {source}")

    document = load_yaml(source)
    includes = document.pop(INCLUDE_KEY, [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigError(f"{source}: '{INCLUDE_KEY}' must be a string or a list of strings")

    merged: dict[str, Any] = {}
    for entry in includes:
        if not isinstance(entry, str):
            raise ConfigError(f"{source}: every '{INCLUDE_KEY}' entry must be a string")
        included_path = (source.parent / entry).resolve()
        merged = deep_merge(
            merged,
            load_document(included_path, _depth=_depth + 1, _seen=_seen | {source}),
        )
    return deep_merge(merged, document)


def apply_overrides(document: Mapping[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``dotted.key=value`` command-line overrides to a config mapping.

    Values are parsed as YAML, so ``runtime.seed=7`` yields an int, ``model.revision=null``
    yields ``None``, and ``experiment.tags=[a,b]`` yields a list.

    Args:
        document: Mapping to override.
        overrides: Strings of the form ``section.key=value``.

    Returns:
        A new mapping with the overrides applied.

    Raises:
        ConfigError: If an override is not of the form ``key=value``.
    """
    result: dict[str, Any] = dict(document)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override {item!r} must be of the form dotted.key=value")
        dotted_key, _, raw_value = item.partition("=")
        keys = [part for part in dotted_key.strip().split(".") if part]
        if not keys:
            raise ConfigError(f"Override {item!r} has an empty key")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as error:
            raise ConfigError(f"Could not parse override value in {item!r}: {error}") from error
        fragment: Any = value
        for key in reversed(keys):
            fragment = {key: fragment}
        result = deep_merge(result, fragment)
    return result


def load_config(
    path: str | Path,
    overrides: Sequence[str] | None = None,
) -> ExperimentConfig:
    """Load, merge, override, and validate a configuration in one call.

    This is the function every script and CLI subcommand uses.

    Args:
        path: Entry-point YAML document.
        overrides: Optional ``dotted.key=value`` strings applied after includes.

    Returns:
        A validated :class:`ExperimentConfig`.

    Raises:
        ConfigError: If the document is missing, malformed, or invalid.
    """
    document = load_document(path)
    if overrides:
        document = apply_overrides(document, overrides)
    config = ExperimentConfig.from_mapping(document)
    LOGGER.debug("Loaded configuration from %s -> %s", path, config.describe())
    return config


# ---------------------------------------------------------------------------
# Generic mapping -> dataclass construction
# ---------------------------------------------------------------------------
def _build_dataclass(target: type[T], data: Mapping[str, Any], *, path: str) -> T:
    """Instantiate a dataclass from a mapping, rejecting unknown keys."""
    if not is_dataclass(target):
        raise ConfigError(f"{target.__name__} is not a dataclass")
    hints = get_type_hints(target)
    known = {declared.name for declared in fields(target)}
    unknown = sorted(set(data) - known)
    if unknown:
        location = path or "<root>"
        raise ConfigError(
            f"Unknown configuration key(s) under {location}: {unknown}. Valid keys: {sorted(known)}"
        )
    kwargs: dict[str, Any] = {}
    for declared in fields(target):
        if declared.name not in data:
            continue
        child_path = f"{path}.{declared.name}".lstrip(".")
        kwargs[declared.name] = _coerce(hints[declared.name], data[declared.name], child_path)
    return target(**kwargs)  # type: ignore[return-value]


def _coerce(annotation: Any, value: Any, path: str) -> Any:
    """Convert a raw YAML value to the type declared on a dataclass field."""
    origin = get_origin(annotation)

    if origin in (Union, UnionType):
        options = [option for option in get_args(annotation) if option is not type(None)]
        allows_none = len(options) != len(get_args(annotation))
        if value is None:
            if allows_none:
                return None
            raise ConfigError(f"{path} may not be null")
        for option in options:
            try:
                return _coerce(option, value, path)
            except (ConfigError, TypeError, ValueError):
                continue
        raise ConfigError(f"{path}={value!r} does not match {annotation}")

    if origin in (list, Sequence):
        if not isinstance(value, list):
            raise ConfigError(f"{path} must be a list, got {type(value).__name__}")
        (item_type,) = get_args(annotation) or (Any,)
        return [_coerce(item_type, item, f"{path}[{index}]") for index, item in enumerate(value)]

    if origin is dict or annotation is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
        return dict(value)

    if is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path} must be a mapping, got {type(value).__name__}")
        return _build_dataclass(annotation, value, path=path)

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return _coerce_enum(annotation, value, path)

    if annotation is Path:
        if not isinstance(value, str | Path):
            raise ConfigError(f"{path} must be a path string, got {type(value).__name__}")
        return Path(value)

    if annotation is bool:
        if isinstance(value, bool):
            return value
        raise ConfigError(f"{path} must be true or false, got {value!r}")

    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path} must be an integer, got {value!r}")
        return value

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        return float(value)

    if annotation is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path} must be a string, got {type(value).__name__}")
        return value

    return value


def _coerce_enum(enum_type: type[enum.Enum], value: Any, path: str) -> enum.Enum:
    """Resolve a YAML scalar to an enum member by value, then by case-insensitive name."""
    if isinstance(value, enum_type):
        return value
    try:
        return enum_type(value)
    except ValueError:
        pass
    if isinstance(value, str):
        for member in enum_type:
            if member.name.lower() == value.strip().lower():
                return member
    allowed = [member.value for member in enum_type]
    raise ConfigError(f"{path}={value!r} is not one of {allowed}")


def _to_plain(value: Any) -> Any:
    """Recursively convert enums and paths to primitives for serialisation."""
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    return value


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------
def _require_positive(name: str, value: int | float) -> None:
    """Raise :class:`ConfigError` unless ``value`` is strictly positive."""
    if value <= 0:
        raise ConfigError(f"{name} must be > 0, got {value}")


def _require_unit_interval(name: str, value: float, *, upper_open: bool = False) -> None:
    """Raise :class:`ConfigError` unless ``value`` lies in [0, 1] (or [0, 1))."""
    upper_ok = value < 1.0 if upper_open else value <= 1.0
    if value < 0.0 or not upper_ok:
        bound = "[0, 1)" if upper_open else "[0, 1]"
        raise ConfigError(f"{name} must lie in {bound}, got {value}")
