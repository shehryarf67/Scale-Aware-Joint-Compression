"""Project-wide constants, enumerations, and default relative paths.

This module is intentionally dependency-free (standard library only) so that it can be
imported from anywhere in the package without pulling in torch or transformers.
"""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Paths
#
# Paths are resolved relative to the project root, never hard-coded to a machine.
# Set SAJC_PROJECT_ROOT to relocate the artefact tree (e.g. to scratch storage on a
# cluster) without editing configuration files.
# ---------------------------------------------------------------------------
PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent
"""Directory containing this package."""

PROJECT_ROOT: Final[Path] = Path(
    os.environ.get("SAJC_PROJECT_ROOT", PACKAGE_ROOT.parents[1])
).resolve()
"""Repository root, or the value of ``SAJC_PROJECT_ROOT`` when set."""

PROJECT_ROOT_ENV_VAR: Final[str] = "SAJC_PROJECT_ROOT"

DEFAULT_CONFIG_DIR: Final[Path] = PROJECT_ROOT / "configs"
DEFAULT_DATA_DIR: Final[Path] = PROJECT_ROOT / "data"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "outputs"
DEFAULT_RESULTS_DIR: Final[Path] = PROJECT_ROOT / "results"

CHECKPOINT_SUBDIR: Final[str] = "checkpoints"
LOG_SUBDIR: Final[str] = "logs"
METRICS_SUBDIR: Final[str] = "metrics"
BENCHMARK_SUBDIR: Final[str] = "benchmarks"
FIGURE_SUBDIR: Final[str] = "figures"
TABLE_SUBDIR: Final[str] = "tables"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class CompressionMethod(StrEnum):
    """The experimental arms compared in the study.

    ``SEQUENTIAL`` is the **primary** order, prune then quantise (§3.5). ``SEQUENTIAL_QP`` is the
    reverse-order ablation run at one representative budget (§3.6), which exists so the joint arm
    is not compared only against a weak ordering. Where both are available, §6.1 defines the
    sequential baseline as the *stronger* of the two and requires the winning order be recorded.
    """

    DENSE = "dense"
    PRUNING = "pruning"
    QUANTISATION = "quantisation"
    SEQUENTIAL = "sequential"
    SEQUENTIAL_QP = "sequential_qp"
    JOINT = "joint"


class ReconstructionSolver(StrEnum):
    """How the layerwise reconstruction objective is minimised.

    Both solvers minimise the same objective and take the same mask, so an arm's result is
    comparable across them -- but only if **one** solver is used for every layer and every arm in a
    results table. Mixing them within a run would mean different layers were optimised by different
    algorithms, which is neither describable in a paper nor fair between arms.
    """

    ALS = "als"
    """Damped alternating refinement. Exact per output channel, ``O(out * |S|^3)``. Reference
    implementation: simple enough to verify by inspection, too slow for wide layers."""

    SWEEP = "sweep"
    """Error-compensated column sweep over a Cholesky factor of ``H^-1``. ``O(in^3 + out * in^2)``,
    which is what makes 1B-scale layers tractable."""


class SaliencyRule(StrEnum):
    """Which weights the pruning saliency is computed on.

    This is decision **D3**, and §3.8 makes it the definition of joint: a mask ranked on weights the
    quantiser has not touched is chosen in ignorance of the grid it will live on.
    """

    DENSE = "dense"
    """Rank ``|W| * ||X_j||`` on the dense weights. Correct for sequential P->Q, where no quantiser
    exists at mask time."""

    QUANTISED = "quantised"
    """Rank ``|Q_b(W)| * ||X_j||`` on the fake-quantised weights. Required for the joint arm."""


class CompressionStage(StrEnum):
    """Named stages a compression pipeline moves through.

    Sequential and joint pipelines are distinguished by *which* stages they visit and in
    what order; see :data:`SEQUENTIAL_STAGES` and :data:`JOINT_STAGES`.
    """

    DENSE = "dense"
    PREPARE = "prepare"
    FAKE_QUANTISATION_PREPARED = "fake_quantisation_prepared"
    PRUNED = "pruned"
    GRADUAL_PRUNING = "gradual_pruning"
    RECOVERED = "recovered"
    JOINTLY_FINE_TUNED = "jointly_fine_tuned"
    CALIBRATED = "calibrated"
    QUANTISED = "quantised"
    CONVERTED = "converted"


class Device(StrEnum):
    """Requested compute device. ``AUTO`` resolves to CUDA when it is available."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class DType(StrEnum):
    """Parameter dtype for loading and training.

    The dense baseline is FP32 by definition; the other values exist for recovery and
    joint training, which may run in reduced precision on GPU.
    """

    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"


class PruningGranularity(StrEnum):
    """Sparsity pattern applied to weight tensors."""

    UNSTRUCTURED = "unstructured"
    SEMI_STRUCTURED_2_4 = "2:4"
    SEMI_STRUCTURED_4_8 = "4:8"
    STRUCTURED_CHANNEL = "channel"


class MaskComparisonGroup(StrEnum):
    """Which weights compete against each other for survival at a given sparsity.

    §3.10 permits "global unstructured or fixed blockwise", so both are within the frozen protocol.
    The choice matters far more than it looks: activation-weighted saliency multiplies every weight in
    an input column by that column's norm, so under ``TENSOR`` a low-energy column scores low
    *everywhere* and can be pruned out entirely — deleting an input feature rather than thinning it.
    """

    TENSOR = "tensor"
    """Rank all weights in the layer against each other. One global threshold per tensor."""

    OUTPUT = "output"
    """Rank within each output channel, so every row keeps exactly the same fraction.

    No input column can be removed wholesale, because its survival is decided independently in each
    row. This is the comparison group Wanda uses.
    """


class PruningScheduleName(StrEnum):
    """How sparsity ramps from its initial to its final value."""

    ONE_SHOT = "one_shot"
    CONSTANT = "constant"
    LINEAR = "linear"
    CUBIC = "cubic"


class QuantisationScheme(StrEnum):
    """Whether the quantisation grid is centred on zero."""

    SYMMETRIC = "symmetric"
    ASYMMETRIC = "asymmetric"


class QuantisationGranularity(StrEnum):
    """Scope over which a single scale/zero-point pair is shared."""

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    PER_GROUP = "per_group"


# ---------------------------------------------------------------------------
# Pipeline definitions
#
# These make the two pipelines under comparison explicit and testable, rather than
# implicit in the control flow of the compressors.
# ---------------------------------------------------------------------------
SEQUENTIAL_STAGES: Final[tuple[CompressionStage, ...]] = (
    CompressionStage.DENSE,
    CompressionStage.PRUNED,
    CompressionStage.RECOVERED,
    CompressionStage.QUANTISED,
    CompressionStage.CONVERTED,
)
"""dense model -> pruning -> recovery -> quantisation -> conversion."""

JOINT_STAGES: Final[tuple[CompressionStage, ...]] = (
    CompressionStage.DENSE,
    CompressionStage.FAKE_QUANTISATION_PREPARED,
    CompressionStage.GRADUAL_PRUNING,
    CompressionStage.JOINTLY_FINE_TUNED,
    CompressionStage.CONVERTED,
)
"""dense model -> fake quantisation preparation -> gradual pruning during optimisation
-> recovery / joint fine-tuning -> final conversion."""


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------
BITS_PER_BYTE: Final[int] = 8
BYTES_PER_MIB: Final[int] = 1024 * 1024
BYTES_PER_MB: Final[int] = 1000 * 1000
FP32_BITS: Final[int] = 32
SUPPORTED_WEIGHT_BITS: Final[tuple[int, ...]] = (2, 3, 4, 8, 16, 32)
EPSILON: Final[float] = 1e-12


# ---------------------------------------------------------------------------
# Result schema
#
# Bump RESULT_SCHEMA_VERSION whenever RESULT_CSV_COLUMNS changes, so that CSV files
# written by different revisions of the code are never silently concatenated.
# ---------------------------------------------------------------------------
RESULT_SCHEMA_VERSION: Final[str] = "2"
"""Bumped to 2 when ``status`` was added: a failed run is now recorded rather than dropped, and
a reader must be able to filter those out."""

RESULT_CSV_COLUMNS: Final[tuple[str, ...]] = (
    "experiment_id",
    "timestamp",
    "git_commit",
    "schema_version",
    "status",
    "model_name",
    "model_size_label",
    "parameter_count",
    "compression_method",
    "sparsity",
    "quantisation_bits",
    "seed",
    "perplexity",
    "accuracy",
    "quality_retention",
    "top1_agreement",
    "latency_mean_ms",
    "latency_median_ms",
    "latency_p95_ms",
    "latency_std_ms",
    "throughput_tokens_per_s",
    "peak_memory_mb",
    "checkpoint_size_mb",
    "compression_ratio",
    "benchmark_num_threads",
    "benchmark_batch_size",
    "benchmark_sequence_length",
    "hardware_cpu_model",
    "software_torch_version",
)
"""Flat column order for the aggregated results CSV.

Nested detail (full hardware metadata, per-run latency samples, resolved config) lives in
the per-run JSON record; the CSV is the flat view used for plotting and tables.
"""

RESULT_JSON_SUFFIX: Final[str] = ".json"
RESULT_CSV_NAME: Final[str] = "results.csv"

DEFAULT_SEED: Final[int] = 1234
DEFAULT_SEEDS: Final[tuple[int, ...]] = (1234, 2345, 3456)
"""Seeds used to repeat each sweep cell; the spread across these is the error bar."""
