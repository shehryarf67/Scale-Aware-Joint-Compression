"""The scale sweep: the same compression arms applied across model sizes.

The sweep is a full grid of ``models x methods x budgets x seeds``. Building the plan as data
before running anything makes two things possible: the plan can be printed and checked before
committing hours of compute, and the joint/sequential pairs can be verified to exist at every
scale. A joint gain is only defined where both arms ran at the same model, budget, and seed, so
:func:`find_comparison_pairs` locates those pairs explicitly rather than trusting the grid to
have been complete.

Execution orders each model's dense run first, because every compressed arm normalises against
the dense perplexity *loaded from that model's record* rather than recomputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import product
from typing import Any

from scale_aware_compression.config import ExperimentConfig, deep_merge
from scale_aware_compression.constants import CompressionMethod
from scale_aware_compression.experiments.runner import (
    ExperimentError,
    ExperimentRecord,
    ExperimentTracker,
    make_experiment_id,
)
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.models.registry import get_model_spec
from scale_aware_compression.protocol import frozen_order_evidence, resolve_sequential_order

LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One planned run: a model, an arm, a budget, and a seed."""

    experiment_id: str
    model_name: str
    size_label: str
    parameter_count: int
    method: CompressionMethod
    budget_label: str
    seed: int
    sparsity: float
    bits: int
    replicate: int | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "size_label": self.size_label,
            "parameter_count": self.parameter_count,
            "method": self.method.value,
            "budget_label": self.budget_label,
            "seed": self.seed,
            "sparsity": self.sparsity,
            "bits": self.bits,
        }


@dataclass(slots=True)
class SweepPlan:
    """The full set of planned runs, with the grid it came from."""

    cells: list[SweepCell] = field(default_factory=list)
    models: list[str] = field(default_factory=list)
    methods: list[CompressionMethod] = field(default_factory=list)
    budgets: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.cells)

    @property
    def num_runs(self) -> int:
        """Total planned runs."""
        return len(self.cells)

    def cells_for(
        self,
        *,
        model_name: str | None = None,
        method: CompressionMethod | None = None,
        budget_label: str | None = None,
        seed: int | None = None,
    ) -> list[SweepCell]:
        """Filter the plan.

        Args:
            model_name: Restrict to one model.
            method: Restrict to one arm.
            budget_label: Restrict to one budget.
            seed: Restrict to one seed.

        Returns:
            The matching cells, in plan order.
        """
        return [
            cell
            for cell in self.cells
            if (model_name is None or cell.model_name == model_name)
            and (method is None or cell.method is method)
            and (budget_label is None or cell.budget_label == budget_label)
            and (seed is None or cell.seed == seed)
        ]

    def summary(self) -> dict[str, Any]:
        """Return a summary suitable for logging before the sweep starts."""
        return {
            "num_runs": self.num_runs,
            "models": self.models,
            "methods": [method.value for method in self.methods],
            "budgets": self.budgets,
            "seeds": self.seeds,
        }


def build_sweep_plan(config: ExperimentConfig) -> SweepPlan:
    """Expand a sweep configuration into an explicit list of runs.

    Args:
        config: An experiment config whose ``sweep`` section holds the grid. When
            ``sweep.models`` or ``sweep.methods`` is empty, the single model and method from the
            base config are used, so the same code path handles a one-off run.

    Returns:
        The expanded plan.

    Raises:
        ExperimentError: If the grid is empty, or a model is not in the registry.
    """
    sweep = config.sweep
    models = sweep.models or [config.model.name]
    methods = sweep.methods or [config.compression.method]
    budgets = sweep.budgets or [config.compression.budget_label]
    seeds = sweep.seeds or [config.runtime.seed]

    # A1 §5.1 withdrew the run-seed axis: seeds are inert here (F-15), so replicating over them
    # produces identical numbers. `sweep.replicates` is the axis that varies the calibration draw and
    # therefore actually produces a compressed model that differs. When it is unset the sweep falls
    # back to the seed axis so pre-amendment configs keep working.
    def replicates_for(model_name: str) -> list[int | None]:
        """Replicate indices for one model, honouring any per-model override.

        A1 §6 sets 8 at 160M and 410M but 5 at 1B. The reduced count takes the FIRST entries of the
        seed table, so the smaller set is a strict subset -- replicate 3 is the same calibration draw
        at every scale, which is what lets a scale trend be read as a scale trend rather than as a
        change of calibration.
        """
        if not sweep.replicates and model_name not in sweep.replicates_by_model:
            return [config.data.calibration_replicate]
        count = sweep.replicates_by_model.get(model_name, sweep.replicates)
        return list(range(count))

    if not models or not methods:
        raise ExperimentError("Sweep grid is empty: set sweep.models and sweep.methods")

    cells: list[SweepCell] = []
    for model_name in models:
        for method, budget_label, seed, replicate in product(
            methods, budgets, seeds, replicates_for(model_name)
        ):
            spec = get_model_spec(model_name)
            overrides = _budget_overrides(config, budget_label)
            # Resolve `sequential` to the FROZEN order for this cell when asked. §6.1 requires joint
            # gain against best-of {P→Q, Q→P}, and one frozen cell -- pythia-1b/moderate -- is Q→P.
            # Without this a confirmatory sweep would run P→Q there, which is the weaker baseline and
            # inflates the joint gain (B-30's fault, in the one run that cannot be redone).
            resolved_method = method
            if sweep.use_frozen_order and method is CompressionMethod.SEQUENTIAL:
                resolved_method = resolve_sequential_order(spec.short_name, budget_label)
                if resolved_method is not method:
                    LOGGER.info(
                        "Frozen order for %s/%s is %s (%s), not %s.",
                        spec.short_name,
                        budget_label,
                        resolved_method.value,
                        frozen_order_evidence(spec.short_name, budget_label),
                        method.value,
                    )
            method = resolved_method
            sparsity, bits = _resolve_budget(config, method, overrides)
            cells.append(
                SweepCell(
                    experiment_id=make_experiment_id(
                        model_name=spec.short_name,
                        method=method,
                        budget_label=budget_label,
                        seed=seed,
                        sparsity=sparsity,
                        bits=bits,
                        replicate=replicate,
                    ),
                    model_name=spec.short_name,
                    size_label=spec.size_label,
                    parameter_count=spec.parameter_count,
                    method=method,
                    budget_label=budget_label,
                    seed=seed,
                    sparsity=sparsity,
                    bits=bits,
                    replicate=replicate,
                    overrides=overrides,
                )
            )

    plan = SweepPlan(
        cells=cells,
        models=[get_model_spec(name).short_name for name in models],
        methods=list(methods),
        budgets=list(budgets),
        seeds=list(seeds),
    )
    LOGGER.info("Planned %d runs: %s", plan.num_runs, plan.summary())
    return plan


def _budget_overrides(config: ExperimentConfig, budget_label: str) -> dict[str, Any]:
    """Return the config fragment overriding settings for one budget."""
    fragment = config.sweep.budget_overrides.get(budget_label, {})
    if not isinstance(fragment, dict):
        raise ExperimentError(
            f"sweep.budget_overrides[{budget_label!r}] must be a mapping, got "
            f"{type(fragment).__name__}"
        )
    return dict(fragment)


def _resolve_budget(
    config: ExperimentConfig,
    method: CompressionMethod,
    overrides: dict[str, Any],
) -> tuple[float, int]:
    """Resolve the sparsity and bit width a cell will actually run at."""
    merged = deep_merge(config.to_dict(), overrides)
    compression = merged.get("compression", {})
    sparsity = float(compression.get("pruning", {}).get("sparsity", 0.0))
    bits = int(compression.get("quantisation", {}).get("bits", 32))
    if method in {CompressionMethod.DENSE, CompressionMethod.QUANTISATION}:
        sparsity = 0.0
    if method in {CompressionMethod.DENSE, CompressionMethod.PRUNING}:
        bits = 32
    return sparsity, bits


def find_comparison_pairs(plan: SweepPlan) -> list[tuple[SweepCell, SweepCell]]:
    """Find the (sequential, joint) pairs a joint gain can be computed from.

    Args:
        plan: The expanded sweep plan.

    Returns:
        Pairs matched on model, budget, seed **and calibration replicate**, ordered by parameter
        count then budget then replicate. Cells without a counterpart are logged and omitted: a
        joint gain computed against a different model, budget, seed or draw is not a joint gain.
    """

    # The REPLICATE is part of the key. Amendment A1 replaced the run-seed axis with paired
    # calibration replicates, and every replicate shares one run seed -- so keying on
    # (model, budget, seed) alone made all R replicates collide, and the dict silently kept only
    # the last. A 3-replicate grid with 6 sequential and 6 joint cells reported *2* pairs, and the
    # symmetric-difference warning below could not fire for the 4 it dropped, because they were
    # never distinct keys (B-38).
    def key_for(cell: SweepCell) -> tuple[Any, ...]:
        return (cell.model_name, cell.budget_label, cell.seed, cell.replicate)

    # The frozen best sequential baseline is Q->P for pythia-1b/moderate. It remains a sequential
    # comparator even though its resolved method enum is ``SEQUENTIAL_QP``; excluding it silently
    # drops five required confirmatory pairs.
    sequential_cells = plan.cells_for(method=CompressionMethod.SEQUENTIAL) + plan.cells_for(
        method=CompressionMethod.SEQUENTIAL_QP
    )
    sequential = {key_for(cell): cell for cell in sequential_cells}
    joint = {key_for(cell): cell for cell in plan.cells_for(method=CompressionMethod.JOINT)}

    pairs: list[tuple[SweepCell, SweepCell]] = []
    for key in sorted(sequential.keys() & joint.keys(), key=repr):
        pairs.append((sequential[key], joint[key]))

    for key in sorted(sequential.keys() ^ joint.keys(), key=repr):
        LOGGER.warning(
            "No joint/sequential counterpart for model=%s budget=%s seed=%s replicate=%s; this "
            "cell cannot contribute a joint gain.",
            *key,
        )
    if pairs:
        pairs.sort(
            key=lambda pair: (
                pair[0].parameter_count,
                pair[0].budget_label,
                pair[0].seed,
                # `replicate` is None outside a replicate grid, and None does not order against int.
                -1 if pair[0].replicate is None else pair[0].replicate,
            )
        )
    return pairs


def run_sweep(
    config: ExperimentConfig,
    plan: SweepPlan | None = None,
    *,
    tracker: ExperimentTracker | None = None,
) -> list[ExperimentRecord]:
    """Execute every cell in a sweep plan.

    Args:
        config: The base experiment config.
        plan: A pre-built plan. Built from ``config`` when omitted.
        tracker: Where records go. Defaults to ``<output_dir>/metrics``.

    Returns:
        The completed records, in execution order. Skipped cells are absent, so the returned list
        can be shorter than the plan.

    Raises:
        ExperimentError: If a cell fails and ``sweep.continue_on_error`` is false.
    """
    from scale_aware_compression.experiments.runner import ExperimentRunner

    plan = plan or build_sweep_plan(config)
    tracker = tracker or ExperimentTracker(config.runtime.output_dir / "metrics")

    # Check the arms will get equal solver budgets BEFORE spending the compute. §3.11's critical
    # fairness point is that a score obtained with more optimisation cannot be attributed to the
    # method, and a sweep is hours long -- discovering the mismatch from the records afterwards means
    # discovering it after the whole grid is invalid.
    _assert_grid_is_fair(config, plan)

    ordered = executable_cells(plan)
    records: list[ExperimentRecord] = []
    failures: list[tuple[SweepCell, Exception]] = []

    for index, cell in enumerate(ordered, start=1):
        cell_config = build_cell_config(config, cell)
        if config.sweep.skip_existing and tracker.exists_valid(cell.experiment_id, cell_config):
            LOGGER.info(
                "[%d/%d] skip %s (valid record present)", index, len(ordered), cell.experiment_id
            )
            continue

        LOGGER.info("[%d/%d] run %s", index, len(ordered), cell.experiment_id)
        try:
            records.append(ExperimentRunner(cell_config, tracker=tracker).run())
        except Exception as error:  # noqa: BLE001 - re-raised below unless told to continue
            if not config.sweep.continue_on_error:
                raise ExperimentError(f"Sweep cell {cell.experiment_id} failed: {error}") from error
            LOGGER.error("Cell %s failed, continuing: %s", cell.experiment_id, error)
            failures.append((cell, error))

    if failures:
        # §3.11 requires a failed run to stay in the log with its reason, so it is visible rather
        # than looking like a cell that was never planned.
        LOGGER.warning(
            "Sweep finished with %d failed cell(s): %s",
            len(failures),
            ", ".join(cell.experiment_id for cell, _ in failures),
        )

    incomplete = []
    for cell_a, cell_b in find_comparison_pairs(plan):
        config_a = build_cell_config(config, cell_a)
        config_b = build_cell_config(config, cell_b)
        if not (
            tracker.exists_valid(cell_a.experiment_id, config_a)
            and tracker.exists_valid(cell_b.experiment_id, config_b)
        ):
            incomplete.append((cell_a.model_name, cell_a.budget_label, cell_a.replicate))
    if incomplete:
        LOGGER.warning(
            "These model/budget cells have no complete sequential-versus-joint pair, so they "
            "cannot contribute a joint gain: %s",
            incomplete,
        )

    return records


def _assert_grid_is_fair(config: ExperimentConfig, plan: SweepPlan) -> None:
    """Pre-flight fairness check for a sweep grid.

    Args:
        config: The base config, for its reconstruction settings.
        plan: The expanded plan, for the arms it will run.

    Raises:
        LayerwiseError: If the grid pits arms with unequal solver budgets against each other.
    """
    from scale_aware_compression.compression.arms import plan_from_config
    from scale_aware_compression.compression.layerwise import assert_arms_can_be_matched

    arms = sorted(
        {cell.method.value for cell in plan.cells if cell.method is not CompressionMethod.DENSE}
    )
    if not arms:
        return
    assert_arms_can_be_matched(plan_from_config(config), arms)
    LOGGER.info("Fairness pre-flight passed: arms %s share one solver budget", arms)


def executable_cells(plan: SweepPlan) -> list[SweepCell]:
    """Return the exact cells the runner executes, in execution order.

    The logical grid deliberately contains dense aliases for every budget and calibration draw so
    its Cartesian structure remains explicit. Dense evaluation depends on neither, however, and
    execution represents it once per ``(model, run seed)``. Manifest generation, audits and the
    runner must all call this function so their definition of completeness cannot drift apart.

    Args:
        plan: The logical sweep plan.

    Returns:
        Dense cells first and deduplicated, followed by every compressed cell.
    """
    return _dense_first(plan.cells)


def _dense_first(cells: Sequence[SweepCell]) -> list[SweepCell]:
    """Order cells so each model's dense run happens before its compressed runs.

    Every compressed arm normalises against its model's dense perplexity, which is *loaded from the
    record* rather than recomputed. Running dense last would leave the whole model's sweep with no
    retention figure and nothing to compare.

    Also deduplicates dense cells on ``(model, seed)``. A dense run does not depend on the budget,
    but the budget label is part of the experiment id, so a four-budget grid plans four dense cells
    with four different ids and identical contents. Left alone that is wasted compute and four
    near-duplicate records, which §10.4 asks the audit to reject.

    Args:
        cells: The planned cells.

    Returns:
        Dense cells first, in model order, then the rest in plan order.
    """
    dense: list[SweepCell] = []
    seen_dense: set[tuple[str, int]] = set()
    rest: list[SweepCell] = []
    for cell in cells:
        if cell.method is CompressionMethod.DENSE:
            key = (cell.model_name, cell.seed)
            if key in seen_dense:
                LOGGER.debug("skipping duplicate dense cell %s", cell.experiment_id)
                continue
            seen_dense.add(key)
            dense.append(cell)
        else:
            rest.append(cell)
    return dense + rest


def build_cell_config(config: ExperimentConfig, cell: SweepCell) -> ExperimentConfig:
    """Materialise one cell as a standalone experiment configuration.

    Args:
        config: The base config the sweep was built from.
        cell: The cell to realise.

    Returns:
        A validated config for exactly this cell.
    """
    document = deep_merge(config.to_dict(), cell.overrides)
    document = deep_merge(
        document,
        {
            "experiment": {"id": cell.experiment_id},
            "model": {"name": cell.model_name, "size_label": cell.size_label},
            "runtime": {"seed": cell.seed},
            "data": {"calibration_replicate": cell.replicate},
            "compression": {
                "method": cell.method.value
                if isinstance(cell.method, CompressionMethod)
                else str(cell.method),
                "budget_label": cell.budget_label,
            },
        },
    )
    # A model's own config file carries its pinned revision, so it has to be re-resolved for the
    # cell's model rather than inherited from whichever model the base config named.
    document["model"].pop("hf_id", None)
    document["model"]["revision"] = _revision_for(cell.model_name, document)
    return ExperimentConfig.from_mapping(document)


def _revision_for(model_name: str, document: dict[str, Any]) -> str | None:
    """Return the pinned revision for ``model_name`` from its shipped model config.

    §2.7 requires every checkpoint pinned to a commit SHA. A sweep changes model between cells, so
    inheriting the base config's revision would pin every cell to the *first* model's SHA -- which
    either fails to load or, worse, silently loads the wrong weights if the SHA happens to exist in
    both repositories.

    Args:
        model_name: Registry short name for this cell.
        document: The merged document, used only for its config directory hint.

    Returns:
        The pinned revision, or ``None`` when no shipped config names one.
    """
    from scale_aware_compression.config import load_document
    from scale_aware_compression.constants import DEFAULT_CONFIG_DIR

    candidate = (
        DEFAULT_CONFIG_DIR / "models" / f"{model_name.replace('-', '_').replace('.', '_')}.yaml"
    )
    if not candidate.is_file():
        LOGGER.warning(
            "No shipped model config for %s at %s; the cell will use whatever revision the base "
            "config named, which may be the wrong checkpoint.",
            model_name,
            candidate,
        )
        return document.get("model", {}).get("revision")
    return load_document(candidate).get("model", {}).get("revision")


# Fields two records must agree on before their difference means anything. §3.11 lists the first
# group; the rest are things that silently changed under this project at least once and produced a
# joint-gain figure that had to be retracted.
COMPARABILITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_name", "model"),
    ("budget_label", "budget"),
    ("sparsity", "target sparsity"),
    ("quantisation_bits", "bit width"),
    ("schema_version", "record schema"),
    ("method_version", "method version"),
)


def _replicate_of(record: dict[str, Any]) -> Any:
    """Return the calibration replicate a record used, or its run seed for pre-A1 records.

    The replicate is what makes a comparison *paired*: arms must be differenced within one draw, never
    across draws. Falling back to the seed keeps records taken before A1 groupable, and since seeds
    were inert those records all share one group -- which is the honest representation of a protocol
    that produced no variation.
    """
    data = (record.get("config", {}) or {}).get("data", {}) or {}
    replicate = data.get("calibration_replicate")
    return f"rep{replicate}" if replicate is not None else f"seed{record.get('seed')}"


def _pair_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """The cell a record belongs to: one model, one budget, one replicate."""
    return (record.get("model_name"), record.get("budget_label"), _replicate_of(record))


def _incomparable(sequential: dict[str, Any], joint: dict[str, Any]) -> list[str]:
    """Reasons two records cannot be differenced, empty when they can.

    Checks the §3.11 invariants plus the things that have actually gone wrong here: unequal solver
    budgets, different calibration draws, different module coverage, different evaluation windows, and
    records produced by different versions of the algorithm.
    """
    reasons: list[str] = []
    for key, label in COMPARABILITY_FIELDS:
        if sequential.get(key) != joint.get(key):
            reasons.append(f"{label} differs ({sequential.get(key)} vs {joint.get(key)})")

    if _replicate_of(sequential) != _replicate_of(joint):
        reasons.append(
            f"calibration replicate differs ({_replicate_of(sequential)} vs "
            f"{_replicate_of(joint)}) -- a joint gain must be differenced within one draw"
        )

    def revision(record: dict[str, Any]) -> Any:
        return (record.get("config", {}).get("model", {}) or {}).get("revision")

    if revision(sequential) != revision(joint):
        reasons.append("model revision differs")

    def stats(record: dict[str, Any]) -> dict[str, Any]:
        return (record.get("compression", {}) or {}).get("statistics", {}) or {}

    sequential_stats, joint_stats = stats(sequential), stats(joint)
    if sequential_stats.get("calibration_fingerprint") != joint_stats.get(
        "calibration_fingerprint"
    ):
        reasons.append("calibration set differs")
    if sequential_stats.get("total_local_steps") != joint_stats.get("total_local_steps"):
        reasons.append(
            f"solver budget differs ({sequential_stats.get('total_local_steps')} vs "
            f"{joint_stats.get('total_local_steps')})"
        )
    sequential_modules = (sequential_stats.get("layerwise", {}) or {}).get("module_names")
    joint_modules = (joint_stats.get("layerwise", {}) or {}).get("module_names")
    if sequential_modules is not None and sequential_modules != joint_modules:
        reasons.append("module coverage differs")

    def window(record: dict[str, Any]) -> tuple[Any, Any]:
        payload = (record.get("quality", {}) or {}).get("perplexity", {}) or {}
        return (payload.get("num_sequences"), payload.get("dataset_fingerprint"))

    if window(sequential) != window(joint):
        reasons.append(f"evaluation window differs ({window(sequential)} vs {window(joint)})")
    return reasons


def _nll(record: dict[str, Any]) -> float | None:
    """Mean negative log-likelihood per token, or ``None`` when it cannot be derived.

    Preferred over perplexity as the comparison quantity. Perplexity is exponential, so a fixed gap
    means different things at different baselines -- five points at a dense perplexity of 20 is not
    five points at 200. NLL is additive, which is what makes a difference comparable across scales.
    """
    payload = (record.get("quality", {}) or {}).get("perplexity", {}) or {}
    total = payload.get("total_nll")
    tokens = payload.get("total_tokens")
    if total and tokens:
        return float(total) / float(tokens)
    perplexity = payload.get("perplexity")
    if perplexity:
        import math

        return math.log(float(perplexity))
    return None


def scale_trend(
    records: list[dict[str, Any]],
    *,
    metric: str = "quality_retention",
) -> list[dict[str, Any]]:
    """Extract the joint-gain-versus-scale trend from completed records.

    The answer to the primary research question, in tabular form -- and deliberately conservative
    about what it will put in that table. A row is emitted only when the two arms are genuinely
    comparable; otherwise the pair is reported with its reasons and no gain, because a difference
    between incomparable runs is not a gain.

    Two metrics are reported per cell:

    * **excess NLL** (``joint_advantage_nll``) is primary. NLL is additive, so a difference is
      comparable across scales where a perplexity difference is not.
    * **retention** is kept as the readable secondary figure.

    The x-axis is **targeted non-embedding parameters** (§2.6), taken from the layerwise report rather
    than the registry's total, because the embedding share falls sharply with scale and would confound
    a scale trend with a trend in how much of each model was compressed.

    Args:
        records: Loaded run records.
        metric: Retention key to read from each record's quality payload.

    Returns:
        One row per (model, budget, replicate), ordered by targeted parameter count. Rows carry
        ``comparable`` and, when false, ``reasons``.
    """
    from collections import defaultdict

    by_cell: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        # A failed or partial run must not silently contribute half a pair.
        if record.get("status") != "success":
            LOGGER.debug(
                "Skipping %s: status %s", record.get("experiment_id"), record.get("status")
            )
            continue
        method = record.get("compression_method")
        if method in {CompressionMethod.SEQUENTIAL.value, CompressionMethod.JOINT.value}:
            by_cell[_pair_key(record)][method] = record

    rows: list[dict[str, Any]] = []
    for key, arms in by_cell.items():
        model_name, budget_label, replicate = key
        sequential = arms.get(CompressionMethod.SEQUENTIAL.value)
        joint = arms.get(CompressionMethod.JOINT.value)
        if sequential is None or joint is None:
            LOGGER.info(
                "%s / %s / %s has no complete pair; no gain is defined",
                model_name,
                budget_label,
                replicate,
            )
            continue

        reasons = _incomparable(sequential, joint)
        stats = (joint.get("compression", {}) or {}).get("statistics", {}) or {}
        targeted = stats.get("targeted_parameters") or joint.get("parameter_count") or 0

        sequential_nll, joint_nll = _nll(sequential), _nll(joint)
        row: dict[str, Any] = {
            "model_name": model_name,
            "budget_label": budget_label,
            "replicate": replicate,
            "targeted_parameters": targeted,
            "total_parameters": joint.get("parameter_count"),
            "sequential_experiment_id": sequential.get("experiment_id"),
            "joint_experiment_id": joint.get("experiment_id"),
            "comparable": not reasons,
        }
        if reasons:
            row["reasons"] = reasons
            rows.append(row)
            continue

        row["sequential_nll"] = sequential_nll
        row["joint_nll"] = joint_nll
        # Positive means joint is better: it achieved the lower NLL.
        row["joint_advantage_nll"] = (
            None if sequential_nll is None or joint_nll is None else sequential_nll - joint_nll
        )
        for label, record in (("sequential", sequential), ("joint", joint)):
            retention = (record.get("quality", {}) or {}).get("retention", {}) or {}
            row[f"{label}_retention"] = retention.get("perplexity_retention")
        if row.get("sequential_retention") is not None and row.get("joint_retention") is not None:
            row["joint_gain_retention_pp"] = row["joint_retention"] - row["sequential_retention"]
        row["metric"] = metric
        rows.append(row)

    rows.sort(
        key=lambda item: (
            item["targeted_parameters"],
            str(item["budget_label"]),
            str(item["replicate"]),
        )
    )
    usable = sum(1 for row in rows if row["comparable"])
    LOGGER.info("Scale trend: %d comparable pair(s) of %d", usable, len(rows))
    return rows
