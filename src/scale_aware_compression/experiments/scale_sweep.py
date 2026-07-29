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

    if not models or not methods:
        raise ExperimentError("Sweep grid is empty: set sweep.models and sweep.methods")

    cells: list[SweepCell] = []
    for model_name, method, budget_label, seed in product(models, methods, budgets, seeds):
        spec = get_model_spec(model_name)
        overrides = _budget_overrides(config, budget_label)
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
                ),
                model_name=spec.short_name,
                size_label=spec.size_label,
                parameter_count=spec.parameter_count,
                method=method,
                budget_label=budget_label,
                seed=seed,
                sparsity=sparsity,
                bits=bits,
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
        Pairs matched on model, budget, and seed, ordered by parameter count then budget. Cells
        without a counterpart are logged and omitted: a joint gain computed against a different
        model, budget, or seed is not a joint gain.
    """
    sequential = {
        (cell.model_name, cell.budget_label, cell.seed): cell
        for cell in plan.cells_for(method=CompressionMethod.SEQUENTIAL)
    }
    joint = {
        (cell.model_name, cell.budget_label, cell.seed): cell
        for cell in plan.cells_for(method=CompressionMethod.JOINT)
    }

    pairs: list[tuple[SweepCell, SweepCell]] = []
    for key in sorted(sequential.keys() & joint.keys()):
        pairs.append((sequential[key], joint[key]))

    for key in sorted(sequential.keys() ^ joint.keys()):
        LOGGER.warning(
            "No joint/sequential counterpart for model=%s budget=%s seed=%s; this cell cannot "
            "contribute a joint gain.",
            *key,
        )
    if pairs:
        pairs.sort(key=lambda pair: (pair[0].parameter_count, pair[0].budget_label, pair[0].seed))
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

    ordered = _dense_first(plan.cells)
    records: list[ExperimentRecord] = []
    failures: list[tuple[SweepCell, Exception]] = []

    for index, cell in enumerate(ordered, start=1):
        if config.sweep.skip_existing and tracker.exists(cell.experiment_id):
            LOGGER.info(
                "[%d/%d] skip %s (already recorded)", index, len(ordered), cell.experiment_id
            )
            continue

        LOGGER.info("[%d/%d] run %s", index, len(ordered), cell.experiment_id)
        try:
            cell_config = build_cell_config(config, cell)
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

    incomplete = [
        (cell_a.model_name, cell_a.budget_label)
        for cell_a, cell_b in find_comparison_pairs(plan)
        if not (tracker.exists(cell_a.experiment_id) and tracker.exists(cell_b.experiment_id))
    ]
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


def scale_trend(
    records: list[dict[str, Any]],
    *,
    metric: str = "quality_retention",
) -> list[dict[str, Any]]:
    """Extract the joint-gain-versus-scale trend from completed records.

    This is the answer to the primary research question, in tabular form.

    Args:
        records: Loaded run records, as returned by :meth:`ExperimentTracker.load_all`.
        metric: Quality metric to compare on.

    Returns:
        One row per (model, budget), each holding the sequential score, the joint score, and the
        gain, ordered by parameter count.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(scale_sweep): group records by (model_name, budget_label), average the metric over
    # seeds within each group, then call metrics.joint_gain.joint_gain_summary() per group.
    # Report the spread across seeds alongside the mean: a gain smaller than the seed-to-seed
    # spread is not a finding, and the scale trend must be read with that in view.
    raise NotImplementedError(
        "scale_trend is not implemented yet; see the TODO in experiments/scale_sweep.py"
    )
