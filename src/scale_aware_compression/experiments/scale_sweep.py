"""The scale sweep: the same compression arms applied across model sizes.

The sweep is a full grid of ``models x methods x budgets x seeds``. Building the plan as data
before running anything makes two things possible: the plan can be printed and checked before
committing hours of compute, and the joint/sequential pairs can be verified to exist at every
scale. A joint gain is only defined where both arms ran at the same model, budget, and seed, so
:func:`find_comparison_pairs` locates those pairs explicitly rather than trusting the grid to
have been complete.

Status: plan construction and pair-matching are implemented; sweep execution is a placeholder.
"""

from __future__ import annotations

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
        The completed records, in plan order.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(scale_sweep): iterate the plan and, per cell,
    #   1. skip when config.sweep.skip_existing and tracker.exists(cell.experiment_id)
    #   2. build a per-cell ExperimentConfig by deep-merging cell.overrides plus the cell's
    #      model, method, budget, and seed onto the base config
    #   3. run it through ExperimentRunner
    #   4. on failure, honour config.sweep.continue_on_error -- log and continue, or re-raise
    # Order the runs dense-first per model: every other arm needs that model's dense perplexity
    # as its retention reference, so running dense last wastes the whole model's sweep.
    # Then verify with find_comparison_pairs() that each scale has a complete joint/sequential
    # pair before reporting the sweep as finished.
    raise NotImplementedError(
        "run_sweep is not implemented yet; see the TODO in experiments/scale_sweep.py"
    )


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
