"""Check the reconstruction sweep against the provable optimum of its own objective.

Amendment A1 §5.5(b) asks for a SparseGPT anchor to validate the reconstruction path -- the half the
Wanda anchor cannot reach, and the half where [F-18](../../docs/findings_log.md)'s B-22 and B-23 lived.
This module does something **stronger than reimplementing SparseGPT**, and it is worth being explicit
about why, because it is a deliberate substitution rather than a shortcut.

For a **fixed mask**, minimising

    L(W_hat) = sum_o (w_o - w_hat_o)^T H (w_o - w_hat_o)

subject to ``w_hat`` being supported on each row's keep-set has a closed-form solution. Differentiating
row ``o`` with keep-set ``S`` gives the masked normal equations

    H[S,S] w_hat[S] = H[S,:] w        (and w_hat[S^c] = 0)

so ``w_hat[S] = H[S,S]^-1 H[S,:] w`` is the **exact minimiser**. Note the right-hand side runs over the
*full* dense row: survivors absorb what the pruned weights were contributing, which is the error
compensation the sweep exists to perform.

That makes this a better check than a second heuristic:

* SparseGPT's contribution is **speed**, not a different objective. Comparing our sweep to another
  approximation tells you the two approximations agree; comparing it to the exact optimum tells you
  how much the approximation actually gives up.
* The reference is short enough to be correct by inspection, which a full SparseGPT port would not be.
* It yields a genuine **lower bound**. A sweep result *below* the optimum is impossible, so if one
  appears it proves a defect -- a mask not respected, or an objective computed inconsistently.

**What it still does not settle.** Whether our absolute quality is in line with published work. That
question -- "is ~57% retention plausible or does it indicate a remaining gap?" -- needs an external run
with comparable numbers, and remains open. This anchor establishes that the solver optimises what it
claims to optimise, not that the method is competitive.

The objective is **separable across output rows**, which is what makes the exact solve tractable at
all: each row is an independent quadratic, so a sample of rows gives complete, valid tests of those
rows rather than an approximation of the whole layer. A full-layer exact solve would cost
``out_features * |S|^3`` and is the very thing the sweep was written to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

LOGGER = get_logger(__name__)


class ExactReconstructionError(RuntimeError):
    """Raised when the exact reference cannot be computed or is given inconsistent shapes."""


@dataclass(frozen=True, slots=True)
class RowComparison:
    """Our sweep against the exact optimum, for one output row."""

    row: int
    kept: int
    naive_objective: float
    ours_objective: float
    optimal_objective: float

    @property
    def improvement_over_naive(self) -> float:
        """Fraction of the naive objective our sweep removed. Negative means it made things worse."""
        if self.naive_objective <= 0.0:
            return 0.0
        return (self.naive_objective - self.ours_objective) / self.naive_objective

    @property
    def available_improvement(self) -> float:
        """Fraction the *exact* optimum removes. The ceiling our sweep is measured against."""
        if self.naive_objective <= 0.0:
            return 0.0
        return (self.naive_objective - self.optimal_objective) / self.naive_objective

    @property
    def efficiency(self) -> float:
        """Share of the achievable gain our sweep captured, in ``[0, 1]`` when well behaved.

        ``1.0`` means the sweep matched the exact optimum. Reported rather than pass/failed on its
        own, because a one-pass sweep is not expected to be exactly optimal -- only close, and never
        better.
        """
        available = self.naive_objective - self.optimal_objective
        if available <= 0.0:
            return 1.0
        return (self.naive_objective - self.ours_objective) / available

    @property
    def beats_the_optimum(self) -> bool:
        """True when our sweep scored *below* the provable minimum, which is impossible.

        A small tolerance absorbs float64 round-off in the solve. Anything beyond it means the mask
        was not respected, or the two objectives were not computed on the same quantity -- exactly
        the class of fault that produced B-22 and B-23.
        """
        scale = max(abs(self.optimal_objective), 1e-30)
        return (self.optimal_objective - self.ours_objective) / scale > 1e-6

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-serialisable mapping."""
        return {
            "row": self.row,
            "kept": self.kept,
            "naive_objective": self.naive_objective,
            "ours_objective": self.ours_objective,
            "optimal_objective": self.optimal_objective,
            "improvement_over_naive": self.improvement_over_naive,
            "available_improvement": self.available_improvement,
            "efficiency": self.efficiency,
            "beats_the_optimum": self.beats_the_optimum,
        }


@dataclass(slots=True)
class ExactReconstructionReport:
    """Verdict across every sampled row of every sampled module."""

    module_rows: dict[str, list[RowComparison]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def _all(self) -> list[RowComparison]:
        return [row for rows in self.module_rows.values() for row in rows]

    @property
    def rows_compared(self) -> int:
        """Total rows compared."""
        return len(self._all())

    @property
    def impossible_rows(self) -> int:
        """Rows scoring below the provable optimum. Must be zero."""
        return sum(1 for row in self._all() if row.beats_the_optimum)

    @property
    def rows_worse_than_naive(self) -> int:
        """Rows where reconstruction *increased* the objective.

        Should be zero: the refinement loop keeps naive rounding as iterate zero and only accepts
        improvements, so a positive count means that guard is not working.
        """
        return sum(1 for row in self._all() if row.improvement_over_naive < -1e-9)

    @property
    def mean_efficiency(self) -> float:
        """Mean share of the achievable gain captured, across all rows."""
        rows = self._all()
        if not rows:
            return 0.0
        return sum(row.efficiency for row in rows) / len(rows)

    @property
    def worst_efficiency(self) -> float:
        """Lowest per-row efficiency, or 0.0 when nothing was compared."""
        return min((row.efficiency for row in self._all()), default=0.0)

    @property
    def mean_improvement_over_naive(self) -> float:
        """Mean fraction of the naive objective the sweep removed."""
        rows = self._all()
        if not rows:
            return 0.0
        return sum(row.improvement_over_naive for row in rows) / len(rows)

    @property
    def passes(self) -> bool:
        """The verdict.

        Two hard invariants, both of which are facts rather than quality judgements:

        * no row may score below its provable optimum;
        * no row may end worse than naive rounding.

        Efficiency is **reported, not gated**. A one-pass sweep giving up some of the achievable gain
        is the documented trade for making wide layers tractable; picking a threshold for it here
        would be inventing an acceptance criterion the plan does not have.
        """
        return bool(self._all()) and self.impossible_rows == 0 and self.rows_worse_than_naive == 0

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as a JSON-serialisable mapping."""
        return {
            "anchor": "exact_masked_reconstruction_optimum",
            "rows_compared": self.rows_compared,
            "modules_compared": len(self.module_rows),
            "passes": self.passes,
            "impossible_rows": self.impossible_rows,
            "rows_worse_than_naive": self.rows_worse_than_naive,
            "mean_efficiency": self.mean_efficiency,
            "worst_efficiency": self.worst_efficiency,
            "mean_improvement_over_naive": self.mean_improvement_over_naive,
            "modules": {
                name: [row.to_dict() for row in rows] for name, rows in self.module_rows.items()
            },
            "notes": list(self.notes),
        }

    def summary_lines(self) -> list[str]:
        """Return a short human-readable verdict."""
        lines = [
            "Exact-optimum reconstruction anchor (A1 §5.5b)",
            f"  modules sampled         : {len(self.module_rows)}",
            f"  rows compared           : {self.rows_compared}",
            f"  rows below the optimum  : {self.impossible_rows} (must be 0)",
            f"  rows worse than naive   : {self.rows_worse_than_naive} (must be 0)",
            f"  mean improvement        : {self.mean_improvement_over_naive:+.4%} of naive",
            f"  mean efficiency         : {self.mean_efficiency:.4f} of achievable",
            f"  worst-row efficiency    : {self.worst_efficiency:.4f}",
            f"  VERDICT                 : {'PASS' if self.passes else 'INVESTIGATE'}",
        ]
        for name, rows in self.module_rows.items():
            broken = [
                row for row in rows if row.beats_the_optimum or row.improvement_over_naive < 0
            ]
            for row in broken[:3]:
                lines.append(
                    f"    {name} row {row.row}: ours={row.ours_objective:.6e} "
                    f"optimal={row.optimal_objective:.6e} naive={row.naive_objective:.6e}"
                )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines


@dataclass(frozen=True, slots=True)
class ArmRowComparison:
    """One output row under both arms' masks, each against its own exact optimum.

    The sharp question this answers is not "how efficient is the solver" but **"does the solver rank
    the two masks the same way the exact optimum does?"** If our sweep says the joint mask is better
    while the optimum says the sequential mask is, then the reported joint gain on that row is a
    solver artefact and not a property of the mask.
    """

    row: int
    sequential: RowComparison
    joint: RowComparison

    @property
    def sweep_prefers_joint(self) -> bool:
        """True when our solver produced a lower objective under the joint mask."""
        return self.joint.ours_objective < self.sequential.ours_objective

    @property
    def optimum_prefers_joint(self) -> bool:
        """True when the *exact* optimum is lower under the joint mask."""
        return self.joint.optimal_objective < self.sequential.optimal_objective

    @property
    def ranking_disagrees(self) -> bool:
        """True when the solver and the optimum disagree about which mask is better.

        This is the smoking gun. A row where it holds is a row whose contribution to the measured
        joint gain has the wrong sign relative to the masks' true quality.
        """
        return self.sweep_prefers_joint != self.optimum_prefers_joint

    @property
    def efficiency_gap(self) -> float:
        """Joint efficiency minus sequential efficiency.

        Positive means the solver does *better* on the joint mask, which would inflate the joint gain
        by an amount that has nothing to do with the mask mechanism.
        """
        return self.joint.efficiency - self.sequential.efficiency

    @property
    def sweep_advantage(self) -> float:
        """Objective reduction the sweep attributes to the joint mask. Positive favours joint."""
        return self.sequential.ours_objective - self.joint.ours_objective

    @property
    def optimal_advantage(self) -> float:
        """Objective reduction genuinely available from the joint mask. Positive favours joint."""
        return self.sequential.optimal_objective - self.joint.optimal_objective

    @property
    def advantage_fidelity(self) -> float:
        """How faithfully the sweep's mask preference reflects the true one, as a magnitude ratio.

        ``optimal_advantage / sweep_advantage``. **Read it together with the sign of
        :attr:`sweep_advantage`, because the ratio alone is ambiguous:**

        * both positive -- the joint mask genuinely wins. ``< 1`` means the sweep *overstates* the win,
          ``> 1`` that it understates it.
        * both negative -- the joint mask genuinely loses. ``< 1`` means the sweep *overstates the
          loss*, ``> 1`` that it understates it.
        * opposite signs -- the ratio is negative and the sweep has the direction wrong, which
          :attr:`ranking_disagrees` reports directly.

        Named for the magnitude relationship rather than for "joint benefit", because an earlier
        version was called the latter and read as "64% of the joint gain is real" on a run where every
        row's advantage was in fact *negative*.
        """
        if abs(self.sweep_advantage) < 1e-30:
            return 0.0
        return self.optimal_advantage / self.sweep_advantage

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-serialisable mapping."""
        return {
            "row": self.row,
            "sequential": self.sequential.to_dict(),
            "joint": self.joint.to_dict(),
            "sweep_prefers_joint": self.sweep_prefers_joint,
            "optimum_prefers_joint": self.optimum_prefers_joint,
            "ranking_disagrees": self.ranking_disagrees,
            "efficiency_gap": self.efficiency_gap,
            "sweep_advantage": self.sweep_advantage,
            "optimal_advantage": self.optimal_advantage,
            "attributable_fraction": self.attributable_fraction,
        }


@dataclass(slots=True)
class ArmSlackReport:
    """Whether solver slack differs between the arms enough to confound the comparison."""

    bits: int
    target_sparsity: float
    module_rows: dict[str, list[ArmRowComparison]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def _all(self) -> list[ArmRowComparison]:
        return [row for rows in self.module_rows.values() for row in rows]

    @property
    def rows_compared(self) -> int:
        """Total rows compared under both masks."""
        return len(self._all())

    @property
    def mean_sequential_efficiency(self) -> float:
        """Mean solver efficiency under the sequential mask."""
        rows = self._all()
        return sum(r.sequential.efficiency for r in rows) / len(rows) if rows else 0.0

    @property
    def mean_joint_efficiency(self) -> float:
        """Mean solver efficiency under the joint mask."""
        rows = self._all()
        return sum(r.joint.efficiency for r in rows) / len(rows) if rows else 0.0

    @property
    def mean_efficiency_gap(self) -> float:
        """Joint minus sequential mean efficiency. The size of the confound."""
        return self.mean_joint_efficiency - self.mean_sequential_efficiency

    @property
    def rows_where_ranking_disagrees(self) -> int:
        """Rows where the solver and the optimum disagree on which mask is better."""
        return sum(1 for r in self._all() if r.ranking_disagrees)

    @property
    def disagreement_rate(self) -> float:
        """Fraction of rows where the solver's mask preference is not the true one."""
        rows = self._all()
        return self.rows_where_ranking_disagrees / len(rows) if rows else 0.0

    @property
    def rows_favouring_joint(self) -> int:
        """Rows where the **exact optimum** gives the joint mask the lower objective.

        Reported alongside the fidelity ratio because that ratio is meaningless without knowing which
        direction it is a ratio of.
        """
        return sum(1 for r in self._all() if r.optimum_prefers_joint)

    @property
    def aggregate_advantage_fidelity(self) -> float:
        """Magnitude ratio of true to swept mask advantage, aggregated over rows.

        Aggregated over objectives rather than averaged over per-row ratios, because a per-row ratio
        blows up on rows where the sweep found almost no advantage. Interpret with
        :attr:`rows_favouring_joint` and the sign of the aggregate advantage -- see
        :attr:`ArmRowComparison.advantage_fidelity`.
        """
        rows = self._all()
        swept = sum(r.sweep_advantage for r in rows)
        if abs(swept) < 1e-30:
            return 0.0
        return sum(r.optimal_advantage for r in rows) / swept

    @property
    def aggregate_sweep_advantage(self) -> float:
        """Total objective reduction the sweep attributes to the joint mask. Negative favours seq."""
        return sum(r.sweep_advantage for r in self._all())

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as a JSON-serialisable mapping."""
        return {
            "anchor": "arm_dependent_solver_slack",
            "bits": self.bits,
            "target_sparsity": self.target_sparsity,
            "rows_compared": self.rows_compared,
            "modules_compared": len(self.module_rows),
            "mean_sequential_efficiency": self.mean_sequential_efficiency,
            "mean_joint_efficiency": self.mean_joint_efficiency,
            "mean_efficiency_gap": self.mean_efficiency_gap,
            "rows_where_ranking_disagrees": self.rows_where_ranking_disagrees,
            "disagreement_rate": self.disagreement_rate,
            "rows_favouring_joint": self.rows_favouring_joint,
            "aggregate_sweep_advantage": self.aggregate_sweep_advantage,
            "aggregate_advantage_fidelity": self.aggregate_advantage_fidelity,
            "modules": {
                name: [row.to_dict() for row in rows] for name, rows in self.module_rows.items()
            },
            "notes": list(self.notes),
        }

    def summary_lines(self) -> list[str]:
        """Return a short human-readable verdict.

        There is deliberately no ``passes``. This anchor measures the size of a confound; what counts
        as tolerable depends on the effect size it is being compared against, and that is a protocol
        judgement rather than something a threshold here should pre-empt.
        """
        return [
            f"Arm-dependent solver slack at {self.target_sparsity:.0%} sparsity, W{self.bits}",
            f"  modules / rows            : {len(self.module_rows)} / {self.rows_compared}",
            f"  mean efficiency, sequential: {self.mean_sequential_efficiency:.4f}",
            f"  mean efficiency, joint     : {self.mean_joint_efficiency:.4f}",
            f"  efficiency gap (joint-seq) : {self.mean_efficiency_gap:+.4f}",
            f"  rows where solver misranks : {self.rows_where_ranking_disagrees} "
            f"({self.disagreement_rate:.1%})",
            f"  optimum favours joint on   : {self.rows_favouring_joint}/{self.rows_compared} rows",
            f"  swept mask advantage       : {self.aggregate_sweep_advantage:+.4e} "
            f"({'favours joint' if self.aggregate_sweep_advantage > 0 else 'favours sequential'})",
            f"  advantage fidelity         : {self.aggregate_advantage_fidelity:.4f} "
            f"(magnitude ratio true/swept; read with the sign above)",
        ] + [f"  note: {note}" for note in self.notes]


def row_objective(
    gram: torch.Tensor,
    dense_row: torch.Tensor,
    candidate_row: torch.Tensor,
) -> float:
    """Evaluate ``d^T H d`` for one output row, where ``d = w - w_hat``.

    Computed here rather than imported, so a fault in the shared loss function cannot make our result
    and the reference agree by cancelling in both.

    Args:
        gram: ``H = X^T X``, shape ``(in_features, in_features)``.
        dense_row: The original row, shape ``(in_features,)``.
        candidate_row: The compressed row, same shape.

    Returns:
        The objective for this row, non-negative up to float error.

    Raises:
        ExactReconstructionError: If the shapes are inconsistent.
    """
    import torch

    if dense_row.shape != candidate_row.shape:
        raise ExactReconstructionError(
            f"row shapes differ: {tuple(dense_row.shape)} vs {tuple(candidate_row.shape)}"
        )
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1] or gram.shape[0] != dense_row.shape[0]:
        raise ExactReconstructionError(
            f"gram {tuple(gram.shape)} does not match a row of width {dense_row.shape[0]}"
        )
    delta = (dense_row - candidate_row).to(torch.float64)
    return float(delta @ gram.to(torch.float64) @ delta)


def exact_masked_row_optimum(
    gram: torch.Tensor,
    dense_row: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    """Return the exact minimiser of ``d^T H d`` for one row, restricted to ``keep``.

    Solves the masked normal equations ``H[S,S] x = H[S,:] w`` in float64, falling back to a
    least-squares solve when the submatrix is singular -- which happens for real calibration data,
    where dead or perfectly correlated input columns are routine.

    Deliberately **no damping**. Damping is a regularisation choice our solver makes for
    conditioning; the reference must minimise the true objective, or it is not a lower bound on it.

    Args:
        gram: ``H = X^T X``, shape ``(in_features, in_features)``.
        dense_row: The original row, shape ``(in_features,)``.
        keep: Boolean keep-mask for this row, same shape.

    Returns:
        The optimal row, zero outside ``keep``, in float64.

    Raises:
        ExactReconstructionError: If shapes are inconsistent.
    """
    import torch

    if dense_row.shape != keep.shape:
        raise ExactReconstructionError(
            f"row and mask shapes differ: {tuple(dense_row.shape)} vs {tuple(keep.shape)}"
        )
    gram64 = gram.to(torch.float64)
    row64 = dense_row.to(torch.float64)
    optimal = torch.zeros_like(row64)

    indices = keep.nonzero(as_tuple=True)[0]
    if indices.numel() == 0:
        return optimal

    submatrix = gram64.index_select(0, indices).index_select(1, indices)
    target = gram64.index_select(0, indices) @ row64

    try:
        solution = torch.linalg.solve(submatrix, target)
        if not bool(torch.isfinite(solution).all()):
            raise RuntimeError("non-finite solution")
    except RuntimeError:
        # A singular H[S,S] means the objective has a flat direction: several rows achieve the same
        # minimum. lstsq returns the minimum-norm one, whose objective value is the same -- and the
        # objective is what this anchor compares.
        LOGGER.debug("H[S,S] is singular for this row; falling back to a least-squares solve")
        solution = torch.linalg.lstsq(submatrix, target.unsqueeze(1)).solution.squeeze(1)

    optimal.index_copy_(0, indices, solution)
    return optimal


def compare_row(
    row: int,
    gram: torch.Tensor,
    dense_row: torch.Tensor,
    ours_row: torch.Tensor,
    keep: torch.Tensor,
) -> RowComparison:
    """Compare our reconstructed row against naive masking and against the exact optimum.

    Args:
        row: Row index, for the report.
        gram: ``H = X^T X``.
        dense_row: The original row.
        ours_row: The row our solver produced.
        keep: Boolean keep-mask for this row.

    Returns:
        The comparison.

    Raises:
        ExactReconstructionError: If our row violates the mask, which would make every objective
            comparison meaningless.
    """
    import torch

    violated = int((~keep & (ours_row != 0)).sum())
    if violated:
        raise ExactReconstructionError(
            f"row {row}: our weight is non-zero at {violated} masked position(s). The objective "
            "cannot be compared against a mask-respecting optimum."
        )

    naive_row = torch.where(keep, dense_row, torch.zeros_like(dense_row))
    optimal_row = exact_masked_row_optimum(gram, dense_row, keep)
    return RowComparison(
        row=row,
        kept=int(keep.sum()),
        naive_objective=row_objective(gram, dense_row, naive_row),
        ours_objective=row_objective(gram, dense_row, ours_row),
        optimal_objective=row_objective(gram, dense_row, optimal_row),
    )
