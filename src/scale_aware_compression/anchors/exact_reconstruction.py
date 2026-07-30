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
