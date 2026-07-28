"""Layerwise reconstruction: choose compressed weights that preserve the layer's output.

Research plan §3.1 sets the objective per targeted linear layer

.. code-block:: text

    L_rec = || X W^T - X (M * Q_b(W))^T ||_F^2

Because ``nn.Linear`` computes ``Y = X W^T``, that expands into a sum over output channels of a
quadratic form in the Gram matrix ``H = X^T X``:

.. code-block:: text

    L_rec = sum_o (w_o - w_hat_o)^T H (w_o - w_hat_o)

which means the objective needs only ``H`` -- never the calibration activations themselves. That is
what keeps the memory cost fixed at one ``(in_features, in_features)`` buffer per layer regardless
of how much calibration data was used.

Two consequences worth being explicit about, because both are easy to get wrong:

**The pruned mass is not discarded.** The right-hand side of the solve is ``H w`` over the *full*
dense row, including the entries about to be zeroed. So the survivors absorb the contribution the
pruned weights were making, rather than the layer simply losing it. This error compensation is most
of what reconstruction buys over naive masking.

**Solver depth is decision D2.** This module implements damped alternating refinement, chosen over
an error-compensated Hessian column sweep because §3.3 makes second-order optional and the
iterative form matches §3.11's "equal total local optimisation steps" fairness unit. The Gram
matrix is accumulated regardless, so the sweep can be dropped in behind
:func:`reconstruct` without touching the capture path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.constants import QuantisationGranularity
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

LOGGER = get_logger(__name__)


class ReconstructionError(ValueError):
    """Raised when a reconstruction problem is malformed."""


@dataclass(slots=True)
class ReconstructionResult:
    """What reconstruction achieved for one layer.

    Attributes:
        weight: The reconstructed weight, masked and (when a bit width was given) on the grid.
        naive_loss: Objective of the naive baseline -- mask and round-to-nearest, no
            reconstruction. This is the number reconstruction has to beat to be worth its cost.
        final_loss: Objective of :attr:`weight`.
        local_steps_used: Refinement iterations actually run.
        accepted_steps: Iterations that improved the objective and were kept.
        history: Objective after each iteration, starting with the naive baseline.
    """

    weight: torch.Tensor
    naive_loss: float
    final_loss: float
    local_steps_used: int
    accepted_steps: int
    history: list[float] = field(default_factory=list)

    @property
    def improvement(self) -> float:
        """Absolute reduction in the objective versus the naive baseline."""
        return self.naive_loss - self.final_loss

    @property
    def relative_improvement(self) -> float:
        """Reduction as a fraction of the naive objective; ``0.0`` when the baseline is exact."""
        if self.naive_loss <= 0:
            return 0.0
        return self.improvement / self.naive_loss

    def report(self) -> dict[str, Any]:
        """Summarise for the run record.

        §7.2 asks for per-layer reconstruction loss (record field A9), and the naive baseline
        alongside it is what makes the number interpretable rather than just large or small.
        """
        return {
            "naive_loss": self.naive_loss,
            "final_loss": self.final_loss,
            "improvement": self.improvement,
            "relative_improvement": self.relative_improvement,
            "local_steps_used": self.local_steps_used,
            "accepted_steps": self.accepted_steps,
        }


def reconstruction_loss(
    gram: torch.Tensor,
    dense_weight: torch.Tensor,
    candidate: torch.Tensor,
) -> float:
    """Evaluate ``|| X W^T - X W_hat^T ||_F^2`` using the Gram matrix.

    Args:
        gram: ``H = X^T X``, shape ``(in_features, in_features)``.
        dense_weight: The original weight, shape ``(out_features, in_features)``.
        candidate: The compressed weight, same shape.

    Returns:
        The objective value, non-negative up to floating-point error.

    Raises:
        ReconstructionError: If the shapes are inconsistent.
    """
    _validate_shapes(gram, dense_weight, candidate)
    delta = (dense_weight - candidate).to(gram.dtype)
    # sum_o d_o^T H d_o, done as one matmul rather than a loop over output channels.
    return float(((delta @ gram) * delta).sum())


def solve_masked_rows(
    gram: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    damping: float = 1e-2,
) -> torch.Tensor:
    """Least-squares solve for the surviving weights of each output channel.

    For output channel ``o`` with keep-set ``S``, this solves

    .. code-block:: text

        (H_SS + lambda I) v = (H t_o)_S

    where ``t_o`` is the target row. Note the right-hand side uses the **whole** target row, so
    the surviving weights absorb the contribution of the entries being pruned.

    Args:
        gram: ``H``, shape ``(in_features, in_features)``.
        targets: Target rows, shape ``(out_features, in_features)``. Pass the dense weight to
            solve from scratch, or a residual to solve for a correction.
        mask: Boolean keep-mask, same shape as ``targets``.
        damping: Ridge coefficient relative to the mean Gram diagonal. Keeps the solve defined
            when a keep-set selects near-collinear input columns, which happens routinely at high
            sparsity.

    Returns:
        The solved rows, zero outside the mask, shaped like ``targets``.

    Raises:
        ReconstructionError: If shapes are inconsistent or damping is negative.

    Note:
        Cost is one dense solve per output channel, so roughly ``out_features * |S|^3``. That is
        fine for the single-layer validation this is written against, and for the small end of the
        sweep, but it will not scale to ``in_features = 8192`` unmodified. Phase 6 needs either
        mask-grouping (rows sharing a keep-set solve together) or the Hessian column sweep that
        D2 defers. Deliberately left simple and obviously correct first.
    """
    import torch

    _validate_shapes(gram, targets, mask)
    if damping < 0:
        raise ReconstructionError(f"damping must be >= 0, got {damping}")

    # Solve in float64: a keep-set can select strongly collinear columns, and a Cholesky-quality
    # failure here would surface as a silently poor reconstruction rather than an error.
    working_dtype = torch.float64
    gram64 = gram.to(working_dtype)
    targets64 = targets.to(working_dtype)

    mean_diagonal = torch.diagonal(gram64).mean()
    if not bool(torch.isfinite(mean_diagonal)) or mean_diagonal <= 0:
        mean_diagonal = torch.ones((), dtype=working_dtype, device=gram64.device)
    ridge = float(damping * mean_diagonal)

    rhs_all = targets64 @ gram64
    solved = torch.zeros_like(targets64)

    for row in range(targets64.shape[0]):
        keep = mask[row]
        num_kept = int(keep.sum())
        if num_kept == 0:
            continue
        indices = keep.nonzero(as_tuple=True)[0]
        submatrix = gram64.index_select(0, indices).index_select(1, indices)
        submatrix = (
            submatrix + torch.eye(num_kept, dtype=working_dtype, device=gram64.device) * ridge
        )
        rhs = rhs_all[row].index_select(0, indices)
        try:
            values = torch.linalg.solve(submatrix, rhs)
        except RuntimeError as error:  # pragma: no cover - needs a singular submatrix
            LOGGER.warning(
                "row %d solve failed (%s); falling back to least squares", row, type(error).__name__
            )
            values = torch.linalg.lstsq(submatrix, rhs.unsqueeze(-1)).solution.squeeze(-1)
        solved[row, indices] = values

    return solved.to(targets.dtype)


def reconstruct(
    gram: torch.Tensor,
    dense_weight: torch.Tensor,
    mask: torch.Tensor,
    *,
    local_steps: int = 4,
    damping: float = 1e-2,
    bits: int | None = None,
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL,
    group_size: int = 128,
    reestimate_scales: bool = True,
) -> ReconstructionResult:
    """Reconstruct a layer's weights under a fixed mask, optionally on a quantisation grid.

    Without ``bits`` this is pure pruning reconstruction and the first solve is already optimal,
    so later iterations find nothing to add. With ``bits`` the problem is discrete and the loop
    alternates: solve a continuous correction against the current residual, project it back onto
    the grid, and keep the result only if the objective actually improved.

    That accept-only-if-better rule is what makes the outcome safe to report. Projection onto a
    discrete grid is not guaranteed to reduce a quadratic objective, so an unguarded loop can end
    up worse than where it started; here the naive baseline is iterate zero and nothing replaces it
    unless it measurably wins.

    Args:
        gram: ``H = X^T X``, shape ``(in_features, in_features)``.
        dense_weight: The original weight, shape ``(out_features, in_features)``.
        mask: Boolean keep-mask, same shape. Fixed for the duration of this call -- mask updates
            belong to the joint arm's outer loop (§3.7), not here.
        local_steps: Refinement iterations. **This is the fairness unit** (§3.11): the sequential
            and joint arms must consume the same total across a model.
        damping: Ridge coefficient relative to the mean Gram diagonal.
        bits: Quantisation bit width, or ``None`` for pruning-only reconstruction.
        granularity: Quantisation granularity, when ``bits`` is set.
        group_size: Elements per quantisation group, for per-group granularity.
        reestimate_scales: Refit scales on the surviving weights at each projection. §3.8 requires
            this for the joint arm; leaving it on for every arm keeps the solver identical across
            arms, which is what makes the comparison about pipeline order alone.

    Returns:
        The result, including the naive baseline it was measured against.

    Raises:
        ReconstructionError: If shapes are inconsistent or ``local_steps`` is negative.
    """
    from scale_aware_compression.compression.quantisation import fake_quantise

    _validate_shapes(gram, dense_weight, mask)
    if local_steps < 0:
        raise ReconstructionError(f"local_steps must be >= 0, got {local_steps}")

    keep = mask.to(dense_weight.dtype)

    def project(candidate: torch.Tensor) -> torch.Tensor:
        """Apply the mask, then the quantisation grid if one was requested."""
        masked = candidate * keep
        if bits is None:
            return masked
        scales = None
        if not reestimate_scales:
            scales = _scales_for(dense_weight, bits, granularity, group_size)
        return (
            fake_quantise(
                masked, bits=bits, granularity=granularity, group_size=group_size, scales=scales
            )
            * keep
        )

    naive = project(dense_weight)
    naive_loss = reconstruction_loss(gram, dense_weight, naive)

    best = naive
    best_loss = naive_loss
    history = [naive_loss]
    accepted = 0

    for step in range(local_steps):
        residual = dense_weight - best
        correction = solve_masked_rows(gram, residual, mask, damping=damping)
        candidate = project(best + correction)
        candidate_loss = reconstruction_loss(gram, dense_weight, candidate)
        history.append(candidate_loss)

        if candidate_loss < best_loss:
            best, best_loss = candidate, candidate_loss
            accepted += 1
        elif bits is None:
            # Pruning-only is a convex problem the first solve closes exactly; a non-improving
            # iterate means we are at the optimum and further steps only burn budget.
            LOGGER.debug("pruning-only reconstruction converged after %d step(s)", step + 1)
            break

    return ReconstructionResult(
        weight=best,
        naive_loss=naive_loss,
        final_loss=best_loss,
        local_steps_used=len(history) - 1,
        accepted_steps=accepted,
        history=history,
    )


def _scales_for(
    weight: torch.Tensor,
    bits: int,
    granularity: QuantisationGranularity,
    group_size: int,
) -> torch.Tensor:
    from scale_aware_compression.compression.quantisation import compute_symmetric_scales

    return compute_symmetric_scales(
        weight, bits=bits, granularity=granularity, group_size=group_size
    )


def _validate_shapes(
    gram: torch.Tensor,
    weight: torch.Tensor,
    other: torch.Tensor,
) -> None:
    """Check the Gram matrix and two weight-shaped tensors agree.

    Args:
        gram: Expected square, ``(in_features, in_features)``.
        weight: Expected ``(out_features, in_features)``.
        other: Expected the same shape as ``weight``.

    Raises:
        ReconstructionError: On any mismatch.
    """
    if weight.ndim != 2:
        raise ReconstructionError(
            f"expected a 2-D (out_features, in_features) weight, got {tuple(weight.shape)}"
        )
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ReconstructionError(f"gram must be square, got {tuple(gram.shape)}")
    if gram.shape[0] != weight.shape[1]:
        raise ReconstructionError(
            f"gram width {gram.shape[0]} does not match in_features {weight.shape[1]}"
        )
    if other.shape != weight.shape:
        raise ReconstructionError(
            f"shape mismatch: {tuple(other.shape)} vs weight {tuple(weight.shape)}"
        )
