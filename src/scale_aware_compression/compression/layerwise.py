"""The layerwise driver: one code path that every experimental arm calls.

Research plan §3.8 defines what qualifies as joint, and the only way to make that checkable in code
rather than by inspection is for every arm to run through the *same* machinery and differ only in
the order it is called. That is what this module provides.

.. code-block:: text

    pruning-only     mask -> reconstruct
    quant-only       quantise -> reconstruct
    sequential P->Q  mask -> reconstruct -> quantise -> reconstruct
    sequential Q->P  quantise -> mask -> reconstruct
    joint            repeat K times:
                       fake-quantise the survivors
                       rescore saliency UNDER the quantised weights   <- §3.8
                       update the mask at target sparsity
                       re-estimate scales on the survivors            <- §3.8
                       reconstruct for a fixed local-step budget
                     freeze M and Q

Blocks are visited in depth order, and within a block the targeted modules are compressed in
**dependency-group order** with activations recaptured between groups. So each layer is fitted against
the inputs it will really see at inference, including the effect of modules compressed earlier in its
own block.

Capturing once per block -- which is what this did originally -- fits an MLP down-projection against
activations the *dense* up-projection produced, inputs that never occur once the up-projection is
compressed. That is blockwise reconstruction, and describing it as layerwise overstates it. The groups
are declared per architecture in ``models/adapters.py``, because the dependency structure differs:
Pythia's parallel residual gives two groups per block, Qwen2's sequential residual gives four.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.activations import ActivationStatistics
from scale_aware_compression.compression.masks import build_mask_from_scores, realised_sparsity
from scale_aware_compression.compression.pruning import (
    activation_weighted_saliency,
    keep_benefit_saliency,
)
from scale_aware_compression.compression.quantisation import (
    compute_symmetric_scales,
    fake_quantise,
    quantise_weight,
    search_clipping_scales,
)
from scale_aware_compression.compression.reconstruct import reconstruct, reconstruction_loss
from scale_aware_compression.constants import (
    MaskComparisonGroup,
    PruningGranularity,
    QuantisationGranularity,
    ReconstructionSolver,
)
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)


class LayerwiseError(RuntimeError):
    """Raised when the layerwise loop cannot proceed."""


@dataclass(slots=True)
class LayerPlan:
    """The compression settings applied to every targeted layer.

    One object per run, shared by every arm. If two arms disagree on any field here, they are not
    running a matched comparison -- which is why :func:`assert_matched_plans` exists.

    Attributes:
        sparsity: Target fraction of targeted weights pruned. ``0.0`` disables pruning.
        bits: Weight bit width, or ``None`` to leave precision untouched.
        granularity: Quantisation granularity.
        group_size: Elements per quantisation group.
        pruning_granularity: Sparsity pattern.
        solver: Which reconstruction minimiser to use.
        local_steps: Refinement iterations per reconstruction call -- the fairness unit (§3.11).
        joint_iterations: Outer alternations, joint arm only.
        damping: Ridge coefficient relative to the mean Gram diagonal.
        block_size: Column block width for the sweep solver.
        activation_order: Whether the sweep visits high-energy columns first.
    """

    retain_masks: bool = False
    """Keep true keep-masks on the report. Set only by the post-hoc recovery ablation."""

    sparsity: float = 0.0
    bits: int | None = None
    granularity: QuantisationGranularity = QuantisationGranularity.PER_CHANNEL
    group_size: int = 128
    pruning_granularity: PruningGranularity = PruningGranularity.UNSTRUCTURED
    comparison_group: MaskComparisonGroup = MaskComparisonGroup.OUTPUT
    """Which weights compete for survival. Per-output by default; see the measurement in
    docs/validity_threats.md."""
    solver: ReconstructionSolver = ReconstructionSolver.SWEEP
    local_steps: int = 1
    joint_iterations: int = 2
    """Outer alternations in the joint arm, and the count that makes it comparable.

    Two, matching the sequential pipeline's two solver calls. This default and
    ``ReconstructionConfig.joint_iterations`` must agree -- they were 2 and 4 for a while, which is
    how a whole screening grid ran with the joint arm on twice the optimisation budget.
    :func:`assert_arms_can_be_matched` now refuses such a grid before it starts.
    """
    damping: float = 1e-2
    block_size: int = 128
    activation_order: bool = True
    scale_search: bool = False
    """Fit scales by minimising *pre-reconstruction* error rather than matching ``max|W|``.

    **Off by default because measurement says it hurts.** It reduces naive quantisation error by
    12.8% at W4 on real Pythia-160M layers, but the layer objective *after* reconstruction gets
    worse (joint-vs-sequential gain at W4: +1.12% with max-abs, -0.99% with the search). The two
    objectives are not the same one: clipping saturates outliers, and a saturated weight cannot be
    repaired by error compensation afterwards.

    Retained as an ablation. A version that searched the clipping ratio against the *post*-
    reconstruction objective would be the principled fix, at the cost of one full sweep per
    candidate ratio.

    When on, it applies to **every** arm -- giving the joint arm a better quantiser than the
    sequential arm would produce a "joint gain" that was really a quantiser difference (§3.11).
    """
    keep_benefit_saliency: bool = False
    """Score the mask by keep-versus-prune benefit rather than quantised magnitude.

    **Off by default because measurement says it hurts badly.** Layer-objective joint gain on real
    Pythia-160M layers falls to -11.83% at W8 and -16.15% at W4, with one layer at -62.9%.

    The likely reason is that the criterion assumes the pruned weight's contribution is simply lost.
    Under error-compensating reconstruction it is not -- the survivors absorb it -- so "this weight
    quantises badly right now" is a poor predictor of whether removing it will hurt after the solve.
    A criterion consistent with reconstruction would need the inverse-Hessian term, not a diagonal
    approximation.

    Retained as an ablation, because "the obvious quantisation-aware criterion is worse than
    magnitude" is a reportable result rather than a dead end.
    """

    def reconstruction_passes(self, arm: str) -> int:
        """How many times ``arm`` calls the solver, per layer.

        The single source of truth for the fairness unit, kept next to the driver that makes the
        calls so the two cannot drift. With the sweep solver ``local_steps`` does not control any
        work -- a sweep is one deterministic pass -- so the meaningful count is *passes*, not steps.

        Args:
            arm: Arm name, as passed to :func:`compress_layer`.

        Returns:
            Solver calls per layer.

        Raises:
            LayerwiseError: If the arm is unknown.
        """
        passes = {
            "pruning": 1,
            "quantisation": 1,
            "sequential": 2,  # mask -> reconstruct, then quantise -> reconstruct
            "sequential_qp": 2,  # quantise -> reconstruct, then mask -> reconstruct
            "joint": self.joint_iterations,
        }
        if arm not in passes:
            raise LayerwiseError(f"unknown arm {arm!r}; expected one of {sorted(passes)}")
        return passes[arm]

    @property
    def prunes(self) -> bool:
        """Whether this plan removes any weights."""
        return self.sparsity > 0.0

    @property
    def quantises(self) -> bool:
        """Whether this plan reduces precision."""
        return self.bits is not None

    def budget_signature(self) -> tuple[Any, ...]:
        """The fields two arms must agree on to be a fair comparison (§3.11).

        Deliberately excludes ``joint_iterations``: only the joint arm has an outer loop, so
        requiring both arms to share it would be meaningless. What must match is the *total* local
        step count, which :class:`LayerwiseReport` records and
        :func:`assert_matched_plans` checks separately.
        """
        return (
            self.sparsity,
            self.bits,
            self.granularity,
            self.group_size,
            self.pruning_granularity,
            self.comparison_group,
            self.solver,
            self.damping,
            # The quantiser must be identical across arms, so the scale-fitting rule is part of the
            # budget rather than a per-arm choice.
            self.scale_search,
        )


@dataclass(slots=True)
class SolvedLayer:
    """One reconstruction result, with everything about it kept together.

    The four fields must describe the *same* model. Carrying them separately is how the joint arm came
    to accept a proposal on one weight and store a different one: acceptance compared a pre-canonical
    weight while conversion packed a post-canonical one, and re-deriving codes is not idempotent in
    floating point.

    Attributes:
        weight: The canonical weight, exactly ``codes x scales`` under the mask when quantised.
        codes: Integer codes, or ``None`` when the arm does not quantise.
        scales: The grid, or ``None`` when the arm does not quantise.
        loss: Objective against the dense weight, measured on :attr:`weight`.
    """

    weight: torch.Tensor
    codes: torch.Tensor | None
    scales: torch.Tensor | None
    loss: float


@dataclass(slots=True)
class LayerResult:
    """What happened to one layer."""

    name: str
    target_sparsity: float
    realised_sparsity: float
    """Fraction of weights that are numerically zero after compression.

    **Not the pruning budget.** Quantisation rounds small survivors to exactly zero, so this is
    strictly larger than :attr:`mask_sparsity` whenever a bit width is set -- measurably so at W4,
    where a 30% mask produced 31.4% numeric zeros on Pythia-410M. Verify the pruning budget against
    :attr:`mask_sparsity`, which is the quantity the target is defined on.
    """
    mask_sparsity: float = 0.0
    """Fraction the pruning mask explicitly removes. This is what the target constrains."""
    zero_code_fraction: float = 0.0
    """Fraction of *surviving* weights that quantisation rounded to the zero code.

    The gap between the mask budget and the realised zeros, attributed. A large value at low bit
    widths means the effective sparsity exceeds what was asked for, which changes the compression
    ratio and is worth reporting rather than absorbing.
    """
    naive_loss: float = 0.0
    final_loss: float = 0.0
    local_steps: int = 0
    num_weights: int = 0
    joint_trace: list[dict[str, Any]] = field(default_factory=list)
    """Per-outer-iteration record for the joint arm: objective before and after, whether the
    proposal was accepted, and how far the mask moved. Empty for every other arm."""

    @property
    def relative_improvement(self) -> float:
        """Objective reduction versus naive rounding."""
        if self.naive_loss <= 0:
            return 0.0
        return (self.naive_loss - self.final_loss) / self.naive_loss

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping (record field A9: per-layer reconstruction loss)."""
        return {
            "name": self.name,
            "target_sparsity": self.target_sparsity,
            # Three separate quantities, deliberately. `realised_sparsity` counts every zero,
            # including survivors that quantisation rounded away, so it overstates how much pruning
            # was applied -- by 1.4 percentage points at W4. The pruning budget is defined on
            # `mask_sparsity` and must be verified against that.
            "realised_sparsity": self.realised_sparsity,
            "mask_sparsity": self.mask_sparsity,
            "zero_code_fraction": self.zero_code_fraction,
            "naive_loss": self.naive_loss,
            "final_loss": self.final_loss,
            "relative_improvement": self.relative_improvement,
            "local_steps": self.local_steps,
            "num_weights": self.num_weights,
            "joint_trace": self.joint_trace,
        }


@dataclass(slots=True)
class LayerwiseReport:
    """Aggregate outcome of a layerwise compression pass.

    The fairness evidence lives here. ``calibration_fingerprint``, ``module_names`` and
    ``total_local_steps`` are exactly the three things §3.11 requires two arms to share, so a
    mismatch is visible in the run record rather than needing to be trusted.
    """

    arm: str
    layers: list[LayerResult] = field(default_factory=list)
    module_names: list[str] = field(default_factory=list)
    calibration_fingerprint: str = ""
    total_local_steps: int = 0
    masks_by_module: dict[str, torch.Tensor] = field(default_factory=dict)
    """True keep-masks per module, populated **only** when ``retain_masks`` is set.

    Off by default, and that is a memory decision as much as a behavioural one: one bool per weight
    is ~85 MB at 160M and ~800 MB at 1B, which the confirmatory grid has no use for. The post-hoc
    recovery ablation needs them because ``weight != 0`` cannot substitute -- quantisation rounds
    surviving weights to exactly zero (1.3% of them at W4), so freezing on the nonzero pattern
    would freeze more than the pruning budget, by a different amount in each arm.
    """

    grids_by_module: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, repr=False
    )
    """``(codes, scales)`` for each layer, keyed by module name.

    Carried through to conversion so packing stores these exact scales instead of refitting
    ``max|W|`` on the reconstructed weight -- refitting can pick a different grid and quietly ship a
    different model from the one that was measured. Excluded from ``repr`` and from
    :meth:`to_dict`: it is tensor data for the conversion step, not a record field.
    """

    @property
    def num_layers(self) -> int:
        """How many layers were compressed."""
        return len(self.layers)

    @property
    def targeted_parameters(self) -> int:
        """Total weights across every compressed layer -- the §2.6 scale x-axis."""
        return sum(layer.num_weights for layer in self.layers)

    def _weighted(self, attribute: str) -> float:
        """Parameter-weighted mean of a per-layer field.

        Weighted rather than a plain mean: layers differ in size by up to 4x, so an unweighted
        average would let a small attention projection count as much as a wide MLP one and the figure
        would not be the model's sparsity.
        """
        total = self.targeted_parameters
        if total == 0:
            return 0.0
        return sum(getattr(x, attribute) * x.num_weights for x in self.layers) / total

    @property
    def mask_sparsity(self) -> float:
        """Fraction the pruning masks remove -- the quantity the budget is defined on.

        Use this to verify the target, not :attr:`realised_sparsity`: that one also counts survivors
        quantisation rounded to zero, so it overstates the pruning applied.
        """
        return self._weighted("mask_sparsity")

    @property
    def zero_code_fraction(self) -> float:
        """Fraction of weights that survived the mask and were then rounded to zero."""
        return self._weighted("zero_code_fraction")

    @property
    def accepted_joint_updates(self) -> int:
        """Joint mask proposals that improved the objective and were kept."""
        return sum(1 for x in self.layers for step in x.joint_trace if step.get("accepted"))

    @property
    def rejected_joint_updates(self) -> int:
        """Joint mask proposals that made the objective worse and were discarded.

        A large count is the guard doing its job. A count of zero across every layer would mean the
        joint arm never revises its mask, which is §3.8's disqualifying case and worth noticing.
        """
        return sum(1 for x in self.layers for step in x.joint_trace if not step.get("accepted"))

    @property
    def realised_sparsity(self) -> float:
        """Weight-averaged realised sparsity across the compressed layers."""
        total = self.targeted_parameters
        if total == 0:
            return 0.0
        return sum(x.realised_sparsity * x.num_weights for x in self.layers) / total

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping for the run record."""
        return {
            "arm": self.arm,
            "num_layers": self.num_layers,
            "targeted_parameters": self.targeted_parameters,
            # Three separate sparsity figures, because a zero has two possible causes and only one of
            # them is the budget under study.
            "mask_sparsity": self.mask_sparsity,
            "zero_code_fraction": self.zero_code_fraction,
            "accepted_joint_updates": self.accepted_joint_updates,
            "rejected_joint_updates": self.rejected_joint_updates,
            "realised_sparsity": self.realised_sparsity,
            "total_local_steps": self.total_local_steps,
            "calibration_fingerprint": self.calibration_fingerprint,
            "module_names": list(self.module_names),
            "layers": [layer.to_dict() for layer in self.layers],
        }


@dataclass(slots=True)
class LayerOutcome:
    """One layer's compressed weight together with the mask that produced it.

    The mask is returned rather than only its effect on the weight because the two are not
    interchangeable: quantisation rounds small survivors to zero, so ``weight != 0`` is a strictly
    coarser signal than the mask. Distinguishing them matters for §3.8 -- "did the mask respond to
    the bit width?" cannot be answered from the nonzero pattern alone.

    Attributes:
        weight: The compressed weight.
        mask: The final keep-mask, ``True`` meaning kept.
        result: Per-layer statistics for the run record.
        local_steps: Reconstruction steps consumed, for the fairness accounting.
    """

    weight: torch.Tensor
    mask: torch.Tensor
    codes: torch.Tensor | None
    """The integer codes of the final weight, or ``None`` when the arm does not quantise.

    Carried so conversion packs *these* rather than re-deriving them. Re-quantising an already
    quantised value is **not** idempotent in floating point: a value sitting on a rounding boundary
    flips a code, which was observed as a 3.7e-03 deviation on the tiny model. Storing the codes
    removes the second rounding entirely rather than tolerating it.
    """
    scales: torch.Tensor | None
    """The quantisation grid the final solve used, or ``None`` when the arm does not quantise.

    Returned so conversion can pack these exact scales. Refitting ``max|W|`` on the reconstructed
    weight is not equivalent -- the sweep can move a row's maximum, so the refitted grid need not be
    the one the weights were solved onto, and packing would round a second time.
    """
    result: LayerResult
    local_steps: int


def _zero_code_fraction(compressed: torch.Tensor, mask: torch.Tensor) -> float:
    """Fraction of *mask survivors* that quantisation rounded to zero.

    Separating this from the mask sparsity is what makes the pruning budget checkable. A weight can be
    zero for two unrelated reasons -- the mask removed it, or rounding collapsed it -- and only the
    first is the budget the study controls. Conflating them overstates how much pruning was applied,
    by 1.4 percentage points at W4 on Pythia-410M.

    Args:
        compressed: The final weight.
        mask: The keep-mask, ``True`` meaning kept.

    Returns:
        Zeroed survivors as a fraction of the whole tensor, so it adds to the mask sparsity to give
        the realised numeric sparsity.
    """
    survivors = int(mask.sum())
    if survivors == 0:
        return 0.0
    rounded_away = int(((compressed == 0) & mask).sum())
    return rounded_away / mask.numel()


def compress_layer(
    weight: torch.Tensor,
    statistics: ActivationStatistics,
    plan: LayerPlan,
    *,
    arm: str,
) -> LayerOutcome:
    """Compress one layer without constructing an autograd graph.

    Sequential and joint compression both enter through this shared function. Their reconstruction
    path forms Gram products and factorises a Hessian-like matrix, none of which participates in
    training, so :func:`torch.no_grad` prevents graph retention over the whole path.

    The intra-op thread cap is a **deadlock mitigation** for OpenMP-backed linear algebra on Windows
    CPU builds, scoped to this call and restored on exit. It is deliberately not applied
    process-wide at an entry point: inter-op threads can only be set once per process, so an early
    global pin silently overrides the frozen ``benchmark`` thread configuration for the rest of the
    run, and latency then gets measured under conditions the run record does not describe. See
    :func:`~scale_aware_compression.hardware.cpu_thread_limit`.
    """
    import torch

    from scale_aware_compression.hardware import SOLVER_INTRA_OP_THREADS, cpu_thread_limit

    with torch.no_grad(), cpu_thread_limit(SOLVER_INTRA_OP_THREADS):
        return _compress_layer_impl(weight, statistics, plan, arm=arm)


def _compress_layer_impl(
    weight: torch.Tensor,
    statistics: ActivationStatistics,
    plan: LayerPlan,
    *,
    arm: str,
) -> LayerOutcome:
    """Compress one weight tensor according to ``arm``.

    Every arm is a different call order over the same three operations -- score, mask, reconstruct
    -- so the arms cannot diverge in anything except that order.

    Args:
        weight: The dense weight, shape ``(out_features, in_features)``. Not modified.
        statistics: Captured activation statistics for this layer's inputs.
        plan: Shared compression settings.
        arm: One of ``pruning``, ``quantisation``, ``sequential``, ``sequential_qp``, ``joint``.

    Returns:
        The :class:`LayerOutcome`.

    Raises:
        LayerwiseError: If ``arm`` is unknown.
    """
    import torch

    gram = statistics.gram()
    column_norms = statistics.column_norms()
    steps_used = 0
    joint_trace: list[dict[str, Any]] = []
    # The naive baseline of the FIRST solve and the objective of the LAST one. Taking the first
    # naive value is what makes the number comparable across arms: it is the cost of the whole
    # pipeline measured against not reconstructing at all, rather than against whatever
    # intermediate state a particular arm happened to reach.
    first_naive_loss: float | None = None
    last_final_loss = 0.0

    def fit_scales(target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor | None:
        """Fit the quantisation grid for the current mask. Identical procedure for every arm."""
        if not plan.quantises:
            return None
        if not plan.scale_search:
            return compute_symmetric_scales(
                target * mask,
                bits=plan.bits,
                granularity=plan.granularity,
                group_size=plan.group_size,
            )
        return search_clipping_scales(
            target,
            gram,
            bits=plan.bits,
            mask=mask,
            granularity=plan.granularity,
            group_size=plan.group_size,
        )

    def solve(
        mask: torch.Tensor,
        *,
        bits: int | None,
        scales: torch.Tensor | None = None,
        scale_source: torch.Tensor | None = None,
    ) -> SolvedLayer:
        """Reconstruct against the **dense** weight, under ``mask`` and optionally a bit width.

        The objective is always ``‖X·W_dense^T − X·Ŵ^T‖²_F`` (§3.1). It is deliberately not a
        parameter: an earlier version passed each arm's *intermediate* weight here, so sequential
        P->Q's second stage minimised distance to the pruned approximation and Q->P's minimised
        distance to the quantised one, while the joint arm minimised distance to dense. The arms were
        optimising three different objectives, and the direction disadvantaged the sequential arms --
        each treated its own intermediate as ground truth and so could not recover the error that
        intermediate already contained. That inflates joint gain, which is exactly the failure §3.11
        exists to prevent.

        What an arm *may* vary is where the quantisation grid comes from, which is a real pipeline
        difference rather than a different objective:

        * ``scales`` -- use this grid exactly, without refitting. Q->P needs it, because fitting once
          on the dense tensor and never revisiting it is what §3.8 says disqualifies an arm from
          being joint.
        * ``scale_source`` -- fit the grid on these weights. P->Q passes its pruned reconstruction, so
          quantisation is "calibrated on the post-pruning distribution" as the method specifies. The
          joint arm passes its current iterate, which is §3.8's re-estimation requirement.
        * neither -- fit on the dense weight under the mask.

        Args:
            mask: Keep-mask for this solve.
            bits: Bit width, or ``None`` for pruning-only reconstruction.
            scales: An exact grid to reuse.
            scale_source: Weights to fit the grid on.

        Returns:
            The canonicalised result: weight, integer codes, scales and objective, all mutually
            consistent.
        """
        nonlocal steps_used, first_naive_loss
        grid = scales
        if bits is not None and grid is None:
            grid = fit_scales(scale_source if scale_source is not None else weight, mask)

        outcome = reconstruct(
            gram,
            weight,
            mask,
            solver=plan.solver,
            local_steps=plan.local_steps,
            damping=plan.damping,
            bits=bits,
            granularity=plan.granularity,
            group_size=plan.group_size,
            block_size=plan.block_size,
            scales=grid,
        )
        steps_used += max(1, plan.local_steps)
        if first_naive_loss is None:
            first_naive_loss = outcome.naive_loss

        # Canonicalise HERE, before the objective is measured, so the number used for acceptance is
        # the number describing the model that actually gets stored and packed. Canonicalising after
        # the joint loop meant a proposal was accepted on one weight and a different one was saved:
        # re-deriving codes is not idempotent in floating point, so a value on a rounding boundary
        # flips between the two.
        canonical = outcome.weight
        codes = None
        if bits is not None and grid is not None:
            quantised = quantise_weight(
                canonical,
                bits=bits,
                granularity=plan.granularity,
                group_size=plan.group_size,
                scales=grid,
            )
            codes = quantised.codes
            canonical = quantised.dequantise() * mask

        return SolvedLayer(
            weight=canonical,
            codes=codes,
            scales=grid,
            loss=reconstruction_loss(gram, weight, canonical),
        )

    def mask_from(scored: torch.Tensor) -> torch.Tensor:
        """Mask from activation-weighted magnitude -- the criterion with no quantiser."""
        if not plan.prunes:
            return torch.ones_like(scored, dtype=torch.bool)
        return build_mask_from_scores(
            activation_weighted_saliency(scored, column_norms),
            sparsity=plan.sparsity,
            granularity=plan.pruning_granularity,
            comparison_group=plan.comparison_group,
        )

    def quantisation_aware_mask(current: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask from keep-versus-prune benefit, which needs a fitted quantiser.

        This is the mechanism §3.8 asks for. Falls back to quantised-magnitude scoring when
        ``keep_benefit_saliency`` is off, so the weaker criterion stays available as an ablation.
        """
        if not plan.prunes or not plan.quantises:
            return mask_from(current)

        scales = fit_scales(current, mask)
        quantised = fake_quantise(
            current * mask,
            bits=plan.bits,
            granularity=plan.granularity,
            group_size=plan.group_size,
            scales=scales,
        )
        if not plan.keep_benefit_saliency:
            return mask_from(quantised)
        return build_mask_from_scores(
            keep_benefit_saliency(weight, quantised, column_norms),
            sparsity=plan.sparsity,
            granularity=plan.pruning_granularity,
            comparison_group=plan.comparison_group,
        )

    keep_all = torch.ones_like(weight, dtype=torch.bool)

    if arm == "pruning":
        mask = mask_from(weight)
        solved = solve(mask, bits=None)

    elif arm == "quantisation":
        mask = keep_all
        solved = solve(mask, bits=plan.bits)

    elif arm == "sequential":
        # P->Q. The mask is chosen on the dense weights because at this point no quantiser exists --
        # that ignorance is what makes the arm sequential (§3.5).
        #
        # Stage 2 reconstructs against the DENSE weight, not against stage 1's output, so its
        # objective matches every other arm's. What stage 1 contributes is the distribution the
        # quantisation grid is fitted to: "quantisation is calibrated on the altered post-pruning
        # distribution", which is the pipeline property the arm is meant to have.
        mask = mask_from(weight)
        pruned = solve(mask, bits=None)
        solved = solve(mask, bits=plan.bits, scale_source=pruned.weight)

    elif arm == "sequential_qp":
        # Q->P reverse ablation (§3.6). The quantiser is fitted first, so the mask does see the grid.
        # The grid is then reused **exactly** -- passed as `scales` rather than refitted -- because
        # not revisiting the scales after the mask moves is precisely what §3.8 says disqualifies an
        # arm from being joint. Refitting here would quietly turn this into a second joint arm.
        quantised = solve(keep_all, bits=plan.bits)
        mask = quantisation_aware_mask(quantised.weight, keep_all)
        solved = solve(mask, bits=plan.bits, scales=quantised.scales)

    elif arm == "joint":
        # §3.7's alternation. Each iteration re-fits the grid for the current mask, rescores the
        # mask against that grid, and reconstructs -- so mask and quantiser each inform the other.
        #
        # The incumbent guard below is load-bearing. The solver's own accept-only-if-better rule
        # protects a *fixed* mask; nothing protected the outer loop across mask *changes*, so an
        # iteration that chose a worse mask replaced a better feasible solution the loop had already
        # found. That is measurable: at 30% + W4 on Pythia-160M, four alternations scored worse than
        # two (73.17 against 71.87 perplexity), which is the signature of a wandering search rather
        # than a converging one.
        #
        # This does not tilt the comparison towards joint. It stops an alternating procedure
        # discarding a solution it had already reached, which is a property of the optimiser, not an
        # extra optimisation budget -- the step count is unchanged.
        mask = mask_from(weight)
        incumbent = solve(mask, bits=plan.bits)
        best_mask = mask

        for iteration in range(1, plan.joint_iterations):
            proposed_mask = quantisation_aware_mask(incumbent.weight, best_mask)
            # §3.8's re-estimation: the grid is refitted on the current iterate for the new mask.
            proposed = solve(proposed_mask, bits=plan.bits, scale_source=incumbent.weight)
            accepted = proposed.loss < incumbent.loss

            joint_trace.append(
                {
                    "iteration": iteration,
                    "loss_before": incumbent.loss,
                    "loss_proposed": proposed.loss,
                    "accepted": accepted,
                    "mask_divergence": float((proposed_mask != best_mask).float().mean()),
                }
            )
            # The whole SolvedLayer moves together -- weight, codes, scales and loss. Keeping them
            # separate is how the accepted candidate and the stored artefact came to disagree.
            if accepted:
                incumbent, best_mask = proposed, proposed_mask

        mask, solved = best_mask, incumbent

    else:
        raise LayerwiseError(
            f"unknown arm {arm!r}; expected one of pruning, quantisation, sequential, "
            "sequential_qp, joint"
        )

    # Already canonical: solve() quantises before it measures, so the weight, codes, scales and loss
    # in `solved` all describe one model. Nothing is re-derived here.
    compressed = solved.weight
    final_codes = solved.codes
    final_scales = solved.scales
    last_final_loss = solved.loss

    result = LayerResult(
        name="",
        target_sparsity=plan.sparsity,
        realised_sparsity=realised_sparsity(compressed != 0),
        mask_sparsity=realised_sparsity(mask),
        zero_code_fraction=_zero_code_fraction(compressed, mask),
        naive_loss=first_naive_loss or 0.0,
        final_loss=last_final_loss,
        joint_trace=joint_trace,
        local_steps=steps_used,
        num_weights=weight.numel(),
    )
    return LayerOutcome(
        weight=compressed,
        mask=mask,
        codes=final_codes,
        scales=final_scales,
        result=result,
        local_steps=steps_used,
    )


def assert_arms_can_be_matched(plan: LayerPlan, arms: Sequence[str]) -> None:
    """Check *before running* that the arms in a grid will consume equal solver budgets.

    :func:`assert_matched_plans` catches a mismatch after the fact, from the records. This catches it
    from the configuration, which is the only useful moment: a sweep is hours of compute, and §3.11's
    critical fairness point is that a score obtained with more optimisation cannot be attributed to
    the method.

    The default ``joint_iterations = 4`` against sequential's two passes is exactly this failure, and
    it went unnoticed through a whole screening grid because nothing called either assertion during a
    real run.

    Args:
        plan: The shared plan every arm will use.
        arms: Arm names that will run against each other.

    Raises:
        LayerwiseError: If two arms that both prune *and* quantise would get different budgets.
    """
    # Only arms doing both techniques are compared for joint gain. A pruning-only arm legitimately
    # does one pass, and comparing its budget against a two-stage pipeline is meaningless.
    comparable = [arm for arm in arms if arm in {"sequential", "sequential_qp", "joint"}]
    if len(comparable) < 2:
        return

    budgets = {arm: plan.reconstruction_passes(arm) for arm in comparable}
    if len(set(budgets.values())) > 1:
        detail = ", ".join(f"{arm}={count}" for arm, count in sorted(budgets.items()))
        raise LayerwiseError(
            f"arms would receive unequal solver budgets ({detail}), violating §3.11. With the sweep "
            "solver each call is one deterministic pass, so passes are the fairness unit. Set "
            f"compression.reconstruction.joint_iterations to "
            f"{plan.reconstruction_passes('sequential')} to match the sequential pipeline's two "
            "stages, or give the sequential arm the same budget explicitly (§3.5 step 6)."
        )


def assert_matched_plans(
    reports: Sequence[LayerwiseReport],
    plans: Sequence[LayerPlan],
) -> None:
    """Check two or more arms were run under conditions §3.11 calls comparable.

    This is the executable form of the plan's critical fairness point: if one arm received more
    local steps, different calibration data, or a different module list, a difference in its score
    cannot be attributed to the pipeline.

    Args:
        reports: One report per arm.
        plans: The plan each arm ran under, in the same order.

    Raises:
        LayerwiseError: On any mismatch, naming which invariant broke.
    """
    if len(reports) != len(plans):
        raise LayerwiseError(
            f"got {len(reports)} report(s) and {len(plans)} plan(s); they must correspond"
        )
    if len(reports) < 2:
        return

    reference_report, reference_plan = reports[0], plans[0]
    for report, plan in zip(reports[1:], plans[1:], strict=True):
        if plan.budget_signature() != reference_plan.budget_signature():
            raise LayerwiseError(
                f"{report.arm!r} and {reference_report.arm!r} ran under different compression "
                "budgets, so their difference does not measure the pipeline"
            )
        if report.module_names != reference_report.module_names:
            raise LayerwiseError(
                f"{report.arm!r} and {reference_report.arm!r} touched different modules; a "
                "coverage difference would masquerade as a method difference"
            )
        if report.calibration_fingerprint != reference_report.calibration_fingerprint:
            raise LayerwiseError(
                f"{report.arm!r} and {reference_report.arm!r} used different calibration data "
                f"({report.calibration_fingerprint!r} vs "
                f"{reference_report.calibration_fingerprint!r})"
            )
        if report.total_local_steps != reference_report.total_local_steps:
            raise LayerwiseError(
                f"{report.arm!r} consumed {report.total_local_steps} local step(s) but "
                f"{reference_report.arm!r} consumed {reference_report.total_local_steps}; extra "
                "optimisation must not be mistaken for a method advantage"
            )


def compress_model_layerwise(
    model: nn.Module,
    calibration_batches: Iterable[torch.Tensor],
    plan: LayerPlan,
    *,
    arm: str,
    module_names: Sequence[str] | None = None,
    calibration_fingerprint: str = "",
    device: torch.device | str | None = None,
    offload_blocks: bool = False,
    retain_masks: bool = False,
    progress: Callable[[str, int, int], None] | None = None,
) -> LayerwiseReport:
    """Compress every targeted linear layer, block by block, in depth order.

    Activations are captured **through the compressed prefix**: after a block is compressed, the
    calibration batches are pushed through it in its compressed state, so the next block sees the
    inputs it will really receive. This is what makes accumulated error realistic.

    Args:
        model: A loaded decoder-only model. Modified in place.
        calibration_batches: Token-id tensors of shape ``(batch, sequence)``. Materialised once and
            reused for every block, so every layer and every arm sees identical data (§3.11).
        plan: Shared compression settings.
        arm: Which pipeline to run; see :func:`compress_layer`.
        module_names: Restrict to these modules. Defaults to the full selection.
        calibration_fingerprint: Recorded so a mismatch between arms is detectable after the fact.
        device: Device to run capture and compression on. GPU is allowed here -- §4.6 restricts
            only the final deployment measurements to CPU. Required when ``offload_blocks`` is set.
        offload_blocks: Hold one decoder block on ``device`` at a time, keeping the rest of the
            model on the host. Costs two host/device transfers per block and changes no number;
            it is what makes a model larger than the card runnable. **The model is left on the
            host when this returns**, because that is where all but one block already was.
        progress: Optional ``(module_name, index, total)`` callback.

        retain_masks: Keep each module's true keep-mask on the report. Off by default; the
            post-hoc recovery ablation needs it and nothing else does.

    Returns:
        The report, including a per-layer reconstruction loss.

    Raises:
        LayerwiseError: If nothing was selected, a targeted module is not 2-D, or offload was
            requested without a device to offload to.
    """
    import torch

    from scale_aware_compression.models.adapters import (
        get_adapter,
        get_decoder_blocks,
        get_linear_modules,
        select_compressible_modules,
    )

    if module_names is None:
        module_names = select_compressible_modules(model).names
    module_names = list(module_names)
    if not module_names:
        raise LayerwiseError("no modules selected; refusing to return the dense model unchanged")

    modules = get_linear_modules(model, module_names)
    blocks = get_decoder_blocks(model)
    # Materialised once: every block and every arm must see byte-identical calibration data, and a
    # one-shot generator would silently give the second block nothing.
    batches = list(calibration_batches)
    if not batches:
        raise LayerwiseError("calibration set is empty; reconstruction needs activations")

    report = LayerwiseReport(
        arm=arm,
        module_names=module_names,
        calibration_fingerprint=calibration_fingerprint,
    )

    # Group the targeted modules by the block they live in, then within a block by dependency order.
    by_block: list[list[str]] = [[] for _ in blocks]
    block_prefixes = [f"{name}." for name, _ in _named_blocks(model, blocks)]
    for name in module_names:
        for index, prefix in enumerate(block_prefixes):
            if name.startswith(prefix):
                by_block[index].append(name)
                break
        else:
            raise LayerwiseError(f"{name!r} does not belong to any decoder block")

    # Capture once per DEPENDENCY GROUP, not once per block. A module whose input is produced by
    # another module in the same block has to be fitted against the inputs it will really see -- that
    # is, after that producer is compressed. Capturing once per block fits an MLP down-projection
    # against activations the *dense* up-projection produced, inputs that never occur at inference.
    # That is blockwise reconstruction described as layerwise.
    #
    # Cost is one extra forward pass over the calibration set per group beyond the first: two passes
    # per block for Pythia's parallel residual, four for Qwen2's sequential one.
    groups = get_adapter(model).grouped_suffixes()
    total = len(module_names)
    state = {"completed": 0}

    host_device = torch.device("cpu")
    compute_device = torch.device(device) if device is not None else None
    if offload_blocks:
        if compute_device is None or compute_device.type == "cpu":
            raise LayerwiseError(
                "offload_blocks=True needs a non-CPU device to offload to; got "
                f"{device!r}. Offloading blocks to the host from the host moves nothing."
            )
        # Everything but the resident block lives on the host. Moving the model here rather than
        # requiring the caller to load it on CPU means the flag is sufficient on its own -- but a
        # caller that already loaded onto the card has paid a full-model transient, so the configs
        # that set this also set `model.device: cpu`.
        if next(model.parameters()).device != host_device:
            LOGGER.info(
                "offload_blocks: moving the model to %s; blocks go to %s one at a time",
                host_device,
                compute_device,
            )
            model.to(host_device)

    # Where the Gram and the solve happen. Under offload that is the device the resident block is
    # on, which is not where the model as a whole lives.
    capture_device = str(compute_device) if offload_blocks else device

    # A live key/value cache would accumulate across the repeated single-block replays below, changing
    # the activations each group is fitted against. B-28 is this exact fault in the external SparseGPT
    # driver, found there first.
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = False

    try:
        # The embedding and everything before block 0 run exactly once. From here every capture is a
        # replay of a single block over cached hidden states.
        #
        # Under offload this capture happens ON THE DEVICE, with only the modules outside the
        # decoder blocks moved there. Two rejected alternatives, in the order they were tried:
        #
        # Capturing on the HOST is wrong. Aborting at block 0 means only the embedding runs, and an
        # embedding lookup is a gather -- but GPT-NeoX also computes the rotary cos/sin in that
        # forward and passes them into every block as replay context, and CPU and CUDA trigonometry
        # disagree in the last bits. That flips near-ties in the saliency ranking, and a flipped
        # mask position is not a small numerical difference: on real Pythia-160M it moved
        # `attention.dense` in block 0 by 2.25 absolute, cascaded through every later block, and
        # more than doubled the measured joint gain (B-34).
        #
        # Moving the WHOLE model for the capture is correct but does not fit. No Gram factorisation
        # is live yet, so it looked affordable -- but 3.77 GiB of 1B weights plus ~0.5 GiB of cached
        # hidden states plus the forward's own activations exceeded the card, and 1B died at the one
        # step offload exists to make possible. Only the pre-block modules are needed, and they are
        # ~0.8 GiB.
        if offload_blocks:
            _move_outside_blocks(model, blocks, compute_device)
        try:
            cached = _capture_block_inputs(
                model, blocks[0], batches, device=compute_device if offload_blocks else None
            )
        finally:
            if offload_blocks:
                _move_outside_blocks(model, blocks, host_device)

        for block_index, names in enumerate(by_block):
            block = blocks[block_index]
            if offload_blocks:
                block.to(compute_device)
            try:
                remaining = list(names)
                for group in groups:
                    in_group = [name for name in remaining if name.endswith(tuple(group))]
                    if not in_group:
                        continue
                    _compress_group(
                        block,
                        modules,
                        in_group,
                        cached=cached,
                        plan=plan,
                        arm=arm,
                        report=report,
                        device=capture_device,
                        block_index=block_index,
                        total=total,
                        state=state,
                        retain_masks=retain_masks,
                        progress=progress,
                    )
                    remaining = [name for name in remaining if name not in set(in_group)]
                if remaining:
                    # A targeted module the adapter's groups do not mention. Compressing it silently
                    # against stale activations is the failure this whole change exists to remove, so
                    # refuse instead.
                    raise LayerwiseError(
                        f"block {block_index}: {remaining} are targeted but appear in no dependency "
                        f"group for {type(model).__name__}. Add them to the adapter's "
                        "dependency_groups, or they would be fitted against activations captured "
                        "before their inputs were compressed."
                    )

                # Advance the cache through this block in its now-compressed state, so the next block
                # sees the inputs it will really receive. This runs even when the block had NO
                # targeted modules: skipping it would leave the next block replaying inputs from the
                # wrong depth, which is a silent correctness failure rather than an error. Skipped
                # after the last block, whose output nothing reads. Must happen while the block is
                # still resident, because it is a forward pass.
                if block_index + 1 < len(blocks):
                    cached = _advance_cache(block, cached)
            finally:
                # In a `finally` so a raise mid-block does not leave weights stranded on the card,
                # which would make the error message the second problem rather than the first.
                if offload_blocks:
                    block.to(host_device)
                    # The recorded grids have to travel with the block. They are captured while it
                    # is resident, so they come back as CUDA tensors while the weights they describe
                    # are on the host -- and `convert` then packs cuda codes against a cpu weight
                    # and dies at the artefact stage, after all the compute is spent. The invariant
                    # is that a report's grids live wherever its weights live.
                    for name in names:
                        grid = report.grids_by_module.get(name)
                        if grid is not None:
                            report.grids_by_module[name] = (
                                grid[0].to(host_device),
                                grid[1].to(host_device),
                            )
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
        if offload_blocks:
            # The embedding, the final norm and the head were left on the device by the capture.
            # Nothing after this reads them there, and at 1B they are ~0.8 GiB that the caller's
            # next stage -- packing, then CPU evaluation -- has no use for.
            model.to(host_device)

    return report


def _compress_group(  # noqa: PLR0913 - one call site; grouping the arguments would hide the flow
    block: nn.Module,
    modules: dict[str, nn.Module],
    names: list[str],
    *,
    cached: list[tuple[torch.Tensor, dict[str, Any]]],
    plan: LayerPlan,
    arm: str,
    report: LayerwiseReport,
    device: str | None,
    block_index: int,
    total: int,
    state: dict[str, int],
    retain_masks: bool,
    progress: Callable[[str, int, int], None] | None,
) -> None:
    """Capture activations for one dependency group, then compress every module in it.

    The modules in a group read tensors none of the others produce, so a single capture serves all of
    them. Groups are compressed in order, and each group's capture happens *after* the previous one
    has been written back -- which is what makes the reconstruction genuinely layerwise rather than
    blockwise.

    Args:
        block: The decoder block these modules live in. Only this block is run -- blocks after it
            cannot influence its inputs, and blocks before it are already folded into ``cached``.
        modules: Resolved modules by name.
        names: The group's module names.
        cached: ``(hidden_states, kwargs)`` entering this block, one pair per calibration batch.
            Byte-identical across groups and arms, which is what §3.11 requires.
        plan: Shared compression settings.
        arm: Which pipeline to run.
        report: Accumulates per-layer results, grids and the step total.
        device: Device for capture and compression.
        block_index: Block ordinal, for logging.
        total: Total modules in the run, for progress.
        state: Mutable completion counter shared across groups.
        progress: Optional progress callback.
        retain_masks: Store each module's keep-mask on the report.

    Raises:
        LayerwiseError: If a weight is not 2-D.
    """
    import torch

    captures = {
        name: ActivationStatistics(
            modules[name].in_features,
            dtype=torch.float32,
            device=device or modules[name].weight.device,
        )
        for name in names
    }
    handles = [
        modules[name].register_forward_pre_hook(_make_hook(captures[name])) for name in names
    ]
    try:
        # Only this block, replayed over the cached inputs. The previous implementation ran the whole
        # model here, once per dependency group: blocks after this one had their outputs discarded,
        # and blocks before it were recomputed from the embedding every time. That is
        # O(blocks x groups) full-model forwards where O(blocks) single-block forwards suffice, and it
        # forced the entire model to stay resident on the capture device -- which is what made
        # Pythia-1B exceed a 6 GiB card. Proven bit-identical by
        # scripts/verify_block_sequential_capture.py; see docs/capture_refactor_rationale.md.
        with torch.no_grad():
            for hidden, forwarded in cached:
                block(hidden, **forwarded)
    finally:
        for handle in handles:
            handle.remove()

    for name in names:
        module = modules[name]
        weight = module.weight
        if weight.ndim != 2:
            raise LayerwiseError(f"{name!r} weight is {weight.ndim}-D; expected 2-D")

        outcome = compress_layer(
            weight.detach().to(torch.float32),
            captures[name],
            plan,
            arm=arm,
        )
        result = outcome.result
        result.name = name
        with torch.no_grad():
            weight.copy_(outcome.weight.to(weight.dtype))

        report.layers.append(result)
        if retain_masks:
            # The TRUE keep-mask, not the nonzero pattern. See LayerwiseReport.masks_by_module.
            report.masks_by_module[name] = outcome.mask.detach().clone()
        report.total_local_steps += outcome.local_steps
        if outcome.scales is not None and outcome.codes is not None:
            report.grids_by_module[name] = (
                outcome.codes.detach().clone(),
                outcome.scales.detach().clone(),
            )

        state["completed"] += 1
        if progress is not None:
            progress(name, state["completed"], total)
        LOGGER.debug(
            "block %d: compressed %s to %.4f mask sparsity",
            block_index,
            name,
            result.mask_sparsity,
        )


class _StopForwardError(Exception):
    """Raised to abort a forward pass once the first block's inputs have been captured."""


def _move_outside_blocks(
    model: nn.Module,
    blocks: list[nn.Module],
    device: torch.device,
) -> None:
    """Move every parameter and buffer that is *not* inside a decoder block onto ``device``.

    Used by block offload to run the block-0 capture on the compute device without making the whole
    model resident. What the capture needs is the embedding and the rotary tables; what it must not
    drag along is 3.77 GiB of decoder weights, which at Pythia-1B is the difference between fitting
    and an out-of-memory at the one step offload exists to make possible.

    Parameters are moved through ``.data`` and buffers through ``setattr`` so the owning module's
    registration is updated. Both are done with ``recurse=False`` per module, because moving a
    parent would take its block children with it.

    Args:
        model: The model to walk.
        blocks: The decoder blocks to leave where they are.
        device: Destination for everything else.
    """
    inside = {id(module) for block in blocks for module in block.modules()}
    for module in model.modules():
        if id(module) in inside:
            continue
        for parameter in module.parameters(recurse=False):
            parameter.data = parameter.data.to(device)
        for name, buffer in list(module.named_buffers(recurse=False)):
            setattr(module, name, buffer.to(device))


def _capture_block_inputs(
    model: nn.Module,
    first_block: nn.Module,
    batches: list[torch.Tensor],
    *,
    device: torch.device | None = None,
) -> list[tuple[torch.Tensor, dict[str, Any]]]:
    """Record the hidden states and replay context entering the first decoder block.

    Everything downstream replays from here, so this is the only time the embedding and whatever
    precedes block 0 are executed.

    The non-hidden keyword arguments -- attention mask, position ids, rotary embeddings -- are kept
    verbatim and passed back on every replay. A block re-run with a different mask sees different
    activations, which would silently change what the solver is fitted to.

    Args:
        model: The model, run once per batch and aborted at block 0.
        first_block: The block whose inputs are wanted.
        batches: Calibration token-id tensors.
        device: Where to place the input ids. Defaults to wherever the first parameter lives, which
            is right for a uniformly placed model and wrong under offload, where the blocks are
            deliberately somewhere else.

    Returns:
        One ``(hidden_states, kwargs)`` pair per batch, in batch order.

    Raises:
        LayerwiseError: If nothing was captured, which means the block signature or call convention
            has changed and every downstream solve would be fitted against nothing.
    """
    import torch

    captured: list[tuple[torch.Tensor, dict[str, Any]]] = []

    def catcher(_module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        hidden = args[0] if args else kwargs.get("hidden_states")
        if hidden is None:
            raise _StopForwardError
        forwarded = {key: value for key, value in kwargs.items() if key != "hidden_states"}
        captured.append((hidden.detach().clone(), forwarded))
        raise _StopForwardError

    handle = first_block.register_forward_pre_hook(catcher, with_kwargs=True)
    if device is None:
        device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for batch in batches:
                with contextlib.suppress(_StopForwardError):
                    model(batch.to(device))
    finally:
        handle.remove()

    if len(captured) != len(batches):
        raise LayerwiseError(
            f"captured {len(captured)} block-0 input(s) for {len(batches)} calibration batch(es). "
            "The decoder block's call convention has changed, and every layer would otherwise be "
            "fitted against missing activations."
        )
    return captured


def _advance_cache(
    block: nn.Module,
    cached: list[tuple[torch.Tensor, dict[str, Any]]],
) -> list[tuple[torch.Tensor, dict[str, Any]]]:
    """Push the cached hidden states through ``block`` in its current (compressed) state.

    Called after every block, including blocks with no targeted modules -- skipping one would leave
    the next block replaying inputs from the wrong depth, which is a silent correctness failure rather
    than an error.

    Args:
        block: The block to run.
        cached: Current ``(hidden_states, kwargs)`` pairs.

    Returns:
        The pairs advanced by one block, keeping each batch's replay context unchanged.
    """
    import torch

    advanced: list[tuple[torch.Tensor, dict[str, Any]]] = []
    with torch.no_grad():
        for hidden, kwargs in cached:
            output = block(hidden, **kwargs)
            if isinstance(output, tuple):
                output = output[0]
            advanced.append((output.detach(), kwargs))
    return advanced


def _named_blocks(model: nn.Module, blocks: list[nn.Module]) -> list[tuple[str, nn.Module]]:
    """Pair each decoder block with its dotted name, so module names can be matched to blocks."""
    lookup = {id(block): block for block in blocks}
    # named_modules walks in registration order, which for a ModuleList is depth order.
    return [(name, module) for name, module in model.named_modules() if id(module) in lookup]


def _make_hook(statistics: ActivationStatistics) -> Callable[..., None]:
    """Build a forward pre-hook that folds a layer's inputs into ``statistics``."""

    def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
        import torch

        if not inputs:
            return
        with torch.no_grad():
            statistics.update(inputs[0].detach())

    return hook
