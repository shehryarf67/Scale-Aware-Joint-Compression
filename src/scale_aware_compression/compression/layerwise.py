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

Blocks are visited in depth order and activations are captured **through the already-compressed
prefix**, so each layer is fitted against the inputs it will really see at inference rather than the
inputs the dense model would have produced. Compressing out of order, or capturing against the dense
model throughout, would understate the accumulated error.
"""

from __future__ import annotations

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
    search_clipping_scales,
)
from scale_aware_compression.compression.reconstruct import reconstruct
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
class LayerResult:
    """What happened to one layer."""

    name: str
    target_sparsity: float
    realised_sparsity: float
    naive_loss: float
    final_loss: float
    local_steps: int
    num_weights: int

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
            "realised_sparsity": self.realised_sparsity,
            "naive_loss": self.naive_loss,
            "final_loss": self.final_loss,
            "relative_improvement": self.relative_improvement,
            "local_steps": self.local_steps,
            "num_weights": self.num_weights,
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

    @property
    def num_layers(self) -> int:
        """How many layers were compressed."""
        return len(self.layers)

    @property
    def targeted_parameters(self) -> int:
        """Total weights across every compressed layer -- the §2.6 scale x-axis."""
        return sum(layer.num_weights for layer in self.layers)

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
    result: LayerResult
    local_steps: int


def compress_layer(
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
        target: torch.Tensor,
        mask: torch.Tensor,
        *,
        bits: int | None,
    ) -> torch.Tensor:
        nonlocal steps_used, first_naive_loss, last_final_loss
        outcome = reconstruct(
            gram,
            target,
            mask,
            solver=plan.solver,
            local_steps=plan.local_steps,
            damping=plan.damping,
            bits=bits,
            granularity=plan.granularity,
            group_size=plan.group_size,
            block_size=plan.block_size,
            scales=fit_scales(target, mask) if bits is not None else None,
        )
        steps_used += max(1, plan.local_steps)
        if first_naive_loss is None:
            first_naive_loss = outcome.naive_loss
        last_final_loss = outcome.final_loss
        return outcome.weight

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
        compressed = solve(weight, mask, bits=None)

    elif arm == "quantisation":
        mask = keep_all
        compressed = solve(weight, mask, bits=plan.bits)

    elif arm == "sequential":
        # P->Q. The mask is chosen on the dense weights because at this point no quantiser exists
        # -- that ignorance is what makes the arm sequential (§3.5).
        mask = mask_from(weight)
        pruned = solve(weight, mask, bits=None)
        compressed = solve(pruned, mask, bits=plan.bits)

    elif arm == "sequential_qp":
        # Q->P reverse ablation (§3.6). The quantiser is fitted first, so the mask does see the
        # grid -- but the scales are fitted once against the dense tensor and never revisited after
        # the mask moves, which is what §3.8 lists as disqualifying it from being joint.
        quantised = solve(weight, keep_all, bits=plan.bits)
        mask = quantisation_aware_mask(quantised, keep_all)
        compressed = solve(quantised, mask, bits=plan.bits)

    elif arm == "joint":
        # §3.7's alternation. Each iteration re-fits the grid for the current mask, rescores the
        # mask against that grid, and reconstructs -- so mask and quantiser each inform the other.
        mask = mask_from(weight)
        current = weight
        for _ in range(plan.joint_iterations):
            mask = quantisation_aware_mask(current, mask)
            # Always solve against the ORIGINAL dense weight. Letting the target drift with the
            # iterate would optimise towards the previous approximation instead of the true layer
            # output, and the objective would stop being comparable between arms.
            current = solve(weight, mask, bits=plan.bits)
        compressed = current

    else:
        raise LayerwiseError(
            f"unknown arm {arm!r}; expected one of pruning, quantisation, sequential, "
            "sequential_qp, joint"
        )

    result = LayerResult(
        name="",
        target_sparsity=plan.sparsity,
        realised_sparsity=realised_sparsity(compressed != 0),
        naive_loss=first_naive_loss or 0.0,
        final_loss=last_final_loss,
        local_steps=steps_used,
        num_weights=weight.numel(),
    )
    return LayerOutcome(weight=compressed, mask=mask, result=result, local_steps=steps_used)


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
            only the final deployment measurements to CPU.
        progress: Optional ``(module_name, index, total)`` callback.

    Returns:
        The report, including a per-layer reconstruction loss.

    Raises:
        LayerwiseError: If nothing was selected, or a targeted module is not 2-D.
    """
    import torch

    from scale_aware_compression.models.adapters import (
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

    # Group the targeted modules by which block they live in, so a block's activations are captured
    # once and used for all of its layers.
    by_block: list[list[str]] = [[] for _ in blocks]
    block_prefixes = [f"{name}." for name, _ in _named_blocks(model, blocks)]
    for name in module_names:
        for index, prefix in enumerate(block_prefixes):
            if name.startswith(prefix):
                by_block[index].append(name)
                break
        else:
            raise LayerwiseError(f"{name!r} does not belong to any decoder block")

    total = len(module_names)
    completed = 0

    for block_index, names in enumerate(by_block):
        if not names:
            continue
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
            with torch.inference_mode():
                for batch in batches:
                    model(batch.to(next(model.parameters()).device))
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
            report.total_local_steps += outcome.local_steps
            completed += 1
            if progress is not None:
                progress(name, completed, total)
            LOGGER.debug(
                "block %d: compressed %s to %.4f sparsity",
                block_index,
                name,
                result.realised_sparsity,
            )

    return report


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
