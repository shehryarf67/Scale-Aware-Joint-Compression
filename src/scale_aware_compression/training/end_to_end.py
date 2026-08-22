r"""End-to-end recovery for the post-hoc ablation: frozen masks, live W4 fake quantisation.

WHY THIS EXISTS
---------------
The layerwise objective improves more clearly than final perplexity does (F-38: at 1B the local
mask divergence, layer gain and layer-objective advantage are all undiminished while the end-to-end
gain collapses to +0.13 pp). That gap is what this module probes: does the joint solution contain
structure the *local* reconstruction objective cannot translate into language-model quality?

Both arms receive the *same* short global recovery phase. This is deliberately not a way to make
joint win -- if the sequential arm catches up, that is the finding.

WHAT IS FROZEN AND WHAT IS TRAINED
----------------------------------
* **The pruning mask is frozen.** Applied inside every forward pass, so a pruned position
  contributes nothing and receives exactly zero gradient (``d(w*m)/dw = m = 0``). No regrowth, no
  reselection, no sparsity drift.
* **W4 fake quantisation stays live.** The forward pass sees grid values through the repository's
  own ``fake_quantise``, with a straight-through estimator so gradients reach the shadow weight.
  This is not FP32 fine-tuning wearing a hat: the model learns under the constraint it is evaluated
  under.
* **The shadow weight is trained.** Full precision, off-grid between steps, re-snapped every
  forward.

ORDER OF OPERATIONS, AND WHY
----------------------------
``fake_quantise(weight * mask)`` -- mask first, then quantise. The deployed artefact is a masked,
quantised weight and its scales are fitted to what survives; quantising first would fit the grid to
values about to be discarded. The two differ only when pruning removes a row's largest weight, which
is rare but real.

SCALES ARE REFITTED EVERY FORWARD, NOT FROZEN
---------------------------------------------
The final artefact's scales are fitted from the final weights, so freezing stale scales during
recovery would optimise against a grid the evaluated model does not use. Refitting keeps the
training-time and evaluation-time constraints identical.

A DEVIATION FROM THE OLDER SCAFFOLD, STATED PLAINLY
---------------------------------------------------
``training/recovery.py``'s ``build_recovery_callbacks`` TODO says to add fake quantisation "only for
the joint arm -- adding it to the sequential arm would turn it into a third method". That was
written for the **superseded** full-model QAT design, where quantisation-aware training *was* the
joint method. Under the current layerwise PTQ design both arms are already quantised before recovery
begins, so applying fake quantisation to only one of them is what would break the comparison. Both
arms get it here, and the fairness assertions enforce it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from scale_aware_compression.compression.quantisation import QuantisationGranularity, fake_quantise
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.training.recovery import RecoveryBudget, RecoveryError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from scale_aware_compression.config import ExperimentConfig

LOGGER = get_logger(__name__)

SPARSITY_TOLERANCE = 1e-9
"""Mask sparsity must not move at all: the mask is a constant buffer, so any drift is a defect."""


class _FakeQuantiseSTE(torch.autograd.Function):
    """Straight-through estimator around the repository's own fake quantisation.

    ``fake_quantise`` rounds, and ``round`` has zero gradient almost everywhere, so a naive call
    would deliver no gradient to the shadow weight at all -- recovery would appear to run and change
    nothing. The forward is the real grid; the backward is the identity.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        weight: torch.Tensor,
        bits: int,
        granularity_value: str,
        group_size: int,
    ) -> torch.Tensor:
        """Snap to the grid using the shared implementation."""
        return fake_quantise(
            weight,
            bits=bits,
            granularity=QuantisationGranularity(granularity_value),
            group_size=group_size,
        )

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        """Pass the gradient straight through to the shadow weight."""
        return grad_output, None, None, None


class QuantisedMaskedLinear(nn.Module):
    """A linear layer that is masked and fake-quantised on every forward pass.

    Installed in place of an existing :class:`torch.nn.Linear`, adopting that layer's own weight
    tensor as the trainable shadow. Nothing is copied, so installing and removing is lossless.
    """

    def __init__(
        self,
        linear: nn.Linear,
        mask: torch.Tensor,
        *,
        bits: int,
        granularity: QuantisationGranularity,
        group_size: int,
    ) -> None:
        """Wrap ``linear``.

        Args:
            linear: The layer to replace. Its weight is adopted, not copied.
            mask: Keep-mask of the same shape as the weight, ``True`` meaning kept.
            bits: Quantisation width held during recovery.
            granularity: Scope one scale covers.
            group_size: Elements per group, for per-group granularity.

        Raises:
            RecoveryError: If the mask shape does not match the weight.
        """
        super().__init__()
        if tuple(mask.shape) != tuple(linear.weight.shape):
            raise RecoveryError(
                f"mask shape {tuple(mask.shape)} does not match weight {tuple(linear.weight.shape)}"
            )
        self.weight = linear.weight
        self.bias = linear.bias
        self.bits = int(bits)
        self.granularity_value = granularity.value
        self.group_size = int(group_size)
        # A buffer, not a parameter: it must never receive a gradient or an optimiser update.
        self.register_buffer("mask", mask.to(dtype=torch.bool, device=linear.weight.device))
        self.forward_calls = 0

    @property
    def effective_weight(self) -> torch.Tensor:
        """The weight the forward pass uses: masked, then snapped to the grid."""
        masked = self.weight * self.mask
        return _FakeQuantiseSTE.apply(masked, self.bits, self.granularity_value, self.group_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the masked, fake-quantised weight."""
        self.forward_calls += 1
        return nn.functional.linear(inputs, self.effective_weight, self.bias)

    @torch.no_grad()
    def bake(self) -> nn.Linear:
        """Write the effective weight into a plain linear layer.

        Called once recovery finishes, so what is evaluated is exactly what the forward pass
        produced rather than a shadow weight that has drifted off the grid.

        Returns:
            A plain ``nn.Linear`` holding the masked, quantised weight.
        """
        restored = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        restored = restored.to(device=self.weight.device, dtype=self.weight.dtype)
        restored.weight.copy_(self.effective_weight.detach())
        if self.bias is not None:
            restored.bias.copy_(self.bias.detach())
        return restored


@dataclass(slots=True)
class RecoveryOutcome:
    """Everything the ablation records about one arm's recovery phase."""

    steps: int
    tokens: int
    duration_seconds: float
    final_loss: float
    mean_loss: float
    optimiser: str
    learning_rate: float
    scheduler: str
    seed: int
    budget: dict[str, Any]
    mask_sparsity_before: float
    mask_sparsity_after: float
    fake_quant_forward_calls: int
    bits: int
    losses: list[float] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "duration_seconds": self.duration_seconds,
            "final_loss": self.final_loss,
            "mean_loss": self.mean_loss,
            "optimiser": self.optimiser,
            "learning_rate": self.learning_rate,
            "scheduler": self.scheduler,
            "seed": self.seed,
            "budget": self.budget,
            "mask_sparsity_before": self.mask_sparsity_before,
            "mask_sparsity_after": self.mask_sparsity_after,
            "fake_quant_forward_calls": self.fake_quant_forward_calls,
            "bits": self.bits,
            "losses": self.losses,
            "trajectory": self.trajectory,
        }


def _named_linears(model: nn.Module, names: list[str]) -> dict[str, nn.Linear]:
    """Resolve module names to linear layers.

    Args:
        model: The model.
        names: Fully qualified module names.

    Returns:
        Name to layer.

    Raises:
        RecoveryError: If a name is missing or is not a linear layer.
    """
    found: dict[str, nn.Linear] = {}
    for name in names:
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise RecoveryError(f"{name!r} is not present in the model") from error
        if not isinstance(module, nn.Linear):
            raise RecoveryError(f"{name!r} is a {type(module).__name__}, expected nn.Linear")
        found[name] = module
    return found


def _replace_submodule(model: nn.Module, name: str, replacement: nn.Module) -> None:
    """Swap a submodule in place by its qualified name."""
    parent_name, _, attribute = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, attribute, replacement)


def install_recovery_modules(
    model: nn.Module,
    masks: dict[str, torch.Tensor],
    *,
    bits: int,
    granularity: QuantisationGranularity,
    group_size: int,
) -> dict[str, QuantisedMaskedLinear]:
    """Replace every masked linear layer with a masked, fake-quantising equivalent.

    Args:
        model: The compressed model. Modified in place.
        masks: True keep-masks per module name, from ``LayerwiseReport.masks_by_module``.
        bits: Quantisation width to hold during recovery.
        granularity: Scope one scale covers.
        group_size: Elements per group.

    Returns:
        The installed wrappers, by module name.

    Raises:
        RecoveryError: If ``masks`` is empty, or a name does not resolve to a linear layer.
    """
    if not masks:
        raise RecoveryError(
            "install_recovery_modules received no masks. Recovery must not fall back to deriving "
            "them from the nonzero pattern: quantisation zeroes surviving weights, so that would "
            "freeze more than the pruning budget and by a different amount in each arm. Run the "
            "compression with retain_masks=True."
        )
    linears = _named_linears(model, list(masks))
    installed: dict[str, QuantisedMaskedLinear] = {}
    for name, linear in linears.items():
        wrapper = QuantisedMaskedLinear(
            linear,
            masks[name],
            bits=bits,
            granularity=granularity,
            group_size=group_size,
        )
        _replace_submodule(model, name, wrapper)
        installed[name] = wrapper
    LOGGER.info(
        "Recovery: wrapped %d module(s) with frozen masks and live W%d fake quantisation",
        len(installed),
        bits,
    )
    return installed


def bake_recovery_modules(model: nn.Module, installed: dict[str, QuantisedMaskedLinear]) -> None:
    """Replace the wrappers with plain linear layers holding the final effective weights.

    Args:
        model: The model. Modified in place.
        installed: The wrappers returned by :func:`install_recovery_modules`.
    """
    for name, wrapper in installed.items():
        _replace_submodule(model, name, wrapper.bake())
    LOGGER.info("Recovery: baked %d module(s) back to plain linear layers", len(installed))


def mask_sparsity(installed: dict[str, QuantisedMaskedLinear]) -> float:
    """Overall fraction of masked-out weights across the wrapped modules.

    Args:
        installed: The wrappers.

    Returns:
        Pruned fraction, or 0.0 when nothing is wrapped.
    """
    total = 0
    pruned = 0
    for wrapper in installed.values():
        mask = wrapper.mask
        total += mask.numel()
        pruned += int((~mask).sum().item())
    return pruned / total if total else 0.0


def assert_masks_still_hold(
    installed: dict[str, QuantisedMaskedLinear], sparsity_before: float
) -> float:
    """Verify recovery changed neither the mask nor the effective sparsity.

    Checks the *effective* weight, not the shadow: the shadow is allowed to hold anything at a
    pruned position (it receives no gradient, so it stays where it started), but what the model
    computes with must be exactly zero there.

    Args:
        installed: The wrappers.
        sparsity_before: Mask sparsity measured before recovery.

    Returns:
        Mask sparsity after recovery.

    Raises:
        RecoveryError: If sparsity moved, or any pruned position is non-zero in the effective
            weight.
    """
    after = mask_sparsity(installed)
    if abs(after - sparsity_before) > SPARSITY_TOLERANCE:
        raise RecoveryError(
            f"mask sparsity moved during recovery: {sparsity_before!r} -> {after!r}. The mask is a "
            "constant buffer, so this means something reselected or regrew weights."
        )
    for name, wrapper in installed.items():
        with torch.no_grad():
            leaked = wrapper.effective_weight[~wrapper.mask]
            if leaked.numel() and bool(leaked.abs().max() > 0):
                raise RecoveryError(
                    f"{name!r} has non-zero values at pruned positions after recovery "
                    f"(max {float(leaked.abs().max())!r})"
                )
    return after


def assert_fake_quantisation_ran(installed: dict[str, QuantisedMaskedLinear]) -> int:
    """Verify the fake-quantising path was actually exercised.

    A recovery that silently trained in full precision would look like a successful run and produce
    a number that means something else entirely, so this is checked rather than assumed.

    Args:
        installed: The wrappers.

    Returns:
        Total forward calls through the fake-quantising path.

    Raises:
        RecoveryError: If any wrapped module was never used in a forward pass.
    """
    calls = {name: wrapper.forward_calls for name, wrapper in installed.items()}
    idle = sorted(name for name, count in calls.items() if count == 0)
    if idle:
        raise RecoveryError(
            f"{len(idle)} wrapped module(s) never ran a fake-quantised forward pass, so recovery "
            f"did not train under the W4 constraint: {idle[:5]}"
        )
    return sum(calls.values())


def assert_budgets_match(first: RecoveryBudget, second: RecoveryBudget) -> None:
    """Verify two arms were given identical recovery budgets.

    Args:
        first: One arm's budget.
        second: The other arm's.

    Raises:
        RecoveryError: If the budgets differ in any respect.
    """
    if first != second:
        raise RecoveryError(
            f"recovery budgets differ between arms: {first.to_dict()} vs {second.to_dict()}. A "
            "joint gain measured across different training budgets is confounded with the budget."
        )


def run_end_to_end_recovery(
    model: nn.Module,
    batches: list[torch.Tensor],
    installed: dict[str, QuantisedMaskedLinear],
    *,
    config: ExperimentConfig,
    budget: RecoveryBudget,
    device: torch.device | str,
    probe: Callable[[int], dict[str, float]] | None = None,
) -> RecoveryOutcome:
    """Run the short global recovery phase under a causal-language-model loss.

    Standard cross-entropy over the whole decoder -- deliberately not the layerwise reconstruction
    objective, since the question is whether a *global* objective can exploit structure the local
    one leaves behind.

    Args:
        model: The compressed model with recovery wrappers installed.
        batches: Token-id tensors, pre-materialised so both arms see identical data in identical
            order. Consumed cyclically if the budget exceeds their number.
        installed: The wrappers, for the safety assertions.
        config: The full experiment config.
        budget: The resolved budget, identical across arms.
        device: Where to train.
        probe: Optional evaluation callback, invoked with the completed step count every
            ``recovery.probe_every_steps`` steps. Whatever it returns is recorded verbatim in
            the trajectory. It must not mutate the model; the loop restores training mode
            afterwards regardless.

    Returns:
        The outcome, including the exact steps and tokens consumed.

    Raises:
        RecoveryError: If no batches are supplied, or a safety check fails.
    """
    if not batches:
        raise RecoveryError("run_end_to_end_recovery received no batches")

    recovery = config.compression.recovery
    sparsity_before = mask_sparsity(installed)

    # Seed immediately before training so the two arms share initialisation and any stochastic
    # element, and so a replicate is reproducible on its own.
    torch.manual_seed(recovery.seed)

    model.to(device)
    model.train()
    if recovery.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimiser = torch.optim.AdamW(
        trainable, lr=recovery.learning_rate, weight_decay=recovery.weight_decay
    )
    warmup = max(1, int(budget.optimiser_steps * recovery.warmup_ratio))

    def learning_rate_at(step: int) -> float:
        """Linear warmup then cosine decay, expressed as a multiplier."""
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, budget.optimiser_steps - warmup)
        return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793)).item())

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, learning_rate_at)

    losses: list[float] = []
    trajectory: list[dict[str, Any]] = []
    started = time.perf_counter()
    cursor = 0
    for step in range(budget.optimiser_steps):
        optimiser.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(budget.gradient_accumulation_steps):
            tokens = batches[cursor % len(batches)].to(device)
            cursor += 1
            outputs = model(tokens, labels=tokens)
            loss = outputs.loss / budget.gradient_accumulation_steps
            loss.backward()
            step_loss += float(loss.detach())
        if recovery.max_grad_norm and recovery.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, recovery.max_grad_norm)
        optimiser.step()
        scheduler.step()
        losses.append(step_loss)
        if (step + 1) % recovery.log_every_steps == 0 or step == 0:
            LOGGER.info(
                "Recovery step %d/%d loss %.4f lr %.3e",
                step + 1,
                budget.optimiser_steps,
                step_loss,
                optimiser.param_groups[0]["lr"],
            )
        every = recovery.probe_every_steps or 0
        if probe is not None and every > 0 and (step + 1) % every == 0:
            model.eval()
            with torch.no_grad():
                measured = probe(step + 1)
            model.train()
            trajectory.append({"step": step + 1, "loss": step_loss, **measured})
            LOGGER.info("Recovery probe at step %d: %s", step + 1, measured)

    duration = time.perf_counter() - started
    model.eval()

    forward_calls = assert_fake_quantisation_ran(installed)
    sparsity_after = assert_masks_still_hold(installed, sparsity_before)

    consumed_tokens = sum(
        int(batches[index % len(batches)].numel())
        for index in range(budget.optimiser_steps * budget.gradient_accumulation_steps)
    )
    return RecoveryOutcome(
        steps=budget.optimiser_steps,
        tokens=consumed_tokens,
        duration_seconds=duration,
        final_loss=losses[-1] if losses else float("nan"),
        mean_loss=sum(losses) / len(losses) if losses else float("nan"),
        optimiser="AdamW",
        learning_rate=recovery.learning_rate,
        scheduler="linear-warmup-cosine-decay",
        seed=recovery.seed,
        budget=budget.to_dict(),
        mask_sparsity_before=sparsity_before,
        mask_sparsity_after=sparsity_after,
        fake_quant_forward_calls=forward_calls,
        bits=config.compression.quantisation.bits,
        losses=losses,
        trajectory=trajectory,
    )
