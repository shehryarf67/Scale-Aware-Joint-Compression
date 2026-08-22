"""The post-hoc end-to-end recovery ablation: frozen masks, live W4, and no confirmatory contact.

Every test here guards a property whose violation would produce a plausible number rather than an
error, which is why they exist as assertions rather than as a note in the methodology.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.requires_torch


@pytest.fixture
def wrapper():
    """A wrapped linear with 25% of its weights pruned."""
    import torch
    from torch import nn

    from scale_aware_compression.compression.quantisation import QuantisationGranularity
    from scale_aware_compression.training.end_to_end import QuantisedMaskedLinear

    torch.manual_seed(0)
    linear = nn.Linear(8, 4)
    mask = torch.ones(4, 8, dtype=torch.bool)
    mask[:, :2] = False
    with torch.no_grad():
        linear.weight[~mask] = 0.0
    return QuantisedMaskedLinear(
        linear,
        mask,
        bits=4,
        granularity=QuantisationGranularity.PER_CHANNEL,
        group_size=128,
    )


class TestTheMaskIsFrozen:
    """No regrowth, no reselection, no sparsity drift -- the whole premise of the ablation."""

    def test_pruned_positions_receive_no_gradient(self, wrapper):
        """``d(w*m)/dw = m = 0``, so the optimiser has nothing to act on."""
        import torch

        wrapper(torch.randn(3, 8)).pow(2).sum().backward()
        assert float(wrapper.weight.grad[~wrapper.mask].abs().sum()) == 0.0
        assert float(wrapper.weight.grad[wrapper.mask].abs().sum()) > 0.0

    def test_sparsity_survives_optimiser_steps_with_weight_decay(self, wrapper):
        """AdamW with decay is the case that could move a zero weight if masking were post-hoc."""
        import torch

        from scale_aware_compression.training.end_to_end import (
            assert_masks_still_hold,
            mask_sparsity,
        )

        before = mask_sparsity({"m": wrapper})
        optimiser = torch.optim.AdamW([wrapper.weight], lr=1e-2, weight_decay=0.01)
        inputs = torch.randn(3, 8)
        for _ in range(5):
            optimiser.zero_grad()
            wrapper(inputs).pow(2).sum().backward()
            optimiser.step()
        assert assert_masks_still_hold({"m": wrapper}, before) == before

    def test_a_leaked_pruned_weight_is_caught(self, wrapper):
        """The assertion must fail on a real violation, not merely pass on a clean run."""
        import torch

        from scale_aware_compression.training.end_to_end import (
            RecoveryError,
            assert_masks_still_hold,
            mask_sparsity,
        )

        before = mask_sparsity({"m": wrapper})
        with torch.no_grad():
            wrapper.mask[0, 0] = True  # simulate regrowth
        with pytest.raises(RecoveryError, match="sparsity moved"):
            assert_masks_still_hold({"m": wrapper}, before)


class TestFakeQuantisationStaysActive:
    """Recovery must not become FP32 fine-tuning: the model learns under the W4 constraint."""

    def test_the_forward_pass_is_on_the_grid(self, wrapper):
        """A 4-bit symmetric grid admits at most 2^4 distinct values per output channel."""
        import torch

        effective = wrapper.effective_weight.detach()
        for row in range(effective.shape[0]):
            assert len(torch.unique(effective[row])) <= 2**4

    def test_gradients_survive_the_rounding(self, wrapper):
        """Without a straight-through estimator ``round`` would deliver no gradient at all, and.

        recovery would run and change nothing.
        """
        import torch

        wrapper(torch.randn(3, 8)).pow(2).sum().backward()
        assert float(wrapper.weight.grad.abs().sum()) > 0.0

    def test_an_unused_module_is_reported(self, wrapper):
        """A module never exercised means the constraint was not applied where it was claimed."""
        from scale_aware_compression.training.end_to_end import (
            RecoveryError,
            assert_fake_quantisation_ran,
        )

        with pytest.raises(RecoveryError, match="never ran a fake-quantised forward"):
            assert_fake_quantisation_ran({"idle": wrapper})

    def test_baking_preserves_the_effective_weight(self, wrapper):
        """What is evaluated must be what the forward pass produced, not a drifted shadow."""
        import torch

        expected = wrapper.effective_weight.detach().clone()
        baked = wrapper.bake()
        assert torch.equal(baked.weight.detach(), expected)


class TestFairnessIsEnforced:
    """A joint gain measured across different budgets is confounded with the budget."""

    def test_mismatched_budgets_are_refused(self):
        from scale_aware_compression.training.end_to_end import assert_budgets_match
        from scale_aware_compression.training.recovery import RecoveryBudget, RecoveryError

        first = RecoveryBudget(200, 2, 4, 512)
        second = RecoveryBudget(400, 2, 4, 512)
        with pytest.raises(RecoveryError, match="budgets differ"):
            assert_budgets_match(first, second)
        assert_budgets_match(first, first)

    def test_masks_are_required_rather_than_derived_from_zeros(self):
        """``weight != 0`` is strictly coarser: quantisation zeroes surviving weights, by a.

        different amount in each arm, so deriving the mask would freeze more than the budget and
        unequally.
        """
        import torch
        from torch import nn

        from scale_aware_compression.compression.quantisation import QuantisationGranularity
        from scale_aware_compression.training.end_to_end import (
            RecoveryError,
            install_recovery_modules,
        )

        with pytest.raises(RecoveryError, match="no masks"):
            install_recovery_modules(
                nn.Linear(4, 4),
                {},
                bits=4,
                granularity=QuantisationGranularity.PER_CHANNEL,
                group_size=128,
            )
        del torch


class TestTheConfirmatoryStudyCannotBeTouched:
    """The frozen grid must not acquire a training stage, or an output directory in common."""

    def test_frozen_configs_do_not_enable_end_to_end_recovery(self):
        """`main_scale_sweep.yaml` sets recovery.enabled TRUE with max_steps 500. Gating a real.

        training loop on that flag would have turned the confirmatory grid into a fine-tuning
        study on its next re-run, which is why the opt-in is a separate field.
        """
        from scale_aware_compression.config import load_config

        for name in ("main_scale_sweep", "qwen_validation", "screening"):
            config = load_config(f"configs/experiments/{name}.yaml")
            assert config.compression.recovery.end_to_end is False, name

    def test_frozen_configs_do_not_retain_masks(self):
        """Mask retention is derived from the opt-in, so the confirmatory path is unchanged in.

        memory as well as in behaviour.
        """
        from scale_aware_compression.compression.arms import plan_from_config
        from scale_aware_compression.config import load_config

        for name in ("main_scale_sweep", "qwen_validation"):
            plan = plan_from_config(load_config(f"configs/experiments/{name}.yaml"))
            assert plan.retain_masks is False, name

    def test_the_ablation_writes_to_its_own_tree(self):
        """Nothing here may be discoverable by the confirmatory resume logic or audit, which read.

        outputs/metrics.
        """
        from scale_aware_compression.config import load_config

        config = load_config("configs/experiments/recovery_ablation_160m_w4.yaml")
        output = Path(config.runtime.output_dir).as_posix()
        assert output.endswith("recovery_ablation")
        assert not output.endswith("metrics")

    def test_the_ablation_is_validation_only_and_labelled(self):
        """Post-hoc work must not spend the reserved test split."""
        from scale_aware_compression.config import load_config

        config = load_config("configs/experiments/recovery_ablation_160m_w4.yaml")
        assert config.data.eval_split == "validation"
        assert config.compression.recovery.end_to_end is True
        for tag in ("exploratory", "post-hoc", "recovery-ablation", "not-confirmatory"):
            assert tag in config.experiment.tags

    def test_the_budget_matches_the_frozen_headline(self):
        """A different compression budget would make the ablation incomparable to F-37."""
        from scale_aware_compression.config import load_config

        config = load_config("configs/experiments/recovery_ablation_160m_w4.yaml")
        assert config.compression.effective_sparsity == pytest.approx(0.3)
        assert config.compression.quantisation.bits == 4


class TestRecoveryDisabledLeavesThePipelineAlone:
    """The arm hook stays a no-op, which is what keeps every existing config byte-identical."""

    def test_the_arm_recover_hook_is_still_inert(self, tiny_causal_lm):
        """`LayerwiseArm.recover` must not acquire behaviour: the ablation drives the stages itself.

        because `convert` packs weights into an untrainable form.
        """
        import copy

        from scale_aware_compression.compression import COMPRESSOR_REGISTRY
        from scale_aware_compression.config import load_config
        from scale_aware_compression.constants import CompressionMethod

        config = load_config("configs/experiments/recovery_ablation_160m_w4.yaml")
        config.compression.method = CompressionMethod.JOINT
        arm = COMPRESSOR_REGISTRY[CompressionMethod.JOINT](config)
        model = copy.deepcopy(tiny_causal_lm)
        before = {name: parameter.clone() for name, parameter in model.named_parameters()}
        returned = arm.recover(model)
        assert returned is model
        for name, parameter in returned.named_parameters():
            assert (parameter == before[name]).all(), name

    def test_the_end_to_end_flag_requires_an_explicit_step_count(self):
        """Deriving the budget from epochs would let the two arms diverge if their loaders ever.

        differed in length.
        """
        from scale_aware_compression.config import ConfigError, RecoveryConfig

        with pytest.raises(ConfigError, match="requires an explicit max_steps"):
            RecoveryConfig(end_to_end=True, max_steps=None)
        RecoveryConfig(end_to_end=True, max_steps=10)
