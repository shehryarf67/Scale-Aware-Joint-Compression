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


class TestTheMidRecoveryProbe:
    """The trajectory exists to separate "lr too high" from "overfits late" (F-42), so it has to.

    fire on schedule, record what it measured, and leave training exactly as it found it.
    """

    @staticmethod
    def _run(monkeypatch, steps, every, probe=None):
        """Drive the real recovery loop over a trivial model, returning the outcome."""
        import torch
        from torch import nn

        from scale_aware_compression.compression.quantisation import QuantisationGranularity
        from scale_aware_compression.config import load_config
        from scale_aware_compression.training.end_to_end import (
            QuantisedMaskedLinear,
            run_end_to_end_recovery,
        )
        from scale_aware_compression.training.recovery import RecoveryBudget

        class Tiny(nn.Module):
            """Minimal causal-LM stand-in: exposes ``loss`` and routes through the wrapper."""

            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.training_modes = []

            def forward(self, tokens, labels=None):
                self.training_modes.append(self.training)
                value = self.wrapped(tokens.float())
                return type("Output", (), {"loss": value.pow(2).mean()})()

        torch.manual_seed(0)
        linear = nn.Linear(4, 4)
        mask = torch.ones(4, 4, dtype=torch.bool)
        mask[:, 0] = False
        with torch.no_grad():
            linear.weight[~mask] = 0.0
        wrapper = QuantisedMaskedLinear(
            linear,
            mask,
            bits=4,
            granularity=QuantisationGranularity.PER_CHANNEL,
            group_size=128,
        )
        model = Tiny(wrapper)

        config = load_config("configs/experiments/recovery_ablation_160m_w4_gentle.yaml")
        config.compression.recovery.max_steps = steps
        config.compression.recovery.probe_every_steps = every
        config.compression.recovery.log_every_steps = 1000
        budget = RecoveryBudget(steps, 1, 1, 4)
        outcome = run_end_to_end_recovery(
            model,
            [torch.randint(0, 4, (1, 4))],
            {"wrapped": wrapper},
            config=config,
            budget=budget,
            device="cpu",
            probe=probe,
        )
        return outcome, model

    def test_the_probe_fires_on_schedule_and_is_recorded(self, monkeypatch):
        """Four checkpoints at 50-step intervals over 200 steps, in order, with the payload kept."""
        seen = []

        def probe(step):
            seen.append(step)
            return {"retention": 50.0 + step, "device": "cpu"}

        outcome, _ = self._run(monkeypatch, steps=8, every=2, probe=probe)
        assert seen == [2, 4, 6, 8]
        assert [entry["step"] for entry in outcome.trajectory] == [2, 4, 6, 8]
        assert [entry["retention"] for entry in outcome.trajectory] == [52.0, 54.0, 56.0, 58.0]
        # The step loss is carried alongside, so a rising loss and falling retention -- the F-42
        # signature -- is visible in one table.
        assert all("loss" in entry for entry in outcome.trajectory)

    def test_training_mode_is_restored_after_a_probe(self, monkeypatch):
        """Left in eval mode the remaining steps would silently train a different model.

        Nothing would raise; the run would simply be wrong from the first checkpoint onward.
        """
        _, model = self._run(monkeypatch, steps=4, every=2, probe=lambda step: {"retention": 1.0})
        assert model.training_modes == [True, True, True, True]

    def test_no_probe_means_no_trajectory(self, monkeypatch):
        """F-42's config leaves this unset, and its records must stay byte-comparable."""
        outcome, model = self._run(monkeypatch, steps=4, every=None, probe=None)
        assert outcome.trajectory == []
        assert model.training_modes == [True, True, True, True]

    def test_a_probe_without_a_schedule_is_never_called(self, monkeypatch):
        """The schedule governs, not the presence of a callback."""
        calls = []
        outcome, _ = self._run(
            monkeypatch, steps=4, every=0, probe=lambda step: calls.append(step) or {}
        )
        assert calls == []
        assert outcome.trajectory == []


class TestTheGentleProbeIsolatesOneVariable:
    """F-42 versus the gentle run is only interpretable if exactly one thing changed."""

    def test_only_the_learning_rate_and_the_probe_schedule_differ(self):
        """Holding the token budget fixed while shrinking the step is what separates step.

        MAGNITUDE from data QUANTITY. If a second knob moved, neither could be attributed.
        """
        from scale_aware_compression.config import load_config

        base = load_config(
            "configs/experiments/recovery_ablation_160m_w4.yaml"
        ).compression.recovery
        gentle = load_config(
            "configs/experiments/recovery_ablation_160m_w4_gentle.yaml"
        ).compression.recovery

        assert gentle.learning_rate == pytest.approx(1e-5)
        assert base.learning_rate == pytest.approx(5e-5)
        assert gentle.probe_every_steps == 50
        assert not base.probe_every_steps

        for field in (
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "weight_decay",
            "warmup_ratio",
            "max_grad_norm",
            "mixed_precision",
            "gradient_checkpointing",
            "seed",
            "end_to_end",
        ):
            assert getattr(gentle, field) == getattr(base, field), field

    def test_the_two_runs_cannot_overwrite_each_other(self):
        """Same output tree, so the experiment ids must differ -- the B-51 class of fault."""
        from scale_aware_compression.config import load_config

        base = load_config("configs/experiments/recovery_ablation_160m_w4.yaml")
        gentle = load_config("configs/experiments/recovery_ablation_160m_w4_gentle.yaml")
        assert base.runtime.output_dir == gentle.runtime.output_dir
        assert base.experiment.id != gentle.experiment.id

    def test_the_gentle_run_is_labelled_an_instrument_check(self):
        """It must not be readable as an effect size: one paired draw, and tagged as such."""
        from scale_aware_compression.config import load_config

        config = load_config("configs/experiments/recovery_ablation_160m_w4_gentle.yaml")
        assert config.sweep.replicates == 1
        assert config.data.eval_split == "validation"
        for tag in ("exploratory", "post-hoc", "instrument-check", "not-confirmatory"):
            assert tag in config.experiment.tags

    def test_the_masks_stay_frozen_in_the_gentle_run_too(self):
        """Unfreezing would let both arms reselect under one gradient signal, overwriting the.

        initialisation difference instead of testing it. There is no config path to it, and this
        asserts the budget is still the frozen headline so the run stays comparable to F-37.
        """
        from scale_aware_compression.config import load_config

        config = load_config("configs/experiments/recovery_ablation_160m_w4_gentle.yaml")
        assert config.compression.effective_sparsity == pytest.approx(0.3)
        assert config.compression.quantisation.bits == 4

    def test_the_frozen_configs_still_have_no_probe_schedule(self):
        """A confirmatory config that started evaluating mid-run would change its own timings."""
        from scale_aware_compression.config import load_config

        for name in ("main_scale_sweep", "qwen_validation", "screening"):
            recovery = load_config(f"configs/experiments/{name}.yaml").compression.recovery
            assert not recovery.probe_every_steps, name


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


class TestResumingCannotAdoptTheWrongRecord:
    """Resuming by filename is how B-45 and B-51 happened: the name matched, the conditions did.

    not. Every condition that would change the number is compared before a record is reused.
    """

    @staticmethod
    def _module():
        """Load the ablation script as a module without executing its main()."""
        import importlib.util

        spec = importlib.util.spec_from_file_location("rra", "scripts/run_recovery_ablation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _config():
        from scale_aware_compression.config import load_config

        return load_config("configs/experiments/recovery_ablation_160m_w4_gentle.yaml")

    def _write(self, tmp_path, config, **overrides):
        """Write a record that matches the config, with optional corruptions."""
        import json

        recovery = config.compression.recovery
        record = {
            "smoke": False,
            "arm": "sequential",
            "calibration_replicate": 0,
            "model_name": config.model.name,
            "model_revision": config.model.revision,
            "eval_split": config.data.eval_split,
            "target_sparsity": config.compression.effective_sparsity,
            "quantisation_bits": config.compression.quantisation.bits,
            "calibration_fingerprint": "cal-fp",
            "dataset_fingerprint": "data-fp",
            "recovery": {
                "steps": recovery.max_steps,
                "learning_rate": recovery.learning_rate,
                "trajectory": [
                    {"step": step}
                    for step in range(
                        recovery.probe_every_steps,
                        recovery.max_steps + 1,
                        recovery.probe_every_steps,
                    )
                ],
            },
        }
        record.update(overrides)
        path = tmp_path / "record.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        return path

    def _check(self, tmp_path, **overrides):
        module, config = self._module(), self._config()
        path = self._write(tmp_path, config, **overrides)
        return module._reusable_record(
            path,
            config=config,
            arm_name="sequential",
            replicate=0,
            calibration_fingerprint="cal-fp",
            dataset_fingerprint="data-fp",
        )

    def test_a_matching_record_is_reused(self, tmp_path):
        """The whole point of resuming: an unchanged cell must not cost 78 minutes again."""
        assert self._check(tmp_path) is not None

    def test_a_smoke_record_is_never_reused(self, tmp_path):
        """A 4-step smoke record under a real id would pair against a real arm and look plausible.

        This is the fault that was actually found on disk, not a hypothetical.
        """
        assert self._check(tmp_path, smoke=True) is None

    def test_a_different_learning_rate_is_not_reused(self, tmp_path):
        """The gentle probe exists BECAUSE the learning rate changed; reusing across it would.

        compare 1e-5 against 5e-5 and attribute the difference to the arm.
        """
        config = self._config()
        module = self._module()
        path = self._write(tmp_path, config)
        config.compression.recovery.learning_rate = 5e-5
        assert (
            module._reusable_record(
                path,
                config=config,
                arm_name="sequential",
                replicate=0,
                calibration_fingerprint="cal-fp",
                dataset_fingerprint="data-fp",
            )
            is None
        )

    def test_a_different_step_count_is_not_reused(self, tmp_path):
        """Steps are the fairness unit; a mismatch is a different experiment."""
        assert self._check(tmp_path, recovery={"steps": 100, "learning_rate": 1e-5}) is None

    def test_a_different_calibration_draw_is_not_reused(self, tmp_path):
        """Paired replicates are only paired if both arms saw the same draw."""
        assert self._check(tmp_path, calibration_fingerprint="other") is None

    def test_a_different_evaluation_window_is_not_reused(self, tmp_path):
        """B-45 in miniature: a record measured on another window is not this cell's number."""
        assert self._check(tmp_path, dataset_fingerprint="other") is None

    def test_the_wrong_arm_is_not_reused(self, tmp_path):
        """Filenames encode the arm, but the record is what gets believed."""
        assert self._check(tmp_path, arm="joint") is None

    def test_a_missing_trajectory_is_not_reused(self, tmp_path):
        """A resumed arm without its probe points would half-populate the comparison."""
        assert (
            self._check(tmp_path, recovery={"steps": 200, "learning_rate": 1e-5, "trajectory": []})
            is None
        )

    def test_a_missing_file_is_not_reused(self, tmp_path):
        """Nothing on disk means nothing to adopt."""
        module, config = self._module(), self._config()
        assert (
            module._reusable_record(
                tmp_path / "absent.json",
                config=config,
                arm_name="sequential",
                replicate=0,
                calibration_fingerprint="cal-fp",
                dataset_fingerprint="data-fp",
            )
            is None
        )

    def test_an_unreadable_record_is_not_reused(self, tmp_path):
        """A truncated write from a killed run must re-run, not raise mid-sweep."""
        module, config = self._module(), self._config()
        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        assert (
            module._reusable_record(
                path,
                config=config,
                arm_name="sequential",
                replicate=0,
                calibration_fingerprint="cal-fp",
                dataset_fingerprint="data-fp",
            )
            is None
        )


class TestTheArmsCompressAgainstCalibrationNotRecoveryData:
    """B-54: the script handed the arms the RECOVERY slice as calibration, inverting the.

    disjointness its own design claimed and coupling ``recovery.max_steps`` to compression. Thirty
    five tests guarded this ablation and not one asserted what the arms were calibrated on -- the
    suite was watching the recovery phase and ignoring the compression before it. These watch it.
    """

    @staticmethod
    def _module():
        import importlib.util

        spec = importlib.util.spec_from_file_location("rra", "scripts/run_recovery_ablation.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def _calibration(population=64, calibration_size=8, batch_size=2):
        """A stub calibration set with a loader, a dataset and disjointness to check against."""
        import torch

        class Dataset:
            def __len__(self):
                return population

            def __getitem__(self, index):
                # Encode the index in the payload so a batch can be traced to its source rows.
                return {"input_ids": torch.full((4,), float(index))}

        indices = list(range(calibration_size))
        dataset = Dataset()
        loader = [
            {
                "input_ids": torch.stack(
                    [dataset[i]["input_ids"] for i in indices[s : s + batch_size]]
                )
            }
            for s in range(0, len(indices), batch_size)
        ]

        class Calibration:
            pass

        calibration = Calibration()
        calibration.loader = loader
        calibration.dataset = dataset
        calibration.indices = indices
        return calibration

    @staticmethod
    def _recovery(steps=3, batch_size=2, accumulation=2, seed=1234):
        from scale_aware_compression.config import RecoveryConfig

        return RecoveryConfig(
            end_to_end=True,
            max_steps=steps,
            batch_size=batch_size,
            gradient_accumulation_steps=accumulation,
            seed=seed,
        )

    def test_calibration_batches_come_from_the_calibration_loader(self):
        """Its size follows data.calibration_samples, which is the whole point."""
        module = self._module()
        calibration = self._calibration(calibration_size=8, batch_size=2)
        batches = module._calibration_batches(calibration)
        assert len(batches) == 4
        rows = {int(value) for batch in batches for value in batch.flatten().tolist()}
        assert rows == set(range(8))

    def test_calibration_batches_do_not_depend_on_the_recovery_budget(self):
        """The B-54 signature: max_steps silently resizing the COMPRESSION data.

        F-43 compressed against 1600 sequences and the R=8 leg against 400 for this reason, which is
        what made their before-recovery gains differ when they had to be bit-identical.
        """
        module = self._module()
        calibration = self._calibration()
        first = module._calibration_batches(calibration)
        recovery_small, recovery_large = self._recovery(steps=2), self._recovery(steps=6)
        module._recovery_slice(calibration, recovery_small, 0)
        module._recovery_slice(calibration, recovery_large, 0)
        assert len(module._calibration_batches(calibration)) == len(first)

    def test_the_recovery_slice_excludes_every_calibration_index(self):
        """The property the design depends on, and the one B-54 inverted."""
        module = self._module()
        calibration = self._calibration(calibration_size=8)
        _, chosen = module._recovery_slice(calibration, self._recovery(steps=3), 0)
        assert set(chosen) & set(calibration.indices) == set()

    def test_the_recovery_slice_and_the_calibration_batches_are_different_data(self):
        """If a future edit reintroduces the swap, these two must not be interchangeable."""
        module = self._module()
        calibration = self._calibration(calibration_size=8)
        recovery_batches, _ = module._recovery_slice(calibration, self._recovery(steps=3), 0)
        calibration_batches = module._calibration_batches(calibration)
        recovery_rows = {int(v) for b in recovery_batches for v in b.flatten().tolist()}
        calibration_rows = {int(v) for b in calibration_batches for v in b.flatten().tolist()}
        assert recovery_rows & calibration_rows == set()

    def test_the_recovery_slice_is_sized_by_the_budget(self):
        """Steps x batch x accumulation sequences, so both arms consume an identical stream."""
        module = self._module()
        calibration = self._calibration()
        recovery = self._recovery(steps=3, batch_size=2, accumulation=2)
        batches, chosen = module._recovery_slice(calibration, recovery, 0)
        assert len(chosen) == 3 * 2 * 2
        assert len(batches) == 6
        assert all(batch.shape[0] == 2 for batch in batches)

    def test_the_slice_is_deterministic_per_replicate_and_differs_between_them(self):
        """Both arms must get byte-identical data; different replicates must not."""
        module = self._module()
        calibration = self._calibration()
        recovery = self._recovery(steps=3)
        first, _ = module._recovery_slice(calibration, recovery, 0)
        again, _ = module._recovery_slice(calibration, recovery, 0)
        other, _ = module._recovery_slice(calibration, recovery, 1)
        import torch

        assert all(torch.equal(a, b) for a, b in zip(first, again, strict=True))
        assert not all(torch.equal(a, b) for a, b in zip(first, other, strict=True))
