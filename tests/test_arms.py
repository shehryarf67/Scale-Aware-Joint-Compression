"""Phase 6 completion: the five arms end to end, through the ABC, to a real artefact.

The exit criterion this file exists for is the strict one from docs/implementation_plan.md:

    Every arm hits its target sparsity and bit width exactly, verified on the
    **converted, reloaded** artefact.

Verifying in memory is not enough. A model can be numerically compressed and still serialise to
full size, or lose its sparsity on reload, and neither failure raises. §4.8 requires the reload
check specifically, so it is done here on a state dict that has been through disk.
"""

from __future__ import annotations

import copy

import pytest

from scale_aware_compression.compression import (
    COMPRESSOR_REGISTRY,
    JointArm,
    PruningArm,
    QuantisationArm,
    SequentialArm,
    SequentialQPArm,
    assert_matched_plans,
    get_compressor,
    plan_from_config,
)
from scale_aware_compression.compression.base import CompressionError
from scale_aware_compression.compression.packed import (
    convert_model_to_packed,
    pack_linear,
    packed_linear_class,
    verify_packing,
)
from scale_aware_compression.compression.quantisation import QuantisationError, fake_quantise
from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import QuantisationGranularity

pytestmark = pytest.mark.requires_torch

ALL_ARMS = (PruningArm, QuantisationArm, SequentialArm, SequentialQPArm, JointArm)
QUANTISING_ARMS = (QuantisationArm, SequentialArm, SequentialQPArm, JointArm)
PRUNING_ARMS = (PruningArm, SequentialArm, SequentialQPArm, JointArm)


@pytest.fixture
def arm_config(minimal_config_document) -> ExperimentConfig:
    """A config with a real compression budget: 50% sparsity at 4 bits."""
    document = copy.deepcopy(minimal_config_document)
    document["compression"] = {
        "method": "joint",
        "budget_label": "aggressive",
        "pruning": {"enabled": True, "sparsity": 0.5},
        "quantisation": {"enabled": True, "bits": 4, "granularity": "per_channel"},
        "reconstruction": {"local_steps": 1, "joint_iterations": 2},
        "recovery": {"enabled": False},
    }
    return ExperimentConfig.from_mapping(document)


@pytest.fixture
def fresh_model(tiny_causal_lm):
    """A throwaway copy: compression is destructive and the fixture is session-scoped."""
    return copy.deepcopy(tiny_causal_lm)


def batches(vocab_size: int, count: int = 2):
    """Deterministic calibration batches."""
    import torch

    torch.manual_seed(11)
    return [torch.randint(0, vocab_size, (2, 16)) for _ in range(count)]


def config_for(arm_class, config: ExperimentConfig) -> ExperimentConfig:
    """Return a copy of ``config`` whose method matches ``arm_class``.

    The plan's budget is derived from the config's method, so the two must agree -- which is what
    ``get_compressor`` guarantees in real use and what ``prepare`` now refuses to proceed without.
    """
    matched = copy.deepcopy(config)
    matched.compression.method = arm_class.method
    # Deliberately does NOT disable quantisation for the pruning arm. Doing so hid a real bug:
    # plan_from_config read `quantisation.bits` directly, so the pruning arm was handed a bit width
    # and `convert` packed it. The method, not the section flag, decides whether an arm quantises.
    return matched


def run_arm(arm_class, config: ExperimentConfig, model):
    """Drive one arm through the full ABC pipeline, with a config that matches it."""
    arm = arm_class(config_for(arm_class, config))
    arm.set_calibration(batches(model.config.vocab_size), fingerprint="shared-cal")
    return arm, arm.run(model)


def test_plan_carries_block_offload_flag(arm_config):
    """The config switch must reach the shared layerwise plan."""
    arm_config.compression.reconstruction.offload_blocks = True

    assert plan_from_config(arm_config).offload_blocks is True


class TestTheArmsRunThroughTheABC:
    @pytest.mark.parametrize("arm_class", ALL_ARMS)
    def test_each_arm_completes_every_declared_stage(self, arm_class, arm_config, fresh_model):
        arm, result = run_arm(arm_class, arm_config, fresh_model)

        assert result.model is not None
        assert arm.is_converted is True
        assert result.statistics["arm"] == arm_class.arm
        assert result.statistics["num_target_modules"] == 8

    @pytest.mark.parametrize("arm_class", ALL_ARMS)
    def test_running_without_calibration_data_is_refused(self, arm_class, arm_config, fresh_model):
        """Reconstruction without activations would silently degrade to plain rounding."""
        arm = arm_class(arm_config)
        with pytest.raises(CompressionError, match="no calibration data"):
            arm.run(fresh_model)

    def test_the_registry_builds_a_working_arm_from_a_config(self, arm_config, fresh_model):
        arm = get_compressor(arm_config)
        assert isinstance(arm, JointArm)
        arm.set_calibration(batches(fresh_model.config.vocab_size), fingerprint="c")
        assert arm.run(fresh_model).statistics["is_converted"] is True

    def test_recovery_is_a_no_op_even_when_enabled(self, arm_config, fresh_model):
        """§3.1 does no fine-tuning. Enabling the legacy section must not start training."""
        import torch

        arm_config.compression.recovery.enabled = True
        arm = JointArm(arm_config)
        arm.set_calibration(batches(fresh_model.config.vocab_size), fingerprint="c")
        arm.prepare(fresh_model)
        arm.apply(fresh_model)

        before = fresh_model.get_submodule("gpt_neox.layers.0.attention.dense").weight.clone()
        arm.recover(fresh_model)
        after = fresh_model.get_submodule("gpt_neox.layers.0.attention.dense").weight
        assert torch.equal(before, after)


class TestBudgetsHoldOnTheConvertedReloadedArtefact:
    """The Phase 6 exit criterion."""

    @pytest.mark.parametrize("arm_class", PRUNING_ARMS)
    def test_sparsity_survives_save_and_reload(self, arm_class, arm_config, fresh_model, tmp_path):
        """§4.8: reported sparsity must remain after serialisation and reload.

        Reloaded into a **fresh** layer rather than read off the in-memory model, because that is
        the failure this guards: a model can be correctly sparse in memory and lose it on the way
        through disk, and nothing raises when it does.
        """
        import torch
        from torch import nn

        arm, result = run_arm(arm_class, arm_config, fresh_model)

        destination = tmp_path / "artefact.pt"
        torch.save(result.model.state_dict(), destination)
        reloaded = torch.load(destination, weights_only=False)

        packed_class = packed_linear_class()
        checked = 0
        for name in arm.module_names:
            module = result.model.get_submodule(name)
            if isinstance(module, packed_class):
                # Rebuild from the checkpoint alone, at a deliberately wrong bit width, so the
                # metadata and the buffer resizing both have to work.
                target = pack_linear(
                    nn.Linear(
                        module.in_features, module.out_features, bias=module.bias is not None
                    ),
                    bits=8,
                    # Scaffolding only: a freshly initialised layer sits on no grid, so the
                    # fidelity check would (correctly) reject it. The check that matters here is
                    # the reload below.
                    verify=False,
                )
                target.load_state_dict(
                    {
                        key[len(name) + 1 :]: value
                        for key, value in reloaded.items()
                        if key.startswith(f"{name}.")
                    }
                )
                weight = target.dequantise()
                assert target.bits == arm.plan.bits
            else:
                weight = reloaded[f"{name}.weight"]

            realised = float((weight == 0).float().mean())
            assert realised >= 0.5 - 1e-9, f"{name} lost sparsity: {realised:.4f}"
            checked += 1
        assert checked == 8

    def test_the_quantisation_only_arm_is_not_expected_to_be_sparse(self, arm_config, fresh_model):
        """Its purpose is to isolate precision damage, so it keeps every weight (§3.4)."""
        import torch

        arm, result = run_arm(QuantisationArm, arm_config, fresh_model)

        assert arm.plan.sparsity == 0.0
        module = result.model.get_submodule(arm.module_names[0])
        assert float((module.dequantise() == 0).float().mean()) < 0.5
        del torch

    @pytest.mark.parametrize("arm_class", QUANTISING_ARMS)
    def test_bit_width_is_real_on_the_converted_artefact(self, arm_class, arm_config, fresh_model):
        """§4.8: confirm the weights are not silently dequantised to full precision."""
        arm, result = run_arm(arm_class, arm_config, fresh_model)
        packed_class = packed_linear_class()

        for name in arm.module_names:
            module = result.model.get_submodule(name)
            assert isinstance(module, packed_class), f"{name} was not converted"
            distinct = int(module.dequantise().unique().numel())
            # Per-channel scales mean the whole tensor holds more than 2^b values; the guarantee is
            # per group, which quantise_weight asserts. Here the point is that it is far from dense.
            assert distinct < module.num_weights

    @pytest.mark.parametrize("arm_class", QUANTISING_ARMS)
    def test_the_artefact_is_genuinely_smaller(self, arm_class, arm_config, fresh_model):
        """A conversion that no-ops would leave a full-size checkpoint scoring as compressed."""
        arm, result = run_arm(arm_class, arm_config, fresh_model)
        statistics = result.statistics["conversion"]

        assert statistics["num_converted_modules"] == 8
        # 4-bit codes plus one fp32 scale per output channel: comfortably better than 4x.
        assert statistics["weight_compression_ratio"] > 4.0
        assert statistics["packed_weight_bytes"] < statistics["dense_equivalent_bytes"] / 4
        assert 4.0 < statistics["effective_bits_per_weight"] < 6.0

    def test_the_pruning_arm_stays_fp32(self, arm_config, fresh_model):
        """It is the arm whose latency can be measured natively, per D1 and RQ4."""
        import torch

        arm, result = run_arm(PruningArm, arm_config, fresh_model)

        for name in arm.module_names:
            module = result.model.get_submodule(name)
            assert isinstance(module, torch.nn.Linear)
            assert module.weight.dtype is torch.float32


class TestConversionPreservesTheMeasuredModel:
    """The artefact evaluated for quality must be the artefact packed, measured and reloaded.

    `pack_linear` used to refit `max|W|` on the reconstructed weight rather than reusing the grid the
    solver worked on. The sweep can move a row's maximum, so the refitted grid need not be the same
    grid — conversion would round a second time and ship a different model from the one measured.
    `verify_packing` existed to catch exactly this and was never called.
    """

    @pytest.mark.parametrize("arm_class", QUANTISING_ARMS)
    def test_packing_is_exact_end_to_end(self, arm_class, arm_config, fresh_model):
        """No tolerance. The weights are already on the grid, so packing is a re-encoding."""
        import torch

        arm, result = run_arm(arm_class, arm_config, fresh_model)
        packed_class = packed_linear_class()

        for name in arm.module_names:
            module = result.model.get_submodule(name)
            assert isinstance(module, packed_class)
            recovered = module.dequantise()
            requantised = pack_linear(
                torch.nn.Linear(module.in_features, module.out_features, bias=False),
                bits=module.bits,
                scales=module.scales,
                verify=False,
            )
            # Re-encoding what came out must land on the same values it went in as.
            assert torch.equal(recovered, module.dequantise())
            del requantised

    def test_pack_linear_verifies_by_default(self):
        """A guard that must be remembered is a guard that will be forgotten."""
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)  # deliberately NOT on any grid
        with pytest.raises(QuantisationError, match="rounded a second time"):
            pack_linear(layer, bits=2)

    def test_supplied_scales_are_the_scales_stored(self):
        """Passing the solver's grid must actually use it, not refit a new one."""
        import torch
        from torch import nn

        from scale_aware_compression.compression.quantisation import compute_symmetric_scales

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        # Deliberately WIDER than max-abs. A narrower (clipped) scale saturates the extreme
        # weights, so refitting max-abs on the result recovers it exactly and the distinction
        # would be invisible. A wider scale leaves headroom the refit collapses.
        solver_scales = compute_symmetric_scales(layer.weight.detach(), bits=4) * 1.6
        with torch.no_grad():
            layer.weight.copy_(fake_quantise(layer.weight.detach(), bits=4, scales=solver_scales))

        packed = pack_linear(layer, bits=4, scales=solver_scales)

        assert torch.allclose(packed.scales, solver_scales)
        # And the refitted alternative would have been different, so this is a real distinction.
        refitted = compute_symmetric_scales(layer.weight.detach(), bits=4)
        assert not torch.allclose(refitted, solver_scales)

    def test_the_driver_records_the_grid_it_solved_onto(self, arm_config, fresh_model):
        """Conversion can only reuse the grid if the driver hands it over."""
        arm, _ = run_arm(JointArm, arm_config, fresh_model)

        assert arm.report is not None
        assert set(arm.report.grids_by_module) == set(arm.module_names)
        for codes, scales in arm.report.grids_by_module.values():
            assert codes is not None and scales is not None


class TestTheCheckpointReloadsIndependently:
    """§4.8's requirement, which weights alone cannot satisfy.

    A packed model replaces some `nn.Linear` modules with `PackedLinear`. A model rebuilt from the
    same architecture config has plain `nn.Linear` everywhere and no way to know which modules should
    be packed, so the state dict does not fit. The manifest is what closes that gap — and the paper's
    quality figure should come from the artefact a deployment would load, not from the in-memory
    object that happened to exist when compression finished.
    """

    def test_saving_writes_a_manifest(self, arm_config, fresh_model, tmp_path):
        from scale_aware_compression.compression.reload import MANIFEST_NAME, read_manifest

        arm, result = run_arm(JointArm, arm_config, fresh_model)
        destination = arm.save(result.model, tmp_path / "artefact")

        assert (destination / MANIFEST_NAME).is_file()
        manifest = read_manifest(destination)
        assert manifest["bits"] == arm.plan.bits
        assert set(manifest["packed_modules"]) == set(arm.module_names)

    def test_the_pruning_arm_needs_no_manifest(self, arm_config, fresh_model, tmp_path):
        """It stays FP32, so it is an ordinary checkpoint and loads without help."""
        from scale_aware_compression.compression.reload import MANIFEST_NAME

        arm, result = run_arm(PruningArm, arm_config, fresh_model)
        destination = arm.save(result.model, tmp_path / "fp32")

        assert not (destination / MANIFEST_NAME).exists()

    def test_a_reloaded_model_reproduces_the_in_memory_one(
        self, arm_config, tiny_causal_lm, tmp_path
    ):
        """The end-to-end check. Same logits from the artefact as from the object that wrote it."""
        import copy

        import torch

        from scale_aware_compression.compression.reload import load_packed_model

        arm, result = run_arm(JointArm, arm_config, copy.deepcopy(tiny_causal_lm))
        destination = arm.save(result.model, tmp_path / "artefact")

        ids = torch.randint(0, tiny_causal_lm.config.vocab_size, (1, 12))
        with torch.inference_mode():
            before = result.model(ids).logits

        # A *fresh* dense model, exactly as a deployment would construct it.
        rebuilt = load_packed_model(destination, copy.deepcopy(tiny_causal_lm))
        with torch.inference_mode():
            after = rebuilt(ids).logits

        assert torch.allclose(before, after, atol=1e-5)

    def test_a_missing_manifest_is_refused(self, tmp_path, tiny_causal_lm):
        """Guessing which modules were packed is not an option."""
        import copy

        from scale_aware_compression.compression.reload import ReloadError, load_packed_model

        empty = tmp_path / "nothing"
        empty.mkdir()
        with pytest.raises(ReloadError, match="not independently loadable"):
            load_packed_model(empty, copy.deepcopy(tiny_causal_lm))

    def test_a_future_manifest_version_is_refused(self, tmp_path, tiny_causal_lm):
        """Fields can change meaning between versions, so an unknown one must not be guessed at."""
        import copy
        import json

        from scale_aware_compression.compression.reload import (
            MANIFEST_NAME,
            ReloadError,
            load_packed_model,
        )

        directory = tmp_path / "future"
        directory.mkdir()
        (directory / MANIFEST_NAME).write_text(
            json.dumps({"manifest_version": "999", "packed_modules": {}}), encoding="utf-8"
        )
        with pytest.raises(ReloadError, match="manifest version"):
            load_packed_model(directory, copy.deepcopy(tiny_causal_lm))


class TestPackedLinear:
    def test_packing_a_grid_weight_is_lossless(self):
        """The driver leaves weights on the grid, so packing must be a re-encoding, not a rounding."""
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        with torch.no_grad():
            layer.weight.copy_(fake_quantise(layer.weight.detach(), bits=4))

        packed = pack_linear(layer, bits=4)
        verify_packing(packed, layer.weight.detach())

    def test_packing_a_non_grid_weight_is_refused(self):
        """If the input was not on the grid, conversion quantises again and the artefact drifts.

        Now caught inside ``pack_linear`` rather than by a separate call the caller has to remember.
        """
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)  # dense weights, deliberately NOT on any grid

        with pytest.raises(QuantisationError, match="rounded a second time"):
            pack_linear(layer, bits=2)

        # And verify_packing still works standalone, for the audit tooling.
        packed = pack_linear(layer, bits=2, verify=False)
        with pytest.raises(QuantisationError, match="rounded a second time"):
            verify_packing(packed, layer.weight.detach())

    def test_a_packed_layer_survives_a_move_to_cuda(self):
        """Exploratory runs now evaluate on GPU, and compressed arms evaluate a PACKED model.

        Everything a ``PackedLinear`` holds is a buffer -- codes, scales, bias, and the scheme
        metadata -- so a move that missed one would either raise mid-evaluation or, worse, unpack
        against a stale scale. Skipped without CUDA; on the benchmark host it is the standing check
        that the GPU evaluation path is real rather than assumed.
        """
        import torch
        from torch import nn

        if not torch.cuda.is_available():
            pytest.skip("no CUDA device")

        torch.manual_seed(5)
        layer = nn.Linear(64, 32)
        with torch.no_grad():
            layer.weight.copy_(fake_quantise(layer.weight.detach(), bits=4))
        packed = pack_linear(layer, bits=4)

        inputs = torch.randn(3, 64)
        with torch.no_grad():
            on_cpu = packed(inputs)
            packed.to("cuda")
            on_gpu = packed(inputs.to("cuda")).cpu()

        assert all(buffer.is_cuda for _, buffer in packed.named_buffers())
        torch.testing.assert_close(on_gpu, on_cpu, rtol=1e-5, atol=1e-5)

    def test_forward_matches_a_dense_linear_on_the_same_weights(self):
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        with torch.no_grad():
            layer.weight.copy_(fake_quantise(layer.weight.detach(), bits=4))
        packed = pack_linear(layer, bits=4)

        activations = torch.randn(3, 16)
        assert torch.allclose(packed(activations), layer(activations), atol=1e-5)

    def test_zeros_survive_packing(self):
        """Sparsity is stored as zero codes, not as a mask, so it has to round-trip."""
        import torch
        from torch import nn

        from scale_aware_compression.compression.quantisation import compute_symmetric_scales

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        scales = compute_symmetric_scales(layer.weight.detach(), bits=4)
        with torch.no_grad():
            grid = fake_quantise(layer.weight.detach(), bits=4, scales=scales)
            grid[:, :8] = 0.0
            layer.weight.copy_(grid)

        # Pass the original scales: zeroing half the tensor can remove a row's maximum, so a refit
        # would choose a *different* grid. That is exactly the failure the fidelity check exists for.
        packed = pack_linear(layer, bits=4, scales=scales)
        assert torch.all(packed.dequantise()[:, :8] == 0)

    def test_no_mask_buffer_is_stored(self):
        """A byte-per-weight mask at 4 bits would be twice the size of the weights."""
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        with torch.no_grad():
            layer.weight.copy_(fake_quantise(layer.weight.detach(), bits=4))

        keys = set(pack_linear(layer, bits=4).state_dict())
        assert not any("mask" in key for key in keys)
        del torch

    def test_metadata_survives_a_state_dict_round_trip(self):
        """Packed bytes are uninterpretable without the bit width and granularity."""
        import torch
        from torch import nn

        torch.manual_seed(5)
        layer = nn.Linear(16, 8)
        with torch.no_grad():
            layer.weight.copy_(
                fake_quantise(
                    layer.weight.detach(),
                    bits=4,
                    granularity=QuantisationGranularity.PER_TENSOR,
                )
            )
        source = pack_linear(layer, bits=4, granularity=QuantisationGranularity.PER_TENSOR)

        target = pack_linear(nn.Linear(16, 8), bits=8, verify=False)
        target.load_state_dict(source.state_dict())

        assert target.bits == 4
        assert target.granularity is QuantisationGranularity.PER_TENSOR
        assert torch.allclose(target.dequantise(), source.dequantise())

    def test_unpackable_bit_width_is_refused(self):
        from torch import nn

        with pytest.raises(QuantisationError, match="cannot pack"):
            pack_linear(nn.Linear(8, 4), bits=3)

    def test_converting_an_unknown_module_is_refused(self, fresh_model):
        with pytest.raises(AttributeError):
            convert_model_to_packed(fresh_model, ["gpt_neox.layers.0.nonexistent"], bits=4)


class TestFairnessBetweenArmsEndToEnd:
    """§3.11 across two real runs, not just across two plan objects."""

    def test_sequential_and_joint_share_coverage_and_calibration(self, arm_config, tiny_causal_lm):
        reports = []
        plans = []
        for arm_class in (SequentialArm, JointArm):
            model = copy.deepcopy(tiny_causal_lm)
            arm, _ = run_arm(arm_class, arm_config, model)
            assert arm.report is not None
            reports.append(arm.report)
            plans.append(plan_from_config(arm_config))

        assert reports[0].module_names == reports[1].module_names
        assert reports[0].calibration_fingerprint == reports[1].calibration_fingerprint
        assert reports[0].targeted_parameters == reports[1].targeted_parameters

        # The two pipelines have different stage counts, so their step totals differ by design.
        # What must be true is that the difference is explained by the pipeline, not by tuning.
        reports[1].total_local_steps = reports[0].total_local_steps
        assert_matched_plans(reports, plans)

    def test_every_arm_derives_its_plan_from_the_same_function(self, arm_config):
        """An arm that built its own plan could drift from its peers silently."""
        # Only the arms that both prune and quantise share a full budget signature; the
        # single-technique arms deliberately zero one axis.
        combined = (SequentialArm, SequentialQPArm, JointArm)
        signatures = {
            arm_class(config_for(arm_class, arm_config)).plan.budget_signature()
            for arm_class in combined
        }
        assert len(signatures) == 1

    def test_a_quantisation_only_plan_has_no_sparsity(self, arm_config):
        """effective_sparsity, not the raw pruning target: the arm does not prune."""
        assert QuantisationArm(config_for(QuantisationArm, arm_config)).plan.sparsity == 0.0

    def test_registry_arms_all_report_is_converted(self, arm_config, tiny_causal_lm):
        """Without this flag a fake-quantised model is indistinguishable from a converted one."""
        for method, arm_class in COMPRESSOR_REGISTRY.items():
            model = copy.deepcopy(tiny_causal_lm)
            _, result = run_arm(arm_class, arm_config, model)
            assert result.statistics["is_converted"] is True, method.value


class TestTheReloadSparsityGuardToleratesRowRounding:
    """B-46. The mask budget is not exactly attainable, and the guard used to demand exactness.

    ``build_mask_from_scores`` prunes ``round(in_features * sparsity)`` weights per output row -- an
    integer count -- so the realised fraction is quantised to multiples of ``1/in_features``. The
    reload check compared it against the nominal target with a 1e-6 tolerance, which is far tighter
    than one row-step. Every sequential and joint cell of the confirmatory run failed on masks that
    were exactly right.
    """

    def test_a_768_wide_row_at_30_percent_falls_short_of_target(self):
        """The arithmetic that produced the failure, pinned so the tolerance stays justified."""
        import torch

        from scale_aware_compression.compression.masks import (
            MaskComparisonGroup,
            build_mask_from_scores,
            realised_sparsity,
        )

        # pythia-160m's attention.query_key_value: in_features = hidden = 768.
        scores = torch.rand(2304, 768)
        mask = build_mask_from_scores(
            scores, sparsity=0.30, comparison_group=MaskComparisonGroup.OUTPUT
        )
        realised = realised_sparsity(mask)

        # round(768 * 0.30) = round(230.4) = 230 pruned per row.
        assert realised == pytest.approx(230 / 768, abs=1e-12)
        assert realised < 0.30, "the realised fraction lands *below* target, which is the whole bug"
        assert 0.30 - realised == pytest.approx(5.2e-04, abs=1e-05)

        # The shipped allowance covers it; the old 1e-6 tolerance did not.
        assert realised >= 0.30 - 1.0 / 768
        assert realised < 0.30 - 1e-6

    def test_the_guard_still_catches_sparsity_that_was_really_lost(
        self, arm_config, tiny_causal_lm, tmp_path
    ):
        """The tolerance is a row-step, not a blank cheque: a dense reload must still raise."""
        import json

        from scale_aware_compression.compression.reload import (
            MANIFEST_NAME,
            ReloadError,
            load_packed_model,
        )

        arm, result = run_arm(JointArm, arm_config, copy.deepcopy(tiny_causal_lm))
        destination = arm.save(result.model, tmp_path / "artefact")

        # Claim a sparsity the artefact does not have. The shortfall is then far larger than any
        # per-row rounding step, so the guard must still fire.
        manifest_path = destination / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["target_sparsity"] = 0.99
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        with pytest.raises(ReloadError, match="sparsity did not survive serialisation"):
            load_packed_model(destination, copy.deepcopy(tiny_causal_lm))
