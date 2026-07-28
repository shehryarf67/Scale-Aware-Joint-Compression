"""The compression interface: abstract methods, declared pipelines, and placeholder behaviour.

No model is loaded. These tests check the *shape* of the interface — that every arm implements the
same five stages, that the sequential and joint pipelines are the stage sequences the study
specifies, and that unimplemented stages fail loudly rather than returning something plausible.
"""

from __future__ import annotations

import inspect

import pytest

from scale_aware_compression.compression import (
    COMPRESSOR_REGISTRY,
    CompressionError,
    Compressor,
    JointArm,
    JointCompressor,
    LayerwiseArm,
    Pruner,
    PruningArm,
    QuantisationArm,
    Quantiser,
    SequentialArm,
    SequentialCompressor,
    SequentialQPArm,
    get_compressor,
)
from scale_aware_compression.compression.base import CompressionResult, StageRecord
from scale_aware_compression.compression.masks import MaskSet, MaskStatistics
from scale_aware_compression.compression.schedules import (
    ScheduleError,
    is_mask_update_step,
    mask_freeze_step,
    schedule_values,
    sparsity_at_step,
)
from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import (
    JOINT_STAGES,
    SEQUENTIAL_STAGES,
    CompressionMethod,
    CompressionStage,
)

STAGE_METHODS = ("prepare", "apply", "recover", "convert", "report_statistics")
ARMS = (Pruner, Quantiser, SequentialCompressor, JointCompressor)


@pytest.fixture
def config() -> ExperimentConfig:
    """A sequential-arm config, valid for constructing any compressor."""
    return ExperimentConfig.from_mapping(
        {
            "experiment": {"id": "interface-test"},
            "compression": {
                "method": "sequential",
                "pruning": {"enabled": True, "sparsity": 0.5},
                "quantisation": {"enabled": True, "bits": 8},
                "recovery": {"enabled": True, "max_steps": 100},
                "joint": {"joint_max_steps": 100},
            },
        }
    )


class TestAbstractBase:
    def test_base_class_cannot_be_instantiated(self, config: ExperimentConfig):
        with pytest.raises(TypeError):
            Compressor(config)  # type: ignore[abstract]

    @pytest.mark.parametrize("name", STAGE_METHODS)
    def test_stage_is_abstract_on_the_base(self, name: str):
        assert name in Compressor.__abstractmethods__

    def test_save_is_concrete(self):
        """Persistence is identical across arms, so it lives on the base class."""
        assert "save" not in Compressor.__abstractmethods__
        assert callable(Compressor.save)

    def test_run_drives_the_stages(self):
        assert callable(Compressor.run)

    def test_a_subclass_missing_a_stage_cannot_be_instantiated(self, config: ExperimentConfig):
        class Incomplete(Compressor):
            def prepare(self, model):  # type: ignore[no-untyped-def]
                return model

        with pytest.raises(TypeError):
            Incomplete(config)  # type: ignore[abstract]


class TestArmsImplementTheInterface:
    @pytest.mark.parametrize("arm", ARMS)
    def test_arm_is_concrete(self, arm: type[Compressor], config: ExperimentConfig):
        assert not arm.__abstractmethods__
        assert isinstance(arm(config), Compressor)

    @pytest.mark.parametrize("arm", ARMS)
    @pytest.mark.parametrize("name", STAGE_METHODS)
    def test_arm_defines_every_stage(self, arm: type[Compressor], name: str):
        assert callable(getattr(arm, name))

    @pytest.mark.parametrize("arm", ARMS)
    def test_arm_declares_its_method_and_stages(self, arm: type[Compressor]):
        assert isinstance(arm.method, CompressionMethod)
        assert arm.pipeline_stages, f"{arm.__name__} declares no pipeline stages"

    @pytest.mark.parametrize("arm", ARMS)
    def test_stage_signatures_are_documented(self, arm: type[Compressor]):
        for name in STAGE_METHODS:
            method = getattr(arm, name)
            assert method.__doc__, f"{arm.__name__}.{name} has no docstring"

    @pytest.mark.parametrize("arm", ARMS)
    def test_arm_name_matches_its_method(self, arm: type[Compressor], config: ExperimentConfig):
        assert arm(config).name == arm.method.value


class TestPipelineDefinitions:
    """The two pipelines under comparison are declared as data, so they can be asserted here."""

    def test_sequential_stages_are_prune_recover_quantise_convert(self):
        assert SEQUENTIAL_STAGES == (
            CompressionStage.DENSE,
            CompressionStage.PRUNED,
            CompressionStage.RECOVERED,
            CompressionStage.QUANTISED,
            CompressionStage.CONVERTED,
        )

    def test_joint_stages_are_fakequant_gradualprune_finetune_convert(self):
        assert JOINT_STAGES == (
            CompressionStage.DENSE,
            CompressionStage.FAKE_QUANTISATION_PREPARED,
            CompressionStage.GRADUAL_PRUNING,
            CompressionStage.JOINTLY_FINE_TUNED,
            CompressionStage.CONVERTED,
        )

    def test_the_two_pipelines_differ(self):
        assert SEQUENTIAL_STAGES != JOINT_STAGES

    def test_both_pipelines_start_dense_and_end_converted(self):
        for pipeline in (SEQUENTIAL_STAGES, JOINT_STAGES):
            assert pipeline[0] is CompressionStage.DENSE
            assert pipeline[-1] is CompressionStage.CONVERTED

    def test_joint_prepares_fake_quantisation_before_pruning(self):
        """The ordering *is* the method: pruning must see quantised weights."""
        stages = list(JOINT_STAGES)
        assert stages.index(CompressionStage.FAKE_QUANTISATION_PREPARED) < stages.index(
            CompressionStage.GRADUAL_PRUNING
        )

    def test_sequential_recovers_before_quantising(self):
        stages = list(SEQUENTIAL_STAGES)
        assert stages.index(CompressionStage.RECOVERED) < stages.index(CompressionStage.QUANTISED)

    def test_arms_use_the_shared_pipeline_constants(self):
        assert SequentialCompressor.pipeline_stages is SEQUENTIAL_STAGES
        assert JointCompressor.pipeline_stages is JOINT_STAGES


class TestRegistry:
    def test_registry_covers_every_non_dense_method(self):
        """Every method except DENSE must be runnable, including the Q->P reverse ablation."""
        assert set(COMPRESSOR_REGISTRY) == {
            CompressionMethod.PRUNING,
            CompressionMethod.QUANTISATION,
            CompressionMethod.SEQUENTIAL,
            CompressionMethod.SEQUENTIAL_QP,
            CompressionMethod.JOINT,
        }

    def test_every_registered_arm_is_a_layerwise_arm(self):
        """All five arms must share one driver, or §3.8 stops being checkable in code.

        The older per-arm classes were written for the superseded fine-tuning design. They are still
        importable so nothing breaks, but registering one would make that arm run a different
        algorithm from its peers.
        """
        for method, arm in COMPRESSOR_REGISTRY.items():
            assert issubclass(arm, LayerwiseArm), f"{method.value} is not a layerwise arm"

    def test_each_arm_declares_a_distinct_call_order(self):
        """The arm name is the only thing that distinguishes them, so it must be unique."""
        names = [arm.arm for arm in COMPRESSOR_REGISTRY.values()]
        assert len(set(names)) == len(names)
        assert all(names)

    def test_dense_has_no_compressor(self, config: ExperimentConfig):
        config.compression.method = CompressionMethod.DENSE
        assert get_compressor(config) is None

    @pytest.mark.parametrize(
        ("method", "expected"),
        [
            (CompressionMethod.PRUNING, PruningArm),
            (CompressionMethod.QUANTISATION, QuantisationArm),
            (CompressionMethod.SEQUENTIAL, SequentialArm),
            (CompressionMethod.SEQUENTIAL_QP, SequentialQPArm),
            (CompressionMethod.JOINT, JointArm),
        ],
    )
    def test_returns_the_right_arm(
        self, config: ExperimentConfig, method: CompressionMethod, expected: type[Compressor]
    ):
        config.compression.method = method
        compressor = get_compressor(config)
        assert isinstance(compressor, expected)

    def test_compression_error_is_an_exception(self):
        assert issubclass(CompressionError, RuntimeError)


class TestPlaceholdersFailLoudly:
    """A placeholder must raise, not return a plausible model.

    A silent no-op would produce an excellent-looking compression result.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_prepare_raises_not_implemented(self, arm: type[Compressor], config: ExperimentConfig):
        with pytest.raises(NotImplementedError):
            arm(config).prepare(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("arm", ARMS)
    def test_apply_raises_not_implemented(self, arm: type[Compressor], config: ExperimentConfig):
        with pytest.raises(NotImplementedError):
            arm(config).apply(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("arm", ARMS)
    def test_convert_raises_not_implemented(self, arm: type[Compressor], config: ExperimentConfig):
        with pytest.raises(NotImplementedError):
            arm(config).convert(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_error_points_at_the_module_to_implement(
        self, arm: type[Compressor], config: ExperimentConfig
    ):
        with pytest.raises(NotImplementedError, match=r"compression/\w+\.py"):
            arm(config).prepare(object())  # type: ignore[arg-type]

    def test_masks_placeholders_raise(self):
        from scale_aware_compression.compression import masks

        with pytest.raises(NotImplementedError, match="masks.py"):
            masks.build_masks({}, sparsity=0.5)


class TestReportStatistics:
    """report_statistics is implemented, not a placeholder.

    It is what reveals whether an eventual implementation actually compressed anything.
    """

    @pytest.mark.parametrize("arm", ARMS)
    def test_reports_without_a_model(self, arm: type[Compressor], config: ExperimentConfig):
        statistics = arm(config).report_statistics(None)
        assert statistics["method"] == arm.method.value
        assert "target_sparsity" in statistics
        assert "target_bits" in statistics
        assert "declared_stages" in statistics

    def test_pruner_reports_its_schedule(self, config: ExperimentConfig):
        statistics = Pruner(config).report_statistics(None)
        assert statistics["schedule"] == "cubic"
        assert statistics["final_scheduled_sparsity"] == pytest.approx(0.5)

    def test_quantiser_reports_conversion_state(self, config: ExperimentConfig):
        statistics = Quantiser(config).report_statistics(None)
        assert statistics["bits"] == 8
        assert statistics["is_converted"] is False

    def test_sequential_reports_both_sub_arms(self, config: ExperimentConfig):
        statistics = SequentialCompressor(config).report_statistics(None)
        assert statistics["pipeline"] == [stage.value for stage in SEQUENTIAL_STAGES]
        assert statistics["pruning"]["method"] == "pruning"
        assert statistics["quantisation"]["method"] == "quantisation"

    def test_joint_reports_the_budget_matching_flag(self, config: ExperimentConfig):
        statistics = JointCompressor(config).report_statistics(None)
        assert statistics["match_sequential_budget"] is True
        assert statistics["optimiser_steps"] == 0
        assert statistics["fake_quantisation"] is True

    def test_measured_sparsity_is_reported_when_a_model_is_given(
        self, config: ExperimentConfig, half_sparse_module
    ):
        statistics = Pruner(config).report_statistics(half_sparse_module)
        assert statistics["measured_sparsity_percentage"] == pytest.approx(50.0)
        assert statistics["measured_total_parameters"] == 10


class TestStageBookkeeping:
    def test_record_stage_appends(self, config: ExperimentConfig):
        pruner = Pruner(config)
        record = pruner.record_stage(
            CompressionStage.PRUNED, 1.5, optimiser_steps=10, extra="value"
        )
        assert isinstance(record, StageRecord)
        assert pruner.stage_records == [record]
        assert record.details == {"extra": "value"}

    def test_result_totals_sum_across_stages(self):
        result = CompressionResult(
            method=CompressionMethod.JOINT,
            model=None,
            stages=[
                StageRecord(CompressionStage.PRUNED, 1.0, optimiser_steps=100),
                StageRecord(CompressionStage.JOINTLY_FINE_TUNED, 2.0, optimiser_steps=400),
            ],
        )
        assert result.total_optimiser_steps == 500
        assert result.total_duration_seconds == pytest.approx(3.0)
        assert result.stage_sequence == (
            CompressionStage.PRUNED,
            CompressionStage.JOINTLY_FINE_TUNED,
        )

    def test_result_serialises_without_the_model(self):
        payload = CompressionResult(method=CompressionMethod.PRUNING, model=object()).to_dict()
        assert "model" not in payload
        assert payload["method"] == "pruning"

    def test_arms_declare_distinct_apply_stages(self, config: ExperimentConfig):
        """The stage log must distinguish a pruning apply from a quantising one."""
        assert Pruner.apply_stage is CompressionStage.PRUNED
        assert Quantiser.apply_stage is CompressionStage.QUANTISED
        assert JointCompressor.apply_stage is CompressionStage.GRADUAL_PRUNING


class TestMaskBookkeeping:
    def test_empty_mask_set(self):
        mask_set = MaskSet()
        assert len(mask_set) == 0
        assert mask_set.total_statistics().sparsity_percentage == 0.0

    def test_aggregates_across_modules(self):
        mask_set = MaskSet(target_sparsity=0.5)
        mask_set.add("layer.0", object(), MaskStatistics(total_elements=100, pruned_elements=50))
        mask_set.add("layer.1", object(), MaskStatistics(total_elements=100, pruned_elements=60))
        total = mask_set.total_statistics()
        assert total.total_elements == 200
        assert total.pruned_elements == 110
        assert total.sparsity_percentage == pytest.approx(55.0)
        assert total.kept_elements == 90

    def test_report_includes_per_module_sparsity(self):
        mask_set = MaskSet(target_sparsity=0.5)
        mask_set.add("layer.0", object(), MaskStatistics(total_elements=10, pruned_elements=5))
        report = mask_set.report()
        assert report["target_sparsity"] == 0.5
        assert report["num_masked_modules"] == 1
        assert report["per_module_sparsity_percentage"]["layer.0"] == pytest.approx(50.0)


class TestSchedules:
    """Both arms share these, so a divergence would silently change the comparison."""

    @pytest.mark.parametrize("schedule", ["linear", "cubic"])
    def test_starts_at_initial_and_ends_at_final(self, schedule: str):
        common = {
            "schedule": schedule,
            "final_sparsity": 0.7,
            "initial_sparsity": 0.1,
            "start_step": 0,
            "end_step": 100,
        }
        assert sparsity_at_step(0, **common) == pytest.approx(0.1)
        assert sparsity_at_step(100, **common) == pytest.approx(0.7)

    @pytest.mark.parametrize("schedule", ["linear", "cubic"])
    def test_is_monotone_non_decreasing(self, schedule: str):
        values = [
            sparsity_at_step(
                step, schedule=schedule, final_sparsity=0.7, initial_sparsity=0.0, end_step=100
            )
            for step in range(0, 101, 5)
        ]
        assert values == sorted(values)

    def test_linear_midpoint(self):
        assert sparsity_at_step(
            50, schedule="linear", final_sparsity=0.8, initial_sparsity=0.0, end_step=100
        ) == pytest.approx(0.4)

    def test_cubic_removes_faster_early_than_linear(self):
        """The cubic ramp is the standard gradual-pruning schedule for exactly this reason."""
        common = {"final_sparsity": 0.8, "initial_sparsity": 0.0, "end_step": 100}
        cubic = sparsity_at_step(25, schedule="cubic", **common)
        linear = sparsity_at_step(25, schedule="linear", **common)
        assert cubic > linear

    def test_holds_after_the_end_step(self):
        assert sparsity_at_step(
            500, schedule="cubic", final_sparsity=0.6, end_step=100
        ) == pytest.approx(0.6)

    def test_one_shot_applies_the_target_immediately(self):
        assert sparsity_at_step(
            0, schedule="one_shot", final_sparsity=0.5, end_step=100
        ) == pytest.approx(0.5)

    def test_constant_holds_the_target_throughout(self):
        for step in (0, 50, 100):
            assert sparsity_at_step(
                step, schedule="constant", final_sparsity=0.5, end_step=100
            ) == pytest.approx(0.5)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"final_sparsity": 1.0},
            {"final_sparsity": -0.1},
            {"final_sparsity": 0.3, "initial_sparsity": 0.5},
        ],
    )
    def test_invalid_bounds_raise(self, kwargs: dict[str, float]):
        with pytest.raises(ScheduleError):
            sparsity_at_step(0, schedule="cubic", **kwargs)  # type: ignore[arg-type]

    def test_negative_step_raises(self):
        with pytest.raises(ScheduleError, match="step"):
            sparsity_at_step(-1, schedule="cubic", final_sparsity=0.5)

    def test_schedule_values_spans_the_window(self):
        points = schedule_values(
            schedule="cubic", final_sparsity=0.5, start_step=0, end_step=100, num_points=5
        )
        assert len(points) == 5
        assert points[0][0] == 0
        assert points[-1][0] == 100
        assert points[-1][1] == pytest.approx(0.5)

    def test_schedule_values_needs_two_points(self):
        with pytest.raises(ScheduleError, match="num_points"):
            schedule_values(schedule="cubic", final_sparsity=0.5, num_points=1)


class TestMaskUpdateCadence:
    def test_updates_on_the_frequency(self):
        assert is_mask_update_step(0, frequency=50, end_step=200)
        assert is_mask_update_step(50, frequency=50, end_step=200)
        assert not is_mask_update_step(51, frequency=50, end_step=200)

    def test_always_updates_at_the_end_step(self):
        """So the final mask matches the final target exactly."""
        assert is_mask_update_step(175, frequency=50, end_step=175)

    def test_no_updates_outside_the_window(self):
        assert not is_mask_update_step(5, frequency=50, start_step=10, end_step=100)
        assert not is_mask_update_step(200, frequency=50, start_step=10, end_step=100)

    def test_non_positive_frequency_raises(self):
        with pytest.raises(ScheduleError, match="frequency"):
            is_mask_update_step(0, frequency=0)

    def test_freeze_step_is_a_fraction_of_training(self):
        assert mask_freeze_step(total_steps=500, freeze_after_ratio=0.8) == 400

    def test_freeze_ratio_of_one_never_freezes_early(self):
        assert mask_freeze_step(total_steps=500, freeze_after_ratio=1.0) == 500

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"total_steps": 0, "freeze_after_ratio": 0.8},
            {"total_steps": 100, "freeze_after_ratio": 1.5},
        ],
    )
    def test_invalid_freeze_arguments_raise(self, kwargs: dict[str, float]):
        with pytest.raises(ScheduleError):
            mask_freeze_step(**kwargs)  # type: ignore[arg-type]


class TestNoImportTimeSideEffects:
    def test_importing_compression_does_not_import_torch(self, imported_after):
        assert imported_after("scale_aware_compression.compression", ["torch"]) == [], (
            "importing the compression subpackage must not import torch: heavy imports are lazy "
            "so config validation and the test suite stay fast"
        )

    def test_stage_methods_take_no_action_at_definition_time(self):
        """Sanity check that the placeholders are functions, not evaluated expressions."""
        for arm in ARMS:
            for name in STAGE_METHODS:
                assert inspect.isfunction(getattr(arm, name))
