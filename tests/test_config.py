"""Configuration loading, include resolution, overrides, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scale_aware_compression.config import (
    BenchmarkConfig,
    CompressionConfig,
    ConfigError,
    DataConfig,
    ExperimentConfig,
    JointConfig,
    PruningConfig,
    QuantisationConfig,
    RecoveryConfig,
    RunMeta,
    RuntimeConfig,
    apply_overrides,
    deep_merge,
    load_config,
    load_document,
    load_yaml,
)
from scale_aware_compression.constants import CompressionMethod, Device, DType, PruningScheduleName


class TestDefaults:
    def test_empty_document_yields_defaults(self):
        config = ExperimentConfig.from_mapping({})
        assert config.compression.method is CompressionMethod.DENSE
        assert config.benchmark.device is Device.CPU
        assert config.model.dtype is DType.FLOAT32

    def test_dense_baseline_has_no_effective_compression(self):
        config = ExperimentConfig.from_mapping({"compression": {"method": "dense"}})
        assert config.compression.effective_sparsity == 0.0
        assert config.compression.effective_bits == 32


class TestParsing:
    def test_builds_every_section(self, minimal_config_document: dict[str, Any]):
        config = ExperimentConfig.from_mapping(minimal_config_document)
        assert isinstance(config.experiment, RunMeta)
        assert isinstance(config.runtime, RuntimeConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.compression, CompressionConfig)
        assert isinstance(config.compression.pruning, PruningConfig)
        assert isinstance(config.compression.quantisation, QuantisationConfig)
        assert isinstance(config.compression.recovery, RecoveryConfig)
        assert isinstance(config.compression.joint, JointConfig)
        assert isinstance(config.benchmark, BenchmarkConfig)

    def test_coerces_enums_from_strings(self, minimal_config_document: dict[str, Any]):
        config = ExperimentConfig.from_mapping(minimal_config_document)
        assert config.compression.method is CompressionMethod.SEQUENTIAL
        assert config.compression.pruning.schedule is PruningScheduleName.CUBIC
        assert config.model.device is Device.CPU

    def test_coerces_enum_from_member_name(self):
        config = ExperimentConfig.from_mapping({"model": {"dtype": "BFLOAT16"}})
        assert config.model.dtype is DType.BFLOAT16

    def test_coerces_paths(self):
        config = ExperimentConfig.from_mapping({"runtime": {"output_dir": "some/where"}})
        assert isinstance(config.runtime.output_dir, Path)

    def test_optional_field_accepts_null(self):
        config = ExperimentConfig.from_mapping({"runtime": {"num_threads": None}})
        assert config.runtime.num_threads is None

    def test_unknown_key_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown configuration key"):
            ExperimentConfig.from_mapping({"runtime": {"nonexistent": 1}})

    def test_unknown_top_level_key_is_rejected(self):
        with pytest.raises(ConfigError, match="Unknown configuration key"):
            ExperimentConfig.from_mapping({"not_a_section": {}})

    def test_wrong_type_is_rejected(self):
        with pytest.raises(ConfigError, match="must be an integer"):
            ExperimentConfig.from_mapping({"runtime": {"seed": "not-a-number"}})

    def test_bool_is_not_accepted_as_int(self):
        with pytest.raises(ConfigError):
            ExperimentConfig.from_mapping({"data": {"batch_size": True}})

    def test_invalid_enum_lists_the_valid_values(self):
        with pytest.raises(ConfigError, match="is not one of"):
            ExperimentConfig.from_mapping({"compression": {"method": "magic"}})


class TestValidation:
    @pytest.mark.parametrize("sparsity", [-0.1, 1.0, 1.5])
    def test_sparsity_must_be_in_unit_interval(self, sparsity: float):
        with pytest.raises(ConfigError, match="sparsity"):
            ExperimentConfig.from_mapping({"compression": {"pruning": {"sparsity": sparsity}}})

    def test_initial_sparsity_cannot_exceed_target(self):
        with pytest.raises(ConfigError, match="initial_sparsity"):
            ExperimentConfig.from_mapping(
                {"compression": {"pruning": {"sparsity": 0.3, "initial_sparsity": 0.5}}}
            )

    def test_two_four_granularity_requires_half_sparsity(self):
        with pytest.raises(ConfigError, match="2:4"):
            ExperimentConfig.from_mapping(
                {"compression": {"pruning": {"granularity": "2:4", "sparsity": 0.7}}}
            )

    def test_unsupported_bit_width_is_rejected(self):
        with pytest.raises(ConfigError, match="bits"):
            ExperimentConfig.from_mapping({"compression": {"quantisation": {"bits": 7}}})

    def test_schedule_window_must_not_run_backwards(self):
        with pytest.raises(ConfigError, match="schedule_end_step"):
            ExperimentConfig.from_mapping(
                {"compression": {"pruning": {"schedule_start_step": 100, "schedule_end_step": 50}}}
            )

    def test_experiment_id_must_be_filename_safe(self):
        with pytest.raises(ConfigError, match="filename-safe"):
            ExperimentConfig.from_mapping({"experiment": {"id": "has spaces"}})

    def test_experiment_id_must_not_be_empty(self):
        with pytest.raises(ConfigError, match="non-empty"):
            ExperimentConfig.from_mapping({"experiment": {"id": "  "}})

    def test_experiment_name_defaults_to_id(self):
        config = ExperimentConfig.from_mapping({"experiment": {"id": "run-7"}})
        assert config.experiment.name == "run-7"

    def test_log_level_must_be_a_level(self):
        with pytest.raises(ConfigError, match="logging level"):
            ExperimentConfig.from_mapping({"runtime": {"log_level": "CHATTY"}})

    def test_log_level_is_upper_cased(self):
        config = ExperimentConfig.from_mapping({"runtime": {"log_level": "debug"}})
        assert config.runtime.log_level == "DEBUG"

    def test_sequential_requires_pruning_enabled(self):
        with pytest.raises(ConfigError, match="requires compression.pruning.enabled"):
            ExperimentConfig.from_mapping(
                {"compression": {"method": "sequential", "pruning": {"enabled": False}}}
            )

    def test_joint_requires_quantisation_enabled(self):
        with pytest.raises(ConfigError, match="quantisation.enabled"):
            ExperimentConfig.from_mapping(
                {"compression": {"method": "joint", "quantisation": {"enabled": False}}}
            )


class TestCpuOnlyBenchmarkPolicy:
    """The CPU-only deployment policy is enforced by the loader, not just documented."""

    @pytest.mark.parametrize("device", ["cuda", "auto"])
    def test_non_cpu_benchmark_device_is_rejected(self, device: str):
        with pytest.raises(ConfigError, match="CPU-only"):
            ExperimentConfig.from_mapping({"benchmark": {"device": device}})

    def test_measured_runs_must_allow_a_std_and_p95(self):
        with pytest.raises(ConfigError, match="measured_runs"):
            ExperimentConfig.from_mapping({"benchmark": {"measured_runs": 1}})

    def test_zero_warmup_is_permitted_but_discouraged(self):
        config = ExperimentConfig.from_mapping({"benchmark": {"warmup_runs": 0}})
        assert config.benchmark.warmup_runs == 0

    def test_negative_warmup_is_rejected(self):
        with pytest.raises(ConfigError, match="warmup_runs"):
            ExperimentConfig.from_mapping({"benchmark": {"warmup_runs": -1}})


class TestDeepMerge:
    def test_merges_nested_mappings(self):
        merged = deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}})
        assert merged == {"a": {"b": 1, "c": 3}}

    def test_replaces_lists_wholesale(self):
        merged = deep_merge({"a": [1, 2, 3]}, {"a": [4]})
        assert merged == {"a": [4]}

    def test_does_not_mutate_inputs(self):
        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 2}})
        assert base == {"a": {"b": 1}}


class TestIncludes:
    def test_include_is_merged_underneath_the_including_document(self, write_yaml, tmp_path: Path):
        write_yaml("base.yaml", {"runtime": {"seed": 1, "log_level": "INFO"}})
        write_yaml("child.yaml", {"include": ["./base.yaml"], "runtime": {"seed": 2}})
        document = load_document(tmp_path / "child.yaml")
        assert document["runtime"] == {"seed": 2, "log_level": "INFO"}
        assert "include" not in document

    def test_includes_are_merged_in_order(self, write_yaml, tmp_path: Path):
        write_yaml("first.yaml", {"runtime": {"seed": 1}})
        write_yaml("second.yaml", {"runtime": {"seed": 2}})
        write_yaml("child.yaml", {"include": ["./first.yaml", "./second.yaml"]})
        document = load_document(tmp_path / "child.yaml")
        assert document["runtime"]["seed"] == 2

    def test_include_accepts_a_bare_string(self, write_yaml, tmp_path: Path):
        write_yaml("base.yaml", {"runtime": {"seed": 5}})
        write_yaml("child.yaml", {"include": "./base.yaml"})
        assert load_document(tmp_path / "child.yaml")["runtime"]["seed"] == 5

    def test_include_cycle_is_detected(self, write_yaml, tmp_path: Path):
        write_yaml("a.yaml", {"include": ["./b.yaml"]})
        write_yaml("b.yaml", {"include": ["./a.yaml"]})
        with pytest.raises(ConfigError, match="cycle"):
            load_document(tmp_path / "a.yaml")

    def test_missing_include_target_is_reported(self, write_yaml, tmp_path: Path):
        write_yaml("child.yaml", {"include": ["./absent.yaml"]})
        with pytest.raises(ConfigError, match="not found"):
            load_document(tmp_path / "child.yaml")

    def test_non_string_include_entry_is_rejected(self, write_yaml, tmp_path: Path):
        write_yaml("child.yaml", {"include": [7]})
        with pytest.raises(ConfigError, match="must be a string"):
            load_document(tmp_path / "child.yaml")


class TestOverrides:
    def test_sets_a_nested_value(self):
        result = apply_overrides({"runtime": {"seed": 1}}, ["runtime.seed=99"])
        assert result["runtime"]["seed"] == 99

    def test_parses_values_as_yaml(self):
        result = apply_overrides({}, ["model.revision=null", "runtime.deterministic=false"])
        assert result["model"]["revision"] is None
        assert result["runtime"]["deterministic"] is False

    def test_parses_list_values(self):
        result = apply_overrides({}, ["sweep.seeds=[1,2,3]"])
        assert result["sweep"]["seeds"] == [1, 2, 3]

    def test_creates_missing_intermediate_keys(self):
        result = apply_overrides({}, ["a.b.c=1"])
        assert result["a"]["b"]["c"] == 1

    def test_malformed_override_is_rejected(self):
        with pytest.raises(ConfigError, match="dotted.key=value"):
            apply_overrides({}, ["no-equals-sign"])

    def test_empty_key_is_rejected(self):
        with pytest.raises(ConfigError, match="empty key"):
            apply_overrides({}, ["=5"])


class TestLoadYaml:
    def test_missing_file_is_reported(self, tmp_path: Path):
        with pytest.raises(ConfigError, match="not found"):
            load_yaml(tmp_path / "absent.yaml")

    def test_empty_file_yields_an_empty_mapping(self, tmp_path: Path):
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")
        assert load_yaml(path) == {}

    def test_non_mapping_document_is_rejected(self, tmp_path: Path):
        path = tmp_path / "list.yaml"
        path.write_text("- one\n- two\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="top-level mapping"):
            load_yaml(path)

    def test_malformed_yaml_is_reported(self, tmp_path: Path):
        path = tmp_path / "bad.yaml"
        path.write_text("key: [unclosed\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="Could not parse YAML"):
            load_yaml(path)


class TestRoundTrip:
    def test_load_then_save_then_reload_is_stable(self, config_file: Path, tmp_path: Path):
        original = load_config(config_file)
        saved = original.save(tmp_path / "resolved.yaml")
        reloaded = load_config(saved)
        assert reloaded.to_dict() == original.to_dict()

    def test_to_dict_is_yaml_serialisable(self, config_file: Path):
        import yaml

        payload = load_config(config_file).to_dict()
        # Enums and Paths must already be primitives, or safe_dump raises.
        assert yaml.safe_dump(payload)

    def test_describe_names_the_run(self, config_file: Path):
        description = load_config(config_file).describe()
        assert "unit-test" in description
        assert "sequential" in description

    def test_run_output_dir_nests_under_the_experiment_id(self, config_file: Path):
        config = load_config(config_file)
        assert config.run_output_dir.name == "unit-test"

    def test_overrides_apply_after_includes(self, config_file: Path):
        config = load_config(config_file, ["runtime.seed=7"])
        assert config.runtime.seed == 7


class TestShippedConfigs:
    """Every configuration in configs/ must load and validate."""

    def _yaml_files(self, directory: Path) -> list[Path]:
        return sorted(directory.glob("*.yaml"))

    def test_model_configs_parse(self, configs_dir: Path):
        files = self._yaml_files(configs_dir / "models")
        assert len(files) == 5, "expected one config per registered model"
        for path in files:
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.model.name
            assert config.model.size_label

    def test_compression_configs_parse(self, configs_dir: Path):
        files = self._yaml_files(configs_dir / "compression")
        assert len(files) == 4
        for path in files:
            ExperimentConfig.from_mapping(load_document(path))

    def test_evaluation_configs_parse(self, configs_dir: Path):
        for path in self._yaml_files(configs_dir / "evaluation"):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.benchmark.device is Device.CPU

    def test_shared_split_cannot_evaluate_past_the_reserved_prefix(
        self, minimal_config_document: dict
    ):
        """§4.1 requires calibration and evaluation disjoint.

        When both are drawn from the same split, ``data.max_eval_samples`` is the prefix reserved for
        evaluation and calibration comes only from beyond it. Evaluating more sequences than were
        reserved therefore walks straight into the calibration set -- silently, and in a way that
        inflates the evaluation score for every arm equally, so no comparison looks wrong.
        """
        import copy

        document = copy.deepcopy(minimal_config_document)
        document["data"].update(
            {"calibration_split": "validation", "eval_split": "validation", "max_eval_samples": 32}
        )
        document.setdefault("evaluation", {})["max_samples"] = 64

        with pytest.raises(ConfigError, match="disjoint"):
            ExperimentConfig.from_mapping(document)

    def test_shared_split_requires_a_reserved_prefix(self, minimal_config_document: dict):
        """A null cap on a shared split means nothing is reserved at all."""
        import copy

        document = copy.deepcopy(minimal_config_document)
        document["data"].update(
            {
                "calibration_split": "validation",
                "eval_split": "validation",
                "max_eval_samples": None,
            }
        )
        document.setdefault("evaluation", {})["max_samples"] = 16

        with pytest.raises(ConfigError, match="cannot be null"):
            ExperimentConfig.from_mapping(document)

    def test_separate_splits_may_set_the_caps_independently(self, minimal_config_document: dict):
        """With calibration on train and evaluation on validation the sets cannot overlap.

        The guard must not fire here, or it would forbid the arrangement every shipped config uses.
        """
        import copy

        document = copy.deepcopy(minimal_config_document)
        document["data"].update(
            {"calibration_split": "train", "eval_split": "validation", "max_eval_samples": 32}
        )
        document.setdefault("evaluation", {})["max_samples"] = 512

        config = ExperimentConfig.from_mapping(document)
        assert config.evaluation.max_samples == 512

    def test_experiment_configs_parse(self, configs_dir: Path):
        files = self._yaml_files(configs_dir / "experiments")
        assert {path.name for path in files} == {
            "pilot.yaml",
            "main_scale_sweep.yaml",
            "extended_scale_sweep.yaml",
            "qwen_validation.yaml",
            "screening.yaml",
            "screening_410m.yaml",
            # A1 step 6: both sequential orderings on validation, so the stronger baseline can be
            # frozen per (model, budget) before any test evaluation.
            "order_selection.yaml",
            # A1 step 7: re-checks the W8 order across paired draws, because it was frozen on a
            # +0.43 pp single-draw margin and the sign of that budget's joint gain depends on it.
            "order_selection_w8_replicates.yaml",
            # Diagnostic: resolves the cross-machine disagreement on 410M aggressive by measuring
            # how far the calibration draw moves that cell.
            "dispute_410m_aggressive.yaml",
            # Diagnostic: does the 160M joint gain survive replication, after F-26 showed the 410M
            # equivalent changes sign between draws?
            "replicate_160m_aggressive.yaml",
            # Verification: per-block GPU offload must reproduce F-23's S5 cell exactly. Residency
            # is not allowed to move a number, and that is measured rather than assumed.
            "verify_offload_160m.yaml",
            # Verification: the peak-memory measurement offload exists for. Compression only, so it
            # produces a GiB figure rather than a perplexity.
            "verify_offload_1b.yaml",
            # A1 step 7 at the third scale: both budgets and both sequential orders on Pythia-1B,
            # runnable only because of per-block offload (F-31).
            "screening_1b.yaml",
            # A1 step 8 §5.4: the quality-matched mechanistic control. 40% + W8 against the
            # aggressive 30% + W4 primary, to separate a precision-specific effect from a
            # compression-severity one. Secondary and never confirmatory.
            "s6_control.yaml",
            # Gap A5 (§4.7): prefill and decode timed separately at two prompt lengths, with IQR
            # and model-order rotation. FP32 arms only, per decision D1.
            "prefill_decode.yaml",
            # Gap A4 (§4.3): HellaSwag, PIQA and ARC-Easy via a pinned harness, with task versions
            # recorded (§4.8). GPU-evaluated and declared as such; `benchmark.device` stays CPU.
            "downstream.yaml",
        }
        for path in files:
            config = load_config(path)
            assert config.experiment.id

    def test_every_shipped_yaml_config_loads(self, configs_dir: Path):
        """Every YAML anywhere under configs/ must parse and validate.

        Catches a config that was edited but never loaded -- including one reachable only as an
        include target.
        """
        paths = sorted(configs_dir.rglob("*.yaml"))
        assert len(paths) >= 14, f"expected the full config set, found {len(paths)}"
        for path in paths:
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.benchmark.device is Device.CPU, f"{path.name} is not CPU-only"

    def test_compression_configs_declare_their_own_method(self, configs_dir: Path):
        expected = {
            "pruning.yaml": CompressionMethod.PRUNING,
            "quantisation.yaml": CompressionMethod.QUANTISATION,
            "sequential.yaml": CompressionMethod.SEQUENTIAL,
            "joint.yaml": CompressionMethod.JOINT,
        }
        for name, method in expected.items():
            config = ExperimentConfig.from_mapping(
                load_document(configs_dir / "compression" / name)
            )
            assert config.compression.method is method

    def test_sequential_and_joint_share_a_compression_budget(self, configs_dir: Path):
        """A mismatched budget would make joint gain meaningless, so it is asserted here."""
        sequential = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "sequential.yaml")
        )
        joint = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "joint.yaml")
        )
        assert sequential.compression.effective_sparsity == joint.compression.effective_sparsity
        assert sequential.compression.effective_bits == joint.compression.effective_bits

    def test_sequential_and_joint_share_an_optimisation_budget(self, configs_dir: Path):
        """A mismatched step count would confound joint gain with extra training."""
        sequential = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "sequential.yaml")
        )
        joint = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "joint.yaml")
        )
        assert joint.compression.joint.joint_max_steps == sequential.compression.recovery.max_steps

    def test_sequential_and_joint_share_a_sparsity_ramp(self, configs_dir: Path):
        """Both arms must ramp sparsity identically.

        Reaching the target at different steps would give one arm more recovery at full sparsity,
        making part of the measured gain a schedule artefact.
        """
        sequential = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "sequential.yaml")
        ).compression.pruning
        joint = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "joint.yaml")
        ).compression.pruning
        assert sequential.schedule is joint.schedule
        assert sequential.schedule_start_step == joint.schedule_start_step
        assert sequential.schedule_end_step == joint.schedule_end_step
        assert sequential.initial_sparsity == joint.initial_sparsity

    def test_joint_mask_freeze_lands_on_the_end_of_the_sparsity_ramp(self, configs_dir: Path):
        """Freezing masks before the ramp finishes would leave the arm short of its target."""
        from scale_aware_compression.compression.schedules import mask_freeze_step

        compression = ExperimentConfig.from_mapping(
            load_document(configs_dir / "compression" / "joint.yaml")
        ).compression
        total_steps = compression.joint.joint_max_steps
        assert total_steps is not None
        freeze = mask_freeze_step(
            total_steps=total_steps,
            freeze_after_ratio=compression.joint.freeze_masks_after_ratio,
        )
        assert freeze == compression.pruning.schedule_end_step

    def test_pilot_config_is_internally_budget_matched(self, configs_dir: Path):
        """The pilot validates the pipeline, so its own budgets must be consistent."""
        compression = load_config(configs_dir / "experiments" / "pilot.yaml").compression
        assert compression.joint.joint_max_steps == compression.recovery.max_steps

    def test_sweep_and_validation_use_identical_budgets(self, configs_dir: Path):
        """Differing budgets would make the external transfer check meaningless."""
        sweep = load_config(configs_dir / "experiments" / "main_scale_sweep.yaml")
        validation = load_config(configs_dir / "experiments" / "qwen_validation.yaml")
        assert sweep.sweep.budget_overrides == validation.sweep.budget_overrides
        assert sweep.sweep.methods == validation.sweep.methods
        # The replicate count, not the seed list. A1 §5.1 withdrew the seed axis as inert (F-15), so
        # matching on `seeds` would now pass with both empty while the two configs ran different
        # numbers of calibration draws -- which is exactly the mismatch this test exists to catch.
        assert sweep.sweep.replicates == validation.sweep.replicates
        assert sweep.sweep.seeds == validation.sweep.seeds

    def test_order_selection_uses_the_frozen_budgets(self, configs_dir: Path):
        """The order chosen on validation must apply to the baseline the confirmatory stage runs.

        A budget that drifted between the two files would freeze a winning sequential order for a
        budget nothing else uses. This exact failure mode -- a value copied into a second config and
        then left behind when the first changed -- has already happened twice here: once during the
        budget freeze and once when the seed axis was withdrawn.
        """
        sweep = load_config(configs_dir / "experiments" / "main_scale_sweep.yaml")
        order = load_config(configs_dir / "experiments" / "order_selection.yaml")

        for label in ("moderate", "aggressive"):
            expected = sweep.sweep.budget_overrides[label]["compression"]
            actual = order.sweep.budget_overrides[label]["compression"]
            assert actual["pruning"]["sparsity"] == expected["pruning"]["sparsity"], (
                f"{label} sparsity differs between main_scale_sweep and order_selection"
            )
            assert actual["quantisation"]["bits"] == expected["quantisation"]["bits"], (
                f"{label} bit width differs between main_scale_sweep and order_selection"
            )

    def test_order_selection_runs_both_sequential_orders(self, configs_dir: Path):
        """§3.6 and §6.1 require best-of {P->Q, Q->P}; running one order cannot select between them."""
        order = load_config(configs_dir / "experiments" / "order_selection.yaml")
        assert CompressionMethod.SEQUENTIAL in order.sweep.methods
        assert CompressionMethod.SEQUENTIAL_QP in order.sweep.methods

    def test_order_selection_stays_on_validation(self, configs_dir: Path):
        """Selecting on test would spend the confirmatory split on a method choice (A1 §5.3)."""
        order = load_config(configs_dir / "experiments" / "order_selection.yaml")
        assert order.data.eval_split == "validation"

    @pytest.mark.parametrize(
        "name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml", "qwen_validation.yaml"]
    )
    def test_confirmatory_configs_report_on_the_test_split(self, configs_dir: Path, name: str):
        """A1 §5.2: the headline must not be computed on the split that chose the budgets.

        Validation became a selection surface the moment budgets were picked by looking at it. This
        was missed when the replicate axis went in -- the confirmatory configs would have run eight
        paired draws and then reported them on the selection surface. Caught by the partner's
        phase7-close-phase8-setup branch, and pinned here so it cannot drift back.
        """
        assert load_config(configs_dir / "experiments" / name).data.eval_split == "test"

    @pytest.mark.parametrize(
        "name",
        [
            "screening.yaml",
            "screening_410m.yaml",
            "order_selection.yaml",
            "order_selection_w8_replicates.yaml",
            "dispute_410m_aggressive.yaml",
            "replicate_160m_aggressive.yaml",
        ],
    )
    def test_exploratory_configs_stay_off_the_test_split(self, configs_dir: Path, name: str):
        """The other half of the same rule, and the easier one to violate by copy-paste.

        Every selection decision -- budgets, sequential order, mechanistic controls -- happens on
        validation. A screening config that drifted onto test would burn the confirmatory split on a
        choice, which is exactly what reserving it was meant to prevent.
        """
        assert load_config(configs_dir / "experiments" / name).data.eval_split == "validation"


class TestSweepScope:
    """The main sweep is three models; 1.4B is confined to the extended sweep."""

    ALL_ARMS = [
        CompressionMethod.DENSE,
        CompressionMethod.PRUNING,
        CompressionMethod.QUANTISATION,
        CompressionMethod.SEQUENTIAL,
        CompressionMethod.JOINT,
    ]

    def _sweep(self, configs_dir: Path, name: str):
        return load_config(configs_dir / "experiments" / name).sweep

    def test_main_sweep_excludes_pythia_1_4b(self, configs_dir: Path):
        """1.4B may need different training settings, which would confound the scale trend."""
        models = self._sweep(configs_dir, "main_scale_sweep.yaml").models
        assert "pythia-1.4b" not in models
        assert models == ["pythia-160m", "pythia-410m", "pythia-1b"]

    def test_extended_sweep_includes_pythia_1_4b(self, configs_dir: Path):
        models = self._sweep(configs_dir, "extended_scale_sweep.yaml").models
        assert "pythia-1.4b" in models
        assert models == ["pythia-160m", "pythia-410m", "pythia-1b", "pythia-1.4b"]

    def test_extended_sweep_only_adds_the_optional_model(self, configs_dir: Path):
        """The extended sweep adds exactly one model and reorders nothing.

        A model present in the extended sweep but absent from the main one would join the trend
        without a validated pipeline behind it.
        """
        main = self._sweep(configs_dir, "main_scale_sweep.yaml").models
        extended = self._sweep(configs_dir, "extended_scale_sweep.yaml").models
        assert set(extended) - set(main) == {"pythia-1.4b"}
        assert extended[: len(main)] == main

    def test_extended_sweep_inherits_the_main_budgets_and_seeds(self, configs_dir: Path):
        """Relaxed settings would stop the 1.4B point being a comparable scale point."""
        main = self._sweep(configs_dir, "main_scale_sweep.yaml")
        extended = self._sweep(configs_dir, "extended_scale_sweep.yaml")
        assert extended.budgets == main.budgets
        assert extended.seeds == main.seeds
        assert extended.methods == main.methods
        assert extended.budget_overrides == main.budget_overrides

    def test_extended_sweep_does_not_relax_the_evaluation_or_benchmark_protocol(
        self, configs_dir: Path
    ):
        main = load_config(configs_dir / "experiments" / "main_scale_sweep.yaml")
        extended = load_config(configs_dir / "experiments" / "extended_scale_sweep.yaml")
        assert extended.data.sequence_length == main.data.sequence_length
        assert extended.evaluation.max_samples == main.evaluation.max_samples
        assert extended.benchmark.num_threads == main.benchmark.num_threads
        assert extended.benchmark.measured_runs == main.benchmark.measured_runs
        assert extended.benchmark.sequence_length == main.benchmark.sequence_length

    @pytest.mark.parametrize("name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml"])
    def test_phase_eight_reserves_the_test_split_for_final_evaluation(
        self, configs_dir: Path, name: str
    ):
        config = load_config(configs_dir / "experiments" / name)
        assert config.data.eval_split == "test"

    def test_sweep_seeds_select_paired_calibration_replicates(self, configs_dir: Path):
        """A replicate varies calibration data, while paired arms retain the same draw."""
        from scale_aware_compression.experiments.scale_sweep import (
            build_cell_config,
            build_sweep_plan,
        )

        config = load_config(configs_dir / "experiments" / "main_scale_sweep.yaml")
        plan = build_sweep_plan(config)
        calibration_seeds = set()
        for cell in plan.cells:
            cell_config = build_cell_config(config, cell)
            assert cell_config.data.calibration_seed == cell.seed
            calibration_seeds.add(cell_config.data.calibration_seed)
        assert calibration_seeds == set(config.sweep.seeds)

    @pytest.mark.parametrize("name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml"])
    def test_sweep_keeps_all_five_arms(self, configs_dir: Path, name: str):
        assert self._sweep(configs_dir, name).methods == self.ALL_ARMS

    @pytest.mark.parametrize("name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml"])
    def test_sweep_keeps_both_budgets(self, configs_dir: Path, name: str):
        assert self._sweep(configs_dir, name).budgets == ["moderate", "aggressive"]

    @pytest.mark.parametrize("name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml"])
    def test_sweep_runs_dense_first(self, configs_dir: Path, name: str):
        """Every other arm needs its model's dense perplexity as the retention reference."""
        assert self._sweep(configs_dir, name).methods[0] is CompressionMethod.DENSE

    @pytest.mark.parametrize("name", ["main_scale_sweep.yaml", "extended_scale_sweep.yaml"])
    def test_sweep_is_resumable(self, configs_dir: Path, name: str):
        assert self._sweep(configs_dir, name).skip_existing is True

    def test_sweep_budgets_are_the_ones_frozen_by_screening(self, configs_dir: Path):
        """The budgets frozen on 2026-07-29 by the Phase 7 grid (findings_log.md F-13).

        Pinned because §6.3 forbids revisiting the choice once results exist, and because the pair
        this replaced — 50% + W8 and 70% + W4 — measured **catastrophic** on Pythia-160M at 22.9% and
        0.8% retention. The budget is a controlled variable across scales, so the smallest model sets
        the ceiling for all three; if these values drift, the sweep silently changes meaning.

        The pair varies precision, not sparsity: both prune 30%. That is deliberate — 4-bit is the
        only regime where the joint mechanism is measurably live (0.46% mask divergence at W8 against
        8.86% at W4), so two 8-bit budgets could not detect the study's primary effect.
        """
        overrides = self._sweep(configs_dir, "main_scale_sweep.yaml").budget_overrides
        assert overrides["moderate"]["compression"]["pruning"]["sparsity"] == 0.3
        assert overrides["moderate"]["compression"]["quantisation"]["bits"] == 8
        assert overrides["aggressive"]["compression"]["pruning"]["sparsity"] == 0.3
        assert overrides["aggressive"]["compression"]["quantisation"]["bits"] == 4

    def test_the_frozen_budgets_differ_only_in_precision(self, configs_dir: Path):
        """Guards the reasoning, not just the numbers.

        If someone later 'fixes' the budgets so they differ in sparsity instead, the study loses its
        only 4-bit condition and becomes structurally unable to show a joint effect.
        """
        overrides = self._sweep(configs_dir, "main_scale_sweep.yaml").budget_overrides
        moderate = overrides["moderate"]["compression"]
        aggressive = overrides["aggressive"]["compression"]
        assert moderate["pruning"]["sparsity"] == aggressive["pruning"]["sparsity"]
        assert moderate["quantisation"]["bits"] != aggressive["quantisation"]["bits"]
        assert aggressive["quantisation"]["bits"] == 4, "the aggressive budget must keep 4-bit"


class TestPilotScope:
    """The pilot validates the pipeline and must not be able to launch the study."""

    @pytest.fixture
    def pilot(self, configs_dir: Path) -> ExperimentConfig:
        return load_config(configs_dir / "experiments" / "pilot.yaml")

    def test_uses_one_model(self, pilot: ExperimentConfig):
        assert pilot.model.name == "pythia-160m"
        # No sweep grid: this config describes exactly one run, so `sajc sweep` against it cannot
        # expand into the full study.
        assert pilot.sweep.models == []

    def test_uses_one_seed(self, pilot: ExperimentConfig):
        assert pilot.sweep.seeds == []
        assert isinstance(pilot.runtime.seed, int)

    def test_uses_one_budget(self, pilot: ExperimentConfig):
        assert pilot.compression.budget_label == "pilot"
        assert pilot.sweep.budgets == ["moderate"]
        assert pilot.sweep.budget_overrides == {}

    def test_expands_to_a_single_run(self, pilot: ExperimentConfig):
        """The strongest form of "must not launch the full study"."""
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        assert build_sweep_plan(pilot).num_runs == 1

    def test_uses_the_moderate_int8_budget(self, pilot: ExperimentConfig):
        """INT8, not 4-bit: a pilot should validate the pipeline, not discover a missing backend."""
        assert pilot.compression.effective_sparsity == 0.5
        assert pilot.compression.quantisation.bits == 8

    def test_evaluation_and_calibration_samples_are_small(self, pilot: ExperimentConfig):
        assert pilot.evaluation.max_samples is not None
        assert pilot.evaluation.max_samples <= 128
        assert pilot.data.max_eval_samples is not None
        assert pilot.data.max_eval_samples <= 128
        assert pilot.data.calibration_samples <= 32
        assert pilot.compression.quantisation.calibration_samples <= 32

    def test_sequence_length_is_short(self, pilot: ExperimentConfig):
        assert pilot.data.sequence_length <= 256
        assert pilot.evaluation.sequence_length <= 256
        assert pilot.benchmark.sequence_length <= 128

    def test_optimisation_budget_is_short(self, pilot: ExperimentConfig):
        assert pilot.compression.recovery.max_steps is not None
        assert pilot.compression.recovery.max_steps <= 100
        assert pilot.compression.joint.joint_max_steps is not None
        assert pilot.compression.joint.joint_max_steps <= 100

    def test_measured_runs_still_allow_median_std_and_p95(self, pilot: ExperimentConfig):
        """Minimal, but the reported statistics must all remain computable."""
        from scale_aware_compression.benchmarking.latency import summarise_latencies

        runs = pilot.benchmark.measured_runs
        assert runs >= 3, "need at least 3 samples for a p95 that interpolates between values"
        assert runs <= 12, "the pilot must stay minimal"

        samples = [0.01 * (index + 1) for index in range(runs)]
        statistics = summarise_latencies(samples)
        assert statistics.median_ms > 0
        assert statistics.std_ms > 0
        assert statistics.p95_ms >= statistics.median_ms

    def test_benchmark_is_still_cpu_only(self, pilot: ExperimentConfig):
        assert pilot.benchmark.device is Device.CPU
        assert pilot.evaluation.device is Device.CPU

    def test_compares_sequential_against_joint_at_a_matched_budget(self, pilot: ExperimentConfig):
        """The pilot exercises the comparison the study is about, budget-matching check included."""
        assert pilot.compression.method is CompressionMethod.SEQUENTIAL
        assert pilot.compression.joint.match_sequential_budget is True
        assert pilot.compression.joint.joint_max_steps == pilot.compression.recovery.max_steps

    def test_is_labelled_as_a_pipeline_validation_run(self, pilot: ExperimentConfig):
        """A future reader must not mistake pilot output for a result."""
        assert "not-a-result" in pilot.experiment.tags
        assert "pipeline-validation" in pilot.experiment.tags
