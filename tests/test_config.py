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

    def test_experiment_configs_parse(self, configs_dir: Path):
        files = self._yaml_files(configs_dir / "experiments")
        assert len(files) == 3
        for path in files:
            config = load_config(path)
            assert config.experiment.id

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
        assert sweep.sweep.seeds == validation.sweep.seeds
        assert sweep.sweep.methods == validation.sweep.methods
