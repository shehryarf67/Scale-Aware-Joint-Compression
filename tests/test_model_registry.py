"""Model registry lookups, aliases, and error handling.

No test here downloads anything: the registry is pure data, and the loader is exercised only through
its validation path, with ``transformers`` never imported.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.models.registry import (
    MODEL_REGISTRY,
    PYTHIA_FAMILY,
    QWEN_FAMILY,
    ModelSpec,
    UnknownModelError,
    get_model_spec,
    list_models,
    normalise_model_name,
    registry_table,
    resolve_model_id,
    scale_sweep_models,
    validation_models,
)

EXPECTED_IDS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}


class TestRequiredEntries:
    """The five models named in the study brief must resolve to the stated Hub identifiers."""

    @pytest.mark.parametrize(("short_name", "hf_id"), sorted(EXPECTED_IDS.items()))
    def test_resolves_to_the_expected_hub_id(self, short_name: str, hf_id: str):
        assert resolve_model_id(short_name) == hf_id

    def test_registry_holds_exactly_these_models(self):
        assert set(MODEL_REGISTRY) == set(EXPECTED_IDS)

    def test_every_entry_is_a_model_spec(self):
        for spec in MODEL_REGISTRY.values():
            assert isinstance(spec, ModelSpec)
            assert spec.short_name
            assert spec.hf_id
            assert spec.size_label
            assert spec.parameter_count > 0
            assert spec.architecture

    def test_registry_is_read_only(self):
        with pytest.raises(TypeError):
            MODEL_REGISTRY["new"] = MODEL_REGISTRY["pythia-160m"]  # type: ignore[index]


class TestLookup:
    def test_returns_the_full_spec(self):
        spec = get_model_spec("pythia-410m")
        assert spec.size_label == "410M"
        assert spec.family == PYTHIA_FAMILY
        assert spec.architecture == "GPTNeoXForCausalLM"

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("pythia-160m", "pythia-160m"),
            ("PYTHIA-160M", "pythia-160m"),
            ("  pythia-160m  ", "pythia-160m"),
            ("pythia_160m", "pythia-160m"),
            ("pythia_1_4b", "pythia-1.4b"),
            ("pythia-1_4b", "pythia-1.4b"),
            ("qwen2_5_0_5b", "qwen2.5-0.5b"),
            ("Qwen2.5-0.5B", "qwen2.5-0.5b"),
        ],
    )
    def test_normalises_aliases_and_case(self, given: str, expected: str):
        assert normalise_model_name(given) == expected
        assert get_model_spec(given).short_name == expected

    def test_filename_safe_alias_matches_the_config_filename(self):
        """configs/models/pythia_1_4b.yaml must be resolvable from its filename stem."""
        assert get_model_spec("pythia_1_4b").short_name == "pythia-1.4b"


class TestInvalidLookup:
    def test_unknown_name_raises(self):
        with pytest.raises(UnknownModelError):
            get_model_spec("gpt2")

    def test_error_message_lists_the_valid_names(self):
        with pytest.raises(UnknownModelError) as info:
            get_model_spec("llama-7b")
        message = str(info.value)
        assert "pythia-160m" in message
        assert "qwen2.5-0.5b" in message

    @pytest.mark.parametrize("name", ["", "   "])
    def test_blank_name_raises(self, name: str):
        with pytest.raises(UnknownModelError, match="non-empty"):
            normalise_model_name(name)

    def test_non_string_name_raises(self):
        with pytest.raises(UnknownModelError):
            normalise_model_name(None)  # type: ignore[arg-type]

    def test_resolve_model_id_raises_for_unknown_names(self):
        with pytest.raises(UnknownModelError):
            resolve_model_id("mistral-7b")


class TestOrdering:
    def test_list_models_is_sorted_by_parameter_count(self):
        counts = [MODEL_REGISTRY[name].parameter_count for name in list_models()]
        assert counts == sorted(counts)

    def test_family_filter(self):
        assert list_models(family=QWEN_FAMILY) == ["qwen2.5-0.5b"]
        assert "qwen2.5-0.5b" not in list_models(family=PYTHIA_FAMILY)

    def test_role_filter(self):
        assert list_models(role="external_validation") == ["qwen2.5-0.5b"]

    def test_registry_table_rows_are_ordered_and_complete(self):
        rows = registry_table()
        assert len(rows) == len(MODEL_REGISTRY)
        assert [row["parameter_count"] for row in rows] == sorted(
            row["parameter_count"]
            for row in rows  # type: ignore[type-var]
        )
        for row in rows:
            assert set(row) == {
                "short_name",
                "hf_id",
                "size_label",
                "parameter_count",
                "family",
                "role",
            }


class TestSweepSelection:
    def test_sweep_excludes_the_optional_model_by_default(self):
        assert scale_sweep_models() == ["pythia-160m", "pythia-410m", "pythia-1b"]

    def test_sweep_includes_the_optional_model_on_request(self):
        assert scale_sweep_models(include_optional=True) == [
            "pythia-160m",
            "pythia-410m",
            "pythia-1b",
            "pythia-1.4b",
        ]

    def test_qwen_is_never_part_of_the_scale_sweep(self):
        """It is a different family, so including it would break the controlled comparison."""
        assert "qwen2.5-0.5b" not in scale_sweep_models(include_optional=True)
        assert validation_models() == ["qwen2.5-0.5b"]

    def test_pythia_1_4b_is_flagged_optional(self):
        assert get_model_spec("pythia-1.4b").is_optional
        assert not get_model_spec("pythia-1b").is_optional

    def test_validation_model_size_sits_inside_the_sweep_range(self):
        """Qwen must be bracketed by the sweep, not extrapolated from it.

        The transfer check interpolates the Pythia trend at Qwen's parameter count.
        """
        qwen = get_model_spec("qwen2.5-0.5b").parameter_count
        sweep = [get_model_spec(name).parameter_count for name in scale_sweep_models()]
        assert min(sweep) < qwen < max(sweep)


class TestLoaderValidation:
    """The loader validates before importing transformers, so a typo fails fast and offline."""

    def test_load_tokenizer_rejects_an_unknown_model_before_any_import(self):
        from scale_aware_compression.config import ModelConfig
        from scale_aware_compression.models.loader import load_tokenizer

        with pytest.raises(UnknownModelError):
            load_tokenizer(ModelConfig(name="not-a-real-model"))

    def test_load_model_rejects_an_unknown_model_before_any_import(self):
        from scale_aware_compression.config import ModelConfig
        from scale_aware_compression.models.loader import load_model

        with pytest.raises(UnknownModelError):
            load_model(ModelConfig(name="also-not-real"))

    def test_importing_the_loader_does_not_import_transformers(self, imported_after):
        assert (
            imported_after("scale_aware_compression.models.loader", ["transformers", "torch"]) == []
        ), (
            "importing the loader must not import transformers or torch; heavy imports are lazy "
            "so that `sajc info` and the test suite stay fast and offline"
        )

    def test_trust_remote_code_defaults_to_false(self):
        from scale_aware_compression.config import ModelConfig

        assert ModelConfig().trust_remote_code is False


class TestArchitectureAdapters:
    def test_every_registered_architecture_has_an_adapter(self):
        from scale_aware_compression.models.adapters import ADAPTERS

        for spec in MODEL_REGISTRY.values():
            assert spec.architecture in ADAPTERS, (
                f"{spec.short_name} has architecture {spec.architecture} with no adapter"
            )

    def test_unknown_architecture_raises_with_guidance(self):
        from scale_aware_compression.models.adapters import (
            UnsupportedArchitectureError,
            get_adapter,
        )

        with pytest.raises(UnsupportedArchitectureError, match="adapters.py"):
            get_adapter("MambaForCausalLM")

    def test_qwen_adapter_records_tied_embeddings(self):
        """Tied embeddings change which modules may be compressed, so the flag must be right."""
        from scale_aware_compression.models.adapters import get_adapter

        assert get_adapter("Qwen2ForCausalLM").tied_embeddings
        assert not get_adapter("GPTNeoXForCausalLM").tied_embeddings

    def test_adapters_exclude_embeddings_and_output_head(self):
        from scale_aware_compression.models.adapters import ADAPTERS

        for adapter in ADAPTERS.values():
            assert adapter.embedding_names
            assert adapter.compressible_suffixes
            overlap = set(adapter.embedding_names) & set(adapter.compressible_suffixes)
            assert not overlap, f"{adapter.architecture} lists {overlap} as both"
