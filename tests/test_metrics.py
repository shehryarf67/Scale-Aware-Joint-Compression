"""Metric utilities: parameter counts, sparsity, compression ratio, retention, joint gain.

These are the functions the study's conclusions are computed from, so they are tested against
hand-checkable values rather than against themselves.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scale_aware_compression.metrics.compression import (
    checkpoint_size_bytes,
    checkpoint_size_mib,
    compression_ratio,
    count_parameters,
    count_zero_parameters,
    count_zeros,
    effective_compression_ratio,
    measure_sparsity,
    size_reduction_percentage,
    sparsity_fraction,
    sparsity_percentage,
    theoretical_size_bytes,
)
from scale_aware_compression.metrics.efficiency import (
    latency_reduction_percentage,
    memory_reduction_percentage,
    sparsity_realisation,
    speedup,
    theoretical_speedup_from_sparsity,
    throughput_gain,
    training_cost_overhead,
)
from scale_aware_compression.metrics.joint_gain import (
    JointGainSummary,
    accuracy_retention,
    joint_gain,
    joint_gain_from_quality_loss,
    joint_gain_summary,
    perplexity_increase_percentage,
    perplexity_retention,
    relative_joint_gain,
)


class TestParameterCounting:
    def test_counts_every_parameter(self, dense_module):
        assert count_parameters(dense_module) == 10

    def test_counts_only_trainable_when_asked(self, frozen_module):
        assert count_parameters(frozen_module) == 10
        assert count_parameters(frozen_module, trainable_only=True) == 4

    def test_counts_zeros(self, half_sparse_module):
        assert count_zero_parameters(half_sparse_module) == 5

    def test_dense_module_has_no_zeros(self, dense_module):
        assert count_zero_parameters(dense_module) == 0

    def test_count_zeros_uses_count_nonzero_when_available(self):
        class Tensor:
            def numel(self) -> int:
                return 4

            def count_nonzero(self) -> int:
                return 2

        assert count_zeros(Tensor()) == 2

    def test_count_zeros_falls_back_to_tolist(self):
        class Tensor:
            def numel(self) -> int:
                return 4

            def tolist(self) -> list[list[float]]:
                return [[1.0, 0.0], [0.0, 2.0]]

        assert count_zeros(Tensor()) == 2

    def test_count_zeros_rejects_an_unusable_object(self):
        with pytest.raises(TypeError, match="count_nonzero"):
            count_zeros(object())

    def test_measure_sparsity_reports_the_full_picture(self, half_sparse_module):
        measured = measure_sparsity(half_sparse_module)
        assert measured == {
            "total_parameters": 10,
            "zero_parameters": 5,
            "nonzero_parameters": 5,
            "sparsity_percentage": 50.0,
        }


class TestSparsity:
    @pytest.mark.parametrize(
        ("total", "zeros", "expected"),
        [(100, 0, 0.0), (100, 50, 50.0), (100, 100, 100.0), (1000, 700, 70.0), (3, 1, 100 / 3)],
    )
    def test_percentage(self, total: int, zeros: int, expected: float):
        assert sparsity_percentage(total, zeros) == pytest.approx(expected)

    def test_fraction_is_the_percentage_over_one_hundred(self):
        assert sparsity_fraction(100, 70) == pytest.approx(0.7)

    def test_zero_denominator_raises(self):
        with pytest.raises(ValueError, match="total_parameters"):
            sparsity_percentage(0, 0)

    def test_negative_zero_count_raises(self):
        with pytest.raises(ValueError, match="zero_parameters"):
            sparsity_percentage(100, -1)

    def test_more_zeros_than_parameters_raises(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            sparsity_percentage(100, 101)


class TestCheckpointSize:
    def test_measures_a_single_file(self, tmp_path: Path):
        path = tmp_path / "model.safetensors"
        path.write_bytes(b"x" * 2048)
        assert checkpoint_size_bytes(path) == 2048

    def test_measures_a_directory_recursively(self, tmp_path: Path):
        (tmp_path / "nested").mkdir()
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
        (tmp_path / "config.json").write_bytes(b"y" * 100)
        (tmp_path / "nested" / "extra.bin").write_bytes(b"z" * 10)
        assert checkpoint_size_bytes(tmp_path) == 1110

    def test_weights_only_excludes_json(self, tmp_path: Path):
        """Tokeniser and config JSON are identical across arms.

        Including them would add a constant that flatters the larger models.
        """
        (tmp_path / "model.safetensors").write_bytes(b"x" * 1000)
        (tmp_path / "tokenizer.json").write_bytes(b"y" * 500)
        assert checkpoint_size_bytes(tmp_path, weights_only=True) == 1000

    def test_reports_mib(self, tmp_path: Path):
        path = tmp_path / "model.pt"
        path.write_bytes(b"x" * (2 * 1024 * 1024))
        assert checkpoint_size_mib(path) == pytest.approx(2.0)

    def test_missing_path_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            checkpoint_size_bytes(tmp_path / "absent")

    @pytest.mark.parametrize(
        ("nonzero", "bits", "expected"),
        [(8, 8, 8), (8, 32, 32), (8, 4, 4), (1, 4, 1), (3, 4, 2)],
    )
    def test_theoretical_size_rounds_up_to_whole_bytes(
        self, nonzero: int, bits: int, expected: int
    ):
        assert theoretical_size_bytes(nonzero, bits) == expected

    def test_theoretical_size_can_include_sparse_index_overhead(self):
        """A sparse format stores indices too, so 50% sparsity does not halve the file."""
        dense = theoretical_size_bytes(1000, 8)
        sparse = theoretical_size_bytes(500, 8, sparse_index_bits=16)
        assert sparse > dense / 2

    def test_negative_parameter_count_raises(self):
        with pytest.raises(ValueError, match="nonzero_parameters"):
            theoretical_size_bytes(-1, 8)

    def test_non_positive_bits_raise(self):
        with pytest.raises(ValueError, match="bits"):
            theoretical_size_bytes(100, 0)


class TestCompressionRatio:
    @pytest.mark.parametrize(
        ("baseline", "compressed", "expected"),
        [(100.0, 25.0, 4.0), (100.0, 100.0, 1.0), (100.0, 200.0, 0.5), (1000.0, 125.0, 8.0)],
    )
    def test_ratio(self, baseline: float, compressed: float, expected: float):
        assert compression_ratio(baseline, compressed) == pytest.approx(expected)

    def test_reduction_percentage(self):
        assert size_reduction_percentage(100.0, 25.0) == pytest.approx(75.0)

    def test_reduction_is_negative_when_the_artefact_grew(self):
        assert size_reduction_percentage(100.0, 150.0) < 0

    @pytest.mark.parametrize(("baseline", "compressed"), [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0)])
    def test_non_positive_sizes_raise(self, baseline: float, compressed: float):
        with pytest.raises(ValueError):
            compression_ratio(baseline, compressed)

    def test_effective_ratio_combines_sparsity_and_precision(self):
        # 50% sparsity at 8 bits from a 32-bit baseline: 32 / (8 * 0.5) = 8x
        assert effective_compression_ratio(100, 50, 8) == pytest.approx(8.0)

    def test_effective_ratio_matches_the_aggressive_budget(self):
        # 70% sparsity at 4 bits: 32 / (4 * 0.3) = 26.67x
        assert effective_compression_ratio(1000, 700, 4) == pytest.approx(32 / (4 * 0.3))

    def test_effective_ratio_of_an_unpruned_eight_bit_model_is_four(self):
        assert effective_compression_ratio(100, 0, 8) == pytest.approx(4.0)

    def test_fully_zeroed_model_raises(self):
        with pytest.raises(ValueError, match="every parameter is zero"):
            effective_compression_ratio(100, 100, 8)


class TestEfficiency:
    def test_speedup(self):
        assert speedup(100.0, 50.0) == pytest.approx(2.0)

    def test_latency_reduction(self):
        assert latency_reduction_percentage(100.0, 75.0) == pytest.approx(25.0)

    def test_latency_reduction_is_negative_when_compression_slowed_things_down(self):
        """A common and reportable outcome for unstructured sparsity on CPU."""
        assert latency_reduction_percentage(100.0, 120.0) < 0

    def test_throughput_gain(self):
        assert throughput_gain(1000.0, 1500.0) == pytest.approx(1.5)

    def test_memory_reduction(self):
        assert memory_reduction_percentage(800.0, 200.0) == pytest.approx(75.0)

    @pytest.mark.parametrize(
        ("sparsity", "expected"), [(0.0, 1.0), (0.5, 2.0), (0.75, 4.0), (0.7, 1 / 0.3)]
    )
    def test_theoretical_speedup_from_sparsity(self, sparsity: float, expected: float):
        assert theoretical_speedup_from_sparsity(sparsity) == pytest.approx(expected)

    def test_theoretical_speedup_rejects_full_sparsity(self):
        with pytest.raises(ValueError, match="sparsity"):
            theoretical_speedup_from_sparsity(1.0)

    def test_full_realisation(self):
        assert sparsity_realisation(2.0, 2.0) == pytest.approx(1.0)

    def test_half_realisation(self):
        assert sparsity_realisation(1.5, 2.0) == pytest.approx(0.5)

    def test_no_realisation_is_the_expected_unstructured_result(self):
        """Dense CPU GEMM kernels do not skip scattered zeros; that is a finding, not a bug."""
        assert sparsity_realisation(1.0, 2.0) == pytest.approx(0.0)

    def test_negative_realisation_when_slower_than_baseline(self):
        assert sparsity_realisation(0.9, 2.0) < 0

    def test_realisation_is_zero_when_no_speedup_was_available(self):
        assert sparsity_realisation(1.0, 1.0) == 0.0

    def test_realisation_rejects_a_bound_below_one(self):
        with pytest.raises(ValueError, match="theoretical_speedup"):
            sparsity_realisation(1.0, 0.5)

    def test_matched_training_budget_is_one(self):
        assert training_cost_overhead(500.0, 500.0) == pytest.approx(1.0)

    def test_unmatched_training_budget_is_visible(self):
        assert training_cost_overhead(1500.0, 500.0) == pytest.approx(3.0)

    def test_zero_sequential_cost_raises(self):
        with pytest.raises(ValueError, match="sequential_cost"):
            training_cost_overhead(100.0, 0.0)


class TestJointGain:
    """The study's central quantity: joint quality minus sequential quality."""

    def test_positive_when_joint_scores_higher(self):
        assert joint_gain(95.0, 90.0) == pytest.approx(5.0)

    def test_negative_when_sequential_scores_higher(self):
        assert joint_gain(85.0, 90.0) == pytest.approx(-5.0)

    def test_zero_when_the_arms_tie(self):
        assert joint_gain(90.0, 90.0) == 0.0

    def test_lower_is_better_reverses_the_subtraction(self):
        """Perplexity is lower-is-better, so a lower joint value must still be a positive gain."""
        assert joint_gain(10.0, 12.0, higher_is_better=False) == pytest.approx(2.0)
        assert joint_gain(12.0, 10.0, higher_is_better=False) == pytest.approx(-2.0)

    def test_quality_loss_form_is_the_lower_is_better_convention(self):
        assert joint_gain_from_quality_loss(3.0, 5.0) == pytest.approx(2.0)
        assert joint_gain_from_quality_loss(3.0, 5.0) == joint_gain(
            3.0, 5.0, higher_is_better=False
        )

    def test_both_conventions_agree_on_the_sign_when_joint_wins(self):
        higher = joint_gain(95.0, 90.0, higher_is_better=True)
        lower = joint_gain_from_quality_loss(5.0, 10.0)
        assert higher > 0
        assert lower > 0

    def test_relative_gain_normalises_by_the_sequential_score(self):
        assert relative_joint_gain(99.0, 90.0) == pytest.approx(10.0)

    def test_relative_gain_lets_scales_be_compared(self):
        """Absolute gains are not comparable across model sizes; relative ones are."""
        small = relative_joint_gain(50.5, 50.0)
        large = relative_joint_gain(101.0, 100.0)
        assert small == pytest.approx(large)

    def test_relative_gain_rejects_a_zero_denominator(self):
        with pytest.raises(ValueError, match="sequential_score is zero"):
            relative_joint_gain(1.0, 0.0)


class TestRetention:
    def test_accuracy_retention(self):
        assert accuracy_retention(0.45, 0.50) == pytest.approx(90.0)

    def test_unchanged_accuracy_retains_one_hundred_percent(self):
        assert accuracy_retention(0.5, 0.5) == pytest.approx(100.0)

    def test_retention_above_one_hundred_is_permitted(self):
        """Mild sparsity sometimes helps; clipping would hide a real effect."""
        assert accuracy_retention(0.55, 0.50) > 100.0

    def test_accuracy_retention_rejects_a_zero_baseline(self):
        with pytest.raises(ValueError, match="dense_accuracy"):
            accuracy_retention(0.5, 0.0)

    def test_perplexity_retention_is_inverted_so_higher_is_better(self):
        assert perplexity_retention(10.0, 12.5) == pytest.approx(80.0)

    def test_unchanged_perplexity_retains_one_hundred_percent(self):
        assert perplexity_retention(10.0, 10.0) == pytest.approx(100.0)

    def test_perplexity_retention_falls_when_perplexity_rises(self):
        assert perplexity_retention(10.0, 20.0) < perplexity_retention(10.0, 11.0)

    def test_perplexity_increase_percentage(self):
        assert perplexity_increase_percentage(10.0, 11.0) == pytest.approx(10.0)

    def test_perplexity_increase_is_negative_when_perplexity_improved(self):
        assert perplexity_increase_percentage(10.0, 9.0) == pytest.approx(-10.0)

    @pytest.mark.parametrize(("dense", "compressed"), [(0.0, 10.0), (10.0, 0.0), (-1.0, 10.0)])
    def test_non_positive_perplexity_raises(self, dense: float, compressed: float):
        with pytest.raises(ValueError):
            perplexity_retention(dense, compressed)


class TestJointGainSummary:
    def test_builds_a_complete_summary(self):
        summary = joint_gain_summary(
            model_name="pythia-410m",
            size_label="410M",
            parameter_count=405_334_016,
            budget_label="moderate",
            metric_name="perplexity_retention",
            joint_score=96.0,
            sequential_score=94.0,
            seed=1234,
        )
        assert isinstance(summary, JointGainSummary)
        assert summary.absolute_gain == pytest.approx(2.0)
        assert summary.relative_gain_percentage == pytest.approx(100 * 2 / 94)
        assert summary.joint_is_better

    def test_records_the_losing_case(self):
        summary = joint_gain_summary(
            model_name="pythia-160m",
            size_label="160M",
            parameter_count=162_322_944,
            budget_label="aggressive",
            metric_name="perplexity_retention",
            joint_score=88.0,
            sequential_score=91.0,
        )
        assert summary.absolute_gain == pytest.approx(-3.0)
        assert not summary.joint_is_better

    def test_records_the_budget_so_mismatched_pairs_are_visible(self):
        """Comparing arms at different budgets is a measurement error, so the budget is carried."""
        summary = joint_gain_summary(
            model_name="pythia-1b",
            size_label="1B",
            parameter_count=1_011_781_632,
            budget_label="aggressive",
            metric_name="perplexity_retention",
            joint_score=80.0,
            sequential_score=75.0,
        )
        assert summary.to_dict()["budget_label"] == "aggressive"

    def test_zero_sequential_score_yields_nan_rather_than_raising(self):
        summary = joint_gain_summary(
            model_name="pythia-160m",
            size_label="160M",
            parameter_count=162_322_944,
            budget_label="moderate",
            metric_name="accuracy",
            joint_score=1.0,
            sequential_score=0.0,
        )
        assert summary.absolute_gain == pytest.approx(1.0)
        assert summary.relative_gain_percentage != summary.relative_gain_percentage  # NaN

    def test_to_dict_is_flat_and_csv_ready(self):
        payload = joint_gain_summary(
            model_name="pythia-1b",
            size_label="1B",
            parameter_count=1_011_781_632,
            budget_label="moderate",
            metric_name="perplexity_retention",
            joint_score=95.0,
            sequential_score=93.0,
        ).to_dict()
        assert all(not isinstance(value, dict | list) for value in payload.values())
        assert payload["joint_is_better"] is True
