"""Replicate aggregation, the sign test, and the paired block bootstrap (A1 §5.1).

These functions decide what the paper is allowed to say, so the tests are mostly about refusing to
overclaim rather than about arithmetic. The load-bearing ones:

* the **exact** sign-test probabilities, because at R = 3..8 a normal approximation is simply wrong and
  A1's whole replicate-count decision rests on the exact values;
* ``significance_was_reachable``, which distinguishes "no effect" from "not enough draws to tell" --
  the distinction that keeps an underpowered design from being written up as evidence of absence;
* the bootstrap resampling **whole windows** and using the **same indices for both arms**, since
  breaking either would silently produce an interval that looks fine and is wrong.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.metrics.replicates import (
    MIN_R_FOR_SIGNIFICANCE,
    ReplicateError,
    compare_scales,
    paired_block_bootstrap,
    sign_test_p_value,
    summarise_replicates,
)


class TestSignTestIsExact:
    """A1's replicate-count decision rests on these exact values, not on an approximation."""

    @pytest.mark.parametrize(
        ("total", "expected"),
        [(3, 0.250), (5, 0.0625), (6, 0.03125), (8, 0.0078125)],
    )
    def test_unanimous_probabilities_match_the_amendment(self, total: int, expected: float):
        assert sign_test_p_value(total, total) == pytest.approx(expected)

    def test_five_draws_cannot_reach_significance_even_if_unanimous(self):
        """The cliff that decided R=8. At R=5 the best possible outcome is still above 0.05."""
        assert sign_test_p_value(5, 5) > 0.05

    def test_six_draws_can(self):
        assert sign_test_p_value(6, 6) < 0.05

    def test_a_split_result_is_not_significant(self):
        assert sign_test_p_value(4, 8) == pytest.approx(1.0)

    def test_it_is_symmetric_in_direction(self):
        assert sign_test_p_value(1, 8) == sign_test_p_value(7, 8)

    @pytest.mark.parametrize(("positive", "total"), [(0, 0), (-1, 5), (6, 5)])
    def test_invalid_counts_are_refused(self, positive: int, total: int):
        with pytest.raises(ReplicateError):
            sign_test_p_value(positive, total)


class TestReplicateSummary:
    def test_reproduces_the_measured_160m_result(self):
        """F-27: three draws, all positive, mean +1.69 pp."""
        summary = summarise_replicates(
            model_name="pythia-160m", budget_label="aggressive", gains=[1.08, 1.65, 2.34]
        )
        assert summary.mean_gain == pytest.approx(1.690, abs=1e-3)
        assert summary.standard_deviation == pytest.approx(0.631, abs=1e-3)
        assert summary.positive_count == 3
        assert summary.consistent_in_sign

    def test_reproduces_the_measured_410m_result(self):
        """F-26: the sign flips, so consistency must be false."""
        summary = summarise_replicates(
            model_name="pythia-410m", budget_label="aggressive", gains=[0.68, -0.50, 0.98]
        )
        assert summary.mean_gain == pytest.approx(0.387, abs=1e-3)
        assert summary.positive_count == 2
        assert not summary.consistent_in_sign

    def test_every_replicate_value_is_retained(self):
        """A1 §5.1 requires replicate-level values reported, not just the summary."""
        gains = [1.08, 1.65, 2.34]
        assert (
            list(summarise_replicates(model_name="m", budget_label="b", gains=gains).gains) == gains
        )

    def test_a_unanimous_result_at_low_r_is_flagged_as_unreachable(self):
        """The point of the flag: 3/3 positive still cannot be called significant."""
        summary = summarise_replicates(model_name="m", budget_label="b", gains=[1.0, 2.0, 3.0])
        assert summary.consistent_in_sign
        assert summary.sign_test_p == pytest.approx(0.25)
        assert not summary.significance_was_reachable

    def test_eight_draws_are_flagged_as_reachable(self):
        summary = summarise_replicates(model_name="m", budget_label="b", gains=[1.0] * 8)
        assert summary.significance_was_reachable
        assert summary.sign_test_p < 0.05

    def test_the_reachability_threshold_matches_the_arithmetic(self):
        """The constant and the sign test must not drift apart."""
        assert sign_test_p_value(MIN_R_FOR_SIGNIFICANCE, MIN_R_FOR_SIGNIFICANCE) < 0.05
        assert sign_test_p_value(MIN_R_FOR_SIGNIFICANCE - 1, MIN_R_FOR_SIGNIFICANCE - 1) > 0.05

    def test_standard_error_shrinks_with_more_replicates(self):
        few = summarise_replicates(model_name="m", budget_label="b", gains=[1.0, 2.0, 3.0])
        many = summarise_replicates(model_name="m", budget_label="b", gains=[1.0, 2.0, 3.0] * 3)
        assert many.standard_error < few.standard_error

    def test_a_single_replicate_has_no_defined_spread(self):
        """One draw is not an estimate of anything -- the whole lesson of F-26."""
        import math

        summary = summarise_replicates(model_name="m", budget_label="b", gains=[1.08])
        assert math.isnan(summary.standard_error)
        assert not summary.significance_was_reachable

    def test_no_gains_is_an_error_rather_than_an_empty_summary(self):
        with pytest.raises(ReplicateError, match="at least one gain"):
            summarise_replicates(model_name="m", budget_label="b", gains=[])


class TestScaleComparison:
    def test_differences_are_taken_replicate_by_replicate(self):
        """Per-draw differencing removes what a shared draw did to both models."""
        smaller = summarise_replicates(
            model_name="pythia-160m", budget_label="aggressive", gains=[1.08, 1.65, 2.34]
        )
        larger = summarise_replicates(
            model_name="pythia-410m", budget_label="aggressive", gains=[0.68, -0.50, 0.98]
        )
        comparison = compare_scales(smaller=smaller, larger=larger)
        assert comparison.differences == pytest.approx([0.40, 2.15, 1.36], abs=1e-9)
        assert comparison.positive_count == 3
        assert comparison.consistent_in_sign

    def test_it_uses_only_the_shared_replicates(self):
        """A1 makes the smaller count a prefix of the larger, so draw r means the same data."""
        smaller = summarise_replicates(model_name="a", budget_label="b", gains=[1.0] * 8)
        larger = summarise_replicates(model_name="c", budget_label="b", gains=[0.5] * 5)
        assert len(compare_scales(smaller=smaller, larger=larger).differences) == 5

    def test_comparing_across_budgets_is_refused(self):
        smaller = summarise_replicates(model_name="a", budget_label="moderate", gains=[1.0])
        larger = summarise_replicates(model_name="b", budget_label="aggressive", gains=[1.0])
        with pytest.raises(ReplicateError, match="across budgets"):
            compare_scales(smaller=smaller, larger=larger)


class TestPairedBlockBootstrap:
    @staticmethod
    def _windows(count: int, advantage: float):
        """Sequential and joint window NLLs where joint is better by a fixed per-token margin."""
        tokens = [511] * count
        sequential = [3.8 * 511 + (index % 7) * 0.5 for index in range(count)]
        joint = [value - advantage * 511 for value in sequential]
        return sequential, joint, tokens

    def test_a_clear_advantage_gives_an_interval_above_zero(self):
        s, j, t = self._windows(200, advantage=0.02)
        interval = paired_block_bootstrap(
            sequential_window_nll=s, joint_window_nll=j, window_tokens=t, resamples=500
        )
        assert interval.point_estimate == pytest.approx(0.02, abs=1e-6)
        assert interval.excludes_zero
        assert interval.lower > 0.0

    def test_no_advantage_gives_an_interval_containing_zero(self):
        s, j, t = self._windows(200, advantage=0.0)
        interval = paired_block_bootstrap(
            sequential_window_nll=s, joint_window_nll=j, window_tokens=t, resamples=500
        )
        assert interval.point_estimate == pytest.approx(0.0, abs=1e-9)
        assert not interval.excludes_zero

    def test_pairing_makes_a_constant_advantage_have_zero_spread(self):
        """The point of pairing: a per-window difference that is constant cannot vary under resampling.

        Resampling the arms independently would break this and produce a wide interval around the same
        point estimate -- an interval inflated by exactly the variance the paired design removes.
        """
        s, j, t = self._windows(150, advantage=0.02)
        interval = paired_block_bootstrap(
            sequential_window_nll=s, joint_window_nll=j, window_tokens=t, resamples=400
        )
        assert interval.upper - interval.lower == pytest.approx(0.0, abs=1e-9)

    def test_it_is_reproducible_for_a_fixed_seed(self):
        s, j, t = self._windows(80, advantage=0.01)
        kwargs = {
            "sequential_window_nll": s,
            "joint_window_nll": j,
            "window_tokens": t,
            "resamples": 200,
            "seed": 7,
        }
        assert paired_block_bootstrap(**kwargs).lower == paired_block_bootstrap(**kwargs).lower

    def test_mismatched_window_counts_are_refused(self):
        with pytest.raises(ReplicateError, match="same length"):
            paired_block_bootstrap(
                sequential_window_nll=[1.0, 2.0],
                joint_window_nll=[1.0],
                window_tokens=[511, 511],
            )

    def test_no_windows_is_refused(self):
        with pytest.raises(ReplicateError, match="at least one window"):
            paired_block_bootstrap(sequential_window_nll=[], joint_window_nll=[], window_tokens=[])

    @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5, -0.1])
    def test_an_out_of_range_confidence_is_refused(self, confidence: float):
        s, j, t = self._windows(10, advantage=0.01)
        with pytest.raises(ReplicateError, match="confidence"):
            paired_block_bootstrap(
                sequential_window_nll=s,
                joint_window_nll=j,
                window_tokens=t,
                confidence=confidence,
            )

    def test_it_consumes_what_compute_perplexity_produces(self):
        """End to end against the real evaluator, so the two halves cannot drift apart."""
        pytest.importorskip("torch")
        import transformers
        from torch.utils.data import DataLoader

        from scale_aware_compression.config import DataConfig, EvaluationConfig
        from scale_aware_compression.data.loaders import TokenBlockDataset, build_dataloader
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        config = transformers.GPTNeoXConfig(
            vocab_size=259,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=128,
        )
        model = transformers.GPTNeoXForCausalLM(config)
        model.eval()

        dataset = TokenBlockDataset([[(i * 7 + s) % 259 for s in range(16)] for i in range(12)])
        loader = build_dataloader(dataset, DataConfig(batch_size=2), batch_size=2)
        assert isinstance(loader, DataLoader)

        result = compute_perplexity(model, loader, EvaluationConfig())
        interval = paired_block_bootstrap(
            sequential_window_nll=result.window_nll,
            joint_window_nll=result.window_nll,
            window_tokens=result.window_tokens,
            resamples=200,
        )
        assert interval.windows == result.num_sequences
        # Identical inputs, so the advantage is exactly zero and the interval must contain it.
        assert interval.point_estimate == pytest.approx(0.0, abs=1e-12)
        assert not interval.excludes_zero
