"""Phase 5 exit tests: the single-layer compression primitives.

The exit criteria from docs/implementation_plan.md, on a synthetic layer:

* reconstruction strictly reduces ``||Y - Y_hat||_F^2`` versus naive round-to-nearest
* realised sparsity is exact
* quantised weights take at most ``2^b`` distinct values per group
* pack -> unpack is lossless

plus Phase 2's activation-capture criterion, that streamed ``H`` matches a direct ``X^T X``.

These are the properties that fail *silently*. A mask that lands at 49.7% instead of 50%, a
quantiser that keeps more levels than its bit width, an unpack that corrupts the last byte -- none
of them raise, and all of them would quietly invalidate a results table.
"""

from __future__ import annotations

import math

import pytest

from scale_aware_compression.compression.activations import (
    ActivationCaptureError,
    ActivationStatistics,
    LinearActivationCapture,
)
from scale_aware_compression.compression.masks import (
    MaskError,
    build_mask_from_scores,
    realised_sparsity,
)
from scale_aware_compression.compression.pruning import SaliencyError, activation_weighted_saliency
from scale_aware_compression.compression.quantisation import (
    QuantisationError,
    compute_symmetric_scales,
    effective_bits_per_weight,
    fake_quantise,
    pack_low_bit,
    quantise_weight,
    unpack_low_bit,
)
from scale_aware_compression.compression.reconstruct import (
    ReconstructionError,
    reconstruct,
    reconstruction_loss,
    solve_masked_rows,
)
from scale_aware_compression.constants import PruningGranularity, QuantisationGranularity

pytestmark = pytest.mark.requires_torch

IN_FEATURES = 16
OUT_FEATURES = 8
NUM_ROWS = 256


@pytest.fixture
def synthetic_layer():
    """A small linear layer with correlated calibration activations.

    Correlation matters: with perfectly independent inputs the Gram matrix is near-diagonal and
    reconstruction has almost nothing to redistribute, so a broken solver would still look fine.
    """
    import torch

    torch.manual_seed(1234)
    basis = torch.randn(NUM_ROWS, IN_FEATURES // 2)
    activations = torch.cat([basis, basis * 0.6 + torch.randn(NUM_ROWS, IN_FEATURES // 2) * 0.4], 1)
    weight = torch.randn(OUT_FEATURES, IN_FEATURES)
    return activations, weight, activations.t() @ activations


class TestActivationCapture:
    def test_streamed_gram_matches_direct_product(self):
        import torch

        torch.manual_seed(0)
        activations = torch.randn(6, 5, IN_FEATURES)

        statistics = ActivationStatistics(IN_FEATURES)
        for chunk in activations.split(2, dim=0):
            statistics.update(chunk)

        flat = activations.reshape(-1, IN_FEATURES)
        assert torch.allclose(statistics.gram(), flat.t() @ flat, atol=1e-3)
        assert statistics.num_rows == 30
        assert statistics.num_batches == 3

    def test_column_norms_are_the_gram_diagonal(self):
        """Saliency and the solver must not disagree about the activation scale."""
        import torch

        torch.manual_seed(0)
        activations = torch.randn(40, IN_FEATURES)
        statistics = ActivationStatistics(IN_FEATURES)
        statistics.update(activations)

        assert torch.allclose(statistics.column_norms(), activations.norm(dim=0), atol=1e-3)

    def test_damping_scales_with_the_gram_diagonal(self):
        """A relative ridge has to behave the same on activations of very different magnitude."""
        import torch

        torch.manual_seed(0)
        # The SAME activations, only rescaled: any other difference between the two would show up
        # in the ratio as sampling noise and make the assertion meaningless.
        base = torch.randn(32, 4)
        small = ActivationStatistics(4)
        small.update(base)
        large = ActivationStatistics(4)
        large.update(base * 100.0)

        def ridge(statistics: ActivationStatistics) -> float:
            damped = statistics.damped_gram(0.1)
            return float(torch.diagonal(damped - statistics.gram()).mean())

        # Scaling activations by 100 scales the Gram diagonal by 100^2, and a relative ridge must
        # track it exactly -- that is the whole point of keying damping off the mean diagonal.
        assert ridge(large) / ridge(small) == pytest.approx(100.0**2, rel=1e-3)

    def test_reading_statistics_before_any_update_raises(self):
        statistics = ActivationStatistics(4)
        with pytest.raises(ActivationCaptureError, match="no activations accumulated"):
            statistics.gram()

    def test_width_mismatch_raises(self):
        import torch

        statistics = ActivationStatistics(4)
        with pytest.raises(ActivationCaptureError, match="does not match in_features"):
            statistics.update(torch.randn(2, 5))

    def test_hook_captures_layer_inputs_and_is_removed_on_exit(self):
        import torch
        from torch import nn

        torch.manual_seed(0)
        layer = nn.Linear(IN_FEATURES, 4)
        with LinearActivationCapture(layer) as capture:
            layer(torch.randn(3, 7, IN_FEATURES))
        assert capture.statistics.num_rows == 21

        # A hook left installed would fold evaluation activations into a calibration statistic.
        layer(torch.randn(3, 7, IN_FEATURES))
        assert capture.statistics.num_rows == 21


class TestSaliency:
    def test_saliency_is_magnitude_weighted_by_activation_norm(self, synthetic_layer):
        import torch

        activations, weight, _ = synthetic_layer
        norms = activations.norm(dim=0)

        scores = activation_weighted_saliency(weight, norms)
        assert torch.allclose(scores, weight.abs() * norms.unsqueeze(0))

    def test_a_dead_input_column_scores_zero(self, synthetic_layer):
        """Weights on an input the calibration data never excites cannot matter to the output."""
        import torch

        _, weight, _ = synthetic_layer
        norms = torch.ones(IN_FEATURES)
        norms[3] = 0.0

        scores = activation_weighted_saliency(weight, norms)
        assert torch.all(scores[:, 3] == 0)
        assert torch.all(scores[:, 4] > 0)

    def test_mismatched_norms_raise(self, synthetic_layer):
        import torch

        _, weight, _ = synthetic_layer
        with pytest.raises(SaliencyError, match="to match in_features"):
            activation_weighted_saliency(weight, torch.ones(IN_FEATURES + 1))


class TestMaskExactness:
    @pytest.mark.parametrize("sparsity", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    def test_realised_sparsity_is_exact(self, sparsity: float):
        import torch

        torch.manual_seed(0)
        scores = torch.rand(16, 32)
        mask = build_mask_from_scores(scores, sparsity=sparsity)

        assert int((~mask).sum()) == round(mask.numel() * sparsity)
        assert realised_sparsity(mask) == pytest.approx(sparsity, abs=1.0 / mask.numel())

    def test_exact_even_when_scores_tie(self):
        """Activation-weighted saliency makes exact ties routine, not pathological.

        A dead input column zeroes a whole column of scores. A quantile threshold would keep or
        drop all of them together and miss the target sparsity.
        """
        import torch

        scores = torch.ones(8, 16)
        mask = build_mask_from_scores(scores, sparsity=0.5)
        assert int((~mask).sum()) == 64

    def test_mask_keeps_the_highest_scores(self):
        import torch

        scores = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        mask = build_mask_from_scores(scores, sparsity=0.5)
        assert torch.equal(mask.reshape(-1).nonzero(as_tuple=True)[0], torch.arange(6, 12))

    @pytest.mark.parametrize(
        ("granularity", "group", "kept"),
        [
            (PruningGranularity.SEMI_STRUCTURED_2_4, 4, 2),
            (PruningGranularity.SEMI_STRUCTURED_4_8, 8, 4),
        ],
    )
    def test_semi_structured_keeps_exactly_k_per_group(
        self, granularity: PruningGranularity, group: int, kept: int
    ):
        import torch

        torch.manual_seed(0)
        mask = build_mask_from_scores(torch.rand(4, 16), sparsity=0.5, granularity=granularity)
        assert torch.all(mask.reshape(-1, group).sum(dim=1) == kept)

    def test_semi_structured_rejects_a_mismatched_sparsity(self):
        import torch

        with pytest.raises(MaskError, match="implies sparsity"):
            build_mask_from_scores(
                torch.rand(4, 16),
                sparsity=0.7,
                granularity=PruningGranularity.SEMI_STRUCTURED_2_4,
            )

    def test_channel_structured_is_refused_rather_than_approximated(self):
        import torch

        with pytest.raises(MaskError, match="future work"):
            build_mask_from_scores(
                torch.rand(4, 16),
                sparsity=0.5,
                granularity=PruningGranularity.STRUCTURED_CHANNEL,
            )

    def test_out_of_range_sparsity_raises(self):
        import torch

        with pytest.raises(MaskError, match=r"\[0, 1\)"):
            build_mask_from_scores(torch.rand(4, 4), sparsity=1.0)


class TestQuantiser:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize(
        "granularity",
        [
            QuantisationGranularity.PER_TENSOR,
            QuantisationGranularity.PER_CHANNEL,
            QuantisationGranularity.PER_GROUP,
        ],
    )
    def test_at_most_two_to_the_bits_distinct_values_per_group(
        self, bits: int, granularity: QuantisationGranularity, synthetic_layer
    ):
        """§4.8: proves the bit width is real rather than silently dequantised."""
        _, weight, _ = synthetic_layer
        quantised = quantise_weight(
            weight, bits=bits, granularity=granularity, group_size=IN_FEATURES // 2
        )
        assert quantised.distinct_values_per_group() <= 2**bits

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_dequantise_round_trips_through_the_grid(self, bits: int, synthetic_layer):
        import torch

        _, weight, _ = synthetic_layer
        quantised = quantise_weight(weight, bits=bits)
        # Quantising an already-quantised tensor must be a no-op, or the grid is not stable.
        again = quantise_weight(quantised.dequantise(), bits=bits, scales=quantised.scales)
        assert torch.equal(quantised.codes, again.codes)

    def test_zero_stays_exactly_zero_under_a_symmetric_grid(self, synthetic_layer):
        """Pruned entries must survive quantisation as exact zeros.

        With an asymmetric grid zero is not necessarily a grid point, so quantising after masking
        could map a pruned zero onto a non-zero value and silently reduce measured sparsity.
        """
        import torch

        _, weight, _ = synthetic_layer
        weight = weight.clone()
        weight[0, :4] = 0.0
        assert torch.all(fake_quantise(weight, bits=4)[0, :4] == 0)

    def test_fake_quantisation_reduces_to_grid_values_only(self, synthetic_layer):
        import torch

        _, weight, _ = synthetic_layer
        quantised = quantise_weight(weight, bits=4, granularity=QuantisationGranularity.PER_TENSOR)
        scale = float(quantised.scales.reshape(-1)[0])
        fake = fake_quantise(weight, bits=4, granularity=QuantisationGranularity.PER_TENSOR)
        residual = fake / scale
        assert torch.allclose(residual, residual.round(), atol=1e-4)

    def test_an_all_zero_group_does_not_produce_nan(self):
        import torch

        weight = torch.zeros(4, 8)
        assert torch.all(torch.isfinite(fake_quantise(weight, bits=4)))
        assert torch.all(fake_quantise(weight, bits=4) == 0)

    def test_ragged_group_size_is_refused(self, synthetic_layer):
        _, weight, _ = synthetic_layer
        with pytest.raises(QuantisationError, match="does not divide"):
            quantise_weight(
                weight,
                bits=4,
                granularity=QuantisationGranularity.PER_GROUP,
                group_size=IN_FEATURES - 1,
            )

    def test_scales_are_strictly_positive(self, synthetic_layer):
        import torch

        _, weight, _ = synthetic_layer
        scales = compute_symmetric_scales(weight, bits=4)
        assert torch.all(scales > 0)

    def test_effective_bits_exceeds_nominal_because_scales_cost_storage(self, synthetic_layer):
        """§4.5 requires effective bits to include scale overhead."""
        _, weight, _ = synthetic_layer
        per_channel = quantise_weight(weight, bits=4)
        assert effective_bits_per_weight(per_channel) > 4.0

    def test_smaller_groups_cost_more_effective_bits(self, synthetic_layer):
        _, weight, _ = synthetic_layer
        coarse = quantise_weight(
            weight, bits=4, granularity=QuantisationGranularity.PER_GROUP, group_size=IN_FEATURES
        )
        fine = quantise_weight(
            weight, bits=4, granularity=QuantisationGranularity.PER_GROUP, group_size=4
        )
        assert effective_bits_per_weight(fine) > effective_bits_per_weight(coarse)


class TestPacking:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("numel", [1, 7, 8, 63, 64, 65])
    def test_pack_unpack_is_bit_exact(self, bits: int, numel: int):
        """Non-multiples of the lane count are included: padding is where this breaks."""
        import torch

        torch.manual_seed(0)
        qmax = (1 << (bits - 1)) - 1
        codes = torch.randint(-qmax, qmax + 1, (numel,), dtype=torch.int8)

        packed = pack_low_bit(codes, bits=bits)
        recovered = unpack_low_bit(packed, bits=bits, numel=numel)
        assert torch.equal(recovered, codes)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("numel", [1, 7, 64, 65])
    def test_packed_buffer_is_the_predicted_size(self, bits: int, numel: int):
        import torch

        codes = torch.zeros(numel, dtype=torch.int8)
        packed = pack_low_bit(codes, bits=bits)
        assert packed.numel() == math.ceil(numel * bits / 8)

    def test_extreme_codes_survive_the_round_trip(self):
        """The range endpoints are where an off-by-one in the sign offset shows up."""
        import torch

        for bits in (2, 4, 8):
            qmax = (1 << (bits - 1)) - 1
            codes = torch.tensor([-qmax, 0, qmax], dtype=torch.int8)
            packed = pack_low_bit(codes, bits=bits)
            assert torch.equal(unpack_low_bit(packed, bits=bits, numel=3), codes)

    def test_a_quantised_weight_survives_pack_and_unpack(self, synthetic_layer):
        import torch

        _, weight, _ = synthetic_layer
        quantised = quantise_weight(weight, bits=4)
        flat = quantised.codes.reshape(-1)

        packed = pack_low_bit(flat, bits=4)
        recovered = unpack_low_bit(packed, bits=4, numel=flat.numel())
        assert torch.equal(recovered, flat)

    def test_out_of_range_codes_are_refused(self):
        import torch

        with pytest.raises(QuantisationError, match="outside the representable range"):
            pack_low_bit(torch.tensor([8], dtype=torch.int8), bits=4)

    def test_unpackable_bit_width_is_refused(self):
        import torch

        with pytest.raises(QuantisationError, match="cannot pack"):
            pack_low_bit(torch.zeros(4, dtype=torch.int8), bits=3)

    def test_a_truncated_buffer_is_refused(self):
        import torch

        with pytest.raises(QuantisationError, match="need"):
            unpack_low_bit(torch.zeros(1, dtype=torch.uint8), bits=4, numel=64)


class TestReconstruction:
    def test_the_gram_objective_equals_the_direct_output_error(self, synthetic_layer):
        """If this drifts, every reconstruction number is measuring the wrong thing."""
        activations, weight, gram = synthetic_layer
        candidate = fake_quantise(weight, bits=4)

        direct = float(((activations @ weight.t()) - (activations @ candidate.t())).pow(2).sum())
        assert reconstruction_loss(gram, weight, candidate) == pytest.approx(direct, rel=1e-4)

    def test_reconstruction_strictly_beats_naive_masking(self, synthetic_layer):
        """The Phase 5 exit criterion, for pruning-only."""
        activations, weight, gram = synthetic_layer
        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, activations.norm(dim=0)), sparsity=0.5
        )

        result = reconstruct(gram, weight, mask, local_steps=3, bits=None)
        assert result.final_loss < result.naive_loss
        assert result.relative_improvement > 0

    def test_reconstruction_beats_naive_rounding_with_quantisation(self, synthetic_layer):
        """The Phase 5 exit criterion, for the combined constraint."""
        activations, weight, gram = synthetic_layer
        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, activations.norm(dim=0)), sparsity=0.5
        )

        result = reconstruct(gram, weight, mask, local_steps=6, bits=4)
        assert result.final_loss < result.naive_loss

    @pytest.mark.parametrize("bits", [None, 2, 4, 8])
    def test_reconstruction_never_returns_something_worse_than_naive(
        self, bits: int | None, synthetic_layer
    ):
        """Projection onto a discrete grid is not monotone, so the loop must be guarded."""
        activations, weight, gram = synthetic_layer
        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, activations.norm(dim=0)), sparsity=0.7
        )

        result = reconstruct(gram, weight, mask, local_steps=5, bits=bits)
        assert result.final_loss <= result.naive_loss
        assert result.history[0] == result.naive_loss
        assert min(result.history) == pytest.approx(result.final_loss)

    def test_reconstruction_respects_the_mask(self, synthetic_layer):
        """Error compensation must not resurrect a pruned weight."""
        import torch

        activations, weight, gram = synthetic_layer
        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, activations.norm(dim=0)), sparsity=0.5
        )

        result = reconstruct(gram, weight, mask, local_steps=3, bits=4)
        assert torch.all(result.weight[~mask] == 0)
        assert realised_sparsity(result.weight != 0) >= 0.5

    def test_reconstruction_stays_on_the_quantisation_grid(self, synthetic_layer):
        """A solver that returned continuous survivors would report an unachievable loss."""
        _, weight, gram = synthetic_layer
        mask = build_mask_from_scores(weight.abs(), sparsity=0.5)

        result = reconstruct(gram, weight, mask, local_steps=4, bits=4)
        requantised = quantise_weight(result.weight, bits=4)
        assert requantised.distinct_values_per_group() <= 2**4

    def test_more_local_steps_never_hurts(self, synthetic_layer):
        """local_steps is the fairness unit, so its effect has to be monotone to be meaningful."""
        activations, weight, gram = synthetic_layer
        mask = build_mask_from_scores(
            activation_weighted_saliency(weight, activations.norm(dim=0)), sparsity=0.6
        )

        losses = [
            reconstruct(gram, weight, mask, local_steps=steps, bits=4).final_loss
            for steps in (0, 1, 4)
        ]
        assert losses[1] <= losses[0]
        assert losses[2] <= losses[1]

    def test_zero_local_steps_is_exactly_the_naive_baseline(self, synthetic_layer):
        _, weight, gram = synthetic_layer
        mask = build_mask_from_scores(weight.abs(), sparsity=0.5)

        result = reconstruct(gram, weight, mask, local_steps=0, bits=4)
        assert result.final_loss == result.naive_loss
        assert result.local_steps_used == 0

    def test_the_solve_redistributes_pruned_mass_onto_survivors(self, synthetic_layer):
        """The right-hand side uses the full dense row, not just the surviving entries.

        Solving against only the survivors would ignore what the pruned weights contributed, which
        is most of what reconstruction is for.
        """
        import torch

        _, weight, gram = synthetic_layer
        mask = build_mask_from_scores(weight.abs(), sparsity=0.5)

        solved = solve_masked_rows(gram, weight, mask, damping=1e-6)
        assert not torch.allclose(solved[mask], weight[mask], atol=1e-3)

    def test_an_empty_mask_row_solves_to_zero(self, synthetic_layer):
        import torch

        _, weight, gram = synthetic_layer
        mask = torch.ones_like(weight, dtype=torch.bool)
        mask[0] = False

        solved = solve_masked_rows(gram, weight, mask)
        assert torch.all(solved[0] == 0)

    def test_shape_mismatches_raise(self, synthetic_layer):
        import torch

        _, weight, gram = synthetic_layer
        with pytest.raises(ReconstructionError, match="does not match in_features"):
            reconstruction_loss(torch.eye(IN_FEATURES + 1), weight, weight)


class TestJointVersusSequentialScoring:
    """Decision D3, as executable checks.

    §3.8 makes "mask decisions evaluated under quantised weights" the definition of joint, and
    lists ranking on untouched weights as *not* qualifying. These tests pin down that the two arms
    genuinely see different information, which is the only thing that makes the comparison
    meaningful.
    """

    def test_scoring_on_quantised_weights_differs_from_scoring_on_dense(self, synthetic_layer):
        import torch

        activations, weight, _ = synthetic_layer
        norms = activations.norm(dim=0)

        dense_scores = activation_weighted_saliency(weight, norms)
        joint_scores = activation_weighted_saliency(fake_quantise(weight, bits=2), norms)
        assert not torch.allclose(dense_scores, joint_scores)

    def test_the_two_scorings_can_select_different_masks(self, synthetic_layer):
        import torch

        activations, weight, _ = synthetic_layer
        norms = activations.norm(dim=0)

        sequential_mask = build_mask_from_scores(
            activation_weighted_saliency(weight, norms), sparsity=0.5
        )
        joint_mask = build_mask_from_scores(
            activation_weighted_saliency(fake_quantise(weight, bits=2), norms), sparsity=0.5
        )

        assert not torch.equal(sequential_mask, joint_mask)
        # Whatever they choose, the budget is identical -- that is the §3.11 fairness requirement.
        assert int(sequential_mask.sum()) == int(joint_mask.sum())

    def test_both_scorings_hit_the_same_sparsity(self, synthetic_layer):
        activations, weight, _ = synthetic_layer
        norms = activations.norm(dim=0)

        for scored in (weight, fake_quantise(weight, bits=4)):
            mask = build_mask_from_scores(activation_weighted_saliency(scored, norms), sparsity=0.7)
            assert int((~mask).sum()) == round(mask.numel() * 0.7)
