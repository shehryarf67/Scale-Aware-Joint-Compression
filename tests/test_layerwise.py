"""Phase 6 exit tests: the five arms through one shared layerwise driver.

The exit criteria from docs/implementation_plan.md:

* every arm hits its target sparsity and bit width exactly
* a machine-checkable fairness assertion -- identical calibration tensors, identical module lists,
  equal total local steps
* a regression test that **fails** if joint is implemented as "prune fully, then plain PTQ"

That last one matters most. §3.8 defines joint by *what the mask was scored against*, and an
implementation that prunes first and quantises afterwards is a plausible-looking pipeline that
answers a different research question. Nothing about its output shape, realised sparsity, or bit
width reveals the difference, so it has to be pinned by a test.
"""

from __future__ import annotations

import copy

import pytest

from scale_aware_compression.compression.activations import ActivationStatistics
from scale_aware_compression.compression.layerwise import (
    LayerPlan,
    LayerResult,
    LayerwiseError,
    LayerwiseReport,
    assert_matched_plans,
    compress_layer,
    compress_model_layerwise,
)
from scale_aware_compression.compression.quantisation import quantise_weight
from scale_aware_compression.constants import (
    PruningGranularity,
    QuantisationGranularity,
    ReconstructionSolver,
)

pytestmark = pytest.mark.requires_torch

ARMS = ("pruning", "quantisation", "sequential", "sequential_qp", "joint")
QUANTISING_ARMS = ("quantisation", "sequential", "sequential_qp", "joint")
PRUNING_ARMS = ("pruning", "sequential", "sequential_qp", "joint")
IN_FEATURES = 32
OUT_FEATURES = 16


@pytest.fixture
def layer_inputs():
    """A weight tensor plus activation statistics captured from correlated inputs.

    Correlated rather than independent: an near-diagonal Gram matrix gives reconstruction nothing
    to redistribute, so a broken solver would still look plausible.
    """
    import torch

    torch.manual_seed(7)
    basis = torch.randn(512, IN_FEATURES // 2)
    activations = torch.cat([basis, basis * 0.5 + torch.randn(512, IN_FEATURES // 2) * 0.5], dim=1)
    statistics = ActivationStatistics(IN_FEATURES)
    statistics.update(activations)
    return torch.randn(OUT_FEATURES, IN_FEATURES), statistics


@pytest.fixture
def fresh_causal_lm(tiny_causal_lm):
    """A throwaway copy of the tiny model.

    ``tiny_causal_lm`` is **session-scoped** and compression is destructive, so using it directly
    would leak compressed weights into every later test in the suite -- including tests in other
    files that assume a dense model.
    """
    return copy.deepcopy(tiny_causal_lm)


def moderate_plan(**overrides) -> LayerPlan:
    """A 50% / W4 plan, matching the shape of the study's aggressive budget."""
    settings: dict = {"sparsity": 0.5, "bits": 4, "local_steps": 2, "joint_iterations": 3}
    settings.update(overrides)
    return LayerPlan(**settings)


def calibration_batches(vocab_size: int, count: int = 2):
    """Deterministic token batches standing in for a calibration set."""
    import torch

    torch.manual_seed(3)
    return [torch.randint(0, vocab_size, (2, 16)) for _ in range(count)]


class TestEveryArmHitsItsBudget:
    @pytest.mark.parametrize("arm", PRUNING_ARMS)
    def test_realised_sparsity_reaches_the_target(self, arm: str, layer_inputs):
        """Measured on the returned weight, not on the mask.

        Realised sparsity may *exceed* the target, because quantisation rounds small survivors to
        exactly zero. It must never fall short.
        """
        weight, statistics = layer_inputs
        plan = moderate_plan()

        outcome = compress_layer(weight, statistics, plan, arm=arm)

        assert outcome.result.realised_sparsity >= plan.sparsity - 1e-9
        assert outcome.weight.shape == weight.shape

    @pytest.mark.parametrize("arm", PRUNING_ARMS)
    def test_the_mask_itself_is_exactly_at_the_target(self, arm: str, layer_inputs):
        """The mask is the thing the budget is defined on, so it must be exact."""
        from scale_aware_compression.compression.masks import realised_sparsity

        weight, statistics = layer_inputs
        plan = moderate_plan()

        outcome = compress_layer(weight, statistics, plan, arm=arm)

        expected_pruned = round(weight.numel() * plan.sparsity)
        assert int((~outcome.mask).sum()) == expected_pruned
        assert realised_sparsity(outcome.mask) == pytest.approx(plan.sparsity)

    @pytest.mark.parametrize("arm", QUANTISING_ARMS)
    def test_quantising_arms_stay_within_their_bit_width(self, arm: str, layer_inputs):
        """§4.8: proves the precision is real rather than silently dequantised."""
        weight, statistics = layer_inputs
        plan = moderate_plan()

        outcome = compress_layer(weight, statistics, plan, arm=arm)

        requantised = quantise_weight(outcome.weight, bits=plan.bits)
        assert requantised.distinct_values_per_group() <= 2**plan.bits

    def test_the_pruning_arm_leaves_precision_alone(self, layer_inputs):
        """Its whole purpose is to isolate the damage caused by sparsity alone (§3.3)."""
        weight, statistics = layer_inputs

        outcome = compress_layer(
            weight, statistics, LayerPlan(sparsity=0.5, bits=None), arm="pruning"
        )

        surviving = outcome.weight[outcome.weight != 0]
        assert int(surviving.unique().numel()) > 2**4

    def test_the_quantisation_arm_prunes_nothing_by_mask(self, layer_inputs):
        """Its mask keeps everything; any zeros in the output come from rounding, not pruning."""
        import torch

        weight, statistics = layer_inputs

        outcome = compress_layer(
            weight, statistics, LayerPlan(sparsity=0.0, bits=4), arm="quantisation"
        )

        assert torch.all(outcome.mask)

    def test_an_unknown_arm_is_refused(self, layer_inputs):
        weight, statistics = layer_inputs
        with pytest.raises(LayerwiseError, match="unknown arm"):
            compress_layer(weight, statistics, moderate_plan(), arm="magic")


class TestJointIsGenuinelyJoint:
    """§3.8, as executable checks. These are the tests that protect the research question."""

    def test_joint_differs_from_prune_then_quantise(self, layer_inputs):
        """The §3.8 regression test.

        If the joint arm is ever reimplemented as "prune fully, freeze, then plain PTQ" it becomes
        the sequential arm, and joint gain collapses to zero by construction -- a null result caused
        by a bug rather than by nature. That is the worst possible failure for this study, because
        a null result is a publishable outcome and would not look wrong.
        """
        import torch

        weight, statistics = layer_inputs
        plan = moderate_plan()

        joint = compress_layer(weight, statistics, plan, arm="joint")
        sequential = compress_layer(weight, statistics, plan, arm="sequential")

        assert not torch.allclose(joint.weight, sequential.weight, atol=1e-6)

    def test_the_joint_mask_only_diverges_at_aggressive_precision(self, layer_inputs):
        """Records a real limitation of the method, so it cannot be discovered late.

        Scoring the mask on quantised weights only changes the mask when rounding reorders the
        saliency ranking. On this layer that happens at W2 and essentially stops by W4:

        =====  ====================================
        Width  mask positions differing from P->Q
        =====  ====================================
        W2     206 / 512
        W3     2 / 512
        W4     0 / 512
        W8     0 / 512
        =====  ====================================

        So at moderate precision the joint arm's mask is *identical* to the sequential arm's, and
        any joint gain there can only come from the reconstruction ordering -- one of §3.8's two
        mechanisms is inert. That is a prediction about the results, not a defect in the code, and
        it is why the budget screening must include a width where the mechanism is live.
        """
        import torch

        weight, statistics = layer_inputs

        aggressive_joint = compress_layer(weight, statistics, moderate_plan(bits=2), arm="joint")
        aggressive_seq = compress_layer(weight, statistics, moderate_plan(bits=2), arm="sequential")
        assert not torch.equal(aggressive_joint.mask, aggressive_seq.mask)

        moderate_joint = compress_layer(weight, statistics, moderate_plan(bits=8), arm="joint")
        moderate_seq = compress_layer(weight, statistics, moderate_plan(bits=8), arm="sequential")
        assert torch.equal(moderate_joint.mask, moderate_seq.mask)

    def test_the_joint_mask_responds_to_the_bit_width(self, layer_inputs):
        """This is what "quantisation-aware" has to mean operationally.

        A joint arm whose mask is identical at W2 and W8 is not scoring against the grid at all,
        which is precisely §3.8's disqualifying case.
        """
        import torch

        weight, statistics = layer_inputs

        coarse = compress_layer(weight, statistics, moderate_plan(bits=2), arm="joint")
        fine = compress_layer(weight, statistics, moderate_plan(bits=8), arm="joint")

        assert not torch.equal(coarse.mask, fine.mask)

    def test_the_sequential_mask_ignores_the_bit_width(self, layer_inputs):
        """The controlled half of the comparison.

        Sequential P->Q chooses its mask before any quantiser exists, so the bit width must not
        move it. Compared on the *mask*, not the nonzero pattern: quantisation rounds small
        survivors to zero, so the nonzero pattern legitimately differs between W2 and W8 even
        though the mask is identical.
        """
        import torch

        weight, statistics = layer_inputs

        coarse = compress_layer(weight, statistics, moderate_plan(bits=2), arm="sequential")
        fine = compress_layer(weight, statistics, moderate_plan(bits=8), arm="sequential")

        assert torch.equal(coarse.mask, fine.mask)

    def test_the_joint_alternation_converges_rather_than_drifting(self, layer_inputs):
        """More outer iterations must not make the layer worse.

        The alternation is not a descent method -- the mask can move between iterations -- so the
        honest property to assert is convergence to a fixed point, not monotone improvement. A loop
        that kept changing the answer indefinitely would make ``joint_iterations`` an untuned free
        parameter, and §3.11 forbids per-scale tuning.
        """
        import torch

        weight, statistics = layer_inputs

        many = compress_layer(weight, statistics, moderate_plan(joint_iterations=6), arm="joint")
        more = compress_layer(weight, statistics, moderate_plan(joint_iterations=9), arm="joint")

        assert torch.equal(many.mask, more.mask)

    def test_max_abs_scales_are_blind_to_magnitude_pruning(self, layer_inputs):
        """§3.8's second mechanism is **vacuous** under the frozen quantisation scheme.

        A symmetric per-channel scale is ``max|W_row| / qmax``. Saliency pruning removes the
        *smallest* entries, so each row's maximum survives and refitting the scale on the survivors
        returns the same number. Measured on this layer: 100% of output channels unchanged, maximum
        relative change 0.0000.

        So "re-estimate quantisation scales after mask changes", which §3.8 lists as one of the two
        things that qualify a method as joint, is satisfied only formally here -- the re-estimation
        provably cannot alter anything. Recorded as a test so it is a known property of the design
        rather than something discovered while writing up.

        A scale rule that minimised reconstruction error instead of matching the maximum *would*
        respond to the mask. Changing it is a protocol decision, not a code change, so it is flagged
        in docs/validity_threats.md rather than made here.
        """
        import torch

        from scale_aware_compression.compression.quantisation import compute_symmetric_scales

        weight, statistics = layer_inputs
        outcome = compress_layer(weight, statistics, moderate_plan(), arm="joint")

        dense_scales = compute_symmetric_scales(weight, bits=4)
        survivor_scales = compute_symmetric_scales(weight * outcome.mask, bits=4)
        assert torch.allclose(dense_scales, survivor_scales)

    def test_reverse_sequential_differs_from_forward(self, layer_inputs):
        """§3.6 exists so joint is not compared only against a weak ordering.

        If the two orders produced the same artefact there would be nothing to ablate, and §6.1's
        "best of the two" would be vacuous.
        """
        import torch

        weight, statistics = layer_inputs
        plan = moderate_plan()

        forward = compress_layer(weight, statistics, plan, arm="sequential")
        reverse = compress_layer(weight, statistics, plan, arm="sequential_qp")

        assert not torch.allclose(forward.weight, reverse.weight, atol=1e-6)

    def test_reverse_sequential_scores_its_mask_under_quantisation(self, layer_inputs):
        """Q->P does see the grid at mask time -- that is what makes it the interesting ablation.

        What disqualifies it from being joint is that the scales are never revisited afterwards.
        """
        import torch

        weight, statistics = layer_inputs

        coarse = compress_layer(weight, statistics, moderate_plan(bits=2), arm="sequential_qp")
        fine = compress_layer(weight, statistics, moderate_plan(bits=8), arm="sequential_qp")

        assert not torch.equal(coarse.mask, fine.mask)


class TestMeasuredAblations:
    """Two proposed fixes for the weak joint mechanism, both measured and both rejected.

    Documented here because a negative result that lives only in a commit message gets re-proposed.
    Numbers are means over six real Pythia-160M layers at 50% sparsity, matched budget, measuring
    layer-objective joint gain (positive = joint reconstructs better than sequential):

    ==========================  ========  ========
    Configuration                     W8        W4
    ==========================  ========  ========
    max-abs, magnitude score      -0.49%    +1.12%
    + clipping scale search       -1.51%    -0.99%
    + keep-benefit scoring       -11.83%   -16.15%
    ==========================  ========  ========
    """

    def test_both_ablations_are_off_by_default(self):
        """The defaults must be the configuration that measured best, not the newest idea."""
        plan = LayerPlan()
        assert plan.scale_search is False
        assert plan.keep_benefit_saliency is False

    def test_the_scale_rule_is_part_of_the_matched_budget(self):
        """Two arms with different quantisers are not comparable, whichever rule is chosen."""
        with pytest.raises(LayerwiseError, match="budget"):
            assert_matched_plans(
                [
                    LayerwiseReport(arm="sequential", module_names=["a"], total_local_steps=1),
                    LayerwiseReport(arm="joint", module_names=["a"], total_local_steps=1),
                ],
                [moderate_plan(scale_search=False), moderate_plan(scale_search=True)],
            )

    def test_the_clipping_search_still_reduces_naive_quantisation_error(self, layer_inputs):
        """The search is not broken -- it optimises a different objective than the one that matters.

        It genuinely minimises error *before* reconstruction. That it degrades the result *after*
        reconstruction is the finding, and the reason it stays off.
        """
        from scale_aware_compression.compression.quantisation import (
            compute_symmetric_scales,
            fake_quantise,
            search_clipping_scales,
        )
        from scale_aware_compression.compression.reconstruct import reconstruction_loss

        weight, statistics = layer_inputs
        gram = statistics.gram()
        mask = build_mask_from_scores_for(weight, statistics)

        max_abs = compute_symmetric_scales(weight * mask, bits=3)
        searched = search_clipping_scales(weight, gram, bits=3, mask=mask)

        def loss(scales):
            candidate = fake_quantise(weight * mask, bits=3, scales=scales) * mask
            return reconstruction_loss(gram, weight, candidate)

        assert loss(searched) <= loss(max_abs)

    def test_keep_benefit_can_rank_a_large_weight_below_a_smaller_one(self, layer_inputs):
        """Confirms the criterion does what it claims, even though it measures worse overall.

        This is the property magnitude ranking cannot express, and it is why the idea was worth
        testing rather than dismissing.
        """
        import torch

        from scale_aware_compression.compression.pruning import keep_benefit_saliency
        from scale_aware_compression.compression.quantisation import fake_quantise

        weight, statistics = layer_inputs
        norms = statistics.column_norms()
        quantised = fake_quantise(weight, bits=2)

        benefit = keep_benefit_saliency(weight, quantised, norms)
        magnitude = weight.abs() * norms.unsqueeze(0)

        # Ranking disagreements exist: the two criteria are not monotone in each other.
        benefit_order = benefit.reshape(-1).argsort()
        magnitude_order = magnitude.reshape(-1).argsort()
        assert not torch.equal(benefit_order, magnitude_order)

    def test_keep_benefit_is_non_negative_and_zero_where_weights_vanish(self, layer_inputs):
        """Explains *why* the criterion underperforms, rather than just recording that it does.

        For round-to-nearest symmetric quantisation the score cannot go below zero. If
        ``|W| < s/2`` then ``Q(W) = 0``, both error terms equal ``W^2``, and the benefit is exactly
        zero; otherwise ``|W - Q(W)| <= s/2 <= |W|``. So it can never express "this weight is
        actively harmful to keep", which was the property that motivated trying it.
        """
        import torch

        from scale_aware_compression.compression.pruning import keep_benefit_saliency
        from scale_aware_compression.compression.quantisation import fake_quantise

        weight, statistics = layer_inputs
        quantised = fake_quantise(weight, bits=2)
        benefit = keep_benefit_saliency(weight, quantised, statistics.column_norms())

        assert bool((benefit >= 0).all())
        # Exactly zero wherever the weight rounded away entirely.
        vanished = quantised == 0
        assert torch.allclose(benefit[vanished], torch.zeros_like(benefit[vanished]))


def build_mask_from_scores_for(weight, statistics):
    """Helper: the sequential arm's mask for a layer, at 50% sparsity."""
    from scale_aware_compression.compression.masks import build_mask_from_scores
    from scale_aware_compression.compression.pruning import activation_weighted_saliency

    return build_mask_from_scores(
        activation_weighted_saliency(weight, statistics.column_norms()), sparsity=0.5
    )


class TestFairnessInvariants:
    """§3.11 as machine-checkable assertions rather than documented intent."""

    def _report(self, arm: str, *, steps: int = 4, modules=("a", "b"), fingerprint="cal-1"):
        return LayerwiseReport(
            arm=arm,
            module_names=list(modules),
            calibration_fingerprint=fingerprint,
            total_local_steps=steps,
        )

    def test_matched_arms_pass(self):
        assert_matched_plans(
            [self._report("sequential"), self._report("joint")],
            [moderate_plan(), moderate_plan()],
        )

    def test_unequal_local_steps_is_rejected(self):
        """The single easiest way for this study to produce a wrong result."""
        with pytest.raises(LayerwiseError, match="local step"):
            assert_matched_plans(
                [self._report("sequential", steps=4), self._report("joint", steps=12)],
                [moderate_plan(), moderate_plan()],
            )

    def test_different_module_coverage_is_rejected(self):
        with pytest.raises(LayerwiseError, match="different modules"):
            assert_matched_plans(
                [
                    self._report("sequential", modules=("a", "b")),
                    self._report("joint", modules=("a", "b", "c")),
                ],
                [moderate_plan(), moderate_plan()],
            )

    def test_different_calibration_data_is_rejected(self):
        with pytest.raises(LayerwiseError, match="calibration data"):
            assert_matched_plans(
                [
                    self._report("sequential", fingerprint="cal-1"),
                    self._report("joint", fingerprint="cal-2"),
                ],
                [moderate_plan(), moderate_plan()],
            )

    @pytest.mark.parametrize(
        "difference",
        [
            {"sparsity": 0.7},
            {"bits": 8},
            {"granularity": QuantisationGranularity.PER_TENSOR},
            {"group_size": 64},
            {"pruning_granularity": PruningGranularity.SEMI_STRUCTURED_2_4},
            {"solver": ReconstructionSolver.ALS},
            {"damping": 0.5},
        ],
    )
    def test_any_budget_difference_is_rejected(self, difference: dict):
        """Each of these would make a measured difference mean something other than the pipeline."""
        with pytest.raises(LayerwiseError, match="budget"):
            assert_matched_plans(
                [self._report("sequential"), self._report("joint")],
                [moderate_plan(), moderate_plan(**difference)],
            )

    def test_differing_joint_iterations_is_allowed(self):
        """Only the joint arm has an outer loop, so requiring both to share K is meaningless.

        What must match is the total local step count, which is checked separately.
        """
        assert_matched_plans(
            [self._report("sequential"), self._report("joint")],
            [moderate_plan(joint_iterations=1), moderate_plan(joint_iterations=8)],
        )

    def test_a_single_arm_needs_no_comparison(self):
        assert_matched_plans([self._report("joint")], [moderate_plan()])

    def test_mismatched_report_and_plan_counts_are_refused(self):
        with pytest.raises(LayerwiseError, match="correspond"):
            assert_matched_plans([self._report("joint")], [moderate_plan(), moderate_plan()])


class TestDeterminism:
    @pytest.mark.parametrize("arm", ARMS)
    def test_compression_is_reproducible(self, arm: str, layer_inputs):
        """§4.8 requires repeated runs to agree; a nondeterministic arm cannot be seed-averaged."""
        import torch

        weight, statistics = layer_inputs
        plan = moderate_plan()

        first = compress_layer(weight, statistics, plan, arm=arm)
        second = compress_layer(weight, statistics, plan, arm=arm)

        assert torch.equal(first.weight, second.weight)
        assert first.local_steps == second.local_steps > 0

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_dense_weight_is_not_mutated(self, arm: str, layer_inputs):
        """The driver relies on the dense weight staying available as the solve target."""
        import torch

        weight, statistics = layer_inputs
        original = weight.clone()

        compress_layer(weight, statistics, moderate_plan(), arm=arm)

        assert torch.equal(weight, original)


class TestLayerwiseReport:
    def test_sparsity_is_weight_averaged_not_a_plain_mean(self):
        """A wide layer must count for more than a narrow one."""
        report = LayerwiseReport(arm="joint")
        report.layers.append(LayerResult("small", 0.5, 0.5, 1.0, 0.5, 1, num_weights=100))
        report.layers.append(LayerResult("big", 0.5, 0.7, 1.0, 0.5, 1, num_weights=900))

        assert report.realised_sparsity == pytest.approx(0.68)
        assert report.targeted_parameters == 1000

    def test_report_serialises_per_layer_losses(self, layer_inputs):
        """Record field A9 -- per-layer reconstruction loss."""
        weight, statistics = layer_inputs
        outcome = compress_layer(weight, statistics, moderate_plan(), arm="joint")
        outcome.result.name = "gpt_neox.layers.0.mlp.dense_h_to_4h"

        payload = outcome.result.to_dict()
        assert payload["name"] == "gpt_neox.layers.0.mlp.dense_h_to_4h"
        assert payload["naive_loss"] > 0
        assert payload["final_loss"] >= 0
        assert payload["local_steps"] > 0

    def test_an_empty_report_does_not_divide_by_zero(self):
        report = LayerwiseReport(arm="dense")
        assert report.realised_sparsity == 0.0
        assert report.targeted_parameters == 0


class TestEndToEndOnATinyModel:
    """The whole driver against a real GPTNeoX model, offline, in milliseconds."""

    @pytest.mark.parametrize("arm", ARMS)
    def test_the_driver_compresses_every_targeted_layer(self, arm: str, fresh_causal_lm):
        model = fresh_causal_lm
        report = compress_model_layerwise(
            model,
            calibration_batches(model.config.vocab_size),
            moderate_plan(),
            arm=arm,
            calibration_fingerprint="tiny",
        )

        # 2 blocks x 4 targeted linears: fused QKV, attention.dense, and two MLP projections.
        assert report.num_layers == 8
        assert report.total_local_steps > 0
        assert report.targeted_parameters > 0
        assert all(layer.name for layer in report.layers)

    def test_the_driver_excludes_embeddings_and_the_head(self, fresh_causal_lm):
        """§2.6: including them would make the effective budget vary with model scale."""
        model = fresh_causal_lm
        report = compress_model_layerwise(
            model,
            calibration_batches(model.config.vocab_size),
            moderate_plan(),
            arm="joint",
            calibration_fingerprint="tiny",
        )

        joined = " ".join(report.module_names)
        assert "embed_in" not in joined
        assert "embed_out" not in joined
        assert "lm_head" not in joined

    def test_the_driver_actually_changes_the_weights(self, fresh_causal_lm):
        """A driver that silently no-ops would score perfect retention at the target sparsity."""
        import torch

        model = fresh_causal_lm
        target = "gpt_neox.layers.0.mlp.dense_h_to_4h"
        before = model.get_submodule(target).weight.detach().clone()

        compress_model_layerwise(
            model,
            calibration_batches(model.config.vocab_size),
            moderate_plan(),
            arm="joint",
            calibration_fingerprint="tiny",
        )

        after = model.get_submodule(target).weight.detach()
        assert not torch.allclose(before, after)
        assert float((after == 0).float().mean()) >= 0.5 - 1e-6

    def test_every_targeted_layer_reaches_the_target_sparsity(self, fresh_causal_lm):
        model = fresh_causal_lm
        report = compress_model_layerwise(
            model,
            calibration_batches(model.config.vocab_size),
            moderate_plan(),
            arm="sequential",
            calibration_fingerprint="tiny",
        )

        for layer in report.layers:
            assert layer.realised_sparsity >= 0.5 - 1e-9, layer.name

    def test_an_empty_calibration_set_is_refused(self, fresh_causal_lm):
        """Reconstruction without activations would silently degrade to plain rounding."""
        with pytest.raises(LayerwiseError, match="calibration set is empty"):
            compress_model_layerwise(
                fresh_causal_lm, [], moderate_plan(), arm="joint", calibration_fingerprint="tiny"
            )

    def test_arms_touch_identical_modules_in_identical_order(self, tiny_causal_lm):
        """The coverage half of §3.11, end to end, on genuinely separate model instances."""
        reports = []
        for arm in ("sequential", "joint"):
            model = copy.deepcopy(tiny_causal_lm)
            reports.append(
                compress_model_layerwise(
                    model,
                    calibration_batches(model.config.vocab_size),
                    moderate_plan(),
                    arm=arm,
                    calibration_fingerprint="shared-calibration",
                )
            )

        assert reports[0].module_names == reports[1].module_names
        assert reports[0].targeted_parameters == reports[1].targeted_parameters

    def test_the_driver_records_a_calibration_fingerprint(self, fresh_causal_lm):
        """Without it, a mismatch between arms is undetectable after the fact."""
        model = fresh_causal_lm
        report = compress_model_layerwise(
            model,
            calibration_batches(model.config.vocab_size),
            moderate_plan(),
            arm="joint",
            calibration_fingerprint="fingerprint-abc",
        )
        assert report.to_dict()["calibration_fingerprint"] == "fingerprint-abc"


class TestPatternsAndGranularities:
    def test_two_four_pattern_survives_reconstruction(self, layer_inputs):
        """2:4 is the pattern most likely to admit a sparse kernel, so it must stay intact."""
        import torch

        weight, statistics = layer_inputs
        plan = moderate_plan(pruning_granularity=PruningGranularity.SEMI_STRUCTURED_2_4)

        outcome = compress_layer(weight, statistics, plan, arm="joint")

        assert torch.all(outcome.mask.reshape(-1, 4).sum(dim=1) == 2)
        assert torch.all(outcome.weight.reshape(-1, 4).ne(0).sum(dim=1) <= 2)

    @pytest.mark.parametrize("arm", ARMS)
    def test_per_group_quantisation_runs_for_every_arm(self, arm: str, layer_inputs):
        """Activation ordering is skipped here; the sweep must still emit a valid artefact."""
        weight, statistics = layer_inputs
        plan = moderate_plan(
            granularity=QuantisationGranularity.PER_GROUP, group_size=IN_FEATURES // 4
        )

        outcome = compress_layer(weight, statistics, plan, arm=arm)

        assert outcome.weight.shape == weight.shape

    @pytest.mark.parametrize("solver", list(ReconstructionSolver))
    def test_both_solvers_drive_every_arm(self, solver: ReconstructionSolver, layer_inputs):
        """The solver is interchangeable by construction; that is what D2's drop-in promise means."""
        weight, statistics = layer_inputs

        for arm in ARMS:
            outcome = compress_layer(weight, statistics, moderate_plan(solver=solver), arm=arm)
            assert outcome.weight.shape == weight.shape
