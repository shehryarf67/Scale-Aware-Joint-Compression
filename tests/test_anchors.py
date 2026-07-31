"""Tests for the independent reference implementations (Amendment A1 §5.5).

An anchor that cannot fail is worthless, so most of these tests inject a *known* fault and assert the
anchor reports it. The agreement tests come second: they only mean something once the disagreement
tests prove the comparison has teeth.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.anchors import (
    AnchorError,
    WandaAnchorReport,
    compare_column_norms,
    compare_masks,
    independent_column_norms,
    independent_wanda_mask,
)

pytestmark = pytest.mark.requires_torch


@pytest.fixture
def weight_and_norms():
    """A small layer with well-separated scores, so no tie can mask a real divergence."""
    import torch

    generator = torch.Generator().manual_seed(20260730)
    weight = torch.randn(8, 16, generator=generator)
    norms = torch.rand(16, generator=generator) + 0.5
    return weight, norms


class TestIndependentWandaMask:
    def test_realises_the_target_sparsity_per_row(self, weight_and_norms):
        weight, norms = weight_and_norms
        mask = independent_wanda_mask(weight, norms, sparsity=0.25)
        pruned_per_row = (~mask).sum(dim=1)
        assert set(pruned_per_row.tolist()) == {4}

    def test_zero_sparsity_keeps_everything(self, weight_and_norms):
        weight, norms = weight_and_norms
        assert bool(independent_wanda_mask(weight, norms, sparsity=0.0).all())

    def test_prunes_the_lowest_scoring_entries(self, weight_and_norms):
        weight, norms = weight_and_norms
        mask = independent_wanda_mask(weight, norms, sparsity=0.5)
        scores = weight.abs() * norms.unsqueeze(0)
        for row in range(scores.shape[0]):
            kept = scores[row][mask[row]]
            dropped = scores[row][~mask[row]]
            assert float(kept.min()) >= float(dropped.max())

    def test_a_dead_column_is_pruned_everywhere(self, weight_and_norms):
        """A zero-norm column scores zero for every row, so it should be dropped in all of them."""
        import torch

        weight, norms = weight_and_norms
        norms = norms.clone()
        norms[3] = 0.0
        mask = independent_wanda_mask(weight, norms, sparsity=0.25)
        assert not bool(mask[:, 3].any())
        assert torch.is_tensor(mask)

    @pytest.mark.parametrize("bad_sparsity", [-0.1, 1.0, 1.5])
    def test_rejects_out_of_range_sparsity(self, weight_and_norms, bad_sparsity):
        weight, norms = weight_and_norms
        with pytest.raises(AnchorError, match="sparsity"):
            independent_wanda_mask(weight, norms, sparsity=bad_sparsity)

    def test_rejects_mismatched_norms(self, weight_and_norms):
        import torch

        weight, _ = weight_and_norms
        with pytest.raises(AnchorError, match="column_norms"):
            independent_wanda_mask(weight, torch.ones(5), sparsity=0.25)


class TestMaskAgreement:
    """The load-bearing case: our mask and the reference must agree on identical inputs."""

    def test_our_mask_matches_the_reference_exactly(self, weight_and_norms):
        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.pruning import activation_weighted_saliency

        weight, norms = weight_and_norms
        scores = activation_weighted_saliency(weight, norms)
        ours = build_mask_from_scores(scores, sparsity=0.25)
        reference = independent_wanda_mask(weight, norms, sparsity=0.25)

        comparison = compare_masks("layer", ours, reference, scores)
        assert comparison.differing_positions == 0
        assert comparison.overlap == 1.0
        assert comparison.explained_by_ties

    def test_a_transposed_norm_broadcast_is_caught(self):
        """Injects a real bug class: weighting by output index instead of input column."""
        import torch

        from scale_aware_compression.compression.masks import build_mask_from_scores

        generator = torch.Generator().manual_seed(7)
        weight = torch.randn(6, 6, generator=generator)
        norms = torch.linspace(0.1, 3.0, 6)

        wrong_scores = weight.abs() * norms.unsqueeze(1)  # per-row, not per-column
        ours = build_mask_from_scores(wrong_scores, sparsity=0.5)
        reference = independent_wanda_mask(weight, norms, sparsity=0.5)

        comparison = compare_masks("layer", ours, reference, wrong_scores)
        assert comparison.differing_positions > 0
        assert not comparison.explained_by_ties

    def test_the_tensor_wide_comparison_group_is_caught(self, weight_and_norms):
        """B-09 was exactly this, and it cost 6.7x perplexity before anyone noticed."""
        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.pruning import activation_weighted_saliency
        from scale_aware_compression.constants import MaskComparisonGroup

        weight, norms = weight_and_norms
        scores = activation_weighted_saliency(weight, norms)
        ours = build_mask_from_scores(
            scores, sparsity=0.25, comparison_group=MaskComparisonGroup.TENSOR
        )
        reference = independent_wanda_mask(weight, norms, sparsity=0.25)

        comparison = compare_masks("layer", ours, reference, scores)
        assert comparison.differing_positions > 0
        assert not comparison.explained_by_ties

    def test_ties_are_reported_separately_from_real_divergence(self):
        """All-equal scores make every choice arbitrary, so a mismatch must not read as a fault."""
        import torch

        weight = torch.ones(4, 8)
        norms = torch.ones(8)
        scores = weight.abs() * norms.unsqueeze(0)
        reference = independent_wanda_mask(weight, norms, sparsity=0.5)
        # A different but equally valid choice among tied scores.
        ours = reference.flip(dims=(1,))

        comparison = compare_masks("layer", ours, reference, scores)
        assert comparison.differing_positions > 0
        assert comparison.explained_by_ties

    def test_shape_mismatch_raises(self):
        import torch

        with pytest.raises(AnchorError, match="shape mismatch"):
            compare_masks(
                "layer",
                torch.ones(2, 2, dtype=torch.bool),
                torch.ones(3, 3, dtype=torch.bool),
                torch.ones(2, 2),
            )


class TestColumnNormAgreement:
    def test_streamed_and_direct_norms_agree(self):
        """Our ``sqrt(diag(X^T X))`` must equal a direct sum of squares."""
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator
        from scale_aware_compression.compression.activations import ActivationStatistics

        generator = torch.Generator().manual_seed(99)
        statistics = ActivationStatistics(12)
        direct = _DirectColumnNormAccumulator(12)
        for _ in range(4):
            batch = torch.randn(5, 7, 12, generator=generator)
            statistics.update(batch)
            direct.update(batch)

        comparison = compare_column_norms("layer", statistics.column_norms(), direct.norms())
        assert comparison.agrees
        assert comparison.max_relative_difference < 1e-4

    def test_a_scaling_fault_is_caught(self):
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        generator = torch.Generator().manual_seed(5)
        direct = _DirectColumnNormAccumulator(6)
        direct.update(torch.randn(10, 6, generator=generator))
        reference = direct.norms()

        comparison = compare_column_norms("layer", reference * 1.01, reference)
        assert not comparison.agrees

    def test_dead_columns_are_counted(self):
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        activations = torch.ones(4, 5)
        activations[:, 2] = 0.0
        direct = _DirectColumnNormAccumulator(5)
        direct.update(activations)
        norms = direct.norms()

        comparison = compare_column_norms("layer", norms, norms)
        assert comparison.dead_columns_ours == 1
        assert comparison.dead_columns_reference == 1

    def test_an_empty_accumulator_raises_rather_than_returning_zeros(self):
        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        with pytest.raises(AnchorError, match="captured nothing"):
            _DirectColumnNormAccumulator(4).norms()

    def test_wrong_width_raises(self):
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        direct = _DirectColumnNormAccumulator(4)
        with pytest.raises(AnchorError, match="last dimension"):
            direct.update(torch.ones(3, 5))


class TestIndependentCaptureOnARealModel:
    def test_captures_every_named_module(self, tiny_causal_lm):
        import torch

        from scale_aware_compression.models.adapters import select_compressible_modules

        model = tiny_causal_lm
        selection = select_compressible_modules(model)
        names = list(selection.names)[:3]
        batches = [{"input_ids": torch.randint(0, 50, (2, 8))} for _ in range(2)]

        norms = independent_column_norms(model, names, batches)
        assert set(norms) == set(names)
        for name in names:
            expected = model.get_submodule(name).in_features
            assert norms[name].shape == (expected,)
            assert bool((norms[name] >= 0).all())

    def test_hooks_are_removed_afterwards(self, tiny_causal_lm):
        import torch

        from scale_aware_compression.models.adapters import select_compressible_modules

        model = tiny_causal_lm
        name = list(select_compressible_modules(model).names)[0]
        module = model.get_submodule(name)
        before = len(module._forward_pre_hooks)

        independent_column_norms(model, [name], [{"input_ids": torch.randint(0, 50, (2, 8))}])
        assert len(module._forward_pre_hooks) == before

    def test_an_unknown_module_name_raises(self, tiny_causal_lm):
        import torch

        model = tiny_causal_lm
        with pytest.raises(AnchorError, match="does not resolve"):
            independent_column_norms(
                model, ["no.such.module"], [{"input_ids": torch.randint(0, 50, (1, 4))}]
            )

    def test_a_batch_without_input_ids_raises(self, tiny_causal_lm):
        from scale_aware_compression.models.adapters import select_compressible_modules

        model = tiny_causal_lm
        name = list(select_compressible_modules(model).names)[0]
        with pytest.raises(AnchorError, match="input_ids"):
            independent_column_norms(model, [name], [{"labels": None}])


class TestAnchorReport:
    def test_an_empty_report_does_not_pass(self):
        """Nothing compared must never read as success."""
        assert not WandaAnchorReport(target_sparsity=0.3).passes

    def test_a_clean_report_passes(self, weight_and_norms):
        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.pruning import activation_weighted_saliency

        weight, norms = weight_and_norms
        scores = activation_weighted_saliency(weight, norms)
        ours = build_mask_from_scores(scores, sparsity=0.25)
        reference = independent_wanda_mask(weight, norms, sparsity=0.25)

        report = WandaAnchorReport(target_sparsity=0.25)
        report.masks.append(compare_masks("a", ours, reference, scores))
        report.norms.append(compare_column_norms("a", norms, norms))
        assert report.passes
        assert report.worst_overlap == 1.0
        assert "PASS" in "\n".join(report.summary_lines())

    def test_a_divergent_report_fails_and_names_the_module(self):
        import torch

        from scale_aware_compression.compression.masks import build_mask_from_scores

        generator = torch.Generator().manual_seed(11)
        weight = torch.randn(4, 8, generator=generator)
        norms = torch.linspace(0.1, 2.0, 8)
        wrong = weight.abs()  # no activation weighting at all
        ours = build_mask_from_scores(wrong, sparsity=0.5)
        reference = independent_wanda_mask(weight, norms, sparsity=0.5)

        report = WandaAnchorReport(target_sparsity=0.5)
        report.masks.append(compare_masks("mlp.dense", ours, reference, wrong))
        report.norms.append(compare_column_norms("mlp.dense", norms, norms))
        assert not report.passes
        summary = "\n".join(report.summary_lines())
        assert "INVESTIGATE" in summary
        assert "mlp.dense" in summary

    def test_the_report_serialises(self, weight_and_norms):
        import json

        from scale_aware_compression.compression.pruning import activation_weighted_saliency

        weight, norms = weight_and_norms
        scores = activation_weighted_saliency(weight, norms)
        reference = independent_wanda_mask(weight, norms, sparsity=0.25)

        report = WandaAnchorReport(target_sparsity=0.25)
        report.masks.append(compare_masks("a", reference, reference, scores))
        report.norms.append(compare_column_norms("a", norms, norms))
        report.notes.append("dense activations both sides")

        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["anchor"] == "wanda_mask_agreement"
        assert payload["modules_compared"] == 1
        assert payload["passes"] is True
        assert payload["notes"] == ["dense activations both sides"]


class TestAnchorsDoNotImportWhatTheyValidate:
    """A reference that calls our code proves only that the call succeeded."""

    def test_the_wanda_module_does_not_import_our_mask_or_saliency_code(self):
        from pathlib import Path

        source = Path("src/scale_aware_compression/anchors/wanda.py").read_text(encoding="utf-8")
        for forbidden in (
            "from scale_aware_compression.compression.masks import",
            "from scale_aware_compression.compression.pruning import",
            "from scale_aware_compression.compression.activations import",
            "build_mask_from_scores(",
            "activation_weighted_saliency(",
        ):
            assert forbidden not in source, (
                f"anchors/wanda.py references {forbidden!r}. An independent reference must "
                "re-derive its quantities, not delegate to the implementation under test."
            )


class TestAccumulatorHandlesRealCaptureConditions:
    """Regression tests for faults hit while running the anchor on a real model."""

    def test_the_buffer_stays_on_cpu_whatever_the_activation_device(self):
        """The first real run died here: a CPU buffer summing CUDA activations.

        Reducing on the activation's device and moving only the length-``in_features`` result keeps
        this correct without paying for float64 arithmetic on a consumer GPU.
        """
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        direct = _DirectColumnNormAccumulator(6)
        direct.update(torch.ones(4, 6))
        assert direct.norms().device.type == "cpu"

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    def test_accumulates_regardless_of_activation_dtype(self, dtype: str):
        import torch

        from scale_aware_compression.anchors.wanda import _DirectColumnNormAccumulator

        activations = torch.full((9, 4), 2.0, dtype=getattr(torch, dtype))
        direct = _DirectColumnNormAccumulator(4)
        direct.update(activations)
        # Each column holds nine 2s, so the norm is sqrt(9 * 4) = 6.
        assert torch.allclose(direct.norms(), torch.full((4,), 6.0), atol=1e-3)

    def test_a_near_dead_column_does_not_read_as_a_huge_relative_error(self):
        """Dividing by a near-zero norm turns float32 noise into a meaningless ratio."""
        import torch

        norms = torch.tensor([100.0, 50.0, 1e-9])
        perturbed = norms.clone()
        perturbed[2] = 2e-9  # a 100% "relative" change on a column carrying no energy

        comparison = compare_column_norms("layer", perturbed, norms)
        assert comparison.agrees

    def test_a_real_error_on_a_live_column_still_fails(self):
        import torch

        norms = torch.tensor([100.0, 50.0, 10.0])
        perturbed = norms.clone()
        perturbed[1] = 50.5

        comparison = compare_column_norms("layer", perturbed, norms)
        assert not comparison.agrees


class TestTheVerdictSeparatesSelectionFromPrecision:
    """The verdict must turn on selection, not on which way float32 breaks a tie.

    The first real run on Pythia-160M disagreed on 4 positions out of 84,934,656. Feeding both
    selectors the same float64 norms dropped that to 0, and the disputed pairs turned out to tie
    exactly in float64 while differing by 2-3 ULPs in float32. So the selectors were identical and
    the old heuristic verdict was wrong, not the code it was judging.
    """

    def _tie_pair(self):
        import torch

        from scale_aware_compression.compression.masks import build_mask_from_scores

        # Two columns with identical scores; exactly one survives, so the choice is arbitrary.
        weight = torch.tensor([[1.0, 1.0, 5.0, 5.0]])
        norms = torch.ones(4)
        scores = weight.abs() * norms.unsqueeze(0)
        ours = build_mask_from_scores(scores, sparsity=0.25)
        flipped = ours.clone()
        first_pruned = (~ours[0]).nonzero(as_tuple=True)[0][0].item()
        other = 1 - first_pruned if first_pruned in (0, 1) else first_pruned
        flipped[0, first_pruned] = True
        flipped[0, other] = False
        return scores, ours, flipped

    def test_a_matched_norm_disagreement_fails_even_when_tie_explained(self):
        """Identical inputs must give identical output; a tie is no longer an excuse here."""
        scores, ours, flipped = self._tie_pair()
        comparison = compare_masks("layer", ours, flipped, scores)
        assert comparison.explained_by_ties  # still true as a diagnostic

        report = WandaAnchorReport(target_sparsity=0.25)
        report.masks.append(comparison)
        report.norms.append(compare_column_norms("layer", scores[0], scores[0]))
        assert not report.masks_agree
        assert not report.passes

    def test_precision_divergence_does_not_gate_the_verdict(self):
        import torch

        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.pruning import activation_weighted_saliency

        weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        norms = torch.ones(4)
        scores = activation_weighted_saliency(weight, norms)
        mask = build_mask_from_scores(scores, sparsity=0.25)

        report = WandaAnchorReport(target_sparsity=0.25)
        report.masks.append(compare_masks("layer", mask, mask, scores))
        report.norms.append(compare_column_norms("layer", norms, norms))
        # A float32-versus-float64 flip, reported but not fatal.
        _, ours, flipped = self._tie_pair()
        report.precision_divergence.append(compare_masks("layer", ours, flipped, scores))

        assert report.precision_sensitive_positions > 0
        assert report.passes, "precision-sensitive ties must not fail the anchor"

    def test_precision_sensitive_positions_sums_across_modules(self):
        scores, ours, flipped = self._tie_pair()
        report = WandaAnchorReport(target_sparsity=0.25)
        for name in ("a", "b"):
            report.precision_divergence.append(compare_masks(name, ours, flipped, scores))
        assert report.precision_sensitive_positions == 2 * int((ours != flipped).sum())

    def test_the_report_exposes_both_comparisons(self):
        import json

        scores, ours, flipped = self._tie_pair()
        report = WandaAnchorReport(target_sparsity=0.25)
        report.masks.append(compare_masks("a", ours, ours, scores))
        report.norms.append(compare_column_norms("a", scores[0], scores[0]))
        report.precision_divergence.append(compare_masks("a", ours, flipped, scores))

        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["passes"] is True
        assert payload["precision_sensitive_positions"] > 0
        assert len(payload["precision_divergence"]) == 1
        assert "precision-sensitive" in "\n".join(report.summary_lines())
