"""Tests for the exact-optimum reconstruction anchor (Amendment A1 §5.5b).

The reference must genuinely be the optimum, or every comparison against it is meaningless. So the
first group verifies the optimum's defining property directly, by random search rather than by
re-deriving the algebra it was built from. The second group injects impossible results and asserts the
invariants catch them.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.anchors import (
    ExactReconstructionError,
    ExactReconstructionReport,
    RowComparison,
    compare_row,
    exact_masked_row_optimum,
    row_objective,
)

pytestmark = pytest.mark.requires_torch


@pytest.fixture
def problem():
    """A small well-conditioned masked least-squares problem."""
    import torch

    generator = torch.Generator().manual_seed(4242)
    activations = torch.randn(64, 10, generator=generator)
    gram = activations.t() @ activations
    weight = torch.randn(5, 10, generator=generator)
    keep = torch.ones(10, dtype=torch.bool)
    keep[[2, 5, 7]] = False
    return gram, weight[0], keep


@pytest.fixture
def layer():
    """A synthetic layer wide enough for the sweep to have something to compensate."""
    import torch

    generator = torch.Generator().manual_seed(77)
    activations = torch.randn(128, 16, generator=generator)
    gram = activations.t() @ activations
    weight = torch.randn(6, 16, generator=generator)
    return gram, weight


class TestExactMaskedOptimum:
    def test_no_feasible_perturbation_can_improve_it(self, problem):
        """The defining property of a minimiser, checked empirically rather than algebraically."""
        import torch

        gram, row, keep = problem
        optimal = exact_masked_row_optimum(gram, row, keep)
        best = row_objective(gram, row, optimal)

        generator = torch.Generator().manual_seed(1)
        for _ in range(40):
            step = torch.randn(10, generator=generator, dtype=torch.float64) * 0.05
            candidate = optimal + step * keep  # stay inside the feasible set
            assert row_objective(gram, row, candidate) >= best - 1e-9

    def test_the_optimum_respects_the_mask(self, problem):
        gram, row, keep = problem
        optimal = exact_masked_row_optimum(gram, row, keep)
        assert not bool((optimal[~keep] != 0).any())

    def test_the_optimum_beats_naive_masking(self, problem):
        import torch

        gram, row, keep = problem
        naive = torch.where(keep, row, torch.zeros_like(row))
        assert row_objective(gram, row, exact_masked_row_optimum(gram, row, keep)) < row_objective(
            gram, row, naive
        )

    def test_it_redistributes_pruned_mass_rather_than_just_zeroing(self, problem):
        """Error compensation is most of what reconstruction buys, so it must actually happen."""
        import torch

        gram, row, keep = problem
        optimal = exact_masked_row_optimum(gram, row, keep)
        naive = torch.where(keep, row, torch.zeros_like(row)).to(torch.float64)
        # Survivors must move, otherwise the solve reduced to plain masking.
        assert not torch.allclose(optimal[keep], naive[keep], atol=1e-8)

    def test_an_all_keep_mask_recovers_the_dense_row(self, problem):
        import torch

        gram, row, _ = problem
        optimal = exact_masked_row_optimum(gram, row, torch.ones(10, dtype=torch.bool))
        assert row_objective(gram, row, optimal) < 1e-12

    def test_an_empty_mask_gives_the_zero_row(self, problem):
        import torch

        gram, row, _ = problem
        optimal = exact_masked_row_optimum(gram, row, torch.zeros(10, dtype=torch.bool))
        assert not bool((optimal != 0).any())

    def test_a_singular_submatrix_falls_back_rather_than_raising(self):
        """Real calibration Grams are routinely rank-deficient; the reference must cope."""
        import torch

        activations = torch.tensor([[1.0, 1.0, 2.0], [2.0, 2.0, 1.0], [0.5, 0.5, 3.0]])
        gram = activations.t() @ activations  # columns 0 and 1 are identical
        row = torch.tensor([1.0, 2.0, 3.0])
        optimal = exact_masked_row_optimum(gram, row, torch.ones(3, dtype=torch.bool))
        assert bool(torch.isfinite(optimal).all())

    def test_shape_mismatches_raise(self, problem):
        import torch

        gram, row, _ = problem
        with pytest.raises(ExactReconstructionError, match="shapes differ"):
            exact_masked_row_optimum(gram, row, torch.ones(3, dtype=torch.bool))
        with pytest.raises(ExactReconstructionError, match="does not match"):
            row_objective(torch.eye(3), row, row)


class TestOurSweepAgainstTheOptimum:
    def test_the_sweep_lands_between_naive_and_the_optimum(self, layer):
        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.reconstruct import sweep_reconstruct

        gram, weight = layer
        mask = build_mask_from_scores(weight.abs(), sparsity=0.3)
        outcome = sweep_reconstruct(gram, weight, mask, bits=None)

        report = ExactReconstructionReport()
        report.module_rows["synthetic"] = [
            compare_row(row, gram, weight[row], outcome.weight[row], mask[row])
            for row in range(weight.shape[0])
        ]
        assert report.impossible_rows == 0
        assert report.rows_worse_than_naive == 0
        assert report.passes

    def test_a_row_below_the_optimum_is_flagged_as_impossible(self):
        """Fault injection. Scoring below a provable minimum means a defect, not a good result."""
        report = ExactReconstructionReport()
        report.module_rows["synthetic"] = [
            RowComparison(
                row=0,
                kept=10,
                naive_objective=100.0,
                ours_objective=1.0,
                optimal_objective=5.0,
            )
        ]
        assert report.impossible_rows == 1
        assert not report.passes
        assert "INVESTIGATE" in "\n".join(report.summary_lines())

    def test_a_row_worse_than_naive_is_flagged(self):
        report = ExactReconstructionReport()
        report.module_rows["synthetic"] = [
            RowComparison(
                row=3,
                kept=10,
                naive_objective=100.0,
                ours_objective=120.0,
                optimal_objective=50.0,
            )
        ]
        assert report.rows_worse_than_naive == 1
        assert not report.passes

    def test_a_mask_violation_raises_rather_than_scoring(self, layer):
        """Objectives over different feasible sets are not comparable."""
        import torch

        gram, weight = layer
        keep = torch.ones(16, dtype=torch.bool)
        keep[0] = False
        bad = weight[0].clone()
        bad[0] = 1.0

        with pytest.raises(ExactReconstructionError, match="masked position"):
            compare_row(0, gram, weight[0], bad, keep)

    def test_an_empty_report_does_not_pass(self):
        assert not ExactReconstructionReport().passes

    def test_efficiency_is_one_when_the_sweep_matches_the_optimum(self):
        comparison = RowComparison(
            row=0, kept=8, naive_objective=10.0, ours_objective=4.0, optimal_objective=4.0
        )
        assert comparison.efficiency == pytest.approx(1.0)
        assert not comparison.beats_the_optimum

    def test_efficiency_below_one_does_not_fail_the_verdict(self):
        """A one-pass sweep giving up some of the optimum is the documented trade, not a defect."""
        report = ExactReconstructionReport()
        report.module_rows["synthetic"] = [
            RowComparison(
                row=0, kept=8, naive_objective=10.0, ours_objective=6.0, optimal_objective=4.0
            )
        ]
        assert report.mean_efficiency == pytest.approx(2 / 3)
        assert report.passes

    def test_the_report_serialises(self, layer):
        import json

        from scale_aware_compression.compression.masks import build_mask_from_scores
        from scale_aware_compression.compression.reconstruct import sweep_reconstruct

        gram, weight = layer
        mask = build_mask_from_scores(weight.abs(), sparsity=0.3)
        outcome = sweep_reconstruct(gram, weight, mask, bits=None)
        report = ExactReconstructionReport()
        report.module_rows["synthetic"] = [
            compare_row(0, gram, weight[0], outcome.weight[0], mask[0])
        ]
        report.notes.append("pruning-only")

        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["anchor"] == "exact_masked_reconstruction_optimum"
        assert payload["rows_compared"] == 1
        assert payload["passes"] is True


class TestStratifiedModuleSampling:
    """The first run of this anchor sampled 6 modules and every one was the same type.

    A GPT-NeoX block contributes four target modules in a fixed order, so a stride of
    ``48 // 6 == 8`` returns ``attention.query_key_value`` six times and never touches an MLP
    projection -- the widest layers, and the ones where a one-pass sweep has most to compensate for.
    The verdict looked fine and covered a quarter of the model.
    """

    @staticmethod
    def _sampler():
        import sys
        from pathlib import Path

        scripts = str(Path("scripts").resolve())
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        from run_reconstruction_anchor import _stratified_sample

        return _stratified_sample

    @staticmethod
    def _names(layers: int = 12) -> list[str]:
        suffixes = (
            "attention.query_key_value",
            "attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        )
        return [f"gpt_neox.layers.{i}.{s}" for i in range(layers) for s in suffixes]

    def test_every_module_type_is_represented(self):
        sampled = self._sampler()(self._names(), 12)
        kinds = {".".join(name.split(".")[-2:]) for name in sampled}
        assert len(kinds) == 4, f"only sampled {kinds}"

    def test_a_plain_stride_would_have_failed_this(self):
        """Pins the actual defect, so the old behaviour cannot come back and still pass."""
        names = self._names()
        strided = names[:: max(1, len(names) // 6)][:6]
        assert len({".".join(n.split(".")[-2:]) for n in strided}) == 1

    def test_the_sample_spreads_through_the_depth(self):
        sampled = self._sampler()(self._names(), 12)
        layers = {int(name.split(".")[2]) for name in sampled}
        assert len(layers) >= 3, f"sampled only layers {layers}"

    def test_it_never_returns_more_than_the_budget(self):
        assert len(self._sampler()(self._names(), 5)) <= 5

    def test_it_handles_a_budget_larger_than_the_model(self):
        names = self._names(layers=1)
        assert set(self._sampler()(names, 99)) == set(names)


class TestTheExactReferenceIsIndependent:
    def test_it_does_not_import_the_solver_it_checks(self):
        from pathlib import Path

        source = Path("src/scale_aware_compression/anchors/exact_reconstruction.py").read_text(
            encoding="utf-8"
        )
        for forbidden in (
            "from scale_aware_compression.compression.reconstruct import",
            "sweep_reconstruct(",
            "als_reconstruct(",
            "solve_masked_rows(",
            "reconstruction_loss(",
        ):
            assert forbidden not in source, (
                f"exact_reconstruction.py references {forbidden!r}. It must re-derive the objective "
                "and the optimum, or a shared fault would cancel in both."
            )
