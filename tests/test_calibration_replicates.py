"""Tests for the paired calibration replicate axis (Protocol Amendment A1 §5.1).

The replicate axis exists because the run seed is inert under this method: two runs at different run
seeds produce bit-identical output (`findings_log.md` F-15), so the three-seed protocol yielded three
identical numbers and a seed spread of exactly zero. Varying the *calibration draw* changes the Gram
matrix, hence the mask and the scales, hence the compressed model.

Two properties carry the whole design and are tested hardest:

* **replicate 0 reproduces the pre-amendment draw**, or every measurement taken before A1 becomes
  incomparable with the confirmatory runs;
* **different replicates give different draws**, or the error bar is fabricated from eight copies of
  one number.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.config import ConfigError, DataConfig, SweepConfig
from scale_aware_compression.constants import (
    CALIBRATION_REPLICATE_SEEDS,
    DEFAULT_SEED,
    CompressionMethod,
)


class TestTheSeedTable:
    def test_replicate_zero_is_the_default_seed(self):
        """Continuity: F-19 through F-22 and every screening number used DEFAULT_SEED."""
        assert CALIBRATION_REPLICATE_SEEDS[0] == DEFAULT_SEED

    def test_there_are_at_least_eight_seeds(self):
        """A1 §6 sets R=8 at 160M and 410M."""
        assert len(CALIBRATION_REPLICATE_SEEDS) >= 8

    def test_the_seeds_are_distinct(self):
        """A duplicated seed would be a duplicated draw, narrowing the error bar for free."""
        assert len(set(CALIBRATION_REPLICATE_SEEDS)) == len(CALIBRATION_REPLICATE_SEEDS)


class TestEffectiveCalibrationSeed:
    def test_unset_replicate_uses_the_configured_seed(self):
        config = DataConfig(calibration_seed=4321)
        assert config.calibration_replicate is None
        assert config.effective_calibration_seed == 4321

    def test_replicate_overrides_a_stale_configured_seed(self):
        """A replicate index is a claim about which draw this is; a leftover seed must not win."""
        config = DataConfig(calibration_seed=999, calibration_replicate=3)
        assert config.effective_calibration_seed == CALIBRATION_REPLICATE_SEEDS[3]

    def test_replicate_zero_reproduces_the_pre_amendment_draw(self):
        assert DataConfig(calibration_replicate=0).effective_calibration_seed == DEFAULT_SEED

    def test_every_replicate_gives_a_distinct_seed(self):
        seeds = {
            DataConfig(calibration_replicate=index).effective_calibration_seed
            for index in range(len(CALIBRATION_REPLICATE_SEEDS))
        }
        assert len(seeds) == len(CALIBRATION_REPLICATE_SEEDS)

    @pytest.mark.parametrize("bad", [-1, len(CALIBRATION_REPLICATE_SEEDS), 999])
    def test_an_out_of_range_replicate_is_refused_at_load_time(self, bad: int):
        with pytest.raises(ConfigError, match="calibration_replicate"):
            DataConfig(calibration_replicate=bad)


class TestTheDrawsActuallyDiffer:
    """The whole error bar rests on this. Eight identical draws would fabricate one."""

    def test_each_replicate_selects_a_different_set_of_sequences(self):
        from scale_aware_compression.data.calibration import select_calibration_indices

        draws = {
            tuple(
                select_calibration_indices(
                    4749,  # the real WikiText-2 train block count
                    128,
                    seed=DataConfig(calibration_replicate=index).effective_calibration_seed,
                )
            )
            for index in range(len(CALIBRATION_REPLICATE_SEEDS))
        }
        assert len(draws) == len(CALIBRATION_REPLICATE_SEEDS), (
            "two replicates drew the identical sequence set, so the error bar would be computed "
            "over identical compressed models"
        )

    def test_replicate_zero_draws_what_the_default_seed_draws(self):
        from scale_aware_compression.data.calibration import select_calibration_indices

        by_replicate = select_calibration_indices(
            4749, 128, seed=DataConfig(calibration_replicate=0).effective_calibration_seed
        )
        by_default = select_calibration_indices(4749, 128, seed=DEFAULT_SEED)
        assert list(by_replicate) == list(by_default)

    def test_the_draws_overlap_but_are_not_nested(self):
        """Independent draws from one corpus should share some sequences without one containing another."""
        from scale_aware_compression.data.calibration import select_calibration_indices

        first = set(select_calibration_indices(4749, 128, seed=CALIBRATION_REPLICATE_SEEDS[0]))
        second = set(select_calibration_indices(4749, 128, seed=CALIBRATION_REPLICATE_SEEDS[1]))
        assert first != second
        assert not first.issubset(second)
        assert not second.issubset(first)

    def test_a_draw_is_reproducible(self):
        """Same replicate, same draw -- otherwise a re-run is not the same experiment."""
        from scale_aware_compression.data.calibration import select_calibration_indices

        seed = DataConfig(calibration_replicate=4).effective_calibration_seed
        assert list(select_calibration_indices(4749, 128, seed=seed)) == list(
            select_calibration_indices(4749, 128, seed=seed)
        )


class TestSweepReplicateValidation:
    def test_replicates_beyond_the_seed_table_are_refused(self):
        with pytest.raises(ConfigError, match="exceeds"):
            SweepConfig(replicates=len(CALIBRATION_REPLICATE_SEEDS) + 1)

    def test_negative_replicates_are_refused(self):
        with pytest.raises(ConfigError, match="must be >= 0"):
            SweepConfig(replicates=-1)

    def test_replicates_and_multiple_seeds_together_are_refused(self):
        """Both axes would multiply the grid by an axis known to produce identical results."""
        with pytest.raises(ConfigError, match="replaced the seed axis"):
            SweepConfig(replicates=8, seeds=[1234, 2345])

    def test_replicates_with_a_single_seed_is_allowed(self):
        assert SweepConfig(replicates=8, seeds=[1234]).replicates == 8

    def test_zero_replicates_disables_the_axis(self):
        assert SweepConfig(replicates=0).replicates == 0


class TestExperimentIdCarriesTheReplicate:
    """Eight replicates of one cell are eight different models and need eight record names."""

    @staticmethod
    def _identifier(**overrides):
        from scale_aware_compression.experiments.runner import make_experiment_id

        arguments = {
            "model_name": "pythia-160m",
            "method": CompressionMethod.JOINT,
            "budget_label": "aggressive",
            "seed": 1234,
            "sparsity": 0.3,
            "bits": 4,
        }
        arguments.update(overrides)
        return make_experiment_id(**arguments)

    def test_each_replicate_gets_its_own_identifier(self):
        identifiers = {self._identifier(replicate=index) for index in range(8)}
        assert len(identifiers) == 8

    def test_the_replicate_appears_in_the_identifier(self):
        assert self._identifier(replicate=5).endswith("_rep5")

    def test_no_replicate_falls_back_to_the_seed(self):
        """Records taken before A1 are named by seed; renaming them would orphan them."""
        assert self._identifier().endswith("_seed1234")

    def test_the_replicate_wins_over_the_seed(self):
        first = self._identifier(seed=1234, replicate=2)
        second = self._identifier(seed=9999, replicate=2)
        assert first == second, "an inert run seed must not split one replicate into two records"


class TestPairingIsWithinADraw:
    """§3.11 requires identical calibration *between arms within a comparison*.

    Across repeats it must differ -- that is the point -- but differencing two arms that saw different
    draws is not a paired comparison, and would silently mix calibration variance into the effect.
    """

    @staticmethod
    def _record(method: CompressionMethod, replicate: int | None, seed: int = 1234):
        return {
            "status": "success",
            "model_name": "pythia-160m",
            "budget_label": "aggressive",
            "seed": seed,
            "compression_method": method.value,
            "config": {"data": {"calibration_replicate": replicate}},
        }

    def test_arms_in_the_same_draw_pair_together(self):
        from scale_aware_compression.experiments.scale_sweep import _pair_key

        sequential = self._record(CompressionMethod.SEQUENTIAL, 3)
        joint = self._record(CompressionMethod.JOINT, 3)
        assert _pair_key(sequential) == _pair_key(joint)

    def test_arms_in_different_draws_do_not_pair(self):
        from scale_aware_compression.experiments.scale_sweep import _pair_key

        sequential = self._record(CompressionMethod.SEQUENTIAL, 3)
        joint = self._record(CompressionMethod.JOINT, 4)
        assert _pair_key(sequential) != _pair_key(joint)

    def test_a_mismatched_draw_is_reported_as_incomparable(self):
        from scale_aware_compression.experiments.scale_sweep import _incomparable

        reasons = _incomparable(
            self._record(CompressionMethod.SEQUENTIAL, 3),
            self._record(CompressionMethod.JOINT, 4),
        )
        assert any("replicate" in reason for reason in reasons)

    def test_an_inert_seed_difference_no_longer_splits_a_pair(self):
        """Seeds vary nothing (F-15), so they must not prevent two arms from being differenced."""
        from scale_aware_compression.experiments.scale_sweep import _incomparable, _pair_key

        sequential = self._record(CompressionMethod.SEQUENTIAL, 3, seed=1234)
        joint = self._record(CompressionMethod.JOINT, 3, seed=9999)
        assert _pair_key(sequential) == _pair_key(joint)
        assert not any("seed" in reason for reason in _incomparable(sequential, joint))

    def test_pre_amendment_records_still_group_by_seed(self):
        from scale_aware_compression.experiments.scale_sweep import _pair_key

        sequential = self._record(CompressionMethod.SEQUENTIAL, None, seed=1234)
        joint = self._record(CompressionMethod.JOINT, None, seed=1234)
        assert _pair_key(sequential) == _pair_key(joint)


class TestPlanCellPairingCountsEveryReplicate:
    """B-38. `find_comparison_pairs` pairs plan *cells*; `_pair_key` pairs finished *records*.

    The record path already keyed on the replicate. The cell path did not, and since A1 gives every
    replicate the same run seed, all R replicates collided on one key and the dict kept only the
    last -- so a 3-replicate grid with 6 sequential and 6 joint cells reported **2** pairs. The
    symmetric-difference warning could not catch it either, because the dropped cells were never
    distinct keys.

    Nothing computed a joint gain through this path, so no published number moved. What it did was
    under-report how many comparisons a grid would yield, on the line a person reads before
    committing hours of compute.
    """

    @staticmethod
    def _plan(replicates: int, budgets: tuple[str, ...] = ("aggressive",)):
        from scale_aware_compression.config import ExperimentConfig
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        config = ExperimentConfig.from_mapping(
            {
                "model": {"name": "pythia-160m"},
                "sweep": {
                    "models": ["pythia-160m"],
                    "methods": ["sequential", "joint"],
                    "budgets": list(budgets),
                    "replicates": replicates,
                    "budget_overrides": {
                        budget: {"compression": {"budget_label": budget}} for budget in budgets
                    },
                },
            }
        )
        return build_sweep_plan(config)

    @pytest.mark.parametrize("replicates", [1, 3, 8])
    def test_one_pair_per_replicate(self, replicates: int):
        from scale_aware_compression.experiments.scale_sweep import find_comparison_pairs

        assert len(find_comparison_pairs(self._plan(replicates))) == replicates

    def test_pairs_multiply_across_budgets_and_replicates(self):
        from scale_aware_compression.experiments.scale_sweep import find_comparison_pairs

        plan = self._plan(3, budgets=("moderate", "aggressive"))
        assert len(find_comparison_pairs(plan)) == 6

    def test_both_halves_of_a_pair_share_a_draw(self):
        """The invariant the key exists to protect, not just the count."""
        from scale_aware_compression.experiments.scale_sweep import find_comparison_pairs

        for sequential, joint in find_comparison_pairs(self._plan(3)):
            assert sequential.replicate == joint.replicate
            assert sequential.budget_label == joint.budget_label
            assert sequential.model_name == joint.model_name

    def test_every_replicate_appears_exactly_once(self):
        from scale_aware_compression.experiments.scale_sweep import find_comparison_pairs

        pairs = find_comparison_pairs(self._plan(3))
        assert sorted(sequential.replicate for sequential, _ in pairs) == [0, 1, 2]

    def test_a_grid_with_no_replicate_axis_still_pairs(self):
        """Pre-amendment grids carry `replicate=None`, which must not break ordering."""
        from scale_aware_compression.config import ExperimentConfig
        from scale_aware_compression.experiments.scale_sweep import (
            build_sweep_plan,
            find_comparison_pairs,
        )

        config = ExperimentConfig.from_mapping(
            {
                "model": {"name": "pythia-160m"},
                "sweep": {
                    "models": ["pythia-160m"],
                    "methods": ["sequential", "joint"],
                    "budgets": ["aggressive"],
                    "budget_overrides": {
                        "aggressive": {"compression": {"budget_label": "aggressive"}}
                    },
                },
            }
        )
        assert len(find_comparison_pairs(build_sweep_plan(config))) == 1


class TestTheSweepExpandsOverReplicates:
    @staticmethod
    def _config(**sweep_overrides):
        from scale_aware_compression.config import ExperimentConfig

        document = {
            "experiment": {"id": "unnamed"},
            "model": {"name": "pythia-160m"},
            "compression": {"method": "joint", "budget_label": "aggressive"},
            "sweep": {
                "models": ["pythia-160m"],
                "methods": ["sequential", "joint"],
                "budgets": ["aggressive"],
                **sweep_overrides,
            },
        }
        return ExperimentConfig.from_mapping(document)

    def test_one_cell_per_arm_per_replicate(self):
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        plan = build_sweep_plan(self._config(replicates=8))
        assert len(plan.cells) == 2 * 8

    def test_every_replicate_index_appears(self):
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        plan = build_sweep_plan(self._config(replicates=5))
        assert {cell.replicate for cell in plan.cells} == set(range(5))

    def test_both_arms_are_present_in_each_replicate(self):
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        plan = build_sweep_plan(self._config(replicates=3))
        for index in range(3):
            arms = {cell.method for cell in plan.cells if cell.replicate == index}
            assert len(arms) == 2, f"replicate {index} is missing an arm and cannot be paired"

    def test_cell_identifiers_are_unique(self):
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        plan = build_sweep_plan(self._config(replicates=8))
        identifiers = [cell.experiment_id for cell in plan.cells]
        assert len(set(identifiers)) == len(identifiers)

    def test_zero_replicates_keeps_the_old_single_cell_behaviour(self):
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        plan = build_sweep_plan(self._config())
        assert len(plan.cells) == 2
        assert all(cell.replicate is None for cell in plan.cells)

    def test_the_cell_config_carries_its_replicate_through(self):
        from scale_aware_compression.experiments.scale_sweep import (
            build_cell_config,
            build_sweep_plan,
        )

        base = self._config(replicates=4)
        for cell in build_sweep_plan(base).cells:
            realised = build_cell_config(base, cell)
            assert realised.data.calibration_replicate == cell.replicate
            assert (
                realised.data.effective_calibration_seed
                == CALIBRATION_REPLICATE_SEEDS[cell.replicate]
            )


class TestDenseIsNotReplicated:
    """Dense depends on neither the calibration draw nor the budget, so replicating it is waste.

    B-12 already caught the budget half of this. The replicate axis reintroduces the same shape at
    eight times the scale: 2 models x 2 budgets x 8 replicates would plan 32 dense cells with 32
    different identifiers and identical contents -- wasted compute plus near-duplicate records that
    §10.4 asks the audit to reject.
    """

    @staticmethod
    def _ordered(replicates: int):
        from scale_aware_compression.config import ExperimentConfig
        from scale_aware_compression.experiments.scale_sweep import _dense_first, build_sweep_plan

        config = ExperimentConfig.from_mapping(
            {
                "experiment": {"id": "unnamed"},
                "model": {"name": "pythia-160m"},
                "compression": {"method": "dense"},
                "sweep": {
                    "models": ["pythia-160m", "pythia-410m"],
                    "methods": ["dense", "sequential", "joint"],
                    "budgets": ["moderate", "aggressive"],
                    "replicates": replicates,
                },
            }
        )
        return _dense_first(build_sweep_plan(config).cells)

    def test_one_dense_run_per_model_regardless_of_replicate_count(self):
        for replicates in (1, 5, 8):
            dense = [
                cell for cell in self._ordered(replicates) if cell.method is CompressionMethod.DENSE
            ]
            assert len(dense) == 2, (
                f"{len(dense)} dense cells at replicates={replicates}; dense does not depend on the "
                "calibration draw, so more than one per model is duplicate compute"
            )

    def test_compressed_arms_keep_every_replicate(self):
        """The dedup must not touch the arms whose output actually depends on the draw."""
        ordered = self._ordered(8)
        for method in (CompressionMethod.SEQUENTIAL, CompressionMethod.JOINT):
            replicates = {
                cell.replicate
                for cell in ordered
                if cell.method is method and cell.model_name == "pythia-160m"
            }
            assert replicates == set(range(8)), f"{method.value} lost replicates: {replicates}"

    def test_dense_still_runs_before_the_compressed_arms(self):
        """Every compressed arm loads its retention reference from the dense record."""
        ordered = self._ordered(8)
        first_compressed = next(
            index
            for index, cell in enumerate(ordered)
            if cell.method is not CompressionMethod.DENSE
        )
        assert all(cell.method is CompressionMethod.DENSE for cell in ordered[:first_compressed])
