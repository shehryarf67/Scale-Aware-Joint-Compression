"""The frozen sequential order, as a machine-readable decision (§3.6, §6.1).

Why this file exists: the frozen order lived only in prose in ``docs/protocol_freeze.md``, and two
consumers got it wrong. ``run_downstream.py`` mapped every sequential arm to P→Q, and
``main_scale_sweep.yaml`` -- the **confirmatory** config -- listed ``sequential`` for every cell with
no resolution at all. At ``pythia-1b``/``moderate``, where Q→P is frozen, both would have used P→Q:
the *weaker* baseline, which inflates the joint gain. That is B-30's fault, and it would have landed
in the one run that cannot be redone.

The tests here are about that: one source of truth, a refusal rather than a default, and a guard that
every confirmatory config honours it.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.constants import CompressionMethod
from scale_aware_compression.protocol import (
    FROZEN_ORDER_EVIDENCE,
    FROZEN_SEQUENTIAL_ORDER,
    ProtocolError,
    frozen_order_evidence,
    resolve_sequential_order,
)

MODELS = ("pythia-160m", "pythia-410m", "pythia-1b")
BUDGETS = ("moderate", "aggressive")


class TestTheFrozenTableMatchesTheDocument:
    """docs/protocol_freeze.md is authoritative for *why*; this table is authoritative for *what*."""

    def test_all_six_cells_are_frozen(self):
        assert set(FROZEN_SEQUENTIAL_ORDER) == {(m, b) for m in MODELS for b in BUDGETS}

    @pytest.mark.parametrize("model", MODELS)
    def test_w4_is_p_to_q_at_every_scale(self, model: str):
        """Margins of +4.26, +6.82 and +2.15 pp. The direction is stable across three scales."""
        assert resolve_sequential_order(model, "aggressive") is CompressionMethod.SEQUENTIAL

    @pytest.mark.parametrize("model", ["pythia-160m", "pythia-410m"])
    def test_w8_is_p_to_q_at_the_smaller_scales_by_fallback(self, model: str):
        """F-28: the sign varied, so the pre-declared fallback picked the §3.6 primary."""
        assert resolve_sequential_order(model, "moderate") is CompressionMethod.SEQUENTIAL

    def test_w8_at_1b_is_q_to_p_and_this_is_the_cell_that_matters(self):
        """The one cell that differs, and the whole reason this table is machine-readable.

        F-32: the sign WAS consistent here (+0.12, +0.05, +0.13), so the same rule took its measured
        branch rather than its fallback. A consumer that assumes `sequential` means P→Q gets the
        weaker baseline here and reports an inflated joint gain.
        """
        assert resolve_sequential_order("pythia-1b", "moderate") is CompressionMethod.SEQUENTIAL_QP

    def test_every_frozen_cell_records_its_evidence(self):
        """A freeze without a citation cannot be audited, and §6.3 makes the provenance part of it."""
        assert set(FROZEN_ORDER_EVIDENCE) == set(FROZEN_SEQUENTIAL_ORDER)
        for key in FROZEN_SEQUENTIAL_ORDER:
            assert "F-" in FROZEN_ORDER_EVIDENCE[key], key

    def test_evidence_lookup_never_invents_a_citation(self):
        assert frozen_order_evidence("nonexistent", "moderate") == "no evidence recorded"


class TestAnUnfrozenCellIsRefusedNotDefaulted:
    """Defaulting to P→Q is precisely the fault. It is the §3.6 primary, so it *looks* safe."""

    @pytest.mark.parametrize(
        ("model", "budget"),
        [("pythia-1.4b", "moderate"), ("qwen2.5-0.5b", "aggressive"), ("pythia-160m", "s6_40_w8")],
    )
    def test_unknown_cells_raise(self, model: str, budget: str):
        with pytest.raises(ProtocolError, match="no sequential order is frozen"):
            resolve_sequential_order(model, budget)

    def test_the_error_names_the_frozen_cells_and_the_hazard(self):
        """An error a reader can act on: what is frozen, and why not to default."""
        with pytest.raises(ProtocolError) as caught:
            resolve_sequential_order("pythia-1.4b", "moderate")
        message = str(caught.value)
        assert "pythia-1b" in message
        assert "weaker baseline" in message


class TestSweepResolution:
    @staticmethod
    def _plan(*, use_frozen_order: bool, models=MODELS, budgets=BUDGETS):
        from scale_aware_compression.config import ExperimentConfig
        from scale_aware_compression.experiments.scale_sweep import build_sweep_plan

        config = ExperimentConfig.from_mapping(
            {
                "model": {"name": "pythia-160m"},
                "sweep": {
                    "models": list(models),
                    "methods": ["sequential", "joint"],
                    "budgets": list(budgets),
                    "use_frozen_order": use_frozen_order,
                    "budget_overrides": {
                        budget: {"compression": {"budget_label": budget}} for budget in budgets
                    },
                },
            }
        )
        return build_sweep_plan(config)

    def test_with_the_flag_only_1b_moderate_becomes_q_to_p(self):
        plan = self._plan(use_frozen_order=True)
        reversed_cells = {
            (cell.model_name, cell.budget_label)
            for cell in plan.cells
            if cell.method is CompressionMethod.SEQUENTIAL_QP
        }
        assert reversed_cells == {("pythia-1b", "moderate")}

    def test_without_the_flag_nothing_is_reversed(self):
        """Exploratory configs ran `sequential` meaning P→Q specifically.

        Often beside an explicit `sequential_qp` as the other half of the order comparison, so
        automatic resolution would change what ~50 existing records mean.
        """
        plan = self._plan(use_frozen_order=False)
        assert not any(cell.method is CompressionMethod.SEQUENTIAL_QP for cell in plan.cells)

    def test_the_joint_arm_is_never_rewritten(self):
        """Only the sequential baseline is resolved. Rewriting joint would be a different experiment."""
        plan = self._plan(use_frozen_order=True)
        joint = [cell for cell in plan.cells if cell.method is CompressionMethod.JOINT]
        assert len(joint) == len(MODELS) * len(BUDGETS)

    def test_an_unfrozen_model_refuses_to_plan(self):
        """Better a config that will not build than one that quietly runs the wrong baseline."""
        with pytest.raises(ProtocolError):
            self._plan(use_frozen_order=True, models=("pythia-1.4b",))

    def test_the_experiment_id_reflects_the_resolved_order(self):
        """Otherwise a Q→P record would be filed under a P→Q id and overwrite it."""
        plan = self._plan(use_frozen_order=True, models=("pythia-1b",), budgets=("moderate",))
        ids = [cell.experiment_id for cell in plan.cells]
        assert any("sequential_qp" in identifier for identifier in ids)


class TestConfirmatoryConfigsHonourTheFreeze:
    """The guard that stops this recurring: a test-split config must resolve the order.

    §6.1 requires best-of {P→Q, Q→P}. A confirmatory config that leaves `use_frozen_order` off runs
    P→Q everywhere, which is the weaker baseline at pythia-1b/moderate.
    """

    def test_every_test_split_config_uses_the_frozen_order(self, configs_dir):
        from scale_aware_compression.config import ExperimentConfig, load_document

        checked = 0
        for path in sorted((configs_dir / "experiments").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            if config.data.eval_split != "test":
                continue
            checked += 1
            assert config.sweep.use_frozen_order, (
                f"{path.name} evaluates the test split but does not set sweep.use_frozen_order, so "
                "a `sequential` cell would mean P→Q even where Q→P is frozen"
            )
        assert checked >= 1, "no test-split config was found; the guard would pass vacuously"


class TestCellIsolationForLongSweeps:
    """B-48. The runner does not release memory between cells, so a long grid exhausts commit.

    Measured on the confirmatory 1B leg: commit free fell 20.24 -> 1.03 GiB over five sequential
    cells, ~4 GiB each, and stopping the process returned all of it. With ``continue_on_error`` on
    the resulting ``MemoryError`` does not stop the run -- it drops a cell and continues, and a
    dropped sequential or joint cell silently removes a whole comparison.

    ``--isolate-cells`` runs every cell in a child process, so the memory is released at the cell
    boundary by construction rather than by finding every retention inside the runner.
    """

    def _run(self, *args: str) -> int:
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_sweep_script", Path("scripts/run_scale_sweep.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.main(list(args))

    def test_isolate_and_only_cell_are_mutually_exclusive(self):
        """One drives the other. Passing both means the caller has misunderstood which is which."""
        code = self._run(
            "--config",
            "configs/experiments/main_scale_sweep.yaml",
            "--isolate-cells",
            "--only-cell",
            "anything",
        )
        assert code == 2

    def test_an_unknown_cell_id_is_refused_rather_than_silently_running_nothing(self):
        """A typo must not look like a completed run: exit 0 with no work is the dangerous case."""
        code = self._run(
            "--config",
            "configs/experiments/main_scale_sweep.yaml",
            "--only-cell",
            "pythia-160m_joint_moderate_s30_b8_rep999",
        )
        assert code == 2

    def test_the_sweep_script_offers_cell_isolation(self):
        """The flag is the B-48 mitigation; losing it silently reopens the failure mode."""
        from pathlib import Path

        source = Path("scripts/run_scale_sweep.py").read_text(encoding="utf-8")
        assert "--isolate-cells" in source
        assert "--only-cell" in source
        assert "subprocess" in source, "isolation must actually spawn a process, not just claim to"


class TestScaleTrendAcceptsTheFrozenQtoPOrder:
    """B-50. ``scale_trend`` filtered on SEQUENTIAL and JOINT and never learned about SEQUENTIAL_QP.

    ``find_comparison_pairs`` was fixed for this (B-42); this function was not. The one cell whose
    frozen order is Q→P -- pythia-1b/moderate -- was therefore dropped from every trend and every
    figure built from records: 37 pairs reported where 42 exist.

    The direction is what makes it serious. The dropped cell is the only one where joint is
    consistently WORSE than sequential, so omitting it flattered joint -- the seventh fault in this
    project to run that way.
    """

    def _record(self, method: str, model: str, budget: str, replicate: int, ppl: float) -> dict:
        return {
            "experiment_id": f"{model}_{method}_{budget}_rep{replicate}",
            "status": "success",
            "compression_method": method,
            "model_name": model,
            "budget_label": budget,
            "seed": 1234,
            "parameter_count": 1_000_000,
            "method_version": "4",
            "config": {"data": {"calibration_replicate": replicate, "eval_split": "test"}},
            "quality": {
                "perplexity": {
                    "perplexity": ppl,
                    "total_nll": ppl * 100.0,
                    "total_tokens": 100,
                    "evaluation_device": "cpu",
                },
                "retention": {"perplexity_retention": 100.0 / ppl},
            },
            "compression": {"statistics": {"targeted_parameters": 1_000_000}},
        }

    def test_a_qp_frozen_cell_still_forms_a_pair(self):
        """The comparator is whichever order was frozen, not the one that happens to be named."""
        from scale_aware_compression.experiments.scale_sweep import scale_trend

        records = [
            self._record("sequential_qp", "pythia-1b", "moderate", 0, 20.0),
            self._record("joint", "pythia-1b", "moderate", 0, 21.0),
        ]
        rows = scale_trend(records)
        assert len(rows) == 1, "a Q→P cell must still yield a comparison"
        assert rows[0]["sequential_experiment_id"].endswith("sequential_qp_moderate_rep0")

    def test_a_pq_frozen_cell_is_unaffected(self):
        """The common case must not regress while fixing the rare one."""
        from scale_aware_compression.experiments.scale_sweep import scale_trend

        records = [
            self._record("sequential", "pythia-160m", "aggressive", 0, 20.0),
            self._record("joint", "pythia-160m", "aggressive", 0, 19.0),
        ]
        rows = scale_trend(records)
        assert len(rows) == 1
