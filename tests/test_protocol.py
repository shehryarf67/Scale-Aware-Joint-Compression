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

    def test_every_primary_sweep_cell_is_frozen(self):
        """The six primary cells must all be frozen; the confirmatory grid cannot run otherwise."""
        primary = {(model, budget) for model in MODELS for budget in BUDGETS}
        assert primary <= set(FROZEN_SEQUENTIAL_ORDER)

    def test_the_external_validation_cells_are_frozen_separately(self):
        """Qwen2.5-0.5B was frozen on 2026-08-11 (F-40) for the external-validation leg.

        Kept as its own assertion rather than folded into the primary set: this leg is exploratory,
        it cannot alter F-37, and a future reader must be able to see at a glance which cells belong
        to the frozen primary experiment and which do not.
        """
        for budget in BUDGETS:
            assert ("qwen2.5-0.5b", budget) in FROZEN_SEQUENTIAL_ORDER
            # P→Q both budgets: measured by 1.351 pp at W4, pre-declared fallback at W8.
            assert resolve_sequential_order("qwen2.5-0.5b", budget) is CompressionMethod.SEQUENTIAL

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
        # qwen2.5-0.5b was one of these until 2026-08-11, when F-40 froze it. Replaced with
        # pythia-1.4b's other budget and an unregistered family, so the guard still has genuinely
        # unfrozen cells to refuse.
        [
            ("pythia-1.4b", "moderate"),
            ("pythia-1.4b", "aggressive"),
            ("qwen2.5-1.5b", "aggressive"),
            ("pythia-160m", "s6_40_w8"),
        ],
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


class TestTheFiguresAndTablesRefuseToMislead:
    """The visualisation layer was scaffold until 2026-08-11; these pin what it must not do.

    Two failure modes are specific to this project and both nearly shipped:

    * a latency ratio built from measurements taken in different sessions, which produced a
      pythia-1b point **above** the theoretical speedup bound -- physically impossible (B-49)
    * an arm coloured by its index within a subplot, so ``sequential_qp`` at 1B took the colour
      ``sequential`` had at the other scales and a reader comparing panels was misled
    """

    def test_cross_session_latency_is_not_generated_by_default(self):
        """The known-invalid B-49 diagnostic must require an explicit opt-in."""
        import importlib.util
        from pathlib import Path

        script = Path(__file__).parents[1] / "scripts" / "generate_plots.py"
        spec = importlib.util.spec_from_file_location("generate_plots", script)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert "2" in module.FIGURE_NUMBERS
        assert "2" not in module.DEFAULT_FIGURE_NUMBERS
        assert module.DEFAULT_FIGURE_NUMBERS == ("1", "3", "4")

    def test_arm_colours_are_keyed_to_the_arm_not_its_position(self):
        """Same arm, same colour, in every panel of every figure."""
        from scale_aware_compression.visualisation.plots import METHOD_COLOURS, _method_colour

        assert _method_colour("joint") == METHOD_COLOURS["joint"]
        assert _method_colour("sequential") != _method_colour("sequential_qp"), (
            "the two frozen orders must be distinguishable; they are different baselines"
        )
        assert _method_colour("joint") not in {
            _method_colour("sequential"),
            _method_colour("sequential_qp"),
        }

    def test_the_joint_gain_table_keeps_the_qp_cell(self):
        """B-50 in table form: a Q→P cell must appear, not be silently dropped."""
        from scale_aware_compression.visualisation.tables import build_joint_gain_table

        trend = [
            {
                "model_name": "pythia-1b",
                "budget_label": "moderate",
                "targeted_parameters": 805306368,
                "comparable": True,
                "sequential_retention": 96.5,
                "joint_retention": 96.3,
                "joint_gain_retention_pp": -0.2,
                "joint_advantage_nll": -0.002,
            }
        ]
        rows = build_joint_gain_table(trend)
        assert len(rows) == 1
        assert rows[0]["positive_replicates"] == "0/1"
        assert rows[0]["sign_consistent"] is False

    def test_the_main_table_reports_replicate_count_per_row(self):
        """A row averaged over one replicate must not read as though it were averaged over eight."""
        from scale_aware_compression.visualisation.tables import build_main_results_table

        records = [
            {
                "status": "success",
                "model_name": "pythia-160m",
                "compression_method": "joint",
                "budget_label": "aggressive",
                "sparsity": 0.3,
                "quantisation_bits": 4,
                "parameter_count": 1000,
                "quality": {"perplexity": {"perplexity": 40.0}, "retention": {}},
                "compression": {"statistics": {"targeted_parameters": 1000}},
            }
        ]
        rows = build_main_results_table(records)
        assert rows[0]["replicates"] == 1


class TestACrossSplitWriteDoesNotDestroyTheOtherRecord:
    """B-51. A record id encodes the cell but NOT the evaluation split.

    So a test-split run writes to the same filename as the validation-split run of the same cell
    and destroys it. `exists_valid` gates *reuse* on the split and correctly refuses to read the
    wrong record; nothing gated the *write*.

    It happened: the Qwen test grid overwrote the validation dense baseline its own order selection
    had been measured against. The figures survived only by luck -- a smoke run had left a copy
    under a different experiment id.
    """

    def _record(self, split: str, perplexity: float):
        from scale_aware_compression.experiments.runner import ExperimentRecord

        return ExperimentRecord(
            experiment_id="m_dense_moderate_s00_b32_rep0",
            model_name="m",
            compression_method="dense",
            budget_label="moderate",
            seed=1234,
            config={"data": {"eval_split": split}},
            quality={"perplexity": {"perplexity": perplexity}},
        )

    def test_the_validation_record_survives_a_test_run(self, tmp_path):
        """The exact Qwen failure: the first record must still be readable afterwards."""
        from scale_aware_compression.experiments.runner import ExperimentTracker

        tracker = ExperimentTracker(tmp_path)
        tracker.save(self._record("validation", 17.7758))
        tracker.save(self._record("test", 17.0962))

        splits = {
            ((r.get("config") or {}).get("data") or {}).get("eval_split"): (
                r.get("quality", {}).get("perplexity", {}).get("perplexity")
            )
            for r in tracker.load_all()
        }
        assert splits == {"validation": 17.7758, "test": 17.0962}, (
            "both splits must survive; overwriting one deletes the reference other records were "
            "normalised against"
        )
        assert any("__split-validation" in p.name for p in tmp_path.glob("*.json"))

    def test_rerunning_the_same_split_still_overwrites(self, tmp_path):
        """Archiving must not fire for the ordinary case, or every re-run leaves litter."""
        from scale_aware_compression.experiments.runner import ExperimentTracker

        tracker = ExperimentTracker(tmp_path)
        tracker.save(self._record("test", 20.0))
        tracker.save(self._record("test", 21.0))

        records = tracker.load_all()
        assert len(records) == 1, "a same-split re-run replaces its record rather than archiving it"
        assert records[0]["quality"]["perplexity"]["perplexity"] == 21.0


class TestEvidencePairingRespectsTheEvaluationSplit:
    """B-52. The exported joint-gain table paired arms across evaluation splits.

    The key was (model, budget, replicate), and the two splits produce the same experiment ids by
    construction, so at qwen2.5-0.5b/moderate/rep0 the VALIDATION Q→P record was selected as
    best-of against the TEST joint record: -0.2143 pp exported where the frozen-order test gain is
    -0.0357 pp. Same root cause as B-51, seen from the analysis side.
    """

    def _row(self, method, split, retention, replicate=0, fingerprint="fp-a"):
        return {
            "model_name": "qwen2.5-0.5b",
            "budget_label": "moderate",
            "calibration_replicate": replicate,
            "compression_method": method,
            "status": "success",
            "perplexity_retention": retention,
            "eval_split": split,
            "dataset_fingerprint": fingerprint,
        }

    def _gains(self, rows):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_export_evidence", Path("scripts/export_evidence.py")
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module._joint_gain_rows(rows)

    def test_a_validation_record_never_becomes_a_test_baseline(self):
        """The exact Qwen failure, pinned to the numbers it produced."""
        rows = [
            self._row("sequential", "test", 95.01302088492389),
            self._row("joint", "test", 94.97733340482291),
            self._row("sequential_qp", "validation", 95.19164673718451),
        ]
        gains = self._gains(rows)
        test_rows = [g for g in gains if g["eval_split"] == "test"]
        assert len(test_rows) == 1
        assert test_rows[0]["joint_gain_pp"] == pytest.approx(-0.03568748, abs=1e-6), (
            "must be the frozen-order test gain, not -0.2143 against a validation Q→P record"
        )
        assert test_rows[0]["baseline_order"] == "sequential"
        assert test_rows[0]["baseline_rule"] == "frozen"

    def test_validation_still_uses_best_of_both_orders(self):
        """Best-of is correct on a selection surface where both orders were actually run."""
        rows = [
            self._row("sequential", "validation", 95.0),
            self._row("sequential_qp", "validation", 95.5),
            self._row("joint", "validation", 95.2),
        ]
        gains = self._gains(rows)
        assert len(gains) == 1
        assert gains[0]["baseline_rule"] == "best-of"
        assert gains[0]["baseline_order"] == "sequential_qp"
        assert gains[0]["joint_gain_pp"] == pytest.approx(-0.3, abs=1e-9)

    def test_a_differing_dataset_fingerprint_does_not_pair(self):
        """Same split label is not enough: the window or corpus can still differ."""
        rows = [
            self._row("sequential", "test", 95.0, fingerprint="fp-a"),
            self._row("joint", "test", 95.5, fingerprint="fp-b"),
        ]
        assert self._gains(rows) == []


class TestTheScaleFigureRefusesNonScalePoints:
    """The x-axis is targeted parameters, so anything drawn on it reads as a scale point.

    qwen2.5-0.5b is not one: a different family, tokeniser and corpus, with 358M targeted
    parameters that land between pythia-410m and pythia-1b. Once its records shared the output
    directory, the DEFAULT plotting command included them -- producing a publishable-looking
    figure that asserts exactly what docs/findings_log.md §6 forbids.
    """

    def _trend(self, model):
        return [
            {
                "model_name": model,
                "budget_label": "aggressive",
                "targeted_parameters": 357826560,
                "comparable": True,
                "joint_gain_retention_pp": 0.42,
            }
        ]

    def test_a_non_primary_model_is_refused(self, tmp_path):
        # The refusal happens before any drawing, but apply_style imports matplotlib, and the
        # no-torch CI job installs neither torch nor matplotlib.
        pytest.importorskip("matplotlib")
        from scale_aware_compression.visualisation import plots

        with pytest.raises(ValueError, match="non-scale-point model"):
            plots.plot_joint_gain_vs_scale(self._trend("qwen2.5-0.5b"), tmp_path)

    def test_a_primary_model_still_plots(self, tmp_path):
        pytest.importorskip("matplotlib")
        from scale_aware_compression.visualisation import plots

        written = plots.plot_joint_gain_vs_scale(self._trend("pythia-410m"), tmp_path, name="ok")
        assert written and all(p.exists() for p in written)

    def test_the_external_panel_accepts_what_the_scale_figure_rejects(self, tmp_path):
        """The categorical alternative must exist, or the refusal just blocks the work."""
        pytest.importorskip("matplotlib")
        from scale_aware_compression.visualisation import plots

        written = plots.plot_external_validation(self._trend("qwen2.5-0.5b"), tmp_path)
        assert written and all(p.exists() for p in written)
