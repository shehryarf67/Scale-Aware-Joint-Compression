"""Downstream task evaluation (plan §4.3, gap A4).

Every test here is **offline**. The harness downloads datasets, so the boundary that gets stubbed is
its *output*: `parse_harness_results` is a pure function over the mapping `simple_evaluate` returns,
which is exactly why it was separated from the call that produces it.

What is worth pinning is not that a task scores well -- that is the harness's job and it is pinned by
version -- but that we cannot **silently lose provenance or silently report less than we asked for**.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.evaluation.common import EvaluationError
from scale_aware_compression.evaluation.downstream import (
    CHANCE_LEVEL,
    DOWNSTREAM_TASKS,
    PRIMARY_METRIC,
    DownstreamReport,
    TaskResult,
    accuracy_retention,
    chance_level,
    parse_harness_results,
)


def _harness_output(**overrides):
    """A realistic `simple_evaluate` return value, in this harness release's shape."""
    payload = {
        "results": {
            "hellaswag": {"acc,none": 0.2841, "acc_stderr,none": 0.0045, "acc_norm,none": 0.3112},
            "piqa": {"acc,none": 0.6142, "acc_stderr,none": 0.0114},
            "arc_easy": {"acc,none": 0.4381, "acc_stderr,none": 0.0102},
        },
        "versions": {"hellaswag": 1.0, "piqa": 1.0, "arc_easy": 1.0},
        "n-samples": {
            "hellaswag": {"original": 10042, "effective": 10042},
            "piqa": {"original": 1838, "effective": 1838},
            "arc_easy": {"original": 2376, "effective": 2376},
        },
    }
    payload.update(overrides)
    return payload


class TestTheTasksArePinned:
    def test_the_three_tasks_the_plan_names(self):
        assert DOWNSTREAM_TASKS == ("hellaswag", "piqa", "arc_easy")

    def test_every_task_has_a_chance_level(self):
        """A score cannot be interpreted without knowing what guessing gets."""
        assert set(CHANCE_LEVEL) == set(DOWNSTREAM_TASKS)

    def test_chance_levels_match_the_choice_counts(self):
        assert chance_level("hellaswag") == 0.25
        assert chance_level("arc_easy") == 0.25
        assert chance_level("piqa") == 0.50

    def test_the_primary_metric_is_fixed_in_advance(self):
        """Both `acc` and `acc_norm` are legitimate, so choosing after seeing results is forbidden."""
        assert PRIMARY_METRIC == "acc"


class TestParsingKeepsProvenance:
    def test_every_requested_task_is_returned_in_order(self):
        results = parse_harness_results(_harness_output())
        assert [r.task for r in results] == list(DOWNSTREAM_TASKS)

    def test_task_versions_are_captured(self):
        """§4.8 requires them: task definitions change between releases."""
        for result in parse_harness_results(_harness_output()):
            assert result.task_version is not None, result.task

    def test_a_version_reported_per_task_entry_is_also_found(self):
        """The harness has moved this field between releases, so both shapes must work."""
        payload = _harness_output(versions={})
        payload["results"]["hellaswag"]["task_version"] = 3
        results = {r.task: r for r in parse_harness_results(payload)}
        assert results["hellaswag"].task_version == "3"

    def test_a_missing_version_is_recorded_as_none_not_defaulted(self):
        """A fabricated version is worse than a visibly absent one."""
        payload = _harness_output(versions={})
        results = {r.task: r for r in parse_harness_results(payload)}
        assert results["piqa"].task_version is None

    def test_metric_keys_with_filter_suffixes_are_read(self):
        """This release emits `acc,none`, not `acc`. A strict lookup would find nothing."""
        results = {r.task: r for r in parse_harness_results(_harness_output())}
        assert results["arc_easy"].accuracy == pytest.approx(0.4381)

    def test_sample_counts_are_captured(self):
        results = {r.task: r for r in parse_harness_results(_harness_output())}
        assert results["hellaswag"].num_samples == 10042


class TestParsingRefusesToUnderReport:
    """The failure mode that matters: a short report read as a complete one."""

    def test_a_missing_task_raises_rather_than_being_omitted(self):
        payload = _harness_output()
        del payload["results"]["piqa"]
        with pytest.raises(EvaluationError, match="absent from the harness results"):
            parse_harness_results(payload)

    def test_a_task_without_the_primary_metric_raises(self):
        """`acc_norm` alone is not the number we said we would report."""
        payload = _harness_output()
        payload["results"]["piqa"] = {"acc_norm,none": 0.7}
        with pytest.raises(EvaluationError, match="no 'acc' metric"):
            parse_harness_results(payload)

    def test_empty_results_raise(self):
        with pytest.raises(EvaluationError):
            parse_harness_results({"results": {}})


class TestChanceLevelIsSurfaced:
    """A model at chance has stopped doing the task, not done it worse."""

    def test_a_score_above_chance_is_flagged_as_such(self):
        result = TaskResult("piqa", 0.62, None, None, "1.0", 100)
        assert result.is_above_chance

    def test_a_score_at_chance_is_not(self):
        result = TaskResult("piqa", 0.50, None, None, "1.0", 100)
        assert not result.is_above_chance

    def test_hellaswag_at_26_percent_is_above_chance_but_barely(self):
        """25% is the floor, so 26% is a real but nearly worthless score -- and 24% is broken."""
        assert TaskResult("hellaswag", 0.26, None, None, "1.0", 10).is_above_chance
        assert not TaskResult("hellaswag", 0.24, None, None, "1.0", 10).is_above_chance

    def test_the_report_lists_tasks_at_chance(self):
        report = DownstreamReport(
            tasks=[
                TaskResult("hellaswag", 0.2501, None, None, "1.0", 10),
                TaskResult("piqa", 0.49, None, None, "1.0", 10),
                TaskResult("arc_easy", 0.24, None, None, "1.0", 10),
            ]
        )
        assert report.tasks_at_chance == ["piqa", "arc_easy"]

    def test_chance_is_serialised_with_the_score(self):
        payload = TaskResult("piqa", 0.62, None, None, "1.0", 10).to_dict()
        assert payload["chance_level"] == 0.50
        assert payload["above_chance"] is True


class TestReportAggregation:
    def test_the_mean_is_unweighted_across_tasks(self):
        """Weighting by size would let HellaSwag alone decide the headline: it is 5x PIQA."""
        report = DownstreamReport(
            tasks=[
                TaskResult("hellaswag", 0.30, None, None, "1.0", 10042),
                TaskResult("piqa", 0.60, None, None, "1.0", 1838),
                TaskResult("arc_easy", 0.45, None, None, "1.0", 2376),
            ]
        )
        assert report.mean_accuracy == pytest.approx(0.45)

    def test_an_empty_report_does_not_divide_by_zero(self):
        assert DownstreamReport().mean_accuracy is None

    def test_the_subsample_limit_is_recorded(self):
        """A subsampled score is not comparable with a published one, so it must be visible."""
        assert DownstreamReport(limit=200).to_dict()["limit"] == 200
        assert DownstreamReport().to_dict()["limit"] is None

    def test_the_device_is_recorded(self):
        assert DownstreamReport(device="cuda").to_dict()["device"] == "cuda"


class TestAccuracyRetention:
    def test_retention_is_a_ratio_against_dense(self):
        assert accuracy_retention(0.45, 0.50) == pytest.approx(0.90)

    def test_retention_above_one_is_allowed(self):
        """It happens at mild compression, and clipping it would hide a real observation."""
        assert accuracy_retention(0.52, 0.50) == pytest.approx(1.04)

    def test_a_non_positive_dense_reference_is_refused(self):
        with pytest.raises(EvaluationError, match="must be > 0"):
            accuracy_retention(0.4, 0.0)


class TestConfigGuards:
    def test_an_unnamed_task_is_refused(self):
        """§4.3 names three. A typo must not reach the harness after a model is loaded."""
        from scale_aware_compression.config import ConfigError, EvaluationConfig

        with pytest.raises(ConfigError, match="does not name"):
            EvaluationConfig(downstream_tasks=["hellaswag", "winogrande"])

    def test_the_named_tasks_are_accepted(self):
        from scale_aware_compression.config import EvaluationConfig

        config = EvaluationConfig(downstream_tasks=list(DOWNSTREAM_TASKS))
        assert config.downstream_tasks == list(DOWNSTREAM_TASKS)

    def test_downstream_device_defaults_to_the_evaluation_device(self):
        from scale_aware_compression.config import EvaluationConfig
        from scale_aware_compression.constants import Device

        assert EvaluationConfig().effective_downstream_device is Device.CPU

    def test_downstream_device_can_differ_from_perplexity(self):
        """~53,000 forwards per model makes CPU ~150 h across the sweep against ~15-20 h on GPU."""
        from scale_aware_compression.config import EvaluationConfig
        from scale_aware_compression.constants import Device

        config = EvaluationConfig(device=Device.CPU, downstream_device=Device.CUDA)
        assert config.effective_downstream_device is Device.CUDA
