"""Repository-level checks: documentation, packaging metadata, CI, and the CPU-only policy.

These guard facts that live outside Python modules and would otherwise go stale silently — a
documented protocol that was deleted, a placeholder URL that shipped, a CI workflow that stopped
running the checks it claims to.

Nothing here downloads a model, imports torch, or needs CUDA.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import yaml

from scale_aware_compression.config import ExperimentConfig, load_document
from scale_aware_compression.constants import Device

REQUIRED_DOCS = (
    "research_question.md",
    "method_definition.md",
    "implementation_plan.md",
    "protocol_freeze.md",
    "findings_log.md",
    "review_brief.md",
    "methodology.md",
    "experiment_protocol.md",
    "benchmarking_protocol.md",
    "validity_threats.md",
    "reproducibility.md",
    "paper_outline.md",
    "STATUS.md",
    # The audit trail of the external review: what was fixed, what was deviated from, what is
    # still open. A review answered only in commit messages is a review whose open items vanish.
    "external_review_response.md",
)


@pytest.fixture(scope="module")
def pyproject(project_root: Path) -> dict:
    return tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def workflow_path(project_root: Path) -> Path:
    path = project_root / ".github" / "workflows" / "ci.yml"
    assert path.is_file(), "CI workflow is missing"
    return path


@pytest.fixture(scope="module")
def workflow_text(workflow_path: Path) -> str:
    return workflow_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def workflow(workflow_text: str) -> dict:
    return yaml.safe_load(workflow_text)


class TestDocumentation:
    @pytest.mark.parametrize("name", REQUIRED_DOCS)
    def test_document_exists_and_is_not_empty(self, project_root: Path, name: str):
        path = project_root / "docs" / name
        assert path.is_file(), f"docs/{name} is missing"
        assert len(path.read_text(encoding="utf-8").strip()) > 500, f"docs/{name} looks like a stub"

    @pytest.mark.parametrize("name", REQUIRED_DOCS)
    def test_document_starts_with_a_heading(self, project_root: Path, name: str):
        first = (project_root / "docs" / name).read_text(encoding="utf-8").lstrip().splitlines()[0]
        assert first.startswith("# "), f"docs/{name} does not start with a top-level heading"

    def test_method_definition_covers_the_required_sections(self, project_root: Path):
        text = (project_root / "docs" / "method_definition.md").read_text(encoding="utf-8")
        for heading in (
            "## Scope",
            "## Compressible Modules",
            "## Pruning Method",
            "## Quantisation Method",
            "## Sequential Pipeline",
            "## Joint Pipeline",
            "## Mask Scoring",
            "## Matched Compression Budget",
            "## Matched Optimisation Budget",
            "## Deployment Backend",
            "## Current Status",
        ):
            assert heading in text, f"method_definition.md is missing {heading!r}"

    def test_validity_threats_covers_the_required_sections(self, project_root: Path):
        text = (project_root / "docs" / "validity_threats.md").read_text(encoding="utf-8")
        for heading in (
            "## Construct Validity",
            "## Internal Validity",
            "## External Validity",
            "## Scale as an Independent Variable",
            "## Sparsity and Hardware Speed",
            "## Dataset and Evaluation Limits",
            "## Statistical Limits",
            "## Backend Limits",
        ):
            assert heading in text, f"validity_threats.md is missing {heading!r}"

    def test_the_authoritative_research_plan_is_committed(self, project_root: Path):
        """The plan is the source document; a repo without it cannot be worked from alone."""
        plan = project_root / "docs" / "research_plan.pdf"
        assert plan.is_file(), "docs/research_plan.pdf is missing"
        assert plan.stat().st_size > 50_000, "research_plan.pdf looks truncated"

    def test_agent_guidance_exists_and_points_at_the_status_file(self, project_root: Path):
        """CLAUDE.md is auto-loaded; it is what makes a session on a fresh machine work."""
        text = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
        assert "docs/STATUS.md" in text
        assert "research_plan.pdf" in text
        assert "implementation_plan.md" in text

    def test_status_file_records_when_it_was_last_updated(self, project_root: Path):
        """A stale handoff is worse than none, so it must be datable at a glance."""
        text = (project_root / "docs" / "STATUS.md").read_text(encoding="utf-8")
        assert "Last updated" in text

    def test_implementation_plan_covers_every_phase(self, project_root: Path):
        text = (project_root / "docs" / "implementation_plan.md").read_text(encoding="utf-8")
        for phase in range(11):
            assert f"### Phase {phase} " in text, f"implementation_plan.md is missing Phase {phase}"
        assert "## Testing plan" in text

    def test_method_definition_marks_the_algorithms_as_placeholders(self, project_root: Path):
        """Until the algorithms exist, the document must say so."""
        text = (project_root / "docs" / "method_definition.md").read_text(encoding="utf-8")
        assert "placeholder" in text.lower()

    def test_method_definition_specifies_the_layerwise_method(self, project_root: Path):
        """Plan §3.1 selects layerwise post-training reconstruction.

        The spec previously described full-model quantisation-aware fine-tuning, which is a
        different method family: the unit of optimisation is local steps per layer, not global
        optimiser steps. A regression here means the superseded design crept back in.
        """
        text = (project_root / "docs" / "method_definition.md").read_text(encoding="utf-8")
        lowered = text.lower()
        assert "layerwise" in lowered
        assert "local steps" in lowered
        assert "reconstruction" in lowered

    def test_mask_scoring_is_settled_and_scored_under_quantisation(self, project_root: Path):
        """Plan §3.8 makes scoring under quantised weights the definition of joint.

        Ranking on untouched FP32 weights is the "prune fully then plain PTQ" failure case that
        §3.8 lists as *not* qualifying as joint, so the spec must not drift back to it.
        """
        text = (project_root / "docs" / "method_definition.md").read_text(encoding="utf-8")
        section = text.split("## Mask Scoring", 1)[1].split("\n## ", 1)[0]
        assert "activation-weighted magnitude" in section.lower()
        assert "quantised weights" in section.lower()
        assert "Q_b(W_ij)" in section

    def test_protocol_freeze_settles_the_previously_open_decisions(self, project_root: Path):
        """§2.7 requires these frozen before the experiments, and §6.3 before seeing results."""
        text = (project_root / "docs" / "protocol_freeze.md").read_text(encoding="utf-8")
        for decision in ("### D1", "### D2", "### D3"):
            assert decision in text, f"protocol_freeze.md is missing {decision!r}"
        for item in (
            "Pythia variant",
            "Benchmark runtime",
            "Practical-importance rule",
            "Scale x-axis",
            "Seeds",
        ):
            assert item in text, f"protocol_freeze.md does not freeze {item!r}"

    def test_findings_log_records_the_conditions_every_number_came_from(self, project_root: Path):
        """A perplexity without its evaluation window is not a result.

        The log is what the paper will be written from, so it has to carry provenance rather than
        bare numbers -- the machine, the pinned versions, the model revision, the calibration
        fingerprint, and the evaluation window.
        """
        text = (project_root / "docs" / "findings_log.md").read_text(encoding="utf-8")
        for item in (
            "i7-13620H",  # the machine, per §4.7
            "2.13.0+cu126",  # the pinned runtime, per §2.7
            "50f5173d",  # the pythia-160m revision, per §2.7
            "20bf57e6b08ed60d",  # the calibration set, per §3.11
            "64 sequences",  # the evaluation window
        ):
            assert item in text, f"findings_log.md does not record {item!r}"

    def test_findings_log_separates_permissible_from_impermissible_claims(self, project_root: Path):
        """§6.7 and §6.8 govern what the paper may say; the log must not invite overclaiming.

        Single-seed, single-model, single-window numbers are easy to quote as though they were
        results. The log states explicitly that they are not.
        """
        text = (project_root / "docs" / "findings_log.md").read_text(encoding="utf-8")
        assert "May not claim" in text
        assert "single seed" in text.lower()

    def test_findings_log_keeps_superseded_measurements(self, project_root: Path):
        """A deleted measurement is one that gets re-argued later.

        The tensor-wide comparison-group numbers are wrong as *results* but are the evidence for why
        the default changed, so they stay on the record marked superseded.
        """
        text = (project_root / "docs" / "findings_log.md").read_text(encoding="utf-8")
        assert "superseded" in text.lower()
        assert "233.94" in text, "the pre-fix pruning perplexity is the evidence for F-07"

    def test_protocol_freeze_records_the_hardware_the_numbers_come_from(self, project_root: Path):
        """§10.2 requires the machine recorded; §4.7 forbids mixing machines in one table."""
        text = (project_root / "docs" / "protocol_freeze.md").read_text(encoding="utf-8")
        for field in ("CPU", "RAM", "GPU", "VRAM", "driver"):
            assert field in text or field.lower() in text, (
                f"protocol_freeze.md does not record {field!r}"
            )

    def test_promotion_checklist_is_documented(self, project_root: Path):
        text = (project_root / "docs" / "reproducibility.md").read_text(encoding="utf-8")
        assert "## Promotion checklist" in text
        for item in (
            "Resolved configuration saved",
            "Git commit recorded",
            "Hardware metadata recorded",
            "Matched sequential and joint budgets",
            "No benchmark anomaly",
            "Consistent backend and output format",
        ):
            assert item in text, f"promotion checklist is missing {item!r}"

    def test_outputs_versus_results_is_documented(self, project_root: Path):
        for name in ("README.md", "docs/reproducibility.md"):
            text = (project_root / name).read_text(encoding="utf-8")
            assert "outputs/" in text and "results/" in text
            assert "promotion" in text.lower() or "Promotion" in text, (
                f"{name} does not explain promotion from outputs/ to results/"
            )

    def test_four_bit_backend_risk_is_flagged_everywhere_it_matters(self, project_root: Path):
        """A reader must not reach the aggressive budget without meeting the caveat."""
        targets = (
            "README.md",
            "docs/benchmarking_protocol.md",
            "docs/experiment_protocol.md",
            "docs/method_definition.md",
            "configs/compression/quantisation.yaml",
            "configs/experiments/main_scale_sweep.yaml",
        )
        for name in targets:
            text = (project_root / name).read_text(encoding="utf-8")
            lowered = text.lower()
            assert "4-bit" in lowered, f"{name} does not mention the 4-bit risk"
            assert "int8" in lowered, f"{name} does not mention INT8"
            assert "backend" in lowered, f"{name} does not mention the backend"

    def test_int8_fallback_plan_is_documented(self, project_root: Path):
        for name in (
            "README.md",
            "docs/benchmarking_protocol.md",
            "docs/method_definition.md",
            "configs/experiments/main_scale_sweep.yaml",
        ):
            text = (project_root / name).read_text(encoding="utf-8").lower()
            assert "fallback" in text, f"{name} does not document the INT8 fallback"


class TestPackagingMetadata:
    def test_project_urls_do_not_contain_the_owner_placeholder(self, pyproject: dict):
        urls = pyproject["project"]["urls"]
        for label, url in urls.items():
            assert "OWNER" not in url, f"project.urls.{label} still contains the OWNER placeholder"

    def test_project_urls_point_at_the_repository(self, pyproject: dict):
        urls = pyproject["project"]["urls"]
        expected_base = "https://github.com/shehryarf67/Scale-Aware-Joint-Compression"
        assert urls["Repository"] == expected_base
        assert urls["Documentation"] == f"{expected_base}/tree/main/docs"
        assert urls["Issues"] == f"{expected_base}/issues"

    def test_no_placeholder_urls_anywhere_in_the_repository(self, project_root: Path):
        offenders: list[str] = []
        for path in sorted(project_root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".toml", ".md", ".yaml", ".yml"}:
                continue
            if any(part.startswith((".venv", ".git")) for part in path.parts):
                continue
            if path.name == "test_repository.py":  # this file names the placeholder on purpose
                continue
            if "github.com/OWNER" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(project_root)))
        assert not offenders, f"placeholder OWNER URLs remain in: {offenders}"

    def test_requires_python_311_or_newer(self, pyproject: dict):
        assert pyproject["project"]["requires-python"] == ">=3.11"

    def test_declares_the_test_markers_ci_deselects(self, pyproject: dict):
        """Every marker CI deselects must be declared here.

        CI runs `-m "not slow and not requires_model and not requires_torch"`, and with
        `--strict-markers` an undeclared marker would fail the whole run.
        """
        declared = {
            entry.split(":", 1)[0]
            for entry in pyproject["tool"]["pytest"]["ini_options"]["markers"]
        }
        assert {"slow", "requires_model", "requires_torch"} <= declared


class TestContinuousIntegration:
    def test_runs_on_push_and_pull_request_to_main(self, workflow: dict):
        # PyYAML parses the bare key `on` as the boolean True, so accept either spelling.
        triggers = workflow.get("on", workflow.get(True))
        assert triggers is not None, "workflow declares no triggers"
        assert triggers["push"]["branches"] == ["main"]
        assert triggers["pull_request"]["branches"] == ["main"]

    def test_uses_python_311(self, workflow_text: str):
        assert '"3.11"' in workflow_text

    def test_runs_lint_format_and_tests(self, workflow_text: str):
        assert "ruff check ." in workflow_text
        assert "ruff format --check ." in workflow_text
        assert "pytest" in workflow_text

    def test_installs_the_project_and_dev_dependencies(self, workflow_text: str):
        assert "pip install -e . -r requirements-dev.txt" in workflow_text

    def test_uses_pip_caching(self, workflow_text: str):
        assert "cache: pip" in workflow_text

    def test_blocks_hugging_face_downloads(self, workflow_text: str):
        """Offline env vars turn an accidental download into a loud failure."""
        assert "HF_HUB_OFFLINE" in workflow_text
        assert "TRANSFORMERS_OFFLINE" in workflow_text

    def test_deselects_slow_and_model_dependent_tests(self, workflow_text: str):
        assert "not slow" in workflow_text
        assert "not requires_model" in workflow_text

    def test_main_job_actually_runs_the_torch_tests(self, workflow_text: str):
        """Torch tests cover the real compression and evaluation paths.

        The minimal-environment step still excludes them -- that step exists to prove the
        library works without torch -- but the main job must not, or the coverage is theatre.
        """
        main_step = workflow_text.split("Confirm the fast suite")[0]
        assert 'pytest -m "not slow and not requires_model"' in main_step
        assert "not requires_torch" not in main_step

    def test_minimal_environment_step_excludes_torch_tests(self, workflow_text: str):
        minimal_step = workflow_text.split("Confirm the fast suite")[-1]
        assert "not requires_torch" in minimal_step

    def test_installs_the_cpu_torch_wheel(self, workflow_text: str):
        """The default index would pull a multi-gigabyte CUDA build CI has no use for."""
        assert "download.pytorch.org/whl/cpu" in workflow_text


class TestCpuOnlyPolicyAcrossConfigs:
    def test_every_config_keeps_the_benchmark_on_cpu(self, configs_dir: Path):
        for path in sorted(configs_dir.rglob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.benchmark.device is Device.CPU, f"{path.name} benchmarks off CPU"

    def test_every_confirmatory_config_evaluates_on_cpu(self, configs_dir: Path):
        """The rule that matters: a REPORTED number comes from CPU.

        This used to assert CPU for every experiment config, which is stricter than the design.
        ``check_evaluation_device`` warns rather than errors precisely because exploratory
        evaluation on GPU is legitimate -- and it is 22.5x faster for a relative difference of
        8.3e-06, the same magnitude as CPU thread-configuration sensitivity (F-29).

        The blanket version also protected nothing the tighter version does not: what must never
        happen is a *confirmatory* config drifting off CPU, and confirmatory is exactly the
        configs that evaluate on the held-out test split (Amendment A1 §5.2).
        """
        for path in sorted((configs_dir / "experiments").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            if config.data.eval_split != "test":
                continue
            assert config.evaluation.device is Device.CPU, (
                f"{path.name} evaluates the test split off CPU; confirmatory numbers are CPU-only"
            )

    def test_gpu_evaluation_is_confined_to_declared_exploratory_configs(self, configs_dir: Path):
        """A GPU-evaluated config must say so in its tags, so a record's provenance is greppable.

        The pairing is the point: GPU evaluation is allowed *because* the run is exploratory, so a
        config that takes the speedup without declaring the status has taken the licence without
        the constraint that justifies it.
        """
        for path in sorted((configs_dir / "experiments").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            if config.evaluation.device is Device.CPU:
                continue
            tags = set(config.experiment.tags)
            assert "exploratory" in tags, (
                f"{path.name} evaluates on {config.evaluation.device.value} without an "
                "'exploratory' tag"
            )
            assert config.data.eval_split != "test", f"{path.name} is confirmatory but not on CPU"

    def test_benchmark_config_pins_a_thread_count(self, configs_dir: Path):
        for path in sorted(configs_dir.rglob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.benchmark.num_threads >= 1

    @pytest.mark.requires_torch
    def test_shipped_quantisation_backend_exists_in_the_installed_torch(self, configs_dir: Path):
        """A backend name torch does not recognise fails at conversion, after the compute is spent.

        The configs originally shipped ``x86``, which every PyTorch tutorial names but which this
        build rejects -- its only engine is ``onednn``. Asserting against the installed torch
        turns a torch upgrade that renames engines into a failed test rather than a failed run.
        """
        import torch

        supported = set(torch.backends.quantized.supported_engines)
        for path in sorted(configs_dir.rglob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            if not config.compression.quantisation.enabled:
                continue
            backend = config.compression.quantisation.backend
            assert backend in supported, (
                f"{path.name} requests quantisation backend {backend!r}, but the installed "
                f"torch {torch.__version__} supports only {sorted(supported)}"
            )


class TestFrozenProtocolMatchesTheConfigs:
    """§2.7 decisions must hold in the configs, not only in the freeze document."""

    def test_every_model_config_pins_a_commit_sha(self, configs_dir: Path):
        """A branch name is not a pin: a Hub repo can be updated in place under it."""
        for path in sorted((configs_dir / "models").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            revision = config.model.revision
            assert revision is not None, f"{path.name} has an unpinned revision"
            assert re.fullmatch(r"[0-9a-f]{40}", revision), (
                f"{path.name} revision {revision!r} is not a 40-character commit SHA"
            )

    def test_every_shipped_config_names_a_namespaced_dataset(self, configs_dir: Path):
        """`datasets` 5.x rejects bare canonical aliases like ``wikitext``.

        Found the hard way: the first real execution of the load path failed with an opaque
        HfUriError about an internal ``hf://`` path. It fails at load time, after the model is
        already resident, so it costs a whole setup. Only *shipped* configs are checked -- test
        fixtures legitimately use stub corpus names that never reach the Hub.
        """
        for path in sorted(configs_dir.rglob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert "/" in config.data.dataset, (
                f"{path.name} names dataset {config.data.dataset!r}, which is not a "
                "'namespace/name' Hub id"
            )

    def test_no_pythia_config_uses_the_deduplicated_variant(self, configs_dir: Path):
        """§2.7 forbids mixing standard and deduplicated Pythia across sizes."""
        for path in sorted((configs_dir / "models").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert "deduped" not in config.model.hf_id, (
                f"{path.name} points at a deduplicated checkpoint: {config.model.hf_id}"
            )


class TestLatencyIsOnlyRecordedWhenItMeansSomething:
    """A number that measures the plumbing is worse than no number.

    Decision D1 makes native INT8 the only latency backend and excludes W4 from latency tables. But
    every quantised arm converts to `PackedLinear`, whose forward unpacks codes, dequantises to FP32
    and calls a dense matmul — slower than the dense model, and attributable to neither sparsity nor
    precision. Until a native INT8 runtime artefact exists, quantised arms record no latency.
    """

    def _runner(self, document, tmp_path):
        import copy

        from scale_aware_compression.experiments.runner import ExperimentRunner

        resolved = copy.deepcopy(document)
        resolved.setdefault("runtime", {})["output_dir"] = str(tmp_path)
        return ExperimentRunner(ExperimentConfig.from_mapping(resolved))

    @pytest.mark.parametrize(
        ("method", "bits", "expected"),
        [
            ("dense", 32, "fp32"),
            ("pruning", 32, "fp32"),
            ("quantisation", 8, "packed_dequantising"),
            ("joint", 4, "packed_dequantising"),
        ],
    )
    def test_the_runtime_representation_is_recorded(
        self, minimal_config_document, tmp_path, method, bits, expected
    ):
        import copy

        document = copy.deepcopy(minimal_config_document)
        document["compression"] = {
            "method": method,
            "pruning": {"enabled": method in {"pruning", "sequential", "joint"}, "sparsity": 0.3},
            "quantisation": {"enabled": bits < 32, "bits": bits if bits < 32 else 8},
        }
        runner = self._runner(document, tmp_path)
        assert runner._runtime_representation() == expected

    def test_only_fp32_artefacts_are_benchmarked(self, minimal_config_document, tmp_path):
        """The pruning-only arm stays FP32, which is what makes RQ4 answerable without a 4-bit kernel."""
        runner = self._runner(minimal_config_document, tmp_path)
        assert runner._latency_is_meaningful("fp32") is True
        assert runner._latency_is_meaningful("packed_dequantising") is False


class TestResumabilityAndRecordHygiene:
    """A sweep that skips a stale cell is worse than one that re-runs it.

    `exists()` returned true whenever the JSON file was present — regardless of whether the run
    succeeded, what weights it used, or what budget it ran. Resuming on that basis silently keeps a
    result that does not answer the question being asked now.
    """

    def _tracker(self, tmp_path):
        from scale_aware_compression.experiments.runner import ExperimentTracker

        return ExperimentTracker(tmp_path)

    def _record(self, config, *, checkpoint_dir=None, **overrides):
        from scale_aware_compression.experiments.runner import ExperimentRecord

        record = ExperimentRecord.from_config(config, capture_environment=False)
        record.status = "success"
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            record.checkpoint_path = checkpoint_dir
            record.checkpoint = {
                "reload_verified": True,
                "artifact_sha256": "test-digest",
                "artifact_retained": True,
            }
        for key, value in overrides.items():
            setattr(record, key, value)
        return record

    @pytest.fixture
    def config(self, minimal_config_document):
        import copy

        document = copy.deepcopy(minimal_config_document)
        document["compression"] = {
            "method": "joint",
            "pruning": {"sparsity": 0.3},
            "quantisation": {"bits": 4},
        }
        document["model"] = {**document.get("model", {}), "revision": "abc123"}
        return ExperimentConfig.from_mapping(document)

    def test_a_successful_matching_record_is_skippable(self, tmp_path, config):
        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True

    def test_an_old_success_without_verified_checkpoint_is_re_run(self, tmp_path, config):
        tracker = self._tracker(tmp_path)
        record = self._record(config)
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is False

    def test_a_failed_record_is_re_run(self, tmp_path, config):
        """A crashed cell must not be mistaken for a completed one."""
        tracker = self._tracker(tmp_path)
        record = self._record(config, status="failure")
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is False

    def test_a_record_from_different_weights_is_re_run(self, tmp_path, config):
        """§2.7 pins revisions precisely so this cannot pass unnoticed."""
        import copy

        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        tracker.save(record)

        moved = copy.deepcopy(config)
        moved.model.revision = "def456"
        assert tracker.exists_valid(record.experiment_id, moved) is False

    def test_a_record_at_a_different_budget_is_re_run(self, tmp_path, config):
        import copy

        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        tracker.save(record)

        rebudgeted = copy.deepcopy(config)
        rebudgeted.compression.pruning.sparsity = 0.5
        assert tracker.exists_valid(record.experiment_id, rebudgeted) is False

    def test_re_running_a_cell_does_not_duplicate_its_csv_row(self, tmp_path, config):
        """Appending left two rows for one run, and anything averaging them weights the stale one."""
        import csv

        tracker = self._tracker(tmp_path)
        first = self._record(config, duration_seconds=1.0)
        tracker.save(first)
        second = self._record(config, duration_seconds=2.0)
        tracker.save(second)

        with tracker.csv_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))

        matching = [row for row in rows if row["experiment_id"] == first.experiment_id]
        assert len(matching) == 1, "the stale row was not replaced"

    def test_distinct_runs_still_get_their_own_rows(self, tmp_path, config):
        """Upsert must key on the run, not collapse the whole file to one row."""
        import copy

        tracker = self._tracker(tmp_path)
        tracker.save(self._record(config))

        other = copy.deepcopy(config)
        other.runtime.seed = 999
        tracker.save(self._record(other))

        with tracker.csv_path.open(encoding="utf-8", newline="") as handle:
            import csv

            assert len(list(csv.DictReader(handle))) == 2

    def test_a_record_from_another_machine_is_re_run(self, tmp_path, config):
        """B-33. Two hosts may now run compression, so two hosts may write into one directory.

        Reusing the other machine's record would put two hosts inside a single comparison, which
        is the unmatched-condition error §3.11 exists to prevent -- and invisible, because the
        difference is floating-point reduction order rather than anything a reader would notice.
        """
        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        record.hardware = {
            "system": "Linux",
            "cpu_model": "x86_64",
            "cpu_count_logical": 8,
            "cuda_device_names": ["Tesla T4"],
        }
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is False

    def test_a_record_from_this_machine_is_still_skippable(self, tmp_path, config):
        """The host guard must not invalidate the ~50 records already on the benchmark host."""
        from scale_aware_compression.hardware import get_hardware_info

        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        record.hardware = get_hardware_info()
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True

    def test_a_record_with_no_hardware_recorded_is_not_invalidated(self, tmp_path, config):
        """Records predating the field report "unknown"; a new guard must not force a recompute."""
        tracker = self._tracker(tmp_path)
        record = self._record(config, checkpoint_dir=tmp_path / "checkpoint")
        record.hardware = {}
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True


class TestNoRunArtefactsCommitted:
    """Git tracks nothing but `.gitkeep` and the declared exceptions under `outputs/`/`results/`.

    Checked against the index rather than the working tree. On a machine that actually runs the
    experiments these directories are *supposed* to be full of run artefacts, so asserting the
    directory is empty would fail for exactly the machine the rule exists to protect. The invariant
    is that nothing gets committed, not that nothing exists.

    **The exception list is deliberately narrow and enumerated here, not pattern-matched.** Adding a
    path to it is a decision that should show up in a diff and need a reason:

    * ``results/evidence/`` -- the committed evidence set. `outputs/` being ignored meant the only
      copy of every number in the findings log lived in an ignored directory, so a fresh clone could
      not recompute a table. Four plain-text files, ~1.9 MB, regenerable by
      `scripts/export_evidence.py`.
    * ``results/summaries/*.md`` -- promoted human-readable summaries, already permitted by
      `.gitignore`.

    * ``results/figures/*.{png,pdf}`` and ``results/tables/*.{md,csv}`` -- the paper's figures and
      tables. Same reasoning as the evidence set, one step further along: a reviewer cannot approve
      a plot that exists only on the machine that produced it. Small, plain outputs of
      `scripts/generate_plots.py` and `scripts/build_paper_tables.py`.

    Everything else -- checkpoints, packed artefacts, logs, raw metrics -- stays out. The extension
    lists matter: they are what stops a stray `.npy` or a 4 GB checkpoint riding along in a
    directory that is otherwise permitted.
    """

    ALLOWED_PREFIXES = ("results/evidence/",)
    ALLOWED_SUFFIXES = (".gitkeep",)
    ALLOWED_BY_DIRECTORY = {
        "results/figures/": (".png", ".pdf"),
        "results/tables/": (".md", ".csv"),
    }

    def _is_permitted(self, path: str) -> bool:
        """Whether a tracked path under these directories is a declared exception."""
        if path.endswith(self.ALLOWED_SUFFIXES):
            return True
        if path.startswith(self.ALLOWED_PREFIXES):
            return True
        for prefix, suffixes in self.ALLOWED_BY_DIRECTORY.items():
            if path.startswith(prefix):
                return path.endswith(suffixes)
        return path.startswith("results/summaries/") and path.endswith(".md")

    @pytest.mark.parametrize("directory", ["outputs", "results", "data"])
    def test_git_tracks_only_gitkeep(self, project_root: Path, directory: str):
        tracked = subprocess.run(
            ["git", "ls-files", directory],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode != 0:
            pytest.skip("not a git checkout")

        unexpected = [
            line
            for line in tracked.stdout.splitlines()
            if line.strip() and not self._is_permitted(line.strip())
        ]
        assert not unexpected, (
            f"git tracks run artefacts under {directory}/, which must never be committed: "
            f"{unexpected}. If this is a deliberate new exception, add it to ALLOWED_PREFIXES with "
            "a reason rather than widening the check."
        )

    def test_no_checkpoint_or_weight_file_is_ever_tracked(self, project_root: Path):
        """The exception list must not become a hole. Size and extension, independent of directory."""
        tracked = subprocess.run(
            ["git", "ls-files", "outputs", "results", "data"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if tracked.returncode != 0:
            pytest.skip("not a git checkout")
        forbidden = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf", ".onnx")
        for line in tracked.stdout.splitlines():
            path = line.strip()
            if not path:
                continue
            assert not path.endswith(forbidden), f"a model artefact is tracked: {path}"

    @pytest.mark.parametrize("directory", ["outputs", "results", "data"])
    def test_directory_is_ignored_so_a_stray_artefact_cannot_be_added(
        self, project_root: Path, directory: str
    ):
        """The rule has to be enforced by .gitignore, not by remembering to check."""
        probe = f"{directory}/__ignore_probe__.json"
        result = subprocess.run(
            ["git", "check-ignore", probe],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{probe} is not git-ignored"


class TestWithdrawnSeedEraRulesCannotCreepBack:
    """Amendment A1 withdrew the run-seed axis. Its vocabulary must not read as current policy.

    The specific hazard: §6.3's original practical-importance rule gated a joint gain on exceeding
    the *seed spread*, and the seed spread is exactly zero because the pipeline is deterministic
    (F-15) -- so the gate excluded nothing while looking strict. The rule survives in the freeze
    table and in the experiment protocol as a **record**, which is correct for an append-only
    history, but every occurrence has to be visibly marked as superseded or withdrawn rather than
    sitting in a checklist someone will follow.

    This is the guard the external review asked for after finding those documents quotable as
    current policy.
    """

    MARKERS = ("supersede", "withdraw", "vacuous", "inert", "f-15", "zero")

    @staticmethod
    def _paragraphs(text: str) -> list[str]:
        return [block for block in text.split("\n\n") if block.strip()]

    @pytest.mark.parametrize(
        "document",
        ["protocol_freeze.md", "experiment_protocol.md", "STATUS.md", "methodology.md"],
    )
    def test_seed_spread_is_never_stated_without_being_marked(
        self, project_root: Path, document: str
    ):
        """Each paragraph mentioning the seed spread must also say it is withdrawn or why."""
        path = project_root / "docs" / document
        if not path.exists():  # methodology.md is optional in some checkouts
            pytest.skip(f"{document} not present")
        for block in self._paragraphs(path.read_text(encoding="utf-8")):
            lowered = block.lower()
            if "seed spread" not in lowered and "seed-to-seed spread" not in lowered:
                continue
            assert any(marker in lowered for marker in self.MARKERS), (
                f"{document} states the seed spread rule without marking it superseded:\n\n{block}"
            )

    def test_no_checklist_item_asks_for_a_seed_spread_comparison(self, project_root: Path):
        """A checklist is followed, not read. A vacuous gate in one is worse than in prose."""
        for document in ("experiment_protocol.md", "benchmarking_protocol.md"):
            path = project_root / "docs" / document
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith("- [ ]") and not stripped.startswith("- [x]"):
                    continue
                lowered = stripped.lower()
                if "seed spread" in lowered:
                    assert "withdrawn" in lowered or "vacuous" in lowered, (
                        f"{document} has a checklist item gating on the seed spread: {stripped}"
                    )

    def test_the_freeze_table_records_the_amended_rule(self, project_root: Path):
        """The replacement must be stated, not merely the withdrawal."""
        text = (project_root / "docs" / "protocol_freeze.md").read_text(encoding="utf-8")
        assert "amended practical-importance rule" in text.lower()
        # The amendment is a LOOSENING of a pre-registered rule, and A1 requires the paper to say so.
        assert "reduction in pre-registered strength" in text.lower()

    def test_the_downstream_rule_is_withdrawn_rather_than_amended(self, project_root: Path):
        """No replacement is claimed for it, and claiming one would be unsupported."""
        text = (project_root / "docs" / "protocol_freeze.md").read_text(encoding="utf-8")
        assert "WITHDRAWN, not amended" in text


class TestTheDownstreamHarnessPinCannotDrift:
    """§4.8 requires task versions recorded, which only means something if the harness is pinned.

    The pin lives in two files -- `requirements.txt` for the research environment and the
    `downstream` extra in `pyproject.toml` -- and a value copied into a second file and then left
    behind when the first changed has already happened twice in this repository (the budget freeze,
    and the seed-axis withdrawal). This asserts they agree, and that neither has drifted to a range.
    """

    PIN = "lm-eval==0.4.12"

    def test_requirements_pins_the_harness_exactly(self, project_root: Path):
        text = (project_root / "requirements.txt").read_text(encoding="utf-8")
        assert self.PIN in text, f"requirements.txt does not pin {self.PIN}"

    def test_the_optional_extra_pins_the_same_version(self, pyproject: dict):
        extras = pyproject["project"]["optional-dependencies"]
        assert "downstream" in extras, "the downstream extra is missing"
        assert self.PIN in extras["downstream"]

    def test_the_harness_is_not_a_core_dependency(self, pyproject: dict):
        """Nothing in the library or the suite imports it, and CI should not install it.

        `evaluation.downstream` imports lm_eval inside the function that calls it and raises a
        message naming requirements.txt if it is absent, so the offline tests stub the harness output
        instead.
        """
        core = " ".join(pyproject["project"]["dependencies"])
        assert "lm-eval" not in core and "lm_eval" not in core

    def test_no_floor_pin_anywhere(self, project_root: Path, pyproject: dict):
        """A range would let a later install score a different task and still call it HellaSwag."""
        haystacks = [
            (project_root / "requirements.txt").read_text(encoding="utf-8"),
            " ".join(pyproject["project"]["optional-dependencies"]["downstream"]),
        ]
        for text in haystacks:
            for line in text.splitlines():
                if "lm-eval" in line and not line.strip().startswith("#"):
                    assert "==" in line, f"lm-eval is not pinned exactly: {line.strip()}"

    def test_importing_the_module_does_not_require_the_harness(self):
        """The property that lets it be optional: the import is inside the calling function."""
        import scale_aware_compression.evaluation.downstream as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            if line.startswith("import lm_eval") or line.startswith("from lm_eval"):
                raise AssertionError(f"module-level harness import: {line!r}")


class TestTheCommittedEvidenceSetIsCurrent:
    """`outputs/` is ignored, so without a committed export the findings log is unverifiable.

    The external review's point: a reviewer with a fresh clone could not recompute a single table in
    `docs/findings_log.md`, because the only copy of every number lived in a git-ignored directory.
    `results/evidence/` closes that -- normalised cells, per-replicate joint gains against
    best-of-sequential, per-window NLL for the paired bootstrap, and a sha256 per source record.

    These tests are about the set not going **stale**, which is the failure mode a committed
    derivative has. They skip when the source records are absent, because tier-1 machines legitimately
    have no `outputs/`.
    """

    @pytest.fixture(scope="class")
    def evidence_dir(self, project_root: Path) -> Path:
        return project_root / "results" / "evidence"

    def test_the_evidence_set_is_committed(self, project_root: Path, evidence_dir: Path):
        """It must be tracked, not merely present -- `results/**` is ignored by default."""
        tracked = subprocess.run(
            ["git", "ls-files", "results/evidence"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        names = {line.split("/")[-1] for line in tracked.stdout.splitlines() if line.strip()}
        for required in ("cells.csv", "joint_gains.csv", "windows.csv", "MANIFEST.json"):
            assert required in names, f"results/evidence/{required} is not tracked by git"

    def test_every_cell_row_carries_its_provenance(self, evidence_dir: Path):
        """A measurement without its commit, revision and draw cannot be reproduced."""
        import csv

        path = evidence_dir / "cells.csv"
        if not path.exists():
            pytest.skip("evidence set not present")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "cells.csv is empty"
        for column in (
            "experiment_id",
            "git_commit",
            "model_revision",
            "method_version",
            "host",
            "eval_split",
            "evaluation_device",
            "dataset_fingerprint",
            "calibration_replicate",
        ):
            assert column in rows[0], f"cells.csv is missing the {column!r} column"

    def test_joint_gains_record_which_baseline_each_gain_used(self, evidence_dir: Path):
        """§6.1 requires best-of {P→Q, Q→P}. B-30 was measuring against the weaker order.

        Making the chosen order an explicit column is what lets a reader check it rather than
        assume it. B-52 added the RULE alongside the order: on the test split the baseline is the
        frozen order, not a maximum, because only one order was ever run there and maximising over
        whatever else is present is how a validation record leaked into a test comparison.
        """
        import csv

        path = evidence_dir / "joint_gains.csv"
        if not path.exists():
            pytest.skip("evidence set not present")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows, "joint_gains.csv is empty"
        for column in ("baseline_order", "baseline_rule", "orders_available", "eval_split"):
            assert column in rows[0], f"joint_gains.csv must export {column!r}"
        for row in rows:
            assert row["baseline_order"] in {"sequential", "sequential_qp"}
            assert row["baseline_rule"] in {"frozen", "best-of"}
            # B-52: a test-split gain must never be taken against a maximum.
            if row["eval_split"] == "test":
                assert row["baseline_rule"] == "frozen"

    def test_joint_gains_never_pair_across_evaluation_splits(self, evidence_dir: Path):
        """B-52. Both splits produce the same experiment ids, so this cannot be eyeballed.

        The exported table once carried -0.2143 pp for qwen2.5-0.5b/moderate/rep0, formed from a
        validation Q→P record and a test joint record. The frozen-order test gain is -0.0357 pp.
        """
        import csv

        path = evidence_dir / "joint_gains.csv"
        if not path.exists():
            pytest.skip("evidence set not present")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            assert row.get("eval_split"), "every gain row must name the split it came from"
            assert row.get("dataset_fingerprint"), (
                "every gain row must name the evaluated data; a shared split label is not proof "
                "the same corpus and window were used"
            )

    def test_the_manifest_hashes_every_source_record(self, evidence_dir: Path):
        """The substitute for committing excluded artefacts: they stay identifiable."""
        path = evidence_dir / "MANIFEST.json"
        if not path.exists():
            pytest.skip("evidence set not present")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["source_records"] == len(manifest["record_sha256"])
        assert manifest["source_records"] > 0
        for digest in manifest["record_sha256"].values():
            assert len(digest) == 64, "not a sha256"
        assert "excluded_and_why" in manifest, "exclusions must be stated, not implied"

    @pytest.mark.slow
    def test_the_committed_set_matches_the_records_on_disk(self, project_root: Path):
        """Guards the staleness failure mode. Skipped where there are no records to compare."""
        metrics = project_root / "outputs" / "metrics"
        if not metrics.is_dir() or not any(metrics.glob("*.json")):
            pytest.skip("no run records on this machine")
        completed = subprocess.run(
            [sys.executable, "scripts/export_evidence.py", "--check"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, (
            "results/evidence/ is stale against outputs/metrics/. Re-run "
            f"scripts/export_evidence.py.\n{completed.stderr}"
        )


class TestTheConfirmatoryManifest:
    """A1 step 9. Step 10 runs once, costs ~38 h, and forbids tuning afterwards.

    The manifest is the artefact that makes "frozen" checkable instead of asserted: every commit,
    revision, cell, replicate, order, device and exclusion rule resolved in one place, with the
    checks that had to pass recorded alongside.
    """

    @pytest.fixture(scope="class")
    def manifest(self, project_root: Path) -> dict | None:
        path = project_root / "results" / "evidence" / "confirmatory_manifest.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def test_the_builder_refuses_a_dirty_tree(self, project_root: Path):
        """A freeze recorded at a -dirty commit cannot be reproduced.

        This project's one unusable result set came from a `-dirty` tree 22 commits behind main, so
        the builder refuses rather than warns. Asserted by reading the source, because the test
        cannot make the working tree dirty on demand without side effects.
        """
        source = (project_root / "scripts" / "build_confirmatory_manifest.py").read_text(
            encoding="utf-8"
        )
        assert "working tree is dirty" in source
        assert "valid_for_freeze" in source

    def test_the_manifest_records_whether_it_is_valid_for_freeze(self, manifest):
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        assert "valid_for_freeze" in manifest
        assert isinstance(manifest["valid_for_freeze"], bool)

    def test_a_frozen_manifest_has_no_failed_checks_and_a_clean_tree(self, manifest):
        """The two conditions that make the artefact mean anything."""
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        if not manifest["valid_for_freeze"]:
            pytest.skip("manifest is marked inspection-only")
        assert manifest["checks_failed"] == []
        assert manifest["tree_clean"] is True
        assert not str(manifest["git_commit"]).endswith("-dirty")

    def test_it_pins_the_confirmatory_conditions(self, manifest):
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        assert manifest["evaluation"]["split"] == "test", "confirmation must use the held-out split"
        assert manifest["evaluation"]["device"] == "cpu", "reported quality must come from CPU"
        assert manifest["benchmark"]["device"] == "cpu", "§4.6 deployment measurements are CPU-only"

    def test_a2_builder_freezes_the_executable_policy(self, project_root: Path):
        """A2 keeps logical coverage explicit while listing only records that can exist."""
        source = (project_root / "scripts" / "build_confirmatory_manifest.py").read_text(
            encoding="utf-8"
        )
        for field in (
            "logical_grid_cell_count",
            "executable_cell_count",
            "deduplicated_dense_slots",
            "dense_policy",
        ):
            assert field in source
        assert "executable_cells(plan)" in source

    def test_a2_has_a_validation_only_timing_pilot(self, project_root: Path):
        from scale_aware_compression.config import load_config

        config = load_config(project_root / "configs/experiments/confirmatory_timing_pilot.yaml")
        assert config.data.eval_split == "validation"
        assert config.runtime.output_dir.as_posix().endswith("outputs/timing_pilot")
        assert config.sweep.models == ["pythia-1b"]
        assert [method.value for method in config.sweep.methods] == ["dense", "joint"]
        assert config.sweep.budgets == ["aggressive"]
        assert config.compression.reconstruction.offload_blocks is True

    def test_a3_freezes_the_verified_offload_path(self, project_root: Path):
        from scale_aware_compression.config import load_config

        config = load_config(project_root / "configs/experiments/main_scale_sweep.yaml")
        assert config.compression.reconstruction.offload_blocks is True
        builder = (project_root / "scripts/build_confirmatory_manifest.py").read_text(
            encoding="utf-8"
        )
        assert "Amendment A3 requires" in builder
        assert '"residency_policy"' in builder

    def test_every_model_revision_is_a_full_sha(self, manifest):
        """§2.7. B-13 was a sweep inheriting one model's revision for every cell."""
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        for model, revision in manifest["model_revisions"].items():
            assert re.fullmatch(r"[0-9a-f]{40}", str(revision)), f"{model}: {revision!r}"

    def test_the_frozen_order_includes_the_one_reversed_cell(self, manifest):
        """If pythia-1b/moderate is not Q→P in the manifest, the resolution did not happen."""
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        assert manifest["frozen_sequential_order"]["pythia-1b/moderate"] == "sequential_qp"

    def test_it_states_where_significance_is_unreachable(self, manifest):
        """R=5 at 1B cannot reach p < 0.05 at any effect size, and that must be on the record."""
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        assert manifest["replicates"]["pythia-1b"]["significance_reachable"] is False
        assert manifest["replicates"]["pythia-160m"]["significance_reachable"] is True

    def test_it_records_the_withdrawn_clause_and_the_loosening(self, manifest):
        """A1 requires the write-up to admit the amended rule is weaker, not neutral."""
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        rule = manifest["practical_importance_rule"]
        assert "withdrawn_clause" in rule
        assert "REDUCTION" in rule["withdrawn_clause"]

    def test_it_states_the_exclusions_rather_than_leaving_gaps(self, manifest):
        if manifest is None:
            pytest.skip("manifest not generated on this machine")
        for key in ("latency", "downstream", "1b_significance", "extended_models"):
            assert key in manifest["exclusion_rules"], f"exclusion {key!r} is not stated"


class TestTheRecoveredQwenOrderSelectionEvidence:
    """B-51 destroyed the validation records that F-40's sequential-order freeze rests on.

    Four were lost from disk -- `sequential` (P→Q) and `joint` at both budgets -- because the test
    grid wrote to the same filenames. They are recoverable from the evidence set committed at
    2832914, before the test grid ran, and are extracted to a dedicated artefact so the freeze does
    not depend on a reader knowing which commit to look in.
    """

    def test_the_recovered_artefact_matches_history(self, project_root: Path):
        """Guards against the artefact drifting from the commit it claims to come from."""
        completed = subprocess.run(
            [sys.executable, "scripts/recover_qwen_order_selection.py", "--check"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if "no qwen2.5-0.5b validation rows" in completed.stderr:
            pytest.skip("source commit not present in this clone")
        assert completed.returncode == 0, completed.stderr

    def test_the_provenance_names_what_was_destroyed(self, project_root: Path):
        """A recovery without provenance is just a file someone typed."""
        path = project_root / "results" / "evidence" / "qwen_order_selection_provenance.json"
        if not path.exists():
            pytest.skip("recovered artefact not present")
        provenance = json.loads(path.read_text(encoding="utf-8"))
        assert provenance["source_commit"], "the source commit must be recorded"
        assert provenance["record_sha256_at_source"], "per-record hashes must be recorded"
        destroyed = set(provenance["destroyed_by_b51"])
        assert any("sequential_" in name and "_qp" not in name for name in destroyed), (
            "the P→Q records were destroyed and the provenance must say so"
        )
        assert any("joint" in name for name in destroyed), (
            "the joint records were destroyed and the provenance must say so"
        )

    def test_the_recovered_values_reproduce_the_frozen_order(self, project_root: Path):
        """The whole point: F-40's margins must be recomputable from the recovered file."""
        import csv

        path = project_root / "results" / "evidence" / "qwen_order_selection.csv"
        if not path.exists():
            pytest.skip("recovered artefact not present")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        by_cell = {
            (row["budget_label"], row["compression_method"]): float(row["perplexity_retention"])
            for row in rows
        }
        for budget in ("moderate", "aggressive"):
            pq = by_cell[(budget, "sequential")]
            qp = by_cell[(budget, "sequential_qp")]
            assert pq > qp, (
                f"F-40 froze P→Q at {budget}; the recovered evidence must still show it winning"
            )
