"""Repository-level checks: documentation, packaging metadata, CI, and the CPU-only policy.

These guard facts that live outside Python modules and would otherwise go stale silently — a
documented protocol that was deleted, a placeholder URL that shipped, a CI workflow that stopped
running the checks it claims to.

Nothing here downloads a model, imports torch, or needs CUDA.
"""

from __future__ import annotations

import re
import subprocess
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

    def test_every_experiment_config_evaluates_on_cpu(self, configs_dir: Path):
        for path in sorted((configs_dir / "experiments").glob("*.yaml")):
            config = ExperimentConfig.from_mapping(load_document(path))
            assert config.evaluation.device is Device.CPU, f"{path.name} evaluates off CPU"

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

    def _record(self, config, **overrides):
        from scale_aware_compression.experiments.runner import ExperimentRecord

        record = ExperimentRecord.from_config(config, capture_environment=False)
        record.status = "success"
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
        record = self._record(config)
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True

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
        record = self._record(config)
        tracker.save(record)

        moved = copy.deepcopy(config)
        moved.model.revision = "def456"
        assert tracker.exists_valid(record.experiment_id, moved) is False

    def test_a_record_at_a_different_budget_is_re_run(self, tmp_path, config):
        import copy

        tracker = self._tracker(tmp_path)
        record = self._record(config)
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
        record = self._record(config)
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
        record = self._record(config)
        record.hardware = get_hardware_info()
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True

    def test_a_record_with_no_hardware_recorded_is_not_invalidated(self, tmp_path, config):
        """Records predating the field report "unknown"; a new guard must not force a recompute."""
        tracker = self._tracker(tmp_path)
        record = self._record(config)
        record.hardware = {}
        tracker.save(record)
        assert tracker.exists_valid(record.experiment_id, config) is True


class TestNoRunArtefactsCommitted:
    """Git tracks nothing but `.gitkeep` under `outputs/` and `results/`.

    Checked against the index rather than the working tree. On a machine that actually runs the
    experiments these directories are *supposed* to be full of run artefacts, so asserting the
    directory is empty would fail for exactly the machine the rule exists to protect. The invariant
    is that nothing gets committed, not that nothing exists.
    """

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
            if line.strip() and not line.endswith(".gitkeep")
        ]
        assert not unexpected, (
            f"git tracks run artefacts under {directory}/, which must never be committed: "
            f"{unexpected}"
        )

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
