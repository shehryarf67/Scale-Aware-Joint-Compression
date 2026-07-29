"""Steps 2 and 3: perplexity, generation, agreement, the benchmark callable, and the runner.

Every test runs against the randomly initialised two-layer GPT-NeoX from ``conftest`` — same
architecture class as Pythia, no download, no pretrained weights. A random model has a terrible
perplexity (around the vocabulary size), which is fine: these tests check that the machinery is
correct, not that the model is good.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from scale_aware_compression.config import (
    BenchmarkConfig,
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
)
from scale_aware_compression.constants import Device
from scale_aware_compression.evaluation.common import EvaluationError
from scale_aware_compression.evaluation.perplexity import (
    PerplexityResult,
    compute_perplexity,
    perplexity_from_nll,
)

pytestmark = pytest.mark.requires_torch


@pytest.fixture
def evaluation_config() -> EvaluationConfig:
    """A small CPU evaluation config."""
    return EvaluationConfig(
        device=Device.CPU,
        batch_size=2,
        sequence_length=16,
        max_samples=8,
        metrics=["perplexity"],
        agreement_samples=8,
        generation_prompts=2,
        generation_max_new_tokens=4,
    )


@pytest.fixture
def eval_loader(token_block_dataset):
    """A deterministic evaluation loader over the tiny token blocks."""
    from scale_aware_compression.data.loaders import build_dataloader

    return build_dataloader(token_block_dataset, DataConfig(batch_size=2), batch_size=2)


class TestComputePerplexity:
    def test_returns_a_finite_positive_perplexity(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        result = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        assert isinstance(result, PerplexityResult)
        assert math.isfinite(result.perplexity)
        assert result.perplexity > 1.0

    def test_untrained_model_scores_near_the_vocabulary_size(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        """An untrained model should score near the vocabulary size.

        It is roughly uniform over the vocabulary, so perplexity should land near |V|. This is
        the sanity check that the loss is computed over the right axis: a shape or shift bug
        lands nowhere near it.
        """
        result = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        vocabulary_size = tiny_causal_lm.config.vocab_size
        assert vocabulary_size / 4 < result.perplexity < vocabulary_size * 4

    def test_counts_one_prediction_fewer_than_tokens(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig, token_block_dataset
    ):
        """Each window of L tokens yields L-1 predictions; the first token has no predecessor."""
        result = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        expected = len(token_block_dataset) * (token_block_dataset.sequence_length - 1)
        assert result.total_tokens == expected

    def test_is_deterministic(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        """The Step 2 exit criterion: the same number twice in a row."""
        first = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        second = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        assert first.perplexity == pytest.approx(second.perplexity, rel=1e-9)
        assert first.total_nll == pytest.approx(second.total_nll, rel=1e-9)

    def test_matches_manual_aggregation(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        """Perplexity must be exp(total NLL / total tokens), not a mean of per-batch values."""
        result = compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
        assert result.perplexity == pytest.approx(
            perplexity_from_nll(result.total_nll, result.total_tokens), rel=1e-9
        )

    def test_records_the_dataset_fingerprint(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        result = compute_perplexity(
            tiny_causal_lm, eval_loader, evaluation_config, dataset_fingerprint="abc123"
        )
        assert result.dataset_fingerprint == "abc123"
        assert result.to_dict()["dataset_fingerprint"] == "abc123"

    def test_restores_training_mode(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        tiny_causal_lm.train()
        try:
            compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)
            assert tiny_causal_lm.training, "evaluation must not silently leave the model in eval"
        finally:
            tiny_causal_lm.eval()

    def test_stride_is_refused_rather_than_ignored(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        """Silently ignoring a stride would make results incomparable with published numbers."""
        evaluation_config.stride = 8
        with pytest.raises(NotImplementedError, match="stride"):
            compute_perplexity(tiny_causal_lm, eval_loader, evaluation_config)

    def test_model_without_logits_is_reported(
        self, eval_loader, evaluation_config: EvaluationConfig
    ):
        import torch

        class NoHead(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, **_: object) -> dict[str, int]:
                return {"nothing": 0}

        with pytest.raises(EvaluationError, match="logits"):
            compute_perplexity(NoHead(), eval_loader, evaluation_config)


class TestGeneration:
    def test_generates_the_requested_number_of_tokens(
        self, tiny_causal_lm, fake_tokenizer, evaluation_config: EvaluationConfig
    ):
        from scale_aware_compression.evaluation.generation import generate_samples

        report = generate_samples(tiny_causal_lm, fake_tokenizer, evaluation_config)
        assert len(report.samples) == evaluation_config.generation_prompts
        for sample in report.samples:
            # min_new_tokens == max_new_tokens, so an early EOS cannot shorten a sample.
            assert sample.num_new_tokens == evaluation_config.generation_max_new_tokens

    def test_reports_degeneracy_diagnostics(
        self, tiny_causal_lm, fake_tokenizer, evaluation_config: EvaluationConfig
    ):
        from scale_aware_compression.evaluation.generation import generate_samples

        report = generate_samples(tiny_causal_lm, fake_tokenizer, evaluation_config)
        assert 0.0 <= report.mean_repetition_rate <= 1.0
        assert 0.0 <= report.mean_distinct_token_ratio <= 1.0
        assert isinstance(report.looks_degenerate, bool)


class TestAgreement:
    def test_a_model_agrees_perfectly_with_itself(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        """The invariant that makes a non-trivial agreement number believable."""
        from scale_aware_compression.evaluation.agreement import compute_agreement

        result = compute_agreement(tiny_causal_lm, tiny_causal_lm, eval_loader, evaluation_config)
        assert result.top1_agreement == pytest.approx(1.0)
        assert result.top5_agreement == pytest.approx(1.0)
        assert result.mean_kl_divergence == pytest.approx(0.0, abs=1e-6)

    def test_a_perturbed_model_agrees_less(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        import copy

        import torch

        from scale_aware_compression.evaluation.agreement import compute_agreement

        perturbed = copy.deepcopy(tiny_causal_lm)
        with torch.no_grad():
            for parameter in perturbed.parameters():
                parameter.add_(torch.randn_like(parameter) * 0.5)

        result = compute_agreement(tiny_causal_lm, perturbed, eval_loader, evaluation_config)
        assert result.top1_agreement < 1.0
        assert result.mean_kl_divergence > 0.0

    def test_respects_the_position_budget(
        self, tiny_causal_lm, eval_loader, evaluation_config: EvaluationConfig
    ):
        from scale_aware_compression.evaluation.agreement import compute_agreement

        evaluation_config.agreement_samples = 5
        result = compute_agreement(tiny_causal_lm, tiny_causal_lm, eval_loader, evaluation_config)
        assert result.num_positions == 5


class TestBenchmarkCallable:
    def test_forward_workload_runs(self, tiny_causal_lm, fake_tokenizer):
        from scale_aware_compression.benchmarking.cpu import build_forward_callable

        config = BenchmarkConfig(batch_size=1, sequence_length=8, num_threads=1)
        run = build_forward_callable(tiny_causal_lm, fake_tokenizer, config)
        assert run() is not None

    def test_input_is_allocated_once_and_reused(self, tiny_causal_lm, fake_tokenizer):
        """Allocation inside the timed region would be measured as model latency."""
        from scale_aware_compression.benchmarking.cpu import build_forward_callable

        config = BenchmarkConfig(batch_size=1, sequence_length=8, num_threads=1)
        run = build_forward_callable(tiny_causal_lm, fake_tokenizer, config)
        first = run().logits
        second = run().logits
        assert (first == second).all(), "the same fixed input must be used on every repetition"

    def test_decode_workload_runs(self, tiny_causal_lm, fake_tokenizer):
        from scale_aware_compression.benchmarking.cpu import build_forward_callable

        config = BenchmarkConfig(batch_size=1, sequence_length=8, generated_tokens=4, num_threads=1)
        run = build_forward_callable(tiny_causal_lm, fake_tokenizer, config)
        generated = run()
        assert generated.shape[1] == 8 + 4

    def test_end_to_end_through_the_runner(self, tiny_causal_lm, fake_tokenizer):
        from scale_aware_compression.benchmarking.cpu import benchmark_model

        config = BenchmarkConfig(
            batch_size=1,
            sequence_length=8,
            num_threads=1,
            warmup_runs=1,
            measured_runs=3,
            fail_on_thread_mismatch=False,
        )
        result = benchmark_model(tiny_causal_lm, fake_tokenizer, config, label="tiny")
        assert result.latency.num_runs == 3
        assert result.latency.median_ms > 0
        assert result.throughput.tokens_per_second > 0
        assert result.device == "cpu"


class TestDenseRunEndToEnd:
    """Step 3's exit criterion: one dense record containing the score and the timings."""

    @pytest.fixture
    def dense_config(self, tmp_path: Path) -> ExperimentConfig:
        return ExperimentConfig.from_mapping(
            {
                "experiment": {"id": "tiny-dense"},
                "runtime": {"seed": 1234, "output_dir": str(tmp_path), "log_level": "WARNING"},
                "model": {"name": "pythia-160m", "device": "cpu", "dtype": "float32"},
                "data": {"sequence_length": 16, "batch_size": 2, "max_eval_samples": 4},
                "compression": {"method": "dense"},
                "evaluation": {
                    "device": "cpu",
                    "batch_size": 2,
                    "sequence_length": 16,
                    "max_samples": 4,
                    "metrics": ["perplexity"],
                },
                "benchmark": {
                    "device": "cpu",
                    "num_threads": 1,
                    "warmup_runs": 1,
                    "measured_runs": 3,
                    "batch_size": 1,
                    "sequence_length": 8,
                    "fail_on_thread_mismatch": False,
                },
            }
        )

    def test_writes_a_complete_dense_record(
        self,
        dense_config: ExperimentConfig,
        tiny_causal_lm,
        fake_tokenizer,
        token_block_dataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        from scale_aware_compression.data.loaders import build_dataloader
        from scale_aware_compression.experiments.runner import ExperimentRunner
        from scale_aware_compression.models.loader import LoadedModel
        from scale_aware_compression.models.registry import get_model_spec

        # Substitute the tiny model for the real download, and the tiny blocks for WikiText.
        # Everything between those two seams is the production code path.
        loaded = LoadedModel(
            model=tiny_causal_lm,
            tokenizer=fake_tokenizer,
            spec=get_model_spec("pythia-160m"),
            device=Device.CPU,
            dtype=dense_config.model.dtype,
            parameter_count=sum(p.numel() for p in tiny_causal_lm.parameters()),
        )
        monkeypatch.setattr(
            "scale_aware_compression.models.loader.load_model_and_tokenizer",
            lambda config: loaded,
        )
        # Patched where it is defined, not where it is used: evaluate_model imports it locally,
        # so patching the importing module would silently miss.
        monkeypatch.setattr(
            "scale_aware_compression.data.loaders.build_evaluation_dataloader",
            lambda data, tokenizer, **kwargs: (
                build_dataloader(token_block_dataset, data, batch_size=2),
                None,
            ),
        )

        record = ExperimentRunner(dense_config).run()

        assert record.status == "success"
        assert record.parameter_count > 0
        assert record.quality["perplexity"]["perplexity"] > 1.0
        assert record.deployment["latency_median_ms"] > 0
        assert record.deployment["throughput_tokens_per_s"] > 0
        assert record.hardware["platform"]
        assert record.duration_seconds > 0

        # The dense arm is its own retention reference, so the column is populated at 100%.
        assert record.quality["retention"]["perplexity_retention"] == pytest.approx(100.0)

        # And it is on disk, in both formats. Keyed by the §5.6 run identifier rather than by the
        # configured `experiment.id`: taking the configured id alone meant two runs differing only in
        # seed shared a filename, and the second silently overwrote the first.
        written = tmp_path / "metrics" / f"{record.experiment_id}.json"
        assert written.is_file()
        assert (tmp_path / "metrics" / "results.csv").is_file()

        # The configured id survives as a grouping label, and the identifier encodes what makes this
        # measurement distinct: model, arm, budget and seed.
        assert record.experiment_group == "tiny-dense"
        assert record.experiment_id.startswith("tiny-dense__")
        for fragment in ("dense", f"seed{record.seed}"):
            assert fragment in record.experiment_id

    def test_csv_row_has_the_declared_columns(self, dense_config: ExperimentConfig, tmp_path: Path):
        from scale_aware_compression.constants import RESULT_CSV_COLUMNS
        from scale_aware_compression.experiments.runner import ExperimentRecord

        row = ExperimentRecord.from_config(dense_config, capture_environment=False).to_csv_row()
        assert tuple(row) == RESULT_CSV_COLUMNS
