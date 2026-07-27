"""Step 1: tokenisation, chunking, caching, and calibration sampling.

Everything here is offline. The corpus is a handful of strings and the tokeniser is the
byte-level stand-in from ``conftest``, so no test downloads WikiText or a real tokeniser.

The properties being defended are the ones that would silently invalidate the whole study:
preparation must be reproducible, the evaluation stream must be fingerprinted, and the
calibration set must be identical across arms and disjoint from evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scale_aware_compression.config import DataConfig
from scale_aware_compression.data.calibration import (
    CalibrationError,
    load_calibration_set,
    select_calibration_indices,
)
from scale_aware_compression.data.errors import DataError
from scale_aware_compression.data.loaders import (
    DatasetSummary,
    TokenBlockDataset,
    build_language_modelling_dataset,
    collate_token_blocks,
)
from scale_aware_compression.data.preprocessing import (
    chunk_sequence,
    fingerprint_token_ids,
    load_prepared_tokens,
    prepare_dataset,
    processed_cache_dir,
    tokenise_corpus,
)


@pytest.fixture
def data_config(tmp_path: Path) -> DataConfig:
    """A small data config pointing at a temporary cache."""
    return DataConfig(
        dataset="tiny-corpus",
        subset=None,
        train_split="train",
        eval_split="validation",
        text_column="text",
        sequence_length=16,
        batch_size=2,
        max_eval_samples=4,
        calibration_samples=3,
        calibration_split="train",
        calibration_seed=1234,
    )


@pytest.fixture
def stub_raw_dataset(monkeypatch: pytest.MonkeyPatch, sample_corpus: list[str]):
    """Replace the Hugging Face loader with an in-memory corpus.

    Keeps the test offline while still exercising the real prepare/chunk/cache path.
    """

    def _stub(config: DataConfig, split: str) -> dict[str, list[str]]:
        # A longer corpus for train than validation, so the two splits differ observably.
        repeats = 6 if split == "train" else 3
        return {config.text_column: sample_corpus * repeats}

    monkeypatch.setattr("scale_aware_compression.data.loaders.load_raw_dataset", _stub)
    monkeypatch.setattr(
        "scale_aware_compression.data.preprocessing.load_raw_dataset", _stub, raising=False
    )
    return _stub


class TestTokeniseCorpus:
    def test_produces_a_flat_id_stream(self, fake_tokenizer, sample_corpus: list[str]):
        token_ids = tokenise_corpus(sample_corpus, fake_tokenizer)
        assert token_ids
        assert all(isinstance(value, int) for value in token_ids)

    def test_drops_blank_documents(self, fake_tokenizer):
        with_blanks = tokenise_corpus(["hello", "", "   ", "world"], fake_tokenizer)
        without = tokenise_corpus(["hello", "world"], fake_tokenizer)
        assert with_blanks == without

    def test_is_deterministic(self, fake_tokenizer, sample_corpus: list[str]):
        assert tokenise_corpus(sample_corpus, fake_tokenizer) == tokenise_corpus(
            sample_corpus, fake_tokenizer
        )

    def test_empty_corpus_raises(self, fake_tokenizer):
        with pytest.raises(DataError, match="empty"):
            tokenise_corpus(["", "  "], fake_tokenizer)


class TestChunking:
    def test_splits_into_equal_windows(self):
        blocks = chunk_sequence(list(range(100)), 16)
        assert len(blocks) == 6
        assert all(len(block) == 16 for block in blocks)

    def test_drops_the_ragged_tail(self):
        """A short final window would change the perplexity denominator between runs."""
        blocks = chunk_sequence(list(range(100)), 16)
        assert 100 - sum(len(block) for block in blocks) == 4

    def test_keeps_the_tail_when_asked(self):
        blocks = chunk_sequence(list(range(100)), 16, drop_last=False)
        assert len(blocks[-1]) == 4


class TestFingerprint:
    def test_is_stable(self):
        assert fingerprint_token_ids([1, 2, 3]) == fingerprint_token_ids([1, 2, 3])

    def test_differs_for_different_streams(self):
        assert fingerprint_token_ids([1, 2, 3]) != fingerprint_token_ids([1, 2, 4])

    def test_differs_for_different_lengths(self):
        """Length is hashed too, so a truncated stream is never mistaken for the full one."""
        assert fingerprint_token_ids(list(range(100))) != fingerprint_token_ids(list(range(50)))


class TestPrepareDataset:
    def test_writes_tokens_and_metadata(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        metadata = prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path)
        directory = processed_cache_dir(data_config, fake_tokenizer, "train", root=tmp_path)
        assert (directory / "tokens.json").is_file()
        assert (directory / "metadata.json").is_file()
        assert metadata["num_blocks"] > 0
        assert metadata["fingerprint"]

    def test_running_twice_is_byte_identical(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        """The Step 1 exit criterion."""
        first = prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path, force=True)
        tokens_path = Path(first["tokens_path"])
        first_bytes = tokens_path.read_bytes()

        second = prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path, force=True)
        assert tokens_path.read_bytes() == first_bytes
        assert second["fingerprint"] == first["fingerprint"]

    def test_second_call_reuses_the_cache(
        self,
        data_config: DataConfig,
        fake_tokenizer,
        stub_raw_dataset,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path)

        def _explode(*_: object, **__: object) -> None:
            raise AssertionError("cache miss: the corpus was re-tokenised")

        monkeypatch.setattr("scale_aware_compression.data.preprocessing.tokenise_corpus", _explode)
        again = prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path)
        assert again["num_blocks"] > 0

    def test_cache_key_separates_sequence_lengths(
        self, data_config: DataConfig, fake_tokenizer, tmp_path: Path
    ):
        short = processed_cache_dir(data_config, fake_tokenizer, "train", root=tmp_path)
        data_config.sequence_length = 32
        long = processed_cache_dir(data_config, fake_tokenizer, "train", root=tmp_path)
        assert short != long

    def test_cache_key_separates_tokenisers(
        self, data_config: DataConfig, fake_tokenizer, tmp_path: Path
    ):
        """Pythia and Qwen produce different streams from identical text."""

        class OtherTokenizer(type(fake_tokenizer)):  # type: ignore[misc]
            name_or_path = "other-tokenizer"

        first = processed_cache_dir(data_config, fake_tokenizer, "train", root=tmp_path)
        second = processed_cache_dir(data_config, OtherTokenizer(), "train", root=tmp_path)
        assert first != second

    def test_missing_column_is_reported(
        self,
        data_config: DataConfig,
        fake_tokenizer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        monkeypatch.setattr(
            "scale_aware_compression.data.loaders.load_raw_dataset",
            lambda config, split: {"wrong_column": ["text"]},
        )
        with pytest.raises(DataError, match="not found"):
            prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path)

    def test_load_prepared_tokens_without_a_cache_raises(
        self, data_config: DataConfig, fake_tokenizer, tmp_path: Path
    ):
        with pytest.raises(DataError, match="No prepared tokens"):
            load_prepared_tokens(data_config, fake_tokenizer, "train", root=tmp_path)


class TestTokenBlockDataset:
    def test_length_and_indexing(self, token_block_dataset):
        pytest.importorskip("torch")
        assert len(token_block_dataset) == 8
        item = token_block_dataset[0]
        assert item["input_ids"].shape == (16,)

    def test_rejects_ragged_blocks(self):
        with pytest.raises(DataError, match="same length"):
            TokenBlockDataset([[1, 2, 3], [1, 2]])

    def test_rejects_an_empty_dataset(self):
        with pytest.raises(DataError, match="at least one block"):
            TokenBlockDataset([])

    def test_subset_preserves_order_and_content(self, token_block_dataset):
        subset = token_block_dataset.subset([3, 1])
        assert len(subset) == 2
        assert subset.blocks[0] == token_block_dataset.blocks[3]
        assert subset.blocks[1] == token_block_dataset.blocks[1]

    def test_subset_rejects_out_of_range(self, token_block_dataset):
        with pytest.raises(DataError, match="out of range"):
            token_block_dataset.subset([99])

    @pytest.mark.requires_torch
    def test_collate_builds_a_batch(self, token_block_dataset):
        pytest.importorskip("torch")
        batch = collate_token_blocks([token_block_dataset[0], token_block_dataset[1]])
        assert batch["input_ids"].shape == (2, 16)
        assert batch["attention_mask"].shape == (2, 16)
        assert (batch["labels"] == batch["input_ids"]).all()


class TestBuildLanguageModellingDataset:
    @pytest.mark.requires_torch
    def test_builds_and_summarises(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        dataset, summary = build_language_modelling_dataset(
            data_config, fake_tokenizer, "train", cache_root=tmp_path
        )
        assert isinstance(summary, DatasetSummary)
        assert len(dataset) == summary.num_sequences
        assert summary.sequence_length == data_config.sequence_length
        assert summary.fingerprint

    @pytest.mark.requires_torch
    def test_truncation_changes_the_fingerprint(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        """A run truncated to 2 windows did not see the same data as one truncated to 4."""
        _, short = build_language_modelling_dataset(
            data_config, fake_tokenizer, "train", max_sequences=2, cache_root=tmp_path
        )
        _, long = build_language_modelling_dataset(
            data_config, fake_tokenizer, "train", max_sequences=4, cache_root=tmp_path
        )
        assert short.num_sequences == 2
        assert long.num_sequences == 4
        assert short.fingerprint != long.fingerprint

    @pytest.mark.requires_torch
    def test_detects_a_corrupted_cache(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        prepare_dataset(data_config, fake_tokenizer, "train", root=tmp_path)
        directory = processed_cache_dir(data_config, fake_tokenizer, "train", root=tmp_path)
        (directory / "tokens.json").write_text(json.dumps([1, 2, 3] * 64), encoding="utf-8")
        with pytest.raises(DataError, match="do not match their metadata"):
            build_language_modelling_dataset(
                data_config, fake_tokenizer, "train", cache_root=tmp_path
            )


class TestCalibrationIndices:
    def test_is_deterministic_in_the_seed(self):
        first = select_calibration_indices(100, 10, seed=1234)
        second = select_calibration_indices(100, 10, seed=1234)
        assert first == second

    def test_different_seeds_differ(self):
        assert select_calibration_indices(100, 10, seed=1) != select_calibration_indices(
            100, 10, seed=2
        )

    def test_returns_sorted_unique_indices(self):
        indices = select_calibration_indices(100, 10, seed=7)
        assert indices == sorted(set(indices))

    def test_respects_the_exclusion_floor(self):
        indices = select_calibration_indices(100, 10, seed=7, exclude_below=50)
        assert min(indices) >= 50

    def test_too_few_sequences_raises(self):
        with pytest.raises(CalibrationError, match="Cannot draw"):
            select_calibration_indices(5, 10, seed=1)

    def test_exclusion_is_accounted_for_in_the_population(self):
        with pytest.raises(CalibrationError, match="reserved for evaluation"):
            select_calibration_indices(100, 60, seed=1, exclude_below=50)


class TestCalibrationSet:
    @pytest.mark.requires_torch
    def test_builds_calibration_and_heldout(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        calibration = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        assert len(calibration) == data_config.calibration_samples
        assert calibration.heldout_indices, "a held-out subset is needed for the overfit check"
        assert not set(calibration.indices) & set(calibration.heldout_indices)

    @pytest.mark.requires_torch
    def test_is_identical_across_arms(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        """The core fairness property: every arm calibrates on the same sequences."""
        first = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        second = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        assert first.indices == second.indices
        assert first.summary.indices_fingerprint == second.summary.indices_fingerprint
        assert first.summary.token_fingerprint == second.summary.token_fingerprint

    @pytest.mark.requires_torch
    def test_does_not_depend_on_the_run_seed(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        """The calibration draw must not move when the run seed moves.

        Varying the run seed is how error bars are produced; if it also changed the calibration
        set, the seed spread would conflate two sources of variation.
        """
        from scale_aware_compression.seed import set_global_seed

        set_global_seed(1)
        first = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        set_global_seed(9999)
        second = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        assert first.indices == second.indices

    @pytest.mark.requires_torch
    def test_shared_split_stays_disjoint_from_evaluation(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        """Calibrating on evaluation text would leak the test set into the quantisation scales."""
        data_config.calibration_split = data_config.eval_split
        calibration = load_calibration_set(data_config, fake_tokenizer, cache_root=tmp_path)
        evaluation_prefix = set(range(data_config.max_eval_samples or 0))
        assert not set(calibration.indices) & evaluation_prefix

    @pytest.mark.requires_torch
    def test_summary_serialises(
        self, data_config: DataConfig, fake_tokenizer, stub_raw_dataset, tmp_path: Path
    ):
        payload = load_calibration_set(
            data_config, fake_tokenizer, cache_root=tmp_path
        ).summary.to_dict()
        assert payload["calibration_num_samples"] == data_config.calibration_samples
        assert payload["calibration_seed"] == data_config.calibration_seed
        assert payload["calibration_indices_fingerprint"]
