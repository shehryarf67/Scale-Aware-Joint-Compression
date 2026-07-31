"""Per-window NLL, the unit of resampling for the paired block bootstrap (A1 §5.1).

A1 requires a paired block bootstrap that resamples **complete evaluation windows** — neighbouring
tokens are dependent, so a token-level bootstrap understates the interval. Before this existed only
aggregates were stored, which made that analysis impossible: it would have been discovered after the
~38 hour confirmatory stage, and every run would have needed repeating.

Two properties carry the design:

* **the blocks must reconstruct the reported number**, or a bootstrap over them describes a different
  quantity from the one in the paper;
* **the block structure must be invariant to batch size**, because batch size differs between scales
  (4 at 160M, 2 at 410M for memory) and a per-batch decomposition would not be comparable across them.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.requires_torch


@pytest.fixture
def loader_factory(token_block_dataset):
    """Build evaluation loaders over one fixed corpus at a chosen batch size."""

    def build(batch_size: int):
        from scale_aware_compression.config import DataConfig
        from scale_aware_compression.data.loaders import build_dataloader

        # build_dataloader rather than a bare DataLoader: the evaluator needs the attention mask the
        # collate function adds, and this is the same path the real runs use.
        return build_dataloader(
            token_block_dataset, DataConfig(batch_size=batch_size), batch_size=batch_size
        )

    return build


@pytest.fixture
def evaluation_config():
    from scale_aware_compression.config import EvaluationConfig

    return EvaluationConfig()


class TestBlocksReconstructTheAggregate:
    def test_window_nll_sums_to_total_nll(self, tiny_causal_lm, loader_factory, evaluation_config):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        result = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        assert result.window_nll
        # rel=1e-5, not tighter: total_nll accumulates float32 tensor reductions per batch while
        # this sum is float64 over the extracted values. See the guard in compute_perplexity.
        assert sum(result.window_nll) == pytest.approx(result.total_nll, rel=1e-5)

    def test_window_tokens_sums_to_total_tokens(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        result = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        assert sum(result.window_tokens) == result.total_tokens

    def test_one_block_per_sequence(self, tiny_causal_lm, loader_factory, evaluation_config):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        result = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        assert len(result.window_nll) == result.num_sequences
        assert len(result.window_tokens) == result.num_sequences

    def test_every_block_is_finite_and_positive(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        import math

        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        result = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        assert all(math.isfinite(value) and value > 0.0 for value in result.window_nll)

    def test_recomputing_perplexity_from_blocks_matches(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        """The whole point: an analysis working only from the blocks gets the reported number."""
        from scale_aware_compression.evaluation.perplexity import (
            compute_perplexity,
            perplexity_from_nll,
        )

        result = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        rebuilt = perplexity_from_nll(sum(result.window_nll), sum(result.window_tokens))
        assert rebuilt == pytest.approx(result.perplexity, rel=1e-5)


class TestBlockStructureIsBatchInvariant:
    """160M evaluates at batch 4 and 410M at batch 2. The blocks must still line up.

    A per-batch decomposition would give 124 blocks at one scale and 247 at another, and the two
    could not be bootstrapped against a common block index. Per-sequence blocks are invariant.
    """

    def test_same_number_of_blocks_at_any_batch_size(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        counts = {
            batch_size: len(
                compute_perplexity(
                    tiny_causal_lm, loader_factory(batch_size), evaluation_config
                ).window_nll
            )
            for batch_size in (1, 2, 4)
        }
        assert len(set(counts.values())) == 1, f"block count varies with batch size: {counts}"

    def test_block_values_agree_across_batch_sizes(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        first = compute_perplexity(tiny_causal_lm, loader_factory(1), evaluation_config).window_nll
        second = compute_perplexity(tiny_causal_lm, loader_factory(4), evaluation_config).window_nll
        for index, (left, right) in enumerate(zip(first, second, strict=True)):
            assert left == pytest.approx(right, rel=1e-4, abs=1e-4), (
                f"window {index} differs between batch 1 and batch 4: {left} vs {right}"
            )


class TestBlocksReachTheRecord:
    """A block decomposition that is computed and then dropped is no use at analysis time."""

    def test_to_dict_carries_both_lists(self, tiny_causal_lm, loader_factory, evaluation_config):
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        payload = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config).to_dict()
        assert payload["window_nll"]
        assert payload["window_tokens"]
        assert len(payload["window_nll"]) == payload["num_sequences"]

    def test_the_payload_is_json_serialisable(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        import json

        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        payload = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config).to_dict()
        restored = json.loads(json.dumps(payload))
        assert restored["window_nll"] == payload["window_nll"]

    def test_a_paired_difference_can_be_formed_from_two_results(
        self, tiny_causal_lm, loader_factory, evaluation_config
    ):
        """The actual downstream use: per-window differences between two arms, same window indices."""
        from scale_aware_compression.evaluation.perplexity import compute_perplexity

        baseline = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)
        candidate = compute_perplexity(tiny_causal_lm, loader_factory(2), evaluation_config)

        differences = [
            left / left_tokens - right / right_tokens
            for left, left_tokens, right, right_tokens in zip(
                baseline.window_nll,
                baseline.window_tokens,
                candidate.window_nll,
                candidate.window_tokens,
                strict=True,
            )
        ]
        assert len(differences) == baseline.num_sequences
        # Identical models, so every paired difference is zero. What is being tested is that the
        # pairing is well-formed and the lists align, not the values.
        assert all(abs(value) < 1e-9 for value in differences)
