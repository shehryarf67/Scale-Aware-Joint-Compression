"""Prefill and decode as separately timed workloads (plan §4.7, gap A5).

The invariant worth testing here is not that the functions return a callable. It is that decode
really is a *decode* -- one token against a primed cache -- because the failure mode is silent:
a decode that quietly re-runs the prompt produces a plausible number that is simply the prefill
measured twice, and no downstream check would notice.
"""

from __future__ import annotations

import pytest

from scale_aware_compression.benchmarking.phases import (
    PROMPT_LENGTHS,
    PhaseBenchmarkError,
    build_decode_callable,
    build_prefill_callable,
    rotate,
)


class TestModelOrderRotation:
    """§4.7 requires the arm order rotated between repetitions, so thermal drift is spread."""

    def test_rotation_preserves_membership(self):
        arms = ["dense", "pruning", "sequential", "joint"]
        for offset in range(len(arms)):
            assert sorted(rotate(arms, offset)) == sorted(arms)

    def test_every_arm_leads_exactly_once_over_a_full_cycle(self):
        """The point of rotating: no arm is always measured on the coldest machine."""
        arms = ["dense", "pruning", "sequential", "joint"]
        leaders = [rotate(arms, offset)[0] for offset in range(len(arms))]
        assert sorted(leaders) == sorted(arms)

    def test_offsets_beyond_the_length_wrap(self):
        arms = ["a", "b", "c"]
        assert rotate(arms, 4) == rotate(arms, 1)

    def test_the_input_is_not_mutated(self):
        arms = ["a", "b", "c"]
        rotate(arms, 2)
        assert arms == ["a", "b", "c"]

    def test_an_empty_list_is_not_an_error(self):
        assert rotate([], 3) == []


class TestPromptLengths:
    def test_the_two_lengths_the_plan_requires(self):
        """§4.7 names 128 and 512 so the prompt-length dependence is visible."""
        assert PROMPT_LENGTHS == (128, 512)


@pytest.mark.requires_torch
class TestPhaseCallables:
    """Against a real GPTNeoX model, offline, in milliseconds."""

    def test_prefill_runs_one_forward_over_the_whole_prompt(self, tiny_causal_lm, fake_tokenizer):
        run = build_prefill_callable(tiny_causal_lm, fake_tokenizer, batch_size=1, prompt_length=16)
        output = run()
        assert output.logits.shape[1] == 16

    def test_decode_processes_exactly_one_token(self, tiny_causal_lm, fake_tokenizer):
        """The whole point of the split.

        A decode step emits logits for ONE position. If this returned `prompt_length` positions,
        the cache was not being used and the measurement would be a second prefill wearing the
        decode label -- a plausible number, and wrong.
        """
        run = build_decode_callable(tiny_causal_lm, fake_tokenizer, batch_size=1, prompt_length=16)
        output = run()
        assert output.logits.shape[1] == 1

    def test_decode_does_not_grow_across_repetitions(self, tiny_causal_lm, fake_tokenizer):
        """A shared, mutated cache would make each repetition slower than the last.

        That turns a flat per-token measurement into a ramp, and the median of a ramp is not a
        per-token latency. Each call must start from the same primed state.
        """
        run = build_decode_callable(tiny_causal_lm, fake_tokenizer, batch_size=1, prompt_length=16)
        lengths = []
        for _ in range(3):
            output = run()
            cache = output.past_key_values
            lengths.append(cache.get_seq_length() if hasattr(cache, "get_seq_length") else None)
        assert len(set(lengths)) == 1, f"cache length drifted across repetitions: {lengths}"

    def test_prefill_and_decode_see_the_same_prompt(self, tiny_causal_lm, fake_tokenizer):
        """Two arms are only comparable if both phases were fed byte-identical input."""
        import torch

        from scale_aware_compression.benchmarking.phases import _synthetic_prompt

        first = _synthetic_prompt(tiny_causal_lm, fake_tokenizer, batch_size=2, prompt_length=8)
        second = _synthetic_prompt(tiny_causal_lm, fake_tokenizer, batch_size=2, prompt_length=8)
        assert torch.equal(first, second)

    def test_a_model_without_a_cache_is_refused(self, fake_tokenizer):
        """Rather than silently timing a second prefill and calling it decode."""
        from torch import nn

        class NoCache(nn.Module):
            config = type("cfg", (), {"vocab_size": 64})()

            def forward(self, **_kwargs):
                return type("out", (), {"past_key_values": None})()

        with pytest.raises(PhaseBenchmarkError, match="no past_key_values"):
            build_decode_callable(NoCache(), fake_tokenizer, batch_size=1, prompt_length=8)


class TestInterquartileRange:
    """§4.7 requires the IQR reported alongside the median.

    It is preferred to the standard deviation for a bounded, right-tailed distribution: one
    scheduler preemption moves the mean and the std, and leaves the IQR alone.
    """

    def test_iqr_is_the_distance_between_the_quartiles(self):
        from scale_aware_compression.benchmarking.latency import summarise_latencies

        stats = summarise_latencies([0.001, 0.002, 0.003, 0.004, 0.005])
        assert stats.iqr_ms == pytest.approx(stats.p75_ms - stats.p25_ms)

    def test_a_single_outlier_moves_the_std_far_more_than_the_iqr(self):
        """The property that makes the IQR the right headline for latency."""
        from scale_aware_compression.benchmarking.latency import summarise_latencies

        clean = [0.010] * 20 + [0.011] * 20
        spiked = clean[:-1] + [0.500]

        before, after = summarise_latencies(clean), summarise_latencies(spiked)
        std_growth = after.std_ms / max(before.std_ms, 1e-9)
        iqr_growth = after.iqr_ms / max(before.iqr_ms, 1e-9)
        assert std_growth > 10 * iqr_growth

    def test_the_iqr_is_serialised(self):
        from scale_aware_compression.benchmarking.latency import summarise_latencies

        payload = summarise_latencies([0.001, 0.002, 0.003]).to_dict()
        for key in ("latency_p25_ms", "latency_p75_ms", "latency_iqr_ms"):
            assert key in payload
