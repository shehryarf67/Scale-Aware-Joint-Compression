"""Prefill and decode as separately timed workloads (plan §4.7, gap A5).

A single end-to-end generation latency hides the two things a deployment actually cares about,
because they scale differently and stress different kernels:

* **prefill** -- one forward pass over the whole prompt. Compute-bound, batched matrix multiplies
  over ``prompt_length`` positions at once. Cost grows with prompt length.
* **decode** -- one forward pass per generated token, with the key/value cache already populated.
  Memory-bandwidth-bound, a matrix-*vector* product per layer. Cost per token is roughly flat in
  prompt length, and it is what dominates a long generation.

Compression changes these differently. Weight-only quantisation helps decode most, because decode
is bandwidth-bound and the weights are what is being moved; it helps prefill less, because prefill
is already compute-bound. Reporting one blended number would average that distinction away, which
is why §4.7 asks for them apart.

**Decode is measured with the cache already primed.** The prompt forward runs once, untimed, and
the timed region is a single-token step against that cache. Timing ``generate`` instead would fold
the prefill into every repetition and report the sum under the decode label -- the exact conflation
this module exists to remove.

CPU-only, like every deployment measurement (§4.6). Nothing here accepts a device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.benchmarking.cpu import (
    BENCHMARK_INPUT_SEED,
    BenchmarkCallable,
    BenchmarkError,
)
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

PROMPT_LENGTHS = (128, 512)
"""The two prompt lengths §4.7 requires, so the prompt-length dependence is visible.

Two points cannot fit a curve. They are enough to show that prefill grows with prompt length and
per-token decode does not, which is the qualitative claim the split supports.
"""


class PhaseBenchmarkError(BenchmarkError):
    """Raised when a phase-separated benchmark cannot be constructed."""


def _synthetic_prompt(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    prompt_length: int,
) -> Any:
    """Allocate the prompt once, outside any timed region.

    Token *values* are synthetic and random: latency depends on tensor shapes and on which kernels
    run, not on what the tokens mean. Seeded so every arm sees byte-identical input, which is what
    makes two arms' timings comparable at all.

    Args:
        model: The model, for its vocabulary size.
        tokenizer: Fallback source of the vocabulary size.
        batch_size: Sequences per batch.
        prompt_length: Tokens per sequence.

    Returns:
        A ``(batch_size, prompt_length)`` int64 tensor of token ids.

    Raises:
        PhaseBenchmarkError: If no usable vocabulary size can be determined.
    """
    import torch

    vocabulary_size = int(
        getattr(getattr(model, "config", None), "vocab_size", None)
        or getattr(tokenizer, "vocab_size", 0)
        or 0
    )
    if vocabulary_size < 2:
        raise PhaseBenchmarkError(
            "Could not determine a usable vocabulary size for the synthetic benchmark prompt"
        )
    generator = torch.Generator().manual_seed(BENCHMARK_INPUT_SEED)
    return torch.randint(
        low=0,
        high=vocabulary_size,
        size=(batch_size, prompt_length),
        generator=generator,
        dtype=torch.long,
    )


def _prepare(model: nn.Module) -> None:
    """Put the model in eval mode on CPU, or say why it cannot be."""
    try:
        model.eval()
        model.to("cpu")
    except Exception as error:  # pragma: no cover - defensive
        raise PhaseBenchmarkError(f"Could not prepare the model on CPU: {error}") from error


def build_prefill_callable(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 1,
    prompt_length: int = 128,
) -> BenchmarkCallable:
    """Build a callable timing one forward pass over a full prompt.

    This is *time to first token* minus sampling: the latency a user waits before generation
    starts.

    Args:
        model: Model to benchmark. Moved to CPU and eval mode.
        tokenizer: Tokeniser, used only for its vocabulary size.
        batch_size: Sequences per batch.
        prompt_length: Prompt tokens.

    Returns:
        A zero-argument callable running exactly one prefill.
    """
    import torch

    _prepare(model)
    input_ids = _synthetic_prompt(
        model, tokenizer, batch_size=batch_size, prompt_length=prompt_length
    )
    attention_mask = torch.ones_like(input_ids)

    def run_prefill() -> Any:
        with torch.inference_mode():
            # `use_cache=False`: building a cache nothing will read is work a real prefill also
            # does, but it is not what this measurement is about, and it makes prefill look worse
            # for long prompts in a way that has nothing to do with compression.
            return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    LOGGER.debug("Prefill workload: one forward over %dx%d", batch_size, prompt_length)
    return run_prefill


def build_decode_callable(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int = 1,
    prompt_length: int = 128,
) -> BenchmarkCallable:
    """Build a callable timing one decode step against an already-populated cache.

    The prompt forward runs **once, here, outside the timed region**, and its cache is reused by
    every repetition. So the timed work is a single-token step -- one matrix-vector product per
    layer -- which is what per-token decode latency means.

    The cache is rebuilt per call from the primed copy rather than shared and mutated, because a
    growing cache would make each repetition slower than the last and turn a flat measurement into
    a ramp.

    Args:
        model: Model to benchmark. Moved to CPU and eval mode.
        tokenizer: Tokeniser, used only for its vocabulary size.
        batch_size: Sequences per batch.
        prompt_length: Prompt tokens the cache is primed with.

    Returns:
        A zero-argument callable running exactly one decode step.

    Raises:
        PhaseBenchmarkError: If the model does not return a usable key/value cache, which would
            otherwise silently degrade this into a second prefill measurement.
    """
    import copy

    import torch

    _prepare(model)
    input_ids = _synthetic_prompt(
        model, tokenizer, batch_size=batch_size, prompt_length=prompt_length
    )
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        primed = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    cache = getattr(primed, "past_key_values", None)
    if cache is None:
        raise PhaseBenchmarkError(
            f"{type(model).__name__} returned no past_key_values, so a decode step cannot be "
            "timed against a primed cache. Without it this would time a second prefill and "
            "report it as decode."
        )

    next_token = input_ids[:, -1:].clone()
    # One position longer than the prompt: the cached prompt plus the token being decoded.
    decode_mask = torch.ones((batch_size, prompt_length + 1), dtype=attention_mask.dtype)

    def run_decode() -> Any:
        # Copied per call so the cache never grows across repetitions. The copy is outside the
        # model call but inside the timed region; it is small relative to a decode step and it is
        # identical for every arm, so it cannot advantage one.
        step_cache = copy.deepcopy(cache)
        with torch.inference_mode():
            return model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=step_cache,
                use_cache=True,
            )

    LOGGER.debug(
        "Decode workload: one token against a cache primed with %dx%d", batch_size, prompt_length
    )
    return run_decode


def rotate(items: list[Any], offset: int) -> list[Any]:
    """Rotate a list left by ``offset``, for model-order rotation (§4.7).

    Benchmarks drift with temperature: whichever arm runs first on a cold machine is measured
    under different conditions from whichever runs last. Rotating the order between repetitions
    spreads that drift across arms instead of loading it all onto one, so it becomes noise in
    every arm rather than a bias in one.

    Args:
        items: Whatever is being ordered -- arms, models, configurations.
        offset: How far to rotate. Taken modulo the length, so any integer is valid.

    Returns:
        A new rotated list. The input is not modified.
    """
    if not items:
        return []
    shift = offset % len(items)
    return items[shift:] + items[:shift]
