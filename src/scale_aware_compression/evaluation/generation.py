"""Greedy generation for qualitative inspection.

Not a headline metric. Its purpose is to catch failure modes that perplexity averages away:
degenerate repetition, collapse to a single token, or fluent output that has lost the prompt.
These show up at aggressive compression budgets and are worth seeing before a number is
believed.

Generation is greedy and the prompt set is fixed, so the output is comparable across arms
rather than being a sample from a distribution.

Status: placeholder for the generation path; the repetition diagnostics are implemented.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import EvaluationConfig
from scale_aware_compression.evaluation.common import EvaluationError
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

DEFAULT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "In 1969, humans first",
    "def add(a, b):\n    return",
    "The main difference between a virus and a bacterium is",
    "Once the model has been pruned, the next step is",
)
"""Fixed prompts, deliberately short and factual. Reused unchanged across every arm and every
model size so generations can be diffed."""


@dataclass(frozen=True, slots=True)
class GenerationSample:
    """One prompt and what a model produced from it."""

    prompt: str
    completion: str
    num_new_tokens: int
    repetition_rate: float
    distinct_token_ratio: float

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "num_new_tokens": self.num_new_tokens,
            "repetition_rate": self.repetition_rate,
            "distinct_token_ratio": self.distinct_token_ratio,
        }


@dataclass(frozen=True, slots=True)
class GenerationReport:
    """Generations from one model, with aggregate degeneracy diagnostics."""

    samples: list[GenerationSample] = field(default_factory=list)
    mean_repetition_rate: float = 0.0
    mean_distinct_token_ratio: float = 0.0
    device: str = "cpu"

    @property
    def looks_degenerate(self) -> bool:
        """Whether the outputs show signs of collapse.

        A distinct-token ratio below 0.2 with a high repetition rate means the model is looping,
        which invalidates any quality claim about the arm that produced it.
        """
        return self.mean_distinct_token_ratio < 0.2 and self.mean_repetition_rate > 0.5

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "samples": [sample.to_dict() for sample in self.samples],
            "mean_repetition_rate": self.mean_repetition_rate,
            "mean_distinct_token_ratio": self.mean_distinct_token_ratio,
            "looks_degenerate": self.looks_degenerate,
            "generation_device": self.device,
        }


def repetition_rate(token_ids: Sequence[int], *, window: int = 4) -> float:
    """Fraction of positions that repeat an n-gram seen earlier in the sequence.

    Args:
        token_ids: Generated token ids.
        window: N-gram length; must be positive.

    Returns:
        Repetition rate in ``[0, 1]``. Zero for sequences shorter than ``window``.

    Raises:
        ValueError: If ``window`` is not positive.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if len(token_ids) < window:
        return 0.0
    seen: set[tuple[int, ...]] = set()
    repeats = 0
    total = 0
    for start in range(len(token_ids) - window + 1):
        gram = tuple(token_ids[start : start + window])
        total += 1
        if gram in seen:
            repeats += 1
        seen.add(gram)
    return repeats / total if total else 0.0


def distinct_token_ratio(token_ids: Sequence[int]) -> float:
    """Unique tokens as a fraction of total tokens.

    Args:
        token_ids: Generated token ids.

    Returns:
        Ratio in ``(0, 1]``. Zero for an empty sequence.
    """
    if not token_ids:
        return 0.0
    return len(set(token_ids)) / len(token_ids)


def generate_samples(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    config: EvaluationConfig,
    *,
    prompts: Sequence[str] | None = None,
) -> GenerationReport:
    """Generate greedy completions for a fixed prompt set.

    Greedy and unsampled, so two arms' outputs differ only because the models differ.

    Args:
        model: The model to sample from.
        tokenizer: Matching tokeniser.
        config: Evaluation section of an experiment config.
        prompts: Prompts to use. Defaults to :data:`DEFAULT_PROMPTS`, truncated to
            ``config.generation_prompts``.

    Returns:
        The generation report.

    Raises:
        EvaluationError: If generation fails or the model has no ``generate`` method.
    """
    import torch

    selected = list(prompts) if prompts is not None else list(DEFAULT_PROMPTS)
    selected = selected[: max(1, config.generation_prompts)]
    if not hasattr(model, "generate"):
        raise EvaluationError(
            f"{type(model).__name__} has no generate(); cannot produce generation diagnostics."
        )

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    samples: list[GenerationSample] = []
    try:
        with torch.inference_mode():
            for prompt in selected:
                encoded = tokenizer(prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                prompt_length = int(input_ids.shape[1])
                try:
                    generated = model.generate(
                        input_ids=input_ids,
                        attention_mask=encoded.get("attention_mask", torch.ones_like(input_ids)).to(
                            device
                        ),
                        do_sample=False,
                        num_beams=1,
                        # min == max: every arm decodes exactly the same number of tokens, so an
                        # early EOS cannot shorten one arm's sample and flatter its repetition
                        # statistics.
                        min_new_tokens=config.generation_max_new_tokens,
                        max_new_tokens=config.generation_max_new_tokens,
                        pad_token_id=pad_token_id,
                    )
                except Exception as error:
                    raise EvaluationError(
                        f"Generation failed for prompt {prompt!r}: {error}"
                    ) from error

                new_ids = generated[0, prompt_length:].tolist()
                samples.append(
                    GenerationSample(
                        prompt=prompt,
                        completion=tokenizer.decode(new_ids, skip_special_tokens=True),
                        num_new_tokens=len(new_ids),
                        # Diagnostics on the generated ids only: including the prompt would
                        # dilute the repetition signal with text the model did not produce.
                        repetition_rate=repetition_rate(new_ids),
                        distinct_token_ratio=distinct_token_ratio(new_ids),
                    )
                )
    finally:
        if was_training:
            model.train()

    report = GenerationReport(
        samples=samples,
        mean_repetition_rate=_mean(sample.repetition_rate for sample in samples),
        mean_distinct_token_ratio=_mean(sample.distinct_token_ratio for sample in samples),
        device=str(device),
    )
    if report.looks_degenerate:
        LOGGER.warning(
            "Generations look degenerate (distinct-token ratio %.2f, repetition %.2f). Any "
            "quality number from this model should be treated as suspect.",
            report.mean_distinct_token_ratio,
            report.mean_repetition_rate,
        )
    else:
        LOGGER.info(
            "Generated %d sample(s): repetition %.2f, distinct-token ratio %.2f",
            len(samples),
            report.mean_repetition_rate,
            report.mean_distinct_token_ratio,
        )
    return report


def _mean(values: Any) -> float:
    """Arithmetic mean of an iterable, returning 0.0 when empty."""
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0
