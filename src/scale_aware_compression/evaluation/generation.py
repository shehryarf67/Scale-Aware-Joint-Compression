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

    Args:
        model: The model to sample from.
        tokenizer: Matching tokeniser.
        config: Evaluation section of an experiment config.
        prompts: Prompts to use. Defaults to :data:`DEFAULT_PROMPTS`.

    Returns:
        The generation report.

    Raises:
        NotImplementedError: Always, in the current scaffold.
    """
    # TODO(evaluation): call model.generate(do_sample=False, num_beams=1,
    # max_new_tokens=config.generation_max_new_tokens) under torch.inference_mode(). Greedy and
    # unsampled, so two arms' outputs differ only because the models differ.
    # Compute repetition_rate and distinct_token_ratio on the *generated* ids only, excluding
    # the prompt, and log a warning when looks_degenerate is true.
    raise NotImplementedError(
        "generate_samples is not implemented yet; see the TODO in evaluation/generation.py"
    )
