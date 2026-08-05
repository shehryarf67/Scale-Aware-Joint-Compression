"""Aggregate a joint gain across paired calibration replicates, with a paired block bootstrap.

Amendment A1 §5.1's last unbuilt requirement. Everything upstream produces **one** number per
(model, budget, replicate); this turns R of those into the quantity the paper reports.

**Why this module exists at all.** Three joint-gain figures were reported and retracted before anyone
measured the spread. When it was finally measured, the 410M cell swung from −0.50 to +0.98 pp across
three calibration draws (`findings_log.md` F-26) — the effect and the noise were the same size. A single
draw is not an estimate of anything here, so the aggregation is not a reporting convenience, it is the
measurement.

**What may and may not be claimed from R draws.** A1 §5.1 is explicit, and the arithmetic is a cliff
rather than a slope:

======  ==================================================
R       Best possible two-sided sign-test probability
======  ==================================================
3       0.250
5       0.0625   -- cannot reach 0.05 even if unanimous
6       0.031
8       0.008
======  ==================================================

So at R ≤ 5 no significance claim exists *at any effect size*, and :func:`summarise_replicates`
reports ``sign_test_p`` alongside a flag saying whether significance was reachable in principle. That
distinction is what stops an underpowered design being written up as a null finding about nature.

**Two uncertainty sources, and they are not interchangeable:**

* **across replicates** — how much the compressed model depends on *which* calibration data it saw.
  :func:`summarise_replicates`.
* **across evaluation windows** — how much the number depends on the finite evaluation corpus.
  :func:`paired_block_bootstrap`, resampling **complete windows** because neighbouring tokens are
  dependent and a token-level bootstrap understates the interval.

Neither replaces the other, and the bootstrap must resample the **same window indices for both arms**
or the pairing that makes the comparison valid is thrown away.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from scale_aware_compression.logging_utils import get_logger

LOGGER = get_logger(__name__)

MIN_R_FOR_SIGNIFICANCE = 6
"""Smallest replicate count at which a two-sided sign test can reach p < 0.05.

At R=5 a unanimous result carries ``2 / 2**5 = 0.0625``. Reported explicitly so an underpowered design
is never mistaken for evidence of absence.
"""


class ReplicateError(ValueError):
    """Raised when replicate aggregation is given inconsistent or insufficient input."""


def split_signs(gains: Sequence[float]) -> tuple[int, int, int]:
    """Count positive, negative and tied gains.

    **Ties are counted, not silently absorbed.** The sign test is a statement about the *direction*
    of a difference, and a difference of exactly zero has no direction -- so the conventional test
    discards ties and reduces n accordingly. Counting a tie as a negative, which is what a bare
    ``sum(g > 0)`` against the full replicate count does, biases the p-value in whichever direction
    happens to be inconvenient and does so invisibly.

    **A tie means exactly 0.0, with no tolerance.** A tolerance would be a threshold on the effect
    being measured, and every candidate value for it is now visible in the recorded results -- so
    choosing one would be selecting an analysis parameter after seeing the data, which §6.3 forbids.
    Exact equality is the only threshold that was not chosen with knowledge of the outcome. On real
    records this has never fired: the smallest recorded 1B gain is +0.0044 pp, which *rounds* to
    +0.00 in a two-decimal table but is genuinely positive.

    Args:
        gains: One gain per replicate.

    Returns:
        ``(positive, negative, ties)``, summing to ``len(gains)``.
    """
    positive = sum(1 for value in gains if value > 0.0)
    negative = sum(1 for value in gains if value < 0.0)
    return positive, negative, len(gains) - positive - negative


def sign_test_p_value(positive: int, total: int) -> float:
    """Exact two-sided sign-test probability under a fair-sign null.

    No normal approximation: at the replicate counts this study uses (3 to 8) an approximation is
    simply wrong, and the exact value is a short sum.

    Args:
        positive: Number of replicates whose gain was positive.
        total: Number of **non-tied** replicates. Pass the count excluding ties; see
            :func:`split_signs` for why.

    Returns:
        The two-sided probability, capped at 1.0.

    Raises:
        ReplicateError: If ``total`` is not positive or ``positive`` exceeds it.
    """
    if total <= 0:
        raise ReplicateError(f"total must be > 0, got {total}")
    if not 0 <= positive <= total:
        raise ReplicateError(f"positive must be in [0, {total}], got {positive}")

    extreme = min(positive, total - positive)
    tail = sum(math.comb(total, k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2**total))


@dataclass(frozen=True, slots=True)
class ReplicateSummary:
    """A joint gain aggregated over paired calibration replicates."""

    model_name: str
    budget_label: str
    replicates: int
    gains: tuple[float, ...]
    mean_gain: float
    median_gain: float
    standard_deviation: float
    minimum: float
    maximum: float
    positive_count: int
    negative_count: int = 0
    tie_count: int = 0
    """Replicates whose gain was exactly zero. Excluded from the sign test, reported anyway.

    A tie is not evidence either way, but it is also not nothing: a cell producing ties says the
    two arms are landing on identical numbers, which is worth seeing rather than folding into a
    negative count."""
    sign_test_p: float = 1.0
    sign_test_n: int = 0
    """Non-tied replicates the sign test was computed over. Reported because a p-value from n=6 with
    two ties is a different claim from one over n=8, and §5.1 requires R reported per cell."""

    @property
    def standard_error(self) -> float:
        """Standard error of the mean gain. ``nan`` when a single replicate makes it undefined."""
        if self.replicates < 2:
            return float("nan")
        return self.standard_deviation / math.sqrt(self.replicates)

    @property
    def consistent_in_sign(self) -> bool:
        """True when every replicate that had a direction agreed on it.

        Ties are excluded rather than counted against consistency, matching the sign test. A cell of
        all ties returns ``False``: there is no direction to be consistent about, and returning
        ``True`` would let a set of identical results satisfy §6.3's consistency clause.
        """
        if self.sign_test_n == 0:
            return False
        return self.positive_count == self.sign_test_n or self.negative_count == self.sign_test_n

    @property
    def significance_was_reachable(self) -> bool:
        """Whether *any* outcome at this effective n could have reached p < 0.05.

        False means a null result says nothing about nature -- only about the replicate count. This is
        the flag that keeps an underpowered design from being written up as evidence of absence.

        Judged on :attr:`sign_test_n`, not the raw replicate count: ties do not contribute to the
        test, so R=8 with three ties has the power of n=5 and cannot reach 0.05 however the rest
        fall. Using the raw count here would overstate the power of exactly the cells most likely to
        tie -- the near-lossless W8 control.
        """
        return self.sign_test_n >= MIN_R_FOR_SIGNIFICANCE

    @property
    def mean_over_sd(self) -> float:
        """Mean gain in units of its own spread. ``nan`` when the spread is undefined or zero."""
        if self.replicates < 2 or self.standard_deviation == 0.0:
            return float("nan")
        return self.mean_gain / self.standard_deviation

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "model_name": self.model_name,
            "budget_label": self.budget_label,
            "replicates": self.replicates,
            "gains": list(self.gains),
            "mean_gain": self.mean_gain,
            "median_gain": self.median_gain,
            "standard_deviation": self.standard_deviation,
            "standard_error": self.standard_error,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "tie_count": self.tie_count,
            "sign_test_n": self.sign_test_n,
            "consistent_in_sign": self.consistent_in_sign,
            "sign_test_p": self.sign_test_p,
            "significance_was_reachable": self.significance_was_reachable,
            "mean_over_sd": self.mean_over_sd,
        }


def summarise_replicates(
    *,
    model_name: str,
    budget_label: str,
    gains: list[float],
) -> ReplicateSummary:
    """Aggregate per-replicate joint gains into the quantity the paper reports.

    Every replicate-level value is retained in :attr:`ReplicateSummary.gains`, because A1 §5.1 requires
    them reported individually. A mean that hides a sign flip is exactly what F-26 caught.

    Args:
        model_name: Registry short name.
        budget_label: The budget both arms ran at.
        gains: One gain per replicate, in replicate order.

    Returns:
        The summary.

    Raises:
        ReplicateError: If ``gains`` is empty.
    """
    if not gains:
        raise ReplicateError(
            "summarise_replicates needs at least one gain. An empty list would produce a summary "
            "describing nothing, which is worse than an error."
        )

    count = len(gains)
    positive, negative, ties = split_signs(gains)
    effective = positive + negative
    deviation = statistics.stdev(gains) if count >= 2 else 0.0

    # The sign test runs over non-tied replicates only. With every replicate tied there is no
    # direction to test, so p = 1.0 is the honest answer rather than an error: the cell ran, it just
    # carries no directional evidence.
    probability = sign_test_p_value(positive, effective) if effective else 1.0

    if ties:
        LOGGER.warning(
            "%s/%s: %d of %d replicates gave a gain of exactly zero. Ties carry no direction, so "
            "the sign test is over n=%d, not n=%d. Report both counts.",
            model_name,
            budget_label,
            ties,
            count,
            effective,
            count,
        )
    if effective and effective < MIN_R_FOR_SIGNIFICANCE:
        LOGGER.info(
            "Effective n=%d for %s/%s: no significance claim is reachable at this count "
            "(best possible two-sided p is %.4f). Report effect size and sign consistency.",
            effective,
            model_name,
            budget_label,
            sign_test_p_value(effective, effective),
        )

    return ReplicateSummary(
        model_name=model_name,
        budget_label=budget_label,
        replicates=count,
        gains=tuple(gains),
        mean_gain=statistics.mean(gains),
        median_gain=statistics.median(gains),
        standard_deviation=deviation,
        minimum=min(gains),
        maximum=max(gains),
        positive_count=positive,
        negative_count=negative,
        tie_count=ties,
        sign_test_p=probability,
        sign_test_n=effective,
    )


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    """A paired block-bootstrap interval over evaluation windows."""

    windows: int
    resamples: int
    point_estimate: float
    lower: float
    upper: float
    confidence: float
    positive_resamples: int

    @property
    def excludes_zero(self) -> bool:
        """True when the interval lies entirely on one side of zero."""
        return (self.lower > 0.0 and self.upper > 0.0) or (self.lower < 0.0 and self.upper < 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "windows": self.windows,
            "resamples": self.resamples,
            "point_estimate": self.point_estimate,
            "lower": self.lower,
            "upper": self.upper,
            "confidence": self.confidence,
            "positive_resamples": self.positive_resamples,
            "excludes_zero": self.excludes_zero,
        }


def paired_block_bootstrap(
    *,
    sequential_window_nll: list[float],
    joint_window_nll: list[float],
    window_tokens: list[int],
    resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 1234,
) -> BootstrapInterval:
    """Bootstrap the mean per-token NLL advantage of joint over sequential, by whole windows.

    Resamples **complete evaluation windows** rather than tokens: neighbouring tokens are statistically
    dependent, so a token-level bootstrap treats correlated observations as independent and returns an
    interval that is too narrow.

    **Paired by construction.** Each resample draws window indices once and applies them to *both*
    arms, so a window that happens to be hard for one arm is hard for the other in the same resample.
    Resampling the arms independently would discard the pairing and inflate the interval with variance
    the design exists to remove.

    Quantifies a different thing from :func:`summarise_replicates`: this is uncertainty from the finite
    evaluation corpus at one calibration draw, not uncertainty from which calibration data was seen.
    Both belong in the paper.

    Args:
        sequential_window_nll: Summed NLL per window for the sequential arm, in loader order.
        joint_window_nll: The same for the joint arm, same order and length.
        window_tokens: Predicted-token count per window, parallel to both.
        resamples: Bootstrap resamples.
        confidence: Interval width, e.g. ``0.95``.
        seed: Fixed so an interval is reproducible.

    Returns:
        The interval. Positive values mean joint achieved the lower NLL.

    Raises:
        ReplicateError: If the inputs disagree in length, are empty, or the arguments are out of range.
    """
    import random

    lengths = {len(sequential_window_nll), len(joint_window_nll), len(window_tokens)}
    if len(lengths) != 1:
        raise ReplicateError(
            f"window arrays must be the same length, got {sorted(lengths)}. Two arms evaluated over "
            "different windows cannot be paired, and pairing is what makes this interval valid."
        )
    count = lengths.pop()
    if count == 0:
        raise ReplicateError("paired_block_bootstrap needs at least one window")
    if resamples < 1:
        raise ReplicateError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ReplicateError(f"confidence must be in (0, 1), got {confidence}")

    # Per-window mean NLL difference. Positive favours joint. Weighted by tokens when aggregated, so
    # windows with fewer predicted tokens do not count equally -- the same reasoning that makes the
    # headline perplexity a token-weighted quantity rather than a mean of per-window perplexities.
    def weighted_advantage(indices: list[int]) -> float:
        total_tokens = sum(window_tokens[i] for i in indices)
        if total_tokens == 0:
            return 0.0
        delta = sum(sequential_window_nll[i] - joint_window_nll[i] for i in indices)
        return delta / total_tokens

    point = weighted_advantage(list(range(count)))

    generator = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        drawn = [generator.randrange(count) for _ in range(count)]
        estimates.append(weighted_advantage(drawn))
    estimates.sort()

    tail = (1.0 - confidence) / 2.0
    lower = estimates[max(0, int(math.floor(tail * resamples)) - 1)]
    upper = estimates[min(resamples - 1, int(math.ceil((1.0 - tail) * resamples)) - 1)]

    return BootstrapInterval(
        windows=count,
        resamples=resamples,
        point_estimate=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        positive_resamples=sum(1 for value in estimates if value > 0.0),
    )


@dataclass(slots=True)
class ScaleComparison:
    """Per-replicate differences between two scales, which is how the scale claim is tested."""

    smaller_model: str
    larger_model: str
    budget_label: str
    differences: list[float] = field(default_factory=list)

    @property
    def mean_difference(self) -> float:
        """Mean of the per-replicate differences."""
        return statistics.mean(self.differences) if self.differences else float("nan")

    @property
    def positive_count(self) -> int:
        """Replicates where the smaller model showed the larger gain."""
        return split_signs(self.differences)[0]

    @property
    def negative_count(self) -> int:
        """Replicates where the larger model showed the larger gain."""
        return split_signs(self.differences)[1]

    @property
    def tie_count(self) -> int:
        """Replicates where both scales gained exactly the same. See :func:`split_signs`."""
        return split_signs(self.differences)[2]

    @property
    def sign_test_n(self) -> int:
        """Non-tied replicates the sign test is computed over."""
        return self.positive_count + self.negative_count

    @property
    def consistent_in_sign(self) -> bool:
        """True when every replicate that had a direction agreed on which scale gained more.

        Ties are excluded rather than counted against consistency, matching the sign test and
        :attr:`ReplicateSummary.consistent_in_sign`. All-ties returns ``False``: identical gains at
        two scales are not evidence of a scale effect in either direction.
        """
        if self.sign_test_n == 0:
            return False
        return self.positive_count == self.sign_test_n or self.negative_count == self.sign_test_n

    @property
    def sign_test_p(self) -> float:
        """Exact two-sided sign-test probability over the non-tied replicates."""
        if self.sign_test_n == 0:
            return float("nan")
        return sign_test_p_value(self.positive_count, self.sign_test_n)

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "smaller_model": self.smaller_model,
            "larger_model": self.larger_model,
            "budget_label": self.budget_label,
            "differences": list(self.differences),
            "replicates": len(self.differences),
            "mean_difference": self.mean_difference,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "tie_count": self.tie_count,
            "sign_test_n": self.sign_test_n,
            "consistent_in_sign": self.consistent_in_sign,
            "sign_test_p": self.sign_test_p,
        }


def compare_scales(
    *,
    smaller: ReplicateSummary,
    larger: ReplicateSummary,
) -> ScaleComparison:
    """Difference the two scales **replicate by replicate**, not summary by summary.

    Per-replicate differencing is the stronger comparison whenever both scales used the *same*
    calibration draws, which A1 §5.1 guarantees by making the smaller replicate count a prefix of the
    larger. Draw ``r`` then means the same calibration data at every scale, so the difference at draw
    ``r`` removes whatever that draw did to both models. Subtracting the two means instead would throw
    that pairing away.

    Args:
        smaller: Summary for the smaller model.
        larger: Summary for the larger model.

    Returns:
        The comparison, over as many replicates as both scales share.

    Raises:
        ReplicateError: If the budgets differ, which would compare two different experiments.
    """
    if smaller.budget_label != larger.budget_label:
        raise ReplicateError(
            f"cannot compare scales across budgets: {smaller.budget_label!r} vs "
            f"{larger.budget_label!r}. A gain at one budget is not comparable with a gain at another."
        )

    shared = min(smaller.replicates, larger.replicates)
    if shared < smaller.replicates or shared < larger.replicates:
        LOGGER.info(
            "Comparing %s and %s over %d shared replicate(s); %s has %d and %s has %d.",
            smaller.model_name,
            larger.model_name,
            shared,
            smaller.model_name,
            smaller.replicates,
            larger.model_name,
            larger.replicates,
        )

    return ScaleComparison(
        smaller_model=smaller.model_name,
        larger_model=larger.model_name,
        budget_label=smaller.budget_label,
        differences=[smaller.gains[i] - larger.gains[i] for i in range(shared)],
    )
