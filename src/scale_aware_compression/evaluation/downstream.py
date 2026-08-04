"""Downstream task accuracy via lm-evaluation-harness (plan §4.3, gap A4).

Perplexity says a compressed model still predicts text. It does not say the model is still *useful*.
§4.3 requires three multiple-choice tasks alongside it:

* **HellaSwag** -- commonsense sentence completion, 4 choices
* **PIQA** -- physical commonsense, 2 choices
* **ARC-Easy** -- grade-school science questions, 4 choices

The harness is used rather than reimplemented, and pinned exactly. Reimplementing three tasks in
repo risks silent scoring differences from the published numbers this exists to be comparable with,
which is a worse failure than one heavy dependency (freeze table, §2.7). **Task versions are
recorded per task** (§4.8): task definitions change between harness releases, so an accuracy without
its task version cannot be compared to anything.

WHAT AN ACCURACY HERE MEANS, AND WHAT IT DOES NOT
-------------------------------------------------
Multiple-choice scoring asks the model for the log-likelihood of each candidate continuation and
takes the argmax. So:

* **There is a floor.** Random guessing scores 25% on HellaSwag and ARC-Easy and 50% on PIQA. A
  compressed model at chance has stopped doing the task rather than doing it worse -- but "at chance"
  has to mean *indistinguishable from* the floor, not merely below it, so
  :attr:`TaskResult.chance_verdict` gives three outcomes over an interval rather than a bare
  comparison. 0.2501 on a four-choice task is above the floor arithmetically and says nothing.
* **The scoring path is partly uncompressed.** §2.6 excludes embeddings and the head from the
  targeted modules, so the logits these accuracies are computed from come out of an FP32 layer at
  every budget. That is correct under the plan's scale-invariant definition of the compression
  budget, and it is worth stating because a reader may reasonably expect otherwise.
* **Report against dense**, the way perplexity retention does. An absolute accuracy without its
  dense reference does not say whether compression cost anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.evaluation.common import EvaluationError
from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)

DOWNSTREAM_TASKS = ("hellaswag", "piqa", "arc_easy")
"""The three tasks §4.3 requires, in the order the write-up reports them."""

CHANCE_LEVEL = {"hellaswag": 0.25, "piqa": 0.50, "arc_easy": 0.25}
"""Accuracy from guessing, per task. One over the number of choices."""

PRIMARY_METRIC = "acc"
"""Unnormalised accuracy is the primary figure.

The harness also reports ``acc_norm``, which divides each candidate's log-likelihood by its byte
length. Both are legitimate and published papers quote both, so both are recorded -- but one has to
be *the* number or the choice becomes available after seeing results, and §6.3 forbids that. `acc` is
the simpler definition and the one the plan names.
"""


def chance_level(task: str) -> float:
    """Accuracy a model achieves by guessing on ``task``.

    Args:
        task: Task name as passed to the harness.

    Returns:
        The chance accuracy, or ``0.0`` for a task with no recorded choice count.
    """
    return CHANCE_LEVEL.get(task, 0.0)


@dataclass(frozen=True, slots=True)
class TaskResult:
    """One task's score for one model."""

    task: str
    accuracy: float
    accuracy_stderr: float | None
    accuracy_norm: float | None
    task_version: str | None
    """Harness task version (§4.8). ``None`` means the harness did not report one, which must be
    recorded as such rather than defaulted -- a missing version is a gap in provenance."""
    num_samples: int | None

    @property
    def is_above_chance(self) -> bool:
        """Whether the score is *arithmetically* above guessing.

        **A literal comparison, and not sufficient on its own.** 0.2501 on a four-choice task is
        above chance by this test and statistically indistinguishable from it. Use
        :attr:`chance_verdict` for anything a reader will interpret; this is kept because a bare
        comparison is still the right thing to sort or filter on.
        """
        return self.accuracy > chance_level(self.task)

    @property
    def chance_verdict(self) -> str:
        """Whether the score is *distinguishable* from guessing, at roughly 95%.

        Three outcomes, because two are not enough to say something honest:

        * ``"above chance"`` -- the interval clears the floor;
        * ``"indistinguishable from chance"`` -- the floor lies inside the interval. The model may
          or may not be doing the task; this measurement cannot tell, and saying "it still performs
          the task" on the strength of accuracy > chance would be an overclaim;
        * ``"below chance"`` -- the interval is under the floor, which usually means a systematic
          scoring problem rather than a merely bad model;
        * ``"unknown (no stderr)"`` -- the harness reported no standard error, so no interval exists.
          Recorded rather than defaulted to a verdict.

        The interval is +/- 2 standard errors. That multiplier is the conventional ~95% default, not
        one chosen to make a particular row read a particular way -- and this labelling is
        **descriptive**: the primary downstream comparison is accuracy retention against dense, not
        a test against chance. Nothing in the paper's claims depends on which side of this line a
        row falls.
        """
        floor = chance_level(self.task)
        if self.accuracy_stderr is None:
            return "unknown (no stderr)"
        margin = 2.0 * self.accuracy_stderr
        if self.accuracy - margin > floor:
            return "above chance"
        if self.accuracy + margin < floor:
            return "below chance"
        return "indistinguishable from chance"

    @property
    def is_demonstrably_above_chance(self) -> bool:
        """True only when the interval clears the floor. The strict form of :attr:`is_above_chance`."""
        return self.chance_verdict == "above chance"

    def to_dict(self) -> dict[str, Any]:
        """Return a flat, serialisable mapping."""
        return {
            "task": self.task,
            "accuracy": self.accuracy,
            "accuracy_stderr": self.accuracy_stderr,
            "accuracy_norm": self.accuracy_norm,
            "task_version": self.task_version,
            "num_samples": self.num_samples,
            "chance_level": chance_level(self.task),
            "above_chance": self.is_above_chance,
            "chance_verdict": self.chance_verdict,
            "demonstrably_above_chance": self.is_demonstrably_above_chance,
        }


@dataclass(slots=True)
class DownstreamReport:
    """Every task score for one model, plus the provenance needed to compare it."""

    tasks: list[TaskResult] = field(default_factory=list)
    harness_version: str | None = None
    device: str = "cpu"
    batch_size: int | None = None
    limit: int | None = None
    """Samples per task, when subsampled. ``None`` means the full task.

    Recorded because a subsampled score is **not** comparable with a published one, and the
    difference has to be visible in the record rather than remembered.
    """

    @property
    def mean_accuracy(self) -> float | None:
        """Unweighted mean across tasks, or ``None`` with no tasks.

        Unweighted on purpose: the three tasks differ in size by 5x, and weighting by size would let
        HellaSwag alone decide the headline.
        """
        if not self.tasks:
            return None
        return sum(result.accuracy for result in self.tasks) / len(self.tasks)

    @property
    def tasks_at_chance(self) -> list[str]:
        """Tasks not *demonstrably* above guessing, i.e. the floor is inside the interval or above it.

        Uses the interval rather than the bare comparison, so a score sitting 0.0001 above the floor
        is reported here rather than counted as a working task.
        """
        return [result.task for result in self.tasks if not result.is_demonstrably_above_chance]

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "harness_version": self.harness_version,
            "device": self.device,
            "batch_size": self.batch_size,
            "limit": self.limit,
            "mean_accuracy": self.mean_accuracy,
            "tasks_at_chance": self.tasks_at_chance,
            "tasks": [result.to_dict() for result in self.tasks],
        }


def accuracy_retention(compressed: float, dense: float) -> float:
    """Compressed accuracy as a fraction of dense accuracy.

    Args:
        compressed: Accuracy of the compressed model.
        dense: Accuracy of the dense baseline.

    Returns:
        ``compressed / dense``. May exceed 1.0, which does happen at mild compression and is worth
        reporting rather than clipping.

    Raises:
        EvaluationError: If ``dense`` is not positive, since the ratio would be undefined.
    """
    if dense <= 0:
        raise EvaluationError(f"dense accuracy must be > 0 to form a retention, got {dense}")
    return compressed / dense


def parse_harness_results(
    results: dict[str, Any],
    *,
    tasks: tuple[str, ...] = DOWNSTREAM_TASKS,
) -> list[TaskResult]:
    """Turn the harness's nested output into flat per-task results.

    Kept separate from the call that produces it so the parsing is testable without a network, a
    dataset download, or a model -- which is what lets this have offline tests at all.

    Args:
        results: The mapping ``lm_eval.simple_evaluate`` returns.
        tasks: Which task names to extract, in report order.

    Returns:
        One :class:`TaskResult` per task present, in ``tasks`` order.

    Raises:
        EvaluationError: If a requested task is missing from the output, or reports no primary
            metric. Both mean the run did not measure what it was asked to, and a silently short
            report would be read as a complete one.
    """
    scores = results.get("results") or {}
    versions = results.get("versions") or {}
    counts = results.get("n-samples") or {}

    parsed: list[TaskResult] = []
    for task in tasks:
        if task not in scores:
            raise EvaluationError(
                f"task {task!r} is absent from the harness results (present: "
                f"{sorted(scores)}). The run did not measure what it was asked to."
            )
        entry = scores[task]
        accuracy = _metric(entry, PRIMARY_METRIC)
        if accuracy is None:
            raise EvaluationError(
                f"task {task!r} reported no {PRIMARY_METRIC!r} metric (keys: {sorted(entry)})"
            )
        sample_entry = counts.get(task)
        parsed.append(
            TaskResult(
                task=task,
                accuracy=float(accuracy),
                accuracy_stderr=_optional_float(_metric(entry, f"{PRIMARY_METRIC}_stderr")),
                accuracy_norm=_optional_float(_metric(entry, "acc_norm")),
                # The harness has moved this between `versions` and the per-task entry across
                # releases, so both are checked rather than assuming one shape.
                task_version=_task_version(entry, versions, task),
                num_samples=_sample_count(sample_entry),
            )
        )
    return parsed


def _metric(entry: dict[str, Any], name: str) -> Any:
    """Read a metric, tolerating the harness's ``metric,filter`` key suffixes."""
    if name in entry:
        return entry[name]
    for key, value in entry.items():
        if isinstance(key, str) and key.split(",")[0] == name:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    """Coerce to float, or ``None`` when absent or non-numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _task_version(entry: dict[str, Any], versions: dict[str, Any], task: str) -> str | None:
    """Find the task version wherever this harness release put it."""
    for candidate in (versions.get(task), entry.get("task_version"), entry.get("version")):
        if candidate is not None:
            return str(candidate)
    return None


def _sample_count(entry: Any) -> int | None:
    """Extract an evaluated-sample count from the harness's varied shapes."""
    if isinstance(entry, dict):
        for key in ("effective", "original"):
            if key in entry:
                try:
                    return int(entry[key])
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return int(entry)
    except (TypeError, ValueError):
        return None


def evaluate_downstream(
    model: nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    *,
    tasks: tuple[str, ...] = DOWNSTREAM_TASKS,
    device: str = "cpu",
    batch_size: int = 4,
    limit: int | None = None,
) -> DownstreamReport:
    """Score ``model`` on the §4.3 downstream tasks.

    Args:
        model: A loaded causal LM, compressed or dense. Moved to ``device``.
        tokenizer: Its tokeniser.
        tasks: Task names, defaulting to the three §4.3 requires.
        device: Where to run. GPU is far faster and legitimate here -- accuracy is
            device-invariant far below the ~1 pp differences being reported -- but the choice is
            recorded either way.
        batch_size: Sequences per forward.
        limit: Samples per task. ``None`` runs the full task. **A subsampled score is not
            comparable with a published number**, which is why it is recorded.

    Returns:
        The report, with per-task accuracy, task versions and the harness version.

    Raises:
        EvaluationError: If the harness is unavailable or a requested task is missing.
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError as error:  # pragma: no cover - exercised by the import guard test
        raise EvaluationError(
            "lm-eval is required for downstream tasks (§4.3) and is not installed. It is pinned in "
            "requirements.txt; install with `uv pip install -r requirements.txt`."
        ) from error

    model.eval()
    model.to(device)
    wrapped = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)

    LOGGER.info(
        "Downstream evaluation on %s: tasks=%s batch=%d limit=%s",
        device,
        ",".join(tasks),
        batch_size,
        limit if limit is not None else "full",
    )
    results = lm_eval.simple_evaluate(
        model=wrapped, tasks=list(tasks), limit=limit, verbosity="WARNING"
    )
    if results is None:
        raise EvaluationError("lm_eval.simple_evaluate returned nothing; no scores were produced")

    report = DownstreamReport(
        tasks=parse_harness_results(results, tasks=tasks),
        harness_version=getattr(lm_eval, "__version__", None),
        device=device,
        batch_size=batch_size,
        limit=limit,
    )
    for result in report.tasks:
        LOGGER.info(
            "  %-10s acc %.4f +/- %s (chance %.2f, %s)  version=%s  n=%s",
            result.task,
            result.accuracy,
            f"{result.accuracy_stderr:.4f}" if result.accuracy_stderr is not None else "?",
            chance_level(result.task),
            result.chance_verdict,
            result.task_version,
            result.num_samples,
        )
    if report.tasks_at_chance:
        # Not an error: a sufficiently damaged model legitimately scores at chance, and that is a
        # result. But it must be loud, because "42%" reads as a weak score rather than as a model
        # that may have stopped doing the task.
        LOGGER.warning(
            "Not demonstrably above chance on %s. Do not write that the model still performs these "
            "tasks -- the interval includes guessing.",
            ", ".join(report.tasks_at_chance),
        )
    return report
