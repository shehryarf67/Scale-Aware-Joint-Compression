"""The compression interface every experimental arm implements.

One abstract base class, five stages. Each arm visits a different subset of the stages, in a
different order, but every arm is driven through :meth:`Compressor.run`, so the pipelines
differ only in what they declare rather than in how the runner calls them. That is what makes
the joint-versus-sequential comparison fair: neither arm can quietly gain an extra recovery
pass or skip a conversion step.

Stages
------
``prepare``
    Inspect the model, select compressible modules, and install anything the arm needs before
    optimisation, such as masks or fake-quantisation observers. No weights change.
``apply``
    Change the weights or their representation: assign masks, zero pruned weights, compute
    quantisation parameters.
``recover``
    Optimise to recover quality lost in ``apply``. May run on GPU. The optimisation budget is
    recorded so joint and sequential arms can be compared at matched cost.
``convert``
    Produce the artefact that gets deployed and measured: real low-precision weights rather
    than fake-quantised FP32, masks folded into the weights rather than held alongside them.
``save`` / ``report_statistics``
    Persist the artefact and describe what was actually achieved, as opposed to what the
    configuration asked for.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.config import CompressionConfig, ExperimentConfig
from scale_aware_compression.constants import CompressionMethod, CompressionStage
from scale_aware_compression.logging_utils import get_logger, log_stage

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class CompressionError(RuntimeError):
    """Raised when a compression stage cannot complete."""


@dataclass(slots=True)
class StageRecord:
    """What happened during one stage, for the run record."""

    stage: CompressionStage
    duration_seconds: float
    device: str = "cpu"
    optimiser_steps: int = 0
    tokens_processed: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping."""
        return {
            "stage": self.stage.value,
            "duration_seconds": self.duration_seconds,
            "device": self.device,
            "optimiser_steps": self.optimiser_steps,
            "tokens_processed": self.tokens_processed,
            "details": self.details,
        }


@dataclass(slots=True)
class CompressionResult:
    """The artefact produced by a compression arm, plus how it was produced."""

    method: CompressionMethod
    model: Any
    """The converted, deployable model."""
    stages: list[StageRecord] = field(default_factory=list)
    statistics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Path | None = None

    @property
    def total_optimiser_steps(self) -> int:
        """Optimiser steps across every stage.

        This is the training-cost figure compared between the joint and sequential arms.
        """
        return sum(record.optimiser_steps for record in self.stages)

    @property
    def total_duration_seconds(self) -> float:
        """Wall-clock seconds across every stage."""
        return sum(record.duration_seconds for record in self.stages)

    @property
    def stage_sequence(self) -> tuple[CompressionStage, ...]:
        """The stages actually visited, in order."""
        return tuple(record.stage for record in self.stages)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable mapping, excluding the model object itself."""
        return {
            "method": self.method.value,
            "stages": [record.to_dict() for record in self.stages],
            "stage_sequence": [stage.value for stage in self.stage_sequence],
            "total_optimiser_steps": self.total_optimiser_steps,
            "total_duration_seconds": self.total_duration_seconds,
            "checkpoint_path": self.checkpoint_path.as_posix() if self.checkpoint_path else None,
            "statistics": self.statistics,
        }


class Compressor(ABC):
    """Base class for every compression arm.

    Subclasses declare :attr:`method` and :attr:`pipeline_stages`, then implement the stage
    methods. :meth:`run` is the template method that drives them, so the ordering of stages
    lives in one place rather than being reimplemented per arm.
    """

    method: CompressionMethod = CompressionMethod.DENSE
    """Which experimental arm this class implements."""

    pipeline_stages: tuple[CompressionStage, ...] = ()
    """The stages this arm visits, in order. Documented, asserted, and reported."""

    apply_stage: CompressionStage = CompressionStage.PREPARE
    """Stage name logged for this arm's :meth:`apply` step, e.g. ``PRUNED`` or ``QUANTISED``."""

    recover_stage: CompressionStage = CompressionStage.RECOVERED
    """Stage name logged for this arm's :meth:`recover` step."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Store the config and initialise the stage log.

        Args:
            config: The full experiment config. The whole object is kept, not just the
                compression section, because recovery needs the data and runtime sections
                too.
        """
        self.config = config
        self.stage_records: list[StageRecord] = []
        self._logger = get_logger(f"{type(self).__module__}.{type(self).__name__}")

    # -- properties ---------------------------------------------------------
    @property
    def compression_config(self) -> CompressionConfig:
        """The compression section of the experiment config."""
        return self.config.compression

    @property
    def name(self) -> str:
        """Human-readable arm name, used in labels and log lines."""
        return self.method.value

    # -- abstract stages ----------------------------------------------------
    @abstractmethod
    def prepare(self, model: nn.Module) -> nn.Module:
        """Inspect the model and install whatever the arm needs before weights change.

        Args:
            model: The dense model, freshly loaded.

        Returns:
            The model, possibly wrapped or annotated. Weights are unchanged.
        """

    @abstractmethod
    def apply(self, model: nn.Module) -> nn.Module:
        """Apply the compression itself: masks, zeroing, or quantisation parameters.

        Args:
            model: The prepared model.

        Returns:
            The compressed model, still in a trainable representation.
        """

    @abstractmethod
    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Fine-tune to recover quality lost during :meth:`apply`.

        May run on GPU: recovery is training, not a deployment measurement.

        Args:
            model: The compressed model.
            tokenizer: Tokeniser for building the recovery data loader.

        Returns:
            The recovered model.
        """

    @abstractmethod
    def convert(self, model: nn.Module) -> nn.Module:
        """Produce the deployable artefact that the CPU benchmark will measure.

        Fake quantisation becomes real low-precision storage, and masks are folded into the
        weights. Without this stage a "quantised" model is still FP32 and its measured size
        and latency mean nothing.

        Args:
            model: The recovered model.

        Returns:
            The converted, CPU-deployable model.
        """

    @abstractmethod
    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Describe what was achieved, as distinct from what was requested.

        Args:
            model: The model to measure. When ``None``, only configuration-derived and
                stage-log fields are returned.

        Returns:
            A serialisable mapping. Implementations should report measured sparsity and size
            next to their targets, so a mismatch is visible in the results table.
        """

    # -- concrete helpers ---------------------------------------------------
    def save(self, model: nn.Module, path: str | Path) -> Path:
        """Persist the converted model.

        Uses ``save_pretrained`` when available, since the checkpoint size measured from that
        layout is what a deployment would actually ship. Falls back to a state dict for models
        that are no longer Transformers instances after conversion.

        Args:
            model: The model to save, normally post-:meth:`convert`.
            path: Destination directory. Created if absent.

        Returns:
            The directory written to.

        Raises:
            CompressionError: If neither save route works.
        """
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)

        save_pretrained = getattr(model, "save_pretrained", None)
        if callable(save_pretrained):
            try:
                save_pretrained(destination)
            except Exception as error:
                raise CompressionError(
                    f"save_pretrained failed for {destination}: {error}"
                ) from error
            self._logger.info("Saved %s artefact to %s", self.name, destination)
            return destination

        try:
            import torch
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise CompressionError(
                "PyTorch is required to save a model without save_pretrained support"
            ) from error
        target = destination / "state_dict.pt"
        torch.save(model.state_dict(), target)
        self._logger.info("Saved %s state dict to %s", self.name, target)
        return destination

    def record_stage(
        self,
        stage: CompressionStage,
        duration_seconds: float,
        *,
        device: str = "cpu",
        optimiser_steps: int = 0,
        tokens_processed: int = 0,
        **details: Any,
    ) -> StageRecord:
        """Append a stage to the log.

        Args:
            stage: Which stage completed.
            duration_seconds: Wall-clock duration.
            device: Device the stage ran on.
            optimiser_steps: Optimiser steps consumed, for the training-cost comparison.
            tokens_processed: Training tokens consumed.
            **details: Extra fields stored verbatim in the record.

        Returns:
            The appended record.
        """
        record = StageRecord(
            stage=stage,
            duration_seconds=duration_seconds,
            device=device,
            optimiser_steps=optimiser_steps,
            tokens_processed=tokens_processed,
            details=details,
        )
        self.stage_records.append(record)
        return record

    def run(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> CompressionResult:
        """Drive the arm through its declared stages and collect the result.

        Args:
            model: The dense model to compress.
            tokenizer: Tokeniser, needed by arms that recover or calibrate.

        Returns:
            The :class:`CompressionResult`, including the stage log and statistics.

        Raises:
            CompressionError: If a stage fails.
        """
        self._logger.info(
            "Running %s pipeline: %s",
            self.name,
            " -> ".join(stage.value for stage in self.pipeline_stages),
        )
        self.stage_records.clear()

        with log_stage(self._logger, f"{self.name}: prepare"):
            started = time.perf_counter()
            model = self.prepare(model)
            self.record_stage(CompressionStage.PREPARE, time.perf_counter() - started)

        with log_stage(self._logger, f"{self.name}: apply"):
            started = time.perf_counter()
            model = self.apply(model)
            self.record_stage(self.apply_stage, time.perf_counter() - started)

        if self.compression_config.recovery.enabled:
            with log_stage(self._logger, f"{self.name}: recover"):
                started = time.perf_counter()
                model = self.recover(model, tokenizer)
                # Stage implementations overwrite this record's optimiser_steps via
                # record_stage; the duration here is the outer wall clock.
                self.record_stage(
                    self.recover_stage,
                    time.perf_counter() - started,
                    device=self.compression_config.recovery.device.value,
                )
        else:
            self._logger.info("Recovery disabled by configuration; skipping")

        with log_stage(self._logger, f"{self.name}: convert"):
            started = time.perf_counter()
            model = self.convert(model)
            self.record_stage(CompressionStage.CONVERTED, time.perf_counter() - started)

        result = CompressionResult(
            method=self.method,
            model=model,
            stages=list(self.stage_records),
            statistics=self.report_statistics(model),
        )
        self._logger.info(
            "%s pipeline complete in %.1fs across %d optimiser steps",
            self.name,
            result.total_duration_seconds,
            result.total_optimiser_steps,
        )
        return result

    def base_statistics(self) -> dict[str, Any]:
        """Configuration-derived statistics shared by every arm.

        Returns:
            Mapping with the arm, its budget, the target sparsity and bit width, and the
            declared stage sequence.
        """
        compression = self.compression_config
        return {
            "method": self.method.value,
            "budget_label": compression.budget_label,
            "target_sparsity": compression.effective_sparsity,
            "target_bits": compression.effective_bits,
            "pruning_enabled": compression.pruning.enabled,
            "quantisation_enabled": compression.quantisation.enabled,
            "recovery_enabled": compression.recovery.enabled,
            "declared_stages": [stage.value for stage in self.pipeline_stages],
            "visited_stages": [record.stage.value for record in self.stage_records],
            "total_optimiser_steps": sum(record.optimiser_steps for record in self.stage_records),
        }
