"""The five experimental arms, as thin declarations over one shared driver.

Each arm is a subclass that sets a name and a stage sequence. All the behaviour lives in
:mod:`~scale_aware_compression.compression.layerwise`, which is the point: §3.8 asks whether a
method "qualifies as joint", and that is only answerable by inspection if the arms share an
implementation. Here the only thing an arm can differ in is the order it calls the solver.

Stage semantics under layerwise post-training reconstruction (plan §3.1):

=====================  =========================================================================
``prepare``            select target modules, materialise the calibration batches
``apply``              the layerwise loop -- the whole algorithm
``recover``            no-op; retained for the optional short-fine-tune ablation
``convert``            pack to real low-bit storage, so size and precision become measurable
``report_statistics``  measured against target, plus per-layer reconstruction losses
=====================  =========================================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import CompressionError, Compressor
from scale_aware_compression.compression.layerwise import (
    LayerPlan,
    LayerwiseReport,
    compress_model_layerwise,
)
from scale_aware_compression.compression.packed import convert_model_to_packed
from scale_aware_compression.config import ExperimentConfig
from scale_aware_compression.constants import (
    FP32_BITS,
    SEQUENTIAL_STAGES,
    CompressionMethod,
    CompressionStage,
    Device,
)
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import measure_sparsity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


def plan_from_config(config: ExperimentConfig) -> LayerPlan:
    """Build the shared :class:`LayerPlan` from an experiment configuration.

    Every arm derives its plan through this one function, so two arms configured from the same
    budget are guaranteed to agree on every field §3.11 requires them to share. An arm that built
    its own plan could drift silently.

    Args:
        config: A validated experiment config.

    Returns:
        The plan, with sparsity and bit width taken from the *effective* values so a
        quantisation-only arm gets ``sparsity=0`` rather than the configured pruning target.
    """
    compression = config.compression
    reconstruction = compression.reconstruction
    # Both axes come from the *effective* values, which are method-aware. Reading
    # `quantisation.bits` directly would hand the pruning-only arm a bit width whenever the shipped
    # config happened to enable quantisation -- and while its `apply` ignores that, `convert` would
    # pack the result, quietly turning the one FP32 arm (the arm that answers RQ4) into a quantised
    # one. FP32 is spelled 32 in `effective_bits` and `None` in a plan.
    effective_bits = compression.effective_bits
    return LayerPlan(
        sparsity=compression.effective_sparsity,
        bits=None if effective_bits >= FP32_BITS else effective_bits,
        granularity=compression.quantisation.granularity,
        group_size=compression.quantisation.group_size,
        pruning_granularity=compression.pruning.granularity,
        comparison_group=reconstruction.comparison_group,
        solver=reconstruction.solver,
        local_steps=reconstruction.local_steps,
        joint_iterations=reconstruction.joint_iterations,
        damping=reconstruction.damping,
        block_size=reconstruction.block_size,
        activation_order=reconstruction.activation_order,
        scale_search=reconstruction.scale_search,
        keep_benefit_saliency=reconstruction.keep_benefit_saliency,
    )


class LayerwiseArm(Compressor):
    """Base for every arm that compresses layer by layer.

    Subclasses declare :attr:`arm`, :attr:`method` and :attr:`pipeline_stages`; nothing else.
    """

    arm: str = ""
    """Which call order to run. Passed straight to the layerwise driver."""

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialise with no calibration data and nothing converted yet."""
        super().__init__(config)
        self.module_names: list[str] = []
        self.calibration_batches: list[Any] = []
        self.calibration_fingerprint: str = ""
        self.report: LayerwiseReport | None = None
        self.is_converted: bool = False
        self.conversion_statistics: dict[str, Any] = {}

    # -- configuration ------------------------------------------------------
    @property
    def plan(self) -> LayerPlan:
        """The shared compression plan for this run."""
        return plan_from_config(self.config)

    def set_calibration(
        self,
        batches: list[Any],
        *,
        fingerprint: str = "",
    ) -> None:
        """Supply the calibration batches every arm must share.

        Injected rather than loaded internally so the runner can hand *the same* tensors to every
        arm. §3.11 requires identical calibration examples, ordering and token count; passing the
        materialised list makes that a fact rather than a hope.

        Args:
            batches: Token-id tensors of shape ``(batch, sequence)``.
            fingerprint: Identifier recorded in the run record, so a mismatch between arms is
                detectable after the fact.
        """
        self.calibration_batches = list(batches)
        self.calibration_fingerprint = fingerprint

    # -- stages -------------------------------------------------------------
    def prepare(self, model: nn.Module) -> nn.Module:
        """Select the modules this arm may touch.

        Args:
            model: The dense model.

        Returns:
            The model, unchanged.

        Raises:
            CompressionError: If no calibration data was supplied.
        """
        from scale_aware_compression.models.adapters import select_compressible_modules

        if not self.calibration_batches:
            raise CompressionError(
                f"{self.name}: no calibration data. Call set_calibration() before run(); "
                "layerwise reconstruction cannot proceed without activations."
            )
        # The plan's sparsity and bit width come from the config's *method*, so an arm instantiated
        # against a config for a different method would silently run on the wrong budget -- a
        # quantisation-only arm handed a pruning target, for instance. get_compressor() can never
        # produce that pairing, but a hand-built one can.
        if self.compression_config.method is not self.method:
            raise CompressionError(
                f"{self.name} was built from a config whose method is "
                f"{self.compression_config.method.value!r}. The budget is derived from the config, "
                "so a mismatched pair would compress to the wrong target."
            )
        selection = select_compressible_modules(
            model,
            target_modules=self.compression_config.pruning.target_modules,
            exclude_patterns=self.compression_config.pruning.exclude_patterns,
        )
        self.module_names = selection.names
        LOGGER.info(
            "%s: %d target module(s), %d targeted parameters",
            self.name,
            selection.count,
            selection.total_parameters,
        )
        return model

    def apply(self, model: nn.Module) -> nn.Module:
        """Run the layerwise loop -- the whole algorithm for this arm.

        Args:
            model: The prepared model. Modified in place.

        Returns:
            The compressed model, still in fake-quantised FP32 storage.

        Raises:
            CompressionError: If block offload is requested but no CUDA device is available.
        """
        offload = self.compression_config.reconstruction.offload_blocks
        device: str | None = None
        if offload:
            from scale_aware_compression.hardware import cuda_available

            if not cuda_available():
                raise CompressionError(
                    f"{self.name}: compression.reconstruction.offload_blocks is set but no CUDA "
                    "device is available. Offloading blocks to the host from the host moves "
                    "nothing; unset the flag to run on CPU."
                )
            device = Device.CUDA.value

        self.report = compress_model_layerwise(
            model,
            self.calibration_batches,
            self.plan,
            arm=self.arm,
            module_names=self.module_names,
            calibration_fingerprint=self.calibration_fingerprint,
            device=device,
            offload_blocks=offload,
        )
        return model

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """No-op. The core method does no fine-tuning (§3.1).

        Kept because the ABC declares it and because the optional short-fine-tune ablation would
        live here. Enabling ``compression.recovery`` does not silently make this arm train.

        Args:
            model: The compressed model.
            tokenizer: Unused.

        Returns:
            The model, unchanged.
        """
        LOGGER.debug("%s: recovery is a no-op under layerwise reconstruction", self.name)
        return model

    def convert(self, model: nn.Module) -> nn.Module:
        """Pack the compressed weights into real low-bit storage.

        Until this runs the model is fake-quantised: numerically correct and exactly as large as the
        dense one. Every size figure in the study depends on this stage having happened, which is
        why ``is_converted`` is reported alongside them.

        Args:
            model: The compressed model.

        Returns:
            The model with its targeted layers replaced by packed equivalents.
        """
        plan = self.plan
        if not plan.quantises:
            # The pruning-only arm stays FP32 by design: it is the arm whose latency *can* be
            # measured natively, and packing it would remove that (see D1, and RQ4).
            LOGGER.info("%s: no quantisation, leaving weights in FP32", self.name)
            self.is_converted = True
            return model

        # Pass the grids the driver actually solved onto. Letting conversion refit them would
        # quantise a second time, so the artefact measured for size and reloaded for verification
        # could differ from the one evaluated for quality.
        self.conversion_statistics = convert_model_to_packed(
            model,
            self.module_names,
            bits=plan.bits,
            granularity=plan.granularity,
            group_size=plan.group_size,
            grids_by_module=self.report.grids_by_module if self.report else None,
        )
        self.is_converted = True
        LOGGER.info(
            "%s: packed %d module(s) at %d bits, %.2fx smaller than FP32 weights",
            self.name,
            self.conversion_statistics["num_converted_modules"],
            plan.bits,
            self.conversion_statistics["weight_compression_ratio"],
        )
        return model

    def save(self, model: nn.Module, path: str | Path) -> Path:
        """Persist the artefact **and** the manifest needed to load it back.

        The base class writes the weights. Weights alone are not an independently loadable artefact
        for a packed model: a model rebuilt from the same architecture config has plain ``nn.Linear``
        everywhere and no way to know which modules should be packed, so the state dict will not fit.
        §4.8 requires the checkpoint reload from disk on its own, which needs the manifest.

        Args:
            model: The converted model.
            path: Destination directory.

        Returns:
            The directory written to.
        """
        from scale_aware_compression.compression.packed import packed_linear_class
        from scale_aware_compression.compression.reload import write_manifest

        destination = super().save(model, path)

        plan = self.plan
        if not plan.quantises:
            # Nothing was packed, so the artefact is an ordinary FP32 checkpoint and needs no
            # manifest to be loadable.
            return destination

        packed_class = packed_linear_class()
        shapes: dict[str, tuple[int, int]] = {}
        packed_names: list[str] = []
        for name in self.module_names:
            module = model.get_submodule(name)
            if isinstance(module, packed_class):
                packed_names.append(name)
                shapes[name] = (module.out_features, module.in_features)

        write_manifest(
            Path(destination),
            module_names=packed_names,
            bits=plan.bits,
            granularity=plan.granularity,
            group_size=plan.group_size,
            shapes=shapes,
            target_sparsity=plan.sparsity,
            method=self.method.value,
        )
        return destination

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Describe what was achieved, next to what was requested.

        Returns:
            A serialisable mapping. Includes ``is_converted`` and the per-layer reconstruction
            losses §7.2 asks for (record field A9).
        """
        plan = self.plan
        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "arm": self.arm,
            "solver": plan.solver.value,
            "local_steps_per_call": plan.local_steps,
            "joint_iterations": plan.joint_iterations,
            "damping": plan.damping,
            "scale_search": plan.scale_search,
            "keep_benefit_saliency": plan.keep_benefit_saliency,
            "num_target_modules": len(self.module_names),
            "calibration_batches": len(self.calibration_batches),
            "calibration_fingerprint": self.calibration_fingerprint,
            "is_converted": self.is_converted,
        }
        if self.report is not None:
            statistics["layerwise"] = self.report.to_dict()
            # `measured_sparsity` was the numeric zero fraction, which conflates pruned weights with
            # survivors quantisation rounded away -- it overstates the pruning applied by ~1.8
            # percentage points at W4. The budget is defined on the mask, so that is what is reported
            # against the target; the conflated figure keeps its own clearly named key.
            statistics["measured_sparsity"] = self.report.mask_sparsity
            statistics["numeric_zero_fraction"] = self.report.realised_sparsity
            statistics["zero_code_fraction"] = self.report.zero_code_fraction
            statistics["accepted_joint_updates"] = self.report.accepted_joint_updates
            statistics["rejected_joint_updates"] = self.report.rejected_joint_updates
            statistics["targeted_parameters"] = self.report.targeted_parameters
            statistics["total_local_steps"] = self.report.total_local_steps
        if self.conversion_statistics:
            statistics["conversion"] = self.conversion_statistics
        if model is not None:
            statistics["model_sparsity"] = measure_sparsity(model)
        return statistics


class PruningArm(LayerwiseArm):
    """Pruning only: mask, then reconstruct the survivors. Precision untouched (§3.3).

    The one arm whose weights stay FP32, which makes it the arm that answers RQ4 -- whether sparsity
    produces a real CPU speedup -- since it can be benchmarked on the native dense kernel.
    """

    arm = "pruning"
    method = CompressionMethod.PRUNING
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.PRUNED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.PRUNED


class QuantisationArm(LayerwiseArm):
    """Quantisation only: fit the grid, then reconstruct. Isolates precision damage (§3.4)."""

    arm = "quantisation"
    method = CompressionMethod.QUANTISATION
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.CALIBRATED,
        CompressionStage.QUANTISED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.QUANTISED


class SequentialArm(LayerwiseArm):
    """Sequential P->Q: mask, reconstruct, quantise, reconstruct again (§3.5).

    The second reconstruction exists specifically so this arm's local-step total can equal the joint
    arm's. Without it the joint arm would receive strictly more optimisation, and §3.11 would be
    violated by construction.
    """

    arm = "sequential"
    method = CompressionMethod.SEQUENTIAL
    pipeline_stages = SEQUENTIAL_STAGES
    apply_stage = CompressionStage.PRUNED


class SequentialQPArm(LayerwiseArm):
    """Sequential Q->P: quantise first, then mask (§3.6).

    The reverse-order ablation, run at one representative budget so the joint arm is not compared
    only against a weak ordering. Its mask *does* see the grid; what disqualifies it from being joint
    is that the scales are never revisited after the mask moves.
    """

    arm = "sequential_qp"
    method = CompressionMethod.SEQUENTIAL_QP
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.QUANTISED,
        CompressionStage.PRUNED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.QUANTISED


class JointArm(LayerwiseArm):
    """Joint: alternate mask selection, scale re-estimation and reconstruction (§3.7).

    Each outer iteration rescores the mask against the current quantised weights and refits the grid
    for the resulting mask, so the two decisions inform each other rather than being taken in
    sequence. That is the §3.8 requirement, and ``tests/test_layerwise.py`` fails if this degenerates
    into prune-then-PTQ.
    """

    arm = "joint"
    method = CompressionMethod.JOINT
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.FAKE_QUANTISATION_PREPARED,
        CompressionStage.GRADUAL_PRUNING,
        CompressionStage.JOINTLY_FINE_TUNED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.JOINTLY_FINE_TUNED
