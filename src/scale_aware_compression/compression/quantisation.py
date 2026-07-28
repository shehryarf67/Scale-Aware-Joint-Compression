"""Quantisation-only arm.

Pipeline::

    dense model -> calibration -> quantisation -> conversion

Post-training quantisation, so no recovery fine-tuning by default: the point of this arm is
to isolate the effect of reduced precision alone, as a reference for the sequential and joint
arms.

Status: placeholder. Stage methods raise :class:`NotImplementedError`;
:meth:`Quantiser.report_statistics` is implemented.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import Compressor
from scale_aware_compression.constants import FP32_BITS, CompressionMethod, CompressionStage
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import count_parameters, theoretical_size_bytes

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class Quantiser(Compressor):
    """Post-training weight quantisation with calibration.

    Two representations are involved and the distinction matters for every number this study
    reports:

    * **fake quantisation** -- weights are rounded to the quantisation grid but stored as
      FP32. Differentiable, so it is what training uses, but it is *not* smaller or faster.
    * **real quantisation** -- weights stored at the target bit width with integer kernels.
      Produced by :meth:`convert`, and the only representation the CPU benchmark may measure.
    """

    method = CompressionMethod.QUANTISATION
    pipeline_stages = (
        CompressionStage.DENSE,
        CompressionStage.CALIBRATED,
        CompressionStage.QUANTISED,
        CompressionStage.CONVERTED,
    )
    apply_stage = CompressionStage.QUANTISED
    recover_stage = CompressionStage.RECOVERED

    def __init__(self, config: Any) -> None:
        """Initialise the arm with empty calibration state."""
        super().__init__(config)
        self.module_names: list[str] = []
        self.calibration_batches: int = 0
        self.is_converted: bool = False

    def prepare(self, model: nn.Module) -> nn.Module:
        """Select modules to quantise and insert observers.

        Args:
            model: The dense model.

        Returns:
            The model with observers attached.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): select modules via select_compressible_modules() with
        # config.compression.quantisation.exclude_patterns, then attach observers
        # (torch.ao.quantization.MinMaxObserver / PerChannelMinMaxObserver, chosen from
        # config.observer and config.granularity). Keep the observer set on self so
        # report_statistics can list which modules were actually instrumented.
        raise NotImplementedError(
            "Quantiser.prepare is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def calibrate(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Run calibration data through the model to populate the observers.

        May run on GPU. Uses the *same* calibration set as the sequential and joint arms:
        different calibration data would be an uncontrolled variable in the comparison.

        Args:
            model: The prepared model with observers attached.
            tokenizer: Tokeniser for building the calibration batches.

        Returns:
            The calibrated model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): draw config.compression.quantisation.calibration_samples
        # sequences from data.calibration.load_calibration_set() under torch.inference_mode()
        # and run forward passes to fill the observers. Record the sample count and the
        # calibration set's fingerprint in the stage record, so a rerun with different
        # calibration data is detectable in the results table.
        raise NotImplementedError(
            "Quantiser.calibrate is not implemented yet; see the TODO in "
            "compression/quantisation.py"
        )

    def apply(self, model: nn.Module) -> nn.Module:
        """Compute quantisation parameters and apply fake quantisation.

        Args:
            model: The calibrated model.

        Returns:
            The fake-quantised model, still FP32 in storage.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): derive scale and zero point from each observer, then round
        # weights onto the grid in place while keeping FP32 storage. For bits < 8, implement
        # the grid directly rather than relying on torch.ao, which only supports 8-bit
        # integer paths on CPU.
        raise NotImplementedError(
            "Quantiser.apply is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Optional quantisation-aware fine-tuning.

        Disabled by default for this arm: leaving it off keeps "quantisation only" a
        post-training method, which is how it is normally deployed. Enable it via
        ``compression.recovery.enabled`` only if the sequential arm's recovery budget is
        matched, or the comparison stops being about compression method.

        Args:
            model: The fake-quantised model.
            tokenizer: Tokeniser for the recovery data loader.

        Returns:
            The fine-tuned model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): if enabled, run training.recovery.run_recovery() with a
        # straight-through estimator on the fake-quantisation nodes.
        raise NotImplementedError(
            "Quantiser.recover is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def convert(self, model: nn.Module) -> nn.Module:
        """Convert fake quantisation into real low-precision storage.

        This is the stage that makes the size and latency measurements meaningful. Skipping it
        yields a model that is numerically quantised but still FP32 on disk and in memory.

        Args:
            model: The fake-quantised model.

        Returns:
            A CPU-deployable, really-quantised model.

        Raises:
            NotImplementedError: Always, in the current scaffold.
        """
        # TODO(quantisation): for 8-bit, use torch.ao.quantization.convert with the backend
        # from config.backend. On the pinned torch the only supported engine is 'onednn'; the
        # familiar 'x86' / 'fbgemm' / 'qnnpack' names raise. For sub-8-bit, pack weights into
        # int8 storage with the scales kept alongside and swap in a custom linear module.
        # Set self.is_converted = True, and verify the checkpoint shrank as expected: a
        # conversion that silently no-ops is the failure mode this whole arm is exposed to.
        raise NotImplementedError(
            "Quantiser.convert is not implemented yet; see the TODO in compression/quantisation.py"
        )

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Report the bit width, scheme, calibration size, and theoretical size.

        Args:
            model: The model to measure. When ``None``, only configuration-derived fields are
                returned.

        Returns:
            A serialisable mapping including ``is_converted``, which distinguishes a genuinely
            quantised artefact from a fake-quantised FP32 one.
        """
        quantisation = self.compression_config.quantisation
        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "bits": quantisation.bits,
            "scheme": quantisation.scheme.value,
            "granularity": quantisation.granularity.value,
            "group_size": quantisation.group_size,
            "backend": quantisation.backend,
            "observer": quantisation.observer,
            "calibration_samples": quantisation.calibration_samples,
            "calibration_batches_seen": self.calibration_batches,
            "quantise_activations": quantisation.quantise_activations,
            "activation_bits": quantisation.activation_bits
            if quantisation.quantise_activations
            else None,
            "num_quantised_modules": len(self.module_names),
            "is_converted": self.is_converted,
        }
        if model is not None:
            parameters = count_parameters(model)
            statistics["measured_total_parameters"] = parameters
            statistics["theoretical_size_bytes"] = theoretical_size_bytes(
                parameters, quantisation.bits
            )
            statistics["theoretical_size_bytes_fp32"] = theoretical_size_bytes(
                parameters, FP32_BITS
            )
        return statistics
