"""Sequential arm: prune, recover, then quantise.

This is the baseline pipeline the joint arm is measured against, and it is the standard
practice in the literature. Its stages are explicit and asserted::

    dense model
        -> pruning
        -> recovery
        -> quantisation
        -> conversion

The ordering matters, and it is worth being precise about how. Pruning first and quantising second
means quantisation is calibrated on the altered post-pruning distribution, including the increased
mass at zero and the remaining non-zero weights. Whether that distribution is *easier* to quantise
is an empirical question and not assumed here: magnitude pruning removes the smallest-magnitude
weights, so the surviving values are not necessarily confined to a narrower range than the dense
ones, and the observed min/max may be unchanged. What the ordering definitely cannot do is the
converse -- the pruning decision is made with no knowledge of where the quantisation grid points
will land. Closing that gap is what the joint arm attempts, so the comparison between the two
isolates one design choice.

Status: placeholder. Stages delegate to :class:`Pruner` and :class:`Quantiser`, which are
themselves placeholders; :meth:`SequentialCompressor.report_statistics` is implemented.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from scale_aware_compression.compression.base import Compressor
from scale_aware_compression.compression.pruning import Pruner
from scale_aware_compression.compression.quantisation import Quantiser
from scale_aware_compression.constants import SEQUENTIAL_STAGES, CompressionMethod, CompressionStage
from scale_aware_compression.logging_utils import get_logger
from scale_aware_compression.metrics.compression import measure_sparsity

if TYPE_CHECKING:  # pragma: no cover - typing only
    from torch import nn
    from transformers import PreTrainedTokenizerBase

LOGGER = get_logger(__name__)


class SequentialCompressor(Compressor):
    """Pruning and quantisation applied one after the other, with recovery in between.

    Composition rather than reimplementation: the pruning and quantisation logic lives in
    :class:`Pruner` and :class:`Quantiser`, so the sequential arm cannot drift away from the
    single-method arms it is supposed to be the composition of.
    """

    method = CompressionMethod.SEQUENTIAL
    pipeline_stages = SEQUENTIAL_STAGES
    apply_stage = CompressionStage.PRUNED
    recover_stage = CompressionStage.RECOVERED

    def __init__(self, config: Any) -> None:
        """Construct the pruning and quantisation sub-compressors."""
        super().__init__(config)
        self.pruner = Pruner(config)
        self.quantiser = Quantiser(config)

    def prepare(self, model: nn.Module) -> nn.Module:
        """Prepare pruning only.

        Quantisation preparation is deferred to :meth:`convert`. Inserting observers before pruning
        would calibrate them against the dense weight distribution, which is a different pipeline
        from the one this arm is defined to be.

        Args:
            model: The dense model.

        Returns:
            The prepared model.

        Raises:
            NotImplementedError: Until :meth:`Pruner.prepare` is implemented.
        """
        # TODO(sequential): delegate to self.pruner.prepare(model). Do not touch the quantiser
        # here; see the docstring above for why the ordering is load-bearing.
        return self.pruner.prepare(model)

    def apply(self, model: nn.Module) -> nn.Module:
        """Stage 1: prune.

        Args:
            model: The prepared model.

        Returns:
            The pruned model.

        Raises:
            NotImplementedError: Until :meth:`Pruner.apply` is implemented.
        """
        return self.pruner.apply(model)

    def recover(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> nn.Module:
        """Stage 2: recover the quality lost to pruning. May run on GPU.

        Args:
            model: The pruned model.
            tokenizer: Tokeniser for the recovery data loader.

        Returns:
            The recovered model.

        Raises:
            NotImplementedError: Until :meth:`Pruner.recover` is implemented.
        """
        # TODO(sequential): delegate to self.pruner.recover(), then copy the pruner's stage
        # records onto self.stage_records so the total optimiser-step count for this arm is
        # complete. Matching that total against the joint arm is what keeps the comparison
        # about method rather than budget.
        return self.pruner.recover(model, tokenizer)

    def convert(self, model: nn.Module) -> nn.Module:
        """Stages 3 and 4: quantise the recovered sparse model, then convert it.

        Args:
            model: The recovered, pruned model.

        Returns:
            The deployable artefact.

        Raises:
            NotImplementedError: Until the quantiser stages are implemented.
        """
        # TODO(sequential): run, in order,
        #   1. self.quantiser.prepare(model)   -- observers on the *sparse* weights
        #   2. self.quantiser.calibrate(model) -- same calibration set as every other arm
        #   3. self.quantiser.apply(model)     -- fake quantisation
        #   4. self.quantiser.convert(model)   -- real low precision
        # then fold the pruning masks in. Order matters at the end too: with an asymmetric scheme
        # zero need not be an exact grid point, so quantising after folding can map a pruned zero
        # onto a non-zero value and silently lower the realised sparsity. Verify measured sparsity
        # after conversion either way rather than assuming the order was sufficient.
        started = time.perf_counter()
        prepared = self.quantiser.prepare(model)
        calibrated = self.quantiser.calibrate(prepared)
        quantised = self.quantiser.apply(calibrated)
        self.record_stage(CompressionStage.QUANTISED, time.perf_counter() - started)
        return self.quantiser.convert(quantised)

    def report_statistics(self, model: nn.Module | None = None) -> dict[str, Any]:
        """Report both sub-arms' statistics under one record.

        Args:
            model: The model to measure. When ``None``, only configuration-derived fields are
                returned.

        Returns:
            A serialisable mapping with nested ``pruning`` and ``quantisation`` sections plus
            the combined budget, so a sequential row can be matched to a joint row.
        """
        statistics: dict[str, Any] = {
            **self.base_statistics(),
            "pipeline": [stage.value for stage in SEQUENTIAL_STAGES],
            "pruning": self.pruner.report_statistics(None),
            "quantisation": self.quantiser.report_statistics(None),
        }
        if model is not None:
            statistics.update(
                {f"measured_{key}": value for key, value in measure_sparsity(model).items()}
            )
        return statistics
