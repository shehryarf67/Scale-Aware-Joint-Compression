"""Calibration activation statistics for layerwise reconstruction.

Research plan §3.1 chooses the layerwise objective

.. code-block:: text

    L_rec = || X W^T - X (M * Q_b(W))^T ||_F^2

so every arm needs two things per targeted layer: the Gram matrix ``H = X^T X`` that the
reconstruction solver inverts, and the per-column activation norms ``||X_j||_2`` that weight the
pruning saliency (§3.3).

Both come from one accumulator, because the column norms are the square root of ``H``'s diagonal:

.. code-block:: text

    H_jj = sum_n X_nj^2 = ||X_j||_2^2

so capturing ``H`` gives the saliency for free. That is the reason this module exists even though
the first solver (damped ALS, decision **D2**) barely needs the off-diagonal terms — the same
capture serves the later Hessian upgrade with no second pass over the calibration set.

Accumulation is **streaming**: activations arrive batch by batch and are folded into a fixed-size
``(in_features, in_features)`` buffer, so peak memory does not grow with the calibration set. The
largest layer in the sweep is Pythia-1B/1.4B ``mlp.dense_4h_to_h`` at ``in_features = 8192``, which
is 256 MiB in fp32 — one layer at a time, comfortably inside the 6 GiB GPU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)


class ActivationCaptureError(RuntimeError):
    """Raised when activation statistics are inconsistent or unusable."""


class ActivationStatistics:
    """Streaming accumulator for ``H = X^T X`` and the per-column activation norms.

    Feed it the *inputs* of a linear layer, one batch at a time. Inputs of any leading shape are
    accepted and flattened to ``(-1, in_features)``, which is what makes it work unchanged for
    ``(batch, sequence, features)`` transformer activations.

    Attributes:
        in_features: Width of the activation vectors this accumulator expects.
        dtype: Accumulation dtype. Defaults to fp32, matching the plan's memory estimate. Use
            fp64 when a very long calibration run makes drift a concern; it doubles the buffer.
    """

    def __init__(
        self,
        in_features: int,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Allocate the Gram buffer.

        Args:
            in_features: Width of the layer input.
            dtype: Accumulation dtype, default ``torch.float32``.
            device: Device to accumulate on. Capture may run on GPU (§4.6); only the final
                deployment measurements are CPU-only.

        Raises:
            ActivationCaptureError: If ``in_features`` is not positive.
        """
        import torch

        if in_features <= 0:
            raise ActivationCaptureError(f"in_features must be > 0, got {in_features}")

        self.in_features = in_features
        self.dtype = dtype or torch.float32
        self._gram = torch.zeros((in_features, in_features), dtype=self.dtype, device=device)
        self._num_rows = 0
        self._num_batches = 0

    @property
    def num_rows(self) -> int:
        """Activation vectors accumulated so far, counting every token separately."""
        return self._num_rows

    @property
    def num_batches(self) -> int:
        """Calls to :meth:`update`."""
        return self._num_batches

    @property
    def is_empty(self) -> bool:
        """Whether anything has been accumulated yet."""
        return self._num_rows == 0

    def update(self, activations: torch.Tensor) -> None:
        """Fold one batch of layer inputs into the accumulator.

        Args:
            activations: Layer input of shape ``(..., in_features)``. Leading dimensions are
                flattened, so ``(batch, sequence, features)`` works directly.

        Raises:
            ActivationCaptureError: If the trailing dimension does not match ``in_features``.
        """
        if activations.shape[-1] != self.in_features:
            raise ActivationCaptureError(
                f"activation width {activations.shape[-1]} does not match in_features "
                f"{self.in_features}"
            )

        flat = activations.reshape(-1, self.in_features).to(
            dtype=self.dtype, device=self._gram.device
        )
        # addmm_ rather than `+= flat.T @ flat`: it accumulates in place, so a long calibration
        # run does not allocate a fresh (in_features, in_features) temporary per batch.
        self._gram.addmm_(flat.t(), flat)
        self._num_rows += flat.shape[0]
        self._num_batches += 1

    def gram(self) -> torch.Tensor:
        """Return ``H = X^T X`` accumulated over every batch seen.

        Returns:
            Symmetric ``(in_features, in_features)`` tensor.

        Raises:
            ActivationCaptureError: If nothing has been accumulated.
        """
        self._require_data("gram")
        return self._gram

    def damped_gram(self, damping: float) -> torch.Tensor:
        """Return ``H + lambda * mean(diag(H)) * I``, the matrix the solver inverts.

        Damping is expressed **relative to the mean diagonal** rather than as an absolute value,
        so one configured number behaves consistently across layers whose activation scales differ
        by orders of magnitude — and across model sizes, which §3.11 requires of any
        hyperparameter not tuned per scale.

        Args:
            damping: Relative ridge coefficient, ``>= 0``.

        Returns:
            A new damped copy of the Gram matrix.

        Raises:
            ActivationCaptureError: If nothing has been accumulated, or damping is negative.
        """
        import torch

        self._require_data("damped_gram")
        if damping < 0:
            raise ActivationCaptureError(f"damping must be >= 0, got {damping}")

        gram = self._gram
        mean_diagonal = torch.diagonal(gram).mean()
        # A layer whose captured activations were all zero has no scale to key off; fall back to
        # a unit ridge so the solve stays defined instead of producing NaNs.
        if not bool(torch.isfinite(mean_diagonal)) or mean_diagonal <= 0:
            LOGGER.warning("Gram diagonal is non-positive; falling back to absolute damping")
            mean_diagonal = torch.ones((), dtype=gram.dtype, device=gram.device)

        ridge = damping * mean_diagonal
        return gram + torch.eye(self.in_features, dtype=gram.dtype, device=gram.device) * ridge

    def column_norms(self) -> torch.Tensor:
        """Return the per-column activation norms ``||X_j||_2``.

        Taken from ``sqrt(diag(H))`` rather than tracked separately: the two are the same
        quantity, and computing one from the other removes any chance of them disagreeing.

        Returns:
            Tensor of shape ``(in_features,)``, non-negative.

        Raises:
            ActivationCaptureError: If nothing has been accumulated.
        """
        import torch

        self._require_data("column_norms")
        # clamp before sqrt: fp32 accumulation of a near-zero column can land marginally
        # negative, and a NaN norm would silently poison every saliency score downstream.
        return torch.sqrt(torch.diagonal(self._gram).clamp_min(0.0))

    def report(self) -> dict[str, Any]:
        """Summarise the capture for the run record.

        Returns:
            Mapping with the width, how much data was seen, and the Gram scale. §4.8 requires
            the calibration set be identifiable after the fact; ``num_rows`` is what makes an
            accidental mismatch between arms visible.
        """
        import torch

        statistics: dict[str, Any] = {
            "in_features": self.in_features,
            "num_rows": self._num_rows,
            "num_batches": self._num_batches,
            "dtype": str(self.dtype),
        }
        if not self.is_empty:
            diagonal = torch.diagonal(self._gram)
            statistics["mean_gram_diagonal"] = float(diagonal.mean())
            statistics["max_gram_diagonal"] = float(diagonal.max())
            statistics["num_dead_columns"] = int((diagonal <= 0).sum())
        return statistics

    def _require_data(self, what: str) -> None:
        if self.is_empty:
            raise ActivationCaptureError(
                f"cannot compute {what}: no activations accumulated. Run the calibration "
                "forward passes before reading statistics."
            )


class LinearActivationCapture:
    """Forward hook that accumulates :class:`ActivationStatistics` for one ``nn.Linear``.

    Used as a context manager so the hook is always removed, including on an exception. A hook
    left installed keeps accumulating during evaluation, which would silently mix evaluation
    activations into a calibration statistic.

    Example:
        >>> import torch
        >>> from torch import nn
        >>> layer = nn.Linear(4, 3)
        >>> with LinearActivationCapture(layer) as capture:
        ...     _ = layer(torch.randn(2, 5, 4))
        >>> capture.statistics.num_rows
        10
    """

    def __init__(
        self,
        layer: nn.Linear,
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Attach to a linear layer without installing the hook yet.

        Args:
            layer: The layer whose *inputs* will be captured.
            dtype: Accumulation dtype for the statistics.
            device: Device to accumulate on; defaults to the layer's weight device.

        Raises:
            ActivationCaptureError: If the module has no usable ``in_features``.
        """
        in_features = getattr(layer, "in_features", None)
        if not isinstance(in_features, int):
            raise ActivationCaptureError(
                f"{type(layer).__name__} has no integer in_features; "
                "LinearActivationCapture only supports linear layers"
            )

        self.layer = layer
        self.statistics = ActivationStatistics(
            in_features,
            dtype=dtype,
            device=device if device is not None else layer.weight.device,
        )
        self._handle: Any | None = None

    def __enter__(self) -> LinearActivationCapture:
        """Install the forward pre-hook."""
        self._handle = self.layer.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, *exception: object) -> None:
        """Remove the hook."""
        self.remove()

    def remove(self) -> None:
        """Detach the hook if it is installed. Safe to call more than once."""
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, _module: nn.Module, inputs: tuple[Any, ...]) -> None:
        import torch

        if not inputs:
            return
        with torch.no_grad():
            self.statistics.update(inputs[0].detach())
