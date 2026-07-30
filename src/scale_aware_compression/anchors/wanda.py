"""An independent Wanda reference, for checking our mask against one we did not write.

Amendment A1 §5.5(a). Our saliency rule *is* the Wanda criterion,

    S_ij = |W_ij| * ||X_j||_2

selected per output row. With the criterion, comparison group, model, activations, module coverage
and target sparsity all matched, an independent implementation must produce **the same mask**. Any
divergence beyond deterministic tie-breaking means one of the two differs in how it collects
activations, groups columns, orders modules, or selects modules -- and we have shipped bugs in three
of those four (B-09 comparison group, B-19 dependency grouping, B-07 module selection).

**Why this is worth doing even though we wrote both sides.** The two paths differ where it matters:

* Our pipeline derives ``||X_j||_2`` from the streamed Gram matrix, as ``sqrt(diag(X^T X))``. This
  module accumulates the column sums of squares **directly**, never forming a Gram matrix. A
  streaming or accumulation fault shows up as disagreement; it cannot cancel.
* Our mask construction computes an exact prune count and removes exactly that many entries. This
  module sorts and takes the top-k per row. Two different routes to the same set.

So this validates the criterion, the norm accumulation and the selection. It does **not** validate
the reconstruction -- that needs the SparseGPT anchor, A1 §5.5(b).

**One methodological difference that is not a bug.** Published Wanda captures activations from the
dense model. Our driver captures through the already-compressed prefix (§3, and the B-19 fix), so
for any layer after the first group the two see genuinely different inputs *by design*. The
comparison here therefore runs both sides on **dense-model activations**, which isolates the
criterion from the propagation scheme. The propagation scheme has its own tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from scale_aware_compression.logging_utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

LOGGER = get_logger(__name__)


class AnchorError(RuntimeError):
    """Raised when an anchor cannot be computed, or is asked to compare mismatched shapes."""


@dataclass(frozen=True, slots=True)
class ColumnNormComparison:
    """Agreement between our streamed column norms and directly accumulated ones."""

    module_name: str
    in_features: int
    max_absolute_difference: float
    max_relative_difference: float
    ours_mean: float
    reference_mean: float
    dead_columns_ours: int
    dead_columns_reference: int

    @property
    def agrees(self) -> bool:
        """True when the two norms match to the tolerance float32 accumulation allows.

        The threshold is on the *relative* difference, because the norms scale with the number of
        calibration rows and an absolute tolerance would mean different things at different window
        sizes.

        ``1e-3`` rather than something tighter, and the slack is real rather than defensive: our Gram
        accumulates in **float32** while this reference sums in **float64**, over ~65k rows at the
        screening window. Blocked summation of that many positive terms in float32 carries relative
        error on the order of ``sqrt(n) * eps``, which lands near 3e-5 but is not guaranteed to. A
        tighter bound would fail on arithmetic rather than on a defect.

        The exact observed difference is reported either way, so the margin is visible and not just
        asserted. Note this is the *diagnostic* half of the anchor -- the strict test is mask
        agreement, and float32 noise this small only flips a top-k decision where scores are
        near-tied, which the tie accounting handles separately.
        """
        return self.max_relative_difference < 1e-3

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-serialisable mapping."""
        return {
            "module_name": self.module_name,
            "in_features": self.in_features,
            "max_absolute_difference": self.max_absolute_difference,
            "max_relative_difference": self.max_relative_difference,
            "ours_mean": self.ours_mean,
            "reference_mean": self.reference_mean,
            "dead_columns_ours": self.dead_columns_ours,
            "dead_columns_reference": self.dead_columns_reference,
            "agrees": self.agrees,
        }


@dataclass(frozen=True, slots=True)
class MaskComparison:
    """Agreement between our mask and the independent Wanda mask for one module."""

    module_name: str
    total_weights: int
    differing_positions: int
    ours_pruned: int
    reference_pruned: int
    tied_at_threshold: int
    rows_differing: int

    @property
    def overlap(self) -> float:
        """Fraction of positions where the two masks agree."""
        if self.total_weights == 0:
            return 1.0
        return 1.0 - self.differing_positions / self.total_weights

    @property
    def explained_by_ties(self) -> bool:
        """True when every disagreement sits on a tied score.

        Ties are expected rather than pathological: activation weighting makes a dead input column
        score exactly zero for every row, so which of the zeros gets pruned is arbitrary and the two
        implementations may legitimately choose differently. A disagreement *not* on a tie is a real
        divergence.
        """
        return self.differing_positions <= self.tied_at_threshold

    def to_dict(self) -> dict[str, Any]:
        """Return the comparison as a JSON-serialisable mapping."""
        return {
            "module_name": self.module_name,
            "total_weights": self.total_weights,
            "differing_positions": self.differing_positions,
            "ours_pruned": self.ours_pruned,
            "reference_pruned": self.reference_pruned,
            "tied_at_threshold": self.tied_at_threshold,
            "rows_differing": self.rows_differing,
            "overlap": self.overlap,
            "explained_by_ties": self.explained_by_ties,
        }


@dataclass(slots=True)
class WandaAnchorReport:
    """Aggregate verdict across every compared module."""

    target_sparsity: float
    norms: list[ColumnNormComparison] = field(default_factory=list)
    masks: list[MaskComparison] = field(default_factory=list)
    precision_divergence: list[MaskComparison] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def modules_compared(self) -> int:
        """How many modules were compared."""
        return len(self.masks)

    @property
    def total_differing_positions(self) -> int:
        """Total mask disagreements across all modules."""
        return sum(item.differing_positions for item in self.masks)

    @property
    def worst_overlap(self) -> float:
        """Lowest per-module mask overlap, or 1.0 when nothing was compared."""
        return min((item.overlap for item in self.masks), default=1.0)

    @property
    def norms_agree(self) -> bool:
        """True when every module's column norms agree."""
        return all(item.agrees for item in self.norms)

    @property
    def masks_agree(self) -> bool:
        """True when the two selectors agree exactly on identical norms.

        ``self.masks`` holds the **matched-norm** comparison: both sides fed the same float64 norms,
        so the only thing under test is the selection. Exact agreement is the right bar there, and
        ties cannot excuse anything, because identical inputs must give identical output.
        """
        return bool(self.masks) and all(item.differing_positions == 0 for item in self.masks)

    @property
    def precision_sensitive_positions(self) -> int:
        """Positions where float32 versus float64 norms flip the selection.

        Informational, not a defect. Two weights whose scores tie in float64 sit within a couple of
        ULPs in float32, and which one survives is then arbitrary. Reported because the count is a
        useful measure of how close the ranking runs to its own precision floor -- a large number
        would mean the mask is not reproducible across arithmetic, which *would* matter.
        """
        return sum(item.differing_positions for item in self.precision_divergence)

    @property
    def passes(self) -> bool:
        """The anchor's verdict.

        Deliberately strict, and it can afford to be: this compares two implementations of the *same*
        formula, so on identical inputs exact agreement is the only acceptable result.

        ``precision_sensitive_positions`` deliberately does **not** gate the verdict. Failing the
        anchor because float32 and float64 break a tie differently would be failing it on arithmetic
        rather than on a defect -- and it would train us to ignore the one signal the anchor exists
        to produce.
        """
        return bool(self.masks) and self.norms_agree and self.masks_agree

    def to_dict(self) -> dict[str, Any]:
        """Return the whole report as a JSON-serialisable mapping."""
        return {
            "anchor": "wanda_mask_agreement",
            "target_sparsity": self.target_sparsity,
            "modules_compared": self.modules_compared,
            "passes": self.passes,
            "norms_agree": self.norms_agree,
            "masks_agree": self.masks_agree,
            "total_differing_positions": self.total_differing_positions,
            "worst_overlap": self.worst_overlap,
            "precision_sensitive_positions": self.precision_sensitive_positions,
            "column_norms": [item.to_dict() for item in self.norms],
            "masks": [item.to_dict() for item in self.masks],
            "precision_divergence": [item.to_dict() for item in self.precision_divergence],
            "notes": list(self.notes),
        }

    def summary_lines(self) -> list[str]:
        """Return a short human-readable verdict, worst modules first."""
        lines = [
            f"Wanda mask-agreement anchor at {self.target_sparsity:.0%} sparsity",
            f"  modules compared      : {self.modules_compared}",
            f"  column norms agree    : {self.norms_agree}",
            f"  masks agree (matched) : {self.masks_agree}",
            f"  differing positions   : {self.total_differing_positions}",
            f"  worst module overlap  : {self.worst_overlap:.6f}",
            f"  precision-sensitive   : {self.precision_sensitive_positions} "
            f"(float32 vs float64 norms; informational)",
            f"  VERDICT               : {'PASS' if self.passes else 'INVESTIGATE'}",
        ]
        offenders = sorted(self.masks, key=lambda item: item.overlap)[:5]
        for item in offenders:
            if item.differing_positions:
                lines.append(
                    f"    {item.module_name}: {item.differing_positions} differing, "
                    f"overlap {item.overlap:.6f}, "
                    f"{'tie-explained' if item.explained_by_ties else 'NOT tie-explained'}"
                )
        for note in self.notes:
            lines.append(f"  note: {note}")
        return lines


class _DirectColumnNormAccumulator:
    """Accumulate ``sum_n x_nj^2`` per input column, without forming a Gram matrix.

    This is the point of the whole module. Our production path takes the square root of the Gram
    diagonal, which means a fault in the Gram accumulation propagates into the saliency unnoticed.
    Summing squares directly cannot share that fault.
    """

    def __init__(self, in_features: int) -> None:
        """Initialise an empty accumulator.

        Args:
            in_features: Number of input columns to track.
        """
        import torch

        self.in_features = int(in_features)
        self._sum_of_squares = torch.zeros(self.in_features, dtype=torch.float64)
        self.rows = 0

    def update(self, activations: torch.Tensor) -> None:
        """Add one batch of activations.

        Args:
            activations: Shape ``(..., in_features)``. Leading dimensions are flattened, matching
                how a linear layer sees its input.

        Raises:
            AnchorError: If the last dimension does not match ``in_features``.
        """
        if activations.shape[-1] != self.in_features:
            raise AnchorError(
                f"expected last dimension {self.in_features}, got {activations.shape[-1]}"
            )
        import torch

        flat = activations.reshape(-1, self.in_features).detach()
        # Reduce on whatever device the activations live on -- capture may run on GPU -- then move
        # only the length-in_features result into the float64 CPU accumulator. Keeping the buffer on
        # CPU sidesteps a device mismatch, and float64 arithmetic on a consumer GPU is slow enough
        # to matter over 48 modules.
        self._sum_of_squares += (flat * flat).sum(dim=0).to(device="cpu", dtype=torch.float64)
        self.rows += flat.shape[0]

    def norms(self) -> torch.Tensor:
        """Return ``||X_j||_2`` per column as float32.

        Raises:
            AnchorError: If no activations were accumulated.
        """
        if self.rows == 0:
            raise AnchorError("no activations were accumulated; the forward pass captured nothing")
        import torch

        return self._sum_of_squares.clamp_min(0.0).sqrt().to(torch.float32)


def independent_column_norms(
    model: nn.Module,
    module_names: list[str],
    batches: list[dict[str, Any]],
    *,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Capture ``||X_j||_2`` for each named linear module by direct accumulation.

    Runs the **dense** model over the calibration batches once, with a forward pre-hook on each
    target module. Deliberately does not import or call our activation-capture code.

    Args:
        model: The model to run. Left unmodified; hooks are removed before returning.
        module_names: Dotted names of the linear modules to instrument.
        batches: Calibration batches, each a mapping with at least ``input_ids``.
        device: Device to run the forward passes on. GPU is allowed -- this is not a measurement.

    Returns:
        Column norms per module name.

    Raises:
        AnchorError: If a name does not resolve, or the forward pass captures nothing.
    """
    import torch

    accumulators: dict[str, _DirectColumnNormAccumulator] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                return
            accumulators[name].update(inputs[0])

        return hook

    for name in module_names:
        try:
            module = model.get_submodule(name)
        except AttributeError as error:
            raise AnchorError(f"{name!r} does not resolve on the model") from error
        in_features = getattr(module, "in_features", None)
        if in_features is None:
            raise AnchorError(f"{name!r} has no in_features; is it a linear layer?")
        accumulators[name] = _DirectColumnNormAccumulator(in_features)
        handles.append(module.register_forward_pre_hook(make_hook(name)))

    try:
        model.eval()
        with torch.no_grad():
            for batch in batches:
                inputs = {
                    key: value.to(device)
                    for key, value in batch.items()
                    if key in {"input_ids", "attention_mask"} and hasattr(value, "to")
                }
                if "input_ids" not in inputs:
                    raise AnchorError("calibration batch has no input_ids")
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    return {name: accumulator.norms() for name, accumulator in accumulators.items()}


def independent_wanda_mask(
    weight: torch.Tensor,
    column_norms: torch.Tensor,
    *,
    sparsity: float,
) -> torch.Tensor:
    """Build a Wanda keep-mask independently of our mask code.

    Per-output-row selection, by sorting each row's scores and keeping the top-k. Our production
    path instead computes an exact prune count and scatters that many ``False`` values -- a
    different route to what should be the same set.

    Args:
        weight: Shape ``(out_features, in_features)``.
        column_norms: Shape ``(in_features,)``.
        sparsity: Fraction to prune, in ``[0, 1)``.

    Returns:
        Boolean keep-mask, ``True`` meaning keep.

    Raises:
        AnchorError: On a shape mismatch or an out-of-range sparsity.
    """
    import torch

    if weight.ndim != 2:
        raise AnchorError(f"expected a 2-D weight, got shape {tuple(weight.shape)}")
    if column_norms.ndim != 1 or column_norms.shape[0] != weight.shape[1]:
        raise AnchorError(
            f"column_norms must have shape ({weight.shape[1]},), got {tuple(column_norms.shape)}"
        )
    if not 0.0 <= sparsity < 1.0:
        raise AnchorError(f"sparsity must be in [0, 1), got {sparsity}")

    scores = weight.abs().to(torch.float32) * column_norms.to(torch.float32).unsqueeze(0)
    out_features, in_features = scores.shape
    prune_per_row = int(round(sparsity * in_features))
    keep = torch.ones_like(scores, dtype=torch.bool)
    if prune_per_row == 0:
        return keep

    # Sort ascending and drop the lowest-scoring `prune_per_row` per row. `torch.argsort` is
    # deterministic but its tie order is unspecified, which is precisely why the comparison reports
    # ties separately rather than treating every mismatch as a fault.
    order = torch.argsort(scores, dim=1, stable=True)
    drop = order[:, :prune_per_row]
    keep.scatter_(1, drop, False)
    return keep


def compare_column_norms(
    module_name: str,
    ours: torch.Tensor,
    reference: torch.Tensor,
) -> ColumnNormComparison:
    """Compare our streamed column norms against directly accumulated ones.

    Args:
        module_name: Name of the module, for the report.
        ours: Norms from our pipeline, via ``sqrt(diag(H))``.
        reference: Norms from direct accumulation.

    Returns:
        The comparison.

    Raises:
        AnchorError: If the shapes differ.
    """
    import torch

    if ours.shape != reference.shape:
        raise AnchorError(
            f"{module_name}: shape mismatch, ours {tuple(ours.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    ours = ours.detach().to(torch.float64)
    reference = reference.detach().to(torch.float64)
    absolute = (ours - reference).abs()
    # Floor the denominator at a small fraction of the largest norm rather than at an absolute
    # epsilon. A column the calibration data barely excites has a norm near zero, and dividing by it
    # turns float32 noise into an enormous "relative error" that says nothing about correctness.
    # Columns that are exactly dead are not swept under the rug -- they are counted separately below,
    # and a disagreement about *which* columns are dead shows up there.
    floor = reference.abs().max() * 1e-6
    scale = torch.maximum(reference.abs(), floor.clamp_min(1e-12))
    return ColumnNormComparison(
        module_name=module_name,
        in_features=int(ours.shape[0]),
        max_absolute_difference=float(absolute.max()),
        max_relative_difference=float((absolute / scale).max()),
        ours_mean=float(ours.mean()),
        reference_mean=float(reference.mean()),
        dead_columns_ours=int((ours == 0).sum()),
        dead_columns_reference=int((reference == 0).sum()),
    )


def compare_masks(
    module_name: str,
    ours: torch.Tensor,
    reference: torch.Tensor,
    scores: torch.Tensor,
) -> MaskComparison:
    """Compare two keep-masks, separating real divergence from tie-breaking.

    A disagreement is attributed to a tie when the position's score equals another score that the
    other implementation chose instead -- approximated here by counting positions whose score ties
    with the row's threshold score. That approximation is deliberately generous to the *reference*:
    it can only ever excuse disagreements, so a reported "not tie-explained" is trustworthy.

    Args:
        module_name: Name of the module, for the report.
        ours: Our keep-mask.
        reference: The independent keep-mask.
        scores: The saliency scores both masks were built from, used to identify ties.

    Returns:
        The comparison.

    Raises:
        AnchorError: If the shapes differ.
    """
    import torch

    if ours.shape != reference.shape:
        raise AnchorError(
            f"{module_name}: shape mismatch, ours {tuple(ours.shape)} vs "
            f"reference {tuple(reference.shape)}"
        )
    differing = ours != reference
    differing_positions = int(differing.sum())

    tied = 0
    if differing_positions:
        # For each row with a disagreement, the threshold is the largest score our implementation
        # pruned. Any position whose score equals it could legitimately have gone either way.
        scores = scores.to(torch.float32)
        rows = differing.any(dim=1).nonzero(as_tuple=True)[0]
        for row in rows.tolist():
            row_scores = scores[row]
            pruned = ~ours[row]
            if not bool(pruned.any()):
                continue
            threshold = row_scores[pruned].max()
            tied += int(((row_scores == threshold) & differing[row]).sum())

    return MaskComparison(
        module_name=module_name,
        total_weights=int(ours.numel()),
        differing_positions=differing_positions,
        ours_pruned=int((~ours).sum()),
        reference_pruned=int((~reference).sum()),
        tied_at_threshold=tied,
        rows_differing=int(differing.any(dim=1).sum()),
    )
