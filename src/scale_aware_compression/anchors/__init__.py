"""Independent reference implementations used to check our own, per Amendment A1 §5.5.

These exist because nothing else in this project can tell a correct pipeline from a quietly broken
one. Four separate faults have already been found by review rather than by any test
(`findings_log.md` B-14, B-17, B-22, B-23), and every one of them produced plausible output. An
anchor is the only check that does not share our assumptions.

The rule for anything in this package: **it must not import the code it validates.** A reference
that calls our saliency function only proves the call succeeded. Each module here re-derives its
quantity from the published definition, by a deliberately different route, and the comparison is
what carries the information.

Anchors are diagnostics, never a source of reported results. They produce agreement statistics and
alarm thresholds, not perplexities.
"""

from __future__ import annotations

from scale_aware_compression.anchors.exact_reconstruction import (
    ExactReconstructionError,
    ExactReconstructionReport,
    RowComparison,
    compare_row,
    exact_masked_row_optimum,
    row_objective,
)
from scale_aware_compression.anchors.wanda import (
    AnchorError,
    ColumnNormComparison,
    MaskComparison,
    WandaAnchorReport,
    compare_column_norms,
    compare_masks,
    independent_column_norms,
    independent_wanda_mask,
)

__all__ = [
    "AnchorError",
    "ColumnNormComparison",
    "ExactReconstructionError",
    "ExactReconstructionReport",
    "MaskComparison",
    "RowComparison",
    "WandaAnchorReport",
    "compare_column_norms",
    "compare_masks",
    "compare_row",
    "exact_masked_row_optimum",
    "independent_column_norms",
    "independent_wanda_mask",
    "row_objective",
]
