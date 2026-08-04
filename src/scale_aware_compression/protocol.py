"""Frozen protocol decisions, in machine-readable form.

Some decisions in this study are *frozen*: selected on the validation split, recorded before any
test evaluation, and not reopenable afterwards. Until now they lived only in prose in
``docs/protocol_freeze.md``, which means every consumer had to re-read a table and get it right --
and one did not. ``scripts/run_downstream.py`` mapped every sequential arm to P→Q, and
``configs/experiments/main_scale_sweep.yaml`` -- the **confirmatory** config -- listed ``sequential``
for every cell with no order resolution at all. At Pythia-1B's moderate budget, where Q→P is frozen,
both would have run P→Q: the *weaker* baseline, which inflates the joint gain. That is
[B-30](../../docs/findings_log.md) recurring, and it would have recurred inside the one run that
cannot be redone.

This module is the single source. The prose table remains authoritative for *why* each cell was
frozen; this is the authority for *what* the code does.
"""

from __future__ import annotations

from scale_aware_compression.constants import CompressionMethod


class ProtocolError(RuntimeError):
    """Raised when a frozen decision is needed and none has been recorded."""


FROZEN_SEQUENTIAL_ORDER: dict[tuple[str, str], CompressionMethod] = {
    # §3.6 and §6.1 require joint gain measured against best-of {P→Q, Q→P}, with the winning order
    # selected on validation and frozen per (model, budget) before any test evaluation.
    #
    # W4 -- P→Q at every scale, on evidence. Q→P reuses the dense-fitted scales without refitting,
    # which is nearly free at W8 and punishing at W4 where a coarse grid is badly matched to the
    # post-pruning distribution. Margins: +4.26 pp at 160M, +6.82 pp at 410M, +2.15 pp at 1B (3/3).
    ("pythia-160m", "aggressive"): CompressionMethod.SEQUENTIAL,
    ("pythia-410m", "aggressive"): CompressionMethod.SEQUENTIAL,
    ("pythia-1b", "aggressive"): CompressionMethod.SEQUENTIAL,
    # W8 at 160M and 410M -- the orders are indistinguishable (F-28: mean margin +0.18 pp, sd 0.19,
    # 4/5 favouring Q→P but the sign not consistent). The rule fixed in the config *before* that
    # measurement says: sign varies → freeze P→Q, the §3.6 pre-registered primary, and record the
    # choice as ARBITRARY rather than measured.
    ("pythia-160m", "moderate"): CompressionMethod.SEQUENTIAL,
    ("pythia-410m", "moderate"): CompressionMethod.SEQUENTIAL,
    # W8 at 1B -- Q→P, on evidence. THE ONE CELL THAT DIFFERS, and the reason this table has to be
    # machine-readable rather than assumed. The sign *was* consistent here (+0.12, +0.05, +0.13, 3/3),
    # so the same rule took its measured branch instead of its fallback. A1 §3 freezes per cell
    # precisely so this is permitted: same rule, different evidence.
    #
    # Two caveats travel with it: the margin is 0.10 pp, and three unanimous draws reach only
    # p = 0.25. It is on the control budget, and nothing in the headline depends on it -- but the
    # baseline is still the stronger of the two, and §6.1 requires the stronger one.
    ("pythia-1b", "moderate"): CompressionMethod.SEQUENTIAL_QP,
}
"""The frozen sequential order per (model, budget). Evidence: findings_log F-24, F-28, F-32."""

FROZEN_ORDER_EVIDENCE: dict[tuple[str, str], str] = {
    ("pythia-160m", "aggressive"): "F-24: P→Q by +4.26 pp",
    ("pythia-410m", "aggressive"): "F-25: P→Q by +6.82 pp",
    ("pythia-1b", "aggressive"): "F-32: P→Q by +2.15 pp, 3/3 draws",
    (
        "pythia-160m",
        "moderate",
    ): "F-28: indistinguishable over 5 draws, P→Q by pre-declared fallback",
    ("pythia-410m", "moderate"): "F-28: indistinguishable, P→Q by the same fallback",
    ("pythia-1b", "moderate"): "F-32: Q→P by +0.10 pp, 3/3 draws",
}
"""Why each cell is frozen as it is, for the record a run writes."""


def resolve_sequential_order(model_name: str, budget_label: str) -> CompressionMethod:
    """Return the frozen sequential order for one cell.

    Args:
        model_name: Registry short name, e.g. ``"pythia-1b"``.
        budget_label: Budget label, e.g. ``"aggressive"``.

    Returns:
        :attr:`CompressionMethod.SEQUENTIAL` for P→Q or
        :attr:`CompressionMethod.SEQUENTIAL_QP` for Q→P.

    Raises:
        ProtocolError: If no order has been frozen for this cell. **Refusing is the point.**
            Defaulting to P→Q is what produced the fault this module exists to prevent: it is the
            §3.6 primary, so it looks like a safe default, and at 1B moderate it is the weaker
            baseline and therefore flatters joint.
    """
    key = (model_name, budget_label)
    if key not in FROZEN_SEQUENTIAL_ORDER:
        raise ProtocolError(
            f"no sequential order is frozen for model={model_name!r} budget={budget_label!r}. "
            f"Frozen cells: {sorted(FROZEN_SEQUENTIAL_ORDER)}. §6.1 requires joint gain measured "
            "against best-of {P→Q, Q→P}; run both orders on validation and freeze the winner "
            "before evaluating this cell. Do not default to P→Q -- at pythia-1b/moderate that is "
            "the weaker baseline and would inflate the joint gain."
        )
    return FROZEN_SEQUENTIAL_ORDER[key]


def frozen_order_evidence(model_name: str, budget_label: str) -> str:
    """Return the recorded justification for a cell's frozen order, or a marker if absent."""
    return FROZEN_ORDER_EVIDENCE.get((model_name, budget_label), "no evidence recorded")
