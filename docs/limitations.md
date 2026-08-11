# Limitations

**Written 2026-08-10, deliberately before any interpretation of the results is drafted.** The order
matters: limitations written after a discussion section tend to be the ones that survive it. These
are written from the record while the result is fresh and nothing has been argued yet.

This is the paper's Limitations section in draft form. Every item states what the limitation is,
what it costs the claim, and — where relevant — what would fix it. Nothing here is hedging for its
own sake; each item changes what may be concluded.

Source of record: [F-37](findings_log.md#f-37) (the result), [F-38](findings_log.md#f-38) (the
mechanism diagnostics), [validity_threats.md](validity_threats.md) (the pre-registered threats).

---

## 1. Three scale points cannot establish a trend, and the trend is not tested

The study has **three** models. The research plan already says three points cannot fit a scaling
law. The confirmatory result makes this worse rather than better in two ways:

- **Two of the three coincide.** 160M and 410M differ by 0.08 pp at W4 (+1.0120 vs +0.9348). The
  entire movement in the trend is the 410M→1B step.
- **No statistical test compares the scales.** Every p-value reported is *within* a cell — a sign
  test over that cell's paired replicates. The differences *between* cells carry no test, and with
  three points none is available.

**Cost to the claim.** The paper may say the advantage *did not increase* with scale, and that the
observed direction was opposite. It may **not** say the decline is established, significant, or
monotone.

## 2. The scale axis is confounded with depth, and 1B is the shallow point

| Model | Blocks | Targeted parameters |
| --- | --- | --- |
| pythia-160m | 12 | 84,934,656 |
| pythia-410m | **24** | 301,989,888 |
| pythia-1b | **16** | 805,306,368 |

**Pythia-1B is shallower than Pythia-410M.** Depth runs 12 → 24 → 16 while width rises monotonically.
The method is layerwise: activations are captured through the already-compressed prefix, so
reconstruction error compounds with depth, and the block count is the number of opportunities for a
joint step to help.

**The one large movement in the trend — the drop at 1B — coincides with a drop in depth.** This
design cannot separate the two. It is a property of the Pythia suite rather than a choice made here,
but the scale axis was defined as targeted non-embedding parameters (§2.6) without accounting for it.

**What would fix it.** A fourth point that is deeper rather than merely wider, or a depth-matched
comparison at fixed width. Neither was run.

## 3. R = 5 at 1B cannot reach significance at any effect size

Replicates are R = 8 at 160M and 410M, **R = 5 at 1B**. At R = 5 the best attainable two-sided exact
sign-test p is **0.0625** — so even a perfectly unanimous 1B cell cannot reach 0.05.

This is visible in the result: 1B/moderate is unanimous (0/5, joint worse) and still reports
p = 0.0625. That is the *strongest possible* outcome at that R and it is not significant.

**Cost to the claim.** Neither 1B cell may carry a significance claim in either direction. The 1B
leg contributes effect sizes and sign consistency only. Accepted knowingly at freeze time to fit the
schedule.

## 4. Only one statistically significant cell, and it is fragile

Six cells were examined. Exactly one reaches significance, and it barely survives correction:

| Cell | Raw p | Holm–Bonferroni |
| --- | --- | --- |
| **410M / W4** | 0.0078 | **0.0469** |
| next best (1B / W8) | 0.0625 | 0.3125 |

**0.0469 clears 0.05 by 0.0031.** The same cell is *also* 0.065 pp short of the pre-registered
practical-importance threshold. Two independent criteria, both barely met or barely missed, on one
cell out of six.

**Cost to the claim.** The effect must be described as **small and statistically fragile**, and the
**adjusted** p-value quoted. "Significant at 410M" without the correction is the false-positive
inflation that [validity_threats.md](validity_threats.md) warned about before any result existed.

## 5. No cell meets the pre-registered practical-importance criterion

§6.3 requires a mean gain **≥ 1.0 pp** *and* a consistent sign across every paired replicate. **No
cell meets both.** The two closest fail on opposite criteria: 160M clears the size bar with one
negative replicate; 410M is unanimous but 0.065 pp short.

Two further points the paper must disclose rather than bury:

- **§6.3 was already loosened before the run.** Amendment A1 replaced the original rule, whose
  binding clause was unmeasurable, with a weaker measurable one. The bar that was not met is the
  *weaker* of the two.
- **The threshold is not renegotiable after the fact.** It was pre-registered precisely for the
  near-miss case that occurred.

## 6. A fixed recipe does not imply matched effective severity

Both budgets prune **30%** at every scale. That holds the *recipe* constant; it does not hold the
*damage* constant. Retention at the aggressive budget runs **55.8% / 57.6% / 89.7%** across
160M / 410M / 1B — the same recipe is far milder at 1B.

**This is a genuine alternative explanation for the scale trend.** A baseline that already retains
89.7% leaves little headroom for a better layer solution to recover, so the same local mechanism buys
less end-to-end. [F-38](findings_log.md#f-38) shows the local mechanism is *undiminished* at 1B —
mask divergence 3.41%, layer objective advantage +2.32% — while the model-level gain collapses to
+0.13 pp, which is consistent with a headroom explanation rather than a mechanism-weakening one.

**Cost to the claim.** "Joint helps less at scale" and "joint helps less when there is less to
recover" are not distinguished by this design. Larger models tolerated compression better, and that
robustness benefited the sequential baseline too.

## 7. The confirmatory baseline is the frozen order, not best-of-both

§3.6 defines joint gain against **best-of {P→Q, Q→P}**. In the confirmatory run only the **frozen**
order was executed per cell — selected on validation, frozen before test, per A1 §3. At 1B/moderate
that is Q→P; everywhere else P→Q.

Where the freeze was recorded as **arbitrary** — the W8 cells, because the two orders were
indistinguishable ([F-28](findings_log.md#f-28)) — the baseline carries an uncertainty of roughly the
order margin measured there, ~0.18 pp at 160M/W8. **That is comparable to the W8 effects themselves**
(+0.038, +0.029, −0.179 pp).

**Cost to the claim.** No W8 conclusion should rest on the sign alone. The W4 cells are unaffected:
P→Q won there by 4–7 pp, far outside any plausible order uncertainty.

## 8. One model family, one corpus, one evaluation window

- **One family.** Pythia only. The Qwen2.5-0.5B external-validity point was registered, pinned and
  adapter-implemented but **not run**.
- **One corpus.** WikiText for calibration *and* evaluation. Perplexity on a single corpus is the
  primary metric.
- **One window.** 512 sequences × 512 tokens on the test split. Not a full test set at 2048.
- **Downstream tasks do not resolve the arms.** [F-35](findings_log.md#f-35) found no arm difference
  resolvable at the evaluation size used, so the downstream results constrain nothing about joint
  versus sequential — they establish only that the harness anchors to published values.

**Cost to the claim.** Every result is a statement about Pythia on WikiText perplexity at this
window. External validity is asserted by architecture similarity, not measured.

## 9. Every implementation fault found in this project flattered the joint arm

Six, in order: **B-14, B-17, B-22, B-23, B-30, B-34**. Not one ran the other way.

The final effect is small and was reached by removing biases that all pointed the same direction.
Three independent anchors bound what remains — the mask is bit-identical to an independent Wanda
implementation over 84,934,656 weights ([F-19](findings_log.md#f-19)), the sweep never falls below
the provable optimum of its own objective ([F-20](findings_log.md#f-20)), and absolute retention is
credible against unmodified SparseGPT ([F-22](findings_log.md#f-22)) — but **bounding is not
elimination**, and the direction of the historical error is itself a reason for caution about a
sub-1 pp positive result.

## 10. Solver slack is bounded in sign but not in magnitude

[F-21](findings_log.md#f-21) established that arm-dependent solver slack is real (0.6409 efficiency
under the sequential mask against 0.5631 under the joint mask) but **never inverted which mask was
better** across 96 rows — so the *direction* of a measured joint gain is not a solver artefact, and
the gap runs *against* joint, understating rather than flattering it.

The effect on **magnitude** remains unquantified, and cannot be closed by the same method: the exact
minimiser exists only for the continuous least-squares problem, and a discrete grid makes the
quantised version an integer program with no closed-form optimum. That anchor also ran at **160M
only**; it was not repeated at 410M or 1B, and [F-38](findings_log.md#f-38)'s per-layer statistics
are *not* a substitute — `relative_improvement` is measured against an arm-specific baseline and is
not comparable across arms.

## 11. Two records have incomplete provenance

`pythia-410m_pruning_aggressive_s30_b32_rep0` and `pythia-1b_joint_aggressive_s30_b4_rep3` carry a
**null `git_commit`**. Their configs match their siblings field for field (modulo the intended
`calibration_replicate`), both carry `method_version 4`, and their timestamps fall between
documentation-only commits, so the code that produced them is bounded.

Provenance is nonetheless weaker for those two than for the other 169, and **one of them is a joint
cell inside a reported pair**.

## 12. The memory accumulation was worked around, not fixed

The runner does not release memory between cells: **~4 GiB of commit per 1B compression cell**,
never returned ([B-48](findings_log.md#4-bugs-found-that-would-have-invalidated-results)). The
confirmatory 1B leg was completed by recycling the sweep process at cell boundaries.

**This is numerically inert** — `skip_existing` re-runs nothing, each record is written whole, and a
recycled run reproduces — so the 171 records are unaffected. But the grid was produced across roughly
a dozen process lifetimes rather than one, and with `continue_on_error` enabled an unnoticed
`MemoryError` would have *dropped* cells and silently removed comparisons rather than stopping.

**Fix before any future long sweep**: run each cell in an isolated child process so memory is
released at the boundary by construction.

## 13. Latency evidence is one sparsity, not a curve

[F-34](findings_log.md#f-34) measured **30% unstructured sparsity at three scales** under the §4.7
protocol and found no commensurate CPU speedup. That is **one point, not a curve**, and RQ4's
sparsity–speedup relationship is not answered.

Two further constraints:

- **W4 never appears in a latency table.** Decision D1 makes PyTorch CPU **INT8** (`onednn`) the sole
  latency backend; the 4-bit path exists for quality and size only. So the budget carrying the
  headline effect is the one with no latency measurement.
- **Any figure derived from F-34 must not be labelled a sparsity curve.** It contains a single
  non-zero sparsity.

## 13b. Per-cell latency measurements are not comparable across arms

Separate from the "one sparsity, not a curve" point above: the sweep benchmarks each cell *inside*
its own run, so every latency is taken at whatever moment that cell executed. The confirmatory grid
spanned six days of varying machine state — commit exhaustion, ~12 process recycles, repeated host
standby.

The failure is visible in the records: pythia-1b **dense** reads 1041 ms and pythia-1b **pruning**
630 ms, an apparent 40% speedup from masking weights, which is impossible — pruned weights stay FP32
and dense in storage. The dense figure was measured 2026-08-05 and the pruning figures 2026-08-07
([B-49](findings_log.md#4-bugs-found-that-would-have-invalidated-results)).

**Cost to the claim.** No latency comparison may be drawn from the per-cell sweep records.
[F-34](findings_log.md#f-34) — a dedicated §4.7 study with model-order rotation, which exists to
control exactly this drift — is the only citable latency evidence.

## 14. Structured sparsity was never run

The mask supports 2:4 and 4:8 semi-structured patterns and they are tested at the tensor level, but
**no end-to-end result uses them**. Unstructured 30% is what every reported number carries, and
unstructured sparsity is precisely the kind least likely to yield hardware speedup — which is the
most likely reason F-34 found none.

## 15. What the exploratory→confirmatory shift implies about the screening numbers

Neither exploratory point estimate replicated: 160M fell **+1.69 → +1.01**, 410M rose **+0.39 →
+0.93**. The two-split design worked exactly as intended — had the study reported the validation
numbers it would have published two wrong point estimates and a "shrinks with scale" narrative that
the test split does not support.

**Cost to the claim.** Every exploratory finding in this log carries the same risk and must be read
as screening, not as a result. In particular [F-32](findings_log.md#f-32)'s monotone decline and
[F-27](findings_log.md#f-27)'s "robust at 160M" are superseded.
