# Protocol Amendment A1

**Date:** 2026-07-30 · **Status:** adopted · **Amends:** research plan §5.5, §6.3, §4.1, §3.6/§6.1

This amendment changes four frozen protocol rules and adds two procedures that change nothing. It
exists because [protocol_freeze.md](protocol_freeze.md) states that "a value in this file changes only
by editing this file, in a commit that says why" and that "nothing here may be revised after results
have been seen (§6.3)". Results **have** been seen. That makes this amendment the delicate case the
rule was written to guard against, so it is written to be judged rather than to be convenient.

**The test every item below must pass:** would this change have been justified by the same argument
*before* any result was seen? If a change only became attractive once we saw the numbers, it is not a
correction — it is selection dressed up as method. Each item states its answer to that question
explicitly, and item 4 is the one that most needs the scrutiny.

---

## 1. What is changing

| # | Item | Status | What it amends |
| --- | --- | --- | --- |
| **1** | Run seeds → **paired calibration replicates** | **Necessary methodological correction** | §5.5 seed policy, §6.3 practical-importance rule |
| **2** | Validation for selection, **test for confirmation** | **Necessary selection-bias correction** | §4.1 evaluation split |
| **3** | **Best sequential order**, chosen on validation | **Enforcement of the original protocol** | Nothing — §3.6/§6.1 already required it; the grid did not implement it |
| **4** | **S6 (40% + W8)** quality-matched control | **Optional secondary addition** | Nothing — an addition, not a change |
| **5** | **External correctness anchors** | **Implementation validation** | Nothing — tests the software, not the hypothesis |

Only items 1 and 2 change a rule that was frozen. Item 3 makes the code do what the documents already
said. Items 4 and 5 add work without altering any existing definition, which is why they are the safest
of the five and are recorded here only so the sequence is auditable.

---

## 2. Why the original rules no longer measure what they intended

### §5.5 and §6.3 — the seed policy

**As frozen:** one seed for screening, three confirmatory seeds for the central comparison; a joint
gain counts as practically important only if it is ≥ 1.0 pp retention, consistent in sign across all
three seeds, **and exceeds the seed spread**.

**Why it measures nothing.** The method is deterministic post-training layerwise reconstruction. Two
runs of the same cell at different run seeds produced perplexity **65.1548** and **65.1548** —
bit-identical. There is no stochastic component: no SGD, no sampling, no shuffling, no dropout; the
column sweep is one deterministic pass and `topk` is deterministic. Calibration indices derive from a
separate `calibration_seed` that is deliberately independent of the run seed, so that every arm at a
given scale calibrates on the same sequences.

Consequently three confirmatory seeds would produce three **identical** numbers, the seed spread would
be exactly **zero**, the "exceeds the seed spread" clause would pass for any nonzero gain whatsoever,
and the paper would carry **no uncertainty estimate at all** while appearing to have followed a
three-seed protocol. Evidence: [findings_log.md F-15](findings_log.md#f-15).

**Root cause, and it is not an error by anyone.** The seed policy was written for the *original*
full-model quantisation-aware-training design, where training is stochastic and seeds genuinely
produce variance. It does not transfer to a deterministic post-training method. This is the same root
cause as the superseded method definition described in F-15 — one design's protocol left attached to a
different design's method.

### §4.1 — the evaluation split

**As frozen:** calibration from WikiText-2 **train**, evaluation on WikiText-2 **validation**.

**Why it no longer measures what it intended.** There is no calibration leakage — calibration is drawn
from train, and that separation held throughout. The defect is different: the six candidate budgets
were screened on validation, we **looked at those retention numbers**, and we used them to select the
two frozen budgets. Validation is therefore now a *selection surface*. Reporting the headline
confirmatory result on the same split means the primary evidence is computed on data that was used to
make a choice, which biases it upward by an unknown amount.

### §3.6 / §6.1 — the sequential baseline

**As specified:** joint gain is computed against **best-of** {P→Q, Q→P}, with the winning order
recorded.

**Why it was not being honoured.** The sweep runs **only P→Q**. `SEQUENTIAL_QP` is implemented,
registered and tested, so this was never a code gap — the grid simply did not schedule it. A joint gain
measured against only one ordering is a gain over a baseline we chose, not over the strongest available
baseline. Given that **every fault found in this project so far has flattered the joint arm**
([F-18](findings_log.md#f-18): B-14, B-17, B-22, B-23, none in the other direction), this is precisely
the asymmetry to be strict about.

---

## 3. Why each correction would have been justified before seeing any results

This is the section that decides whether this amendment is legitimate.

| # | The argument, and whether it depends on any result |
| --- | --- |
| **1** | The pipeline is **provably deterministic** — verifiable by reading the solver, the mask construction and the calibration draw, with no run required. The two identical perplexities are a *demonstration* of a fact that was already inspectable, not the discovery of it. The argument stands with every result deleted. |
| **2** | "Do not report a headline on the split used to select the configuration" is a general principle of experimental design. It would have applied identically had we written it down before screening — indeed it *should* have been written down then. Its force does not depend on which budgets won or what the joint gain was. |
| **3** | The documents already required best-of-two. Enforcing a rule that was frozen at the outset cannot be post-hoc by construction. |
| **4** | ⚠️ **The weakest of the five, and it must be labelled as such.** S6 became interesting *because* we observed that 40% + W8 lands at nearly the same retention as 30% + W4. That is a result. What makes the addition defensible is that it tests a **mechanism** ([F-05](findings_log.md#f-05): the mask mechanism is live at W4 and inert at W8) rather than improving a headline, and that it is declared **secondary and non-confirmatory** below. It is recorded here with its provenance visible rather than presented as though it had been planned from the start. |
| **5** | Verifying an implementation against an independent one is always justified and never depends on what the implementation produced. |

**On item 1 specifically:** the change must be justified on the mechanism — seeds provably do nothing —
and **not** on wanting error bars around a number we like. The 410M sign flip in
[F-14](findings_log.md#f-14) is a result, and it is what makes this decision both urgent and delicate.
Note that the seed defect was identified and written up in F-15 *by checking rather than by failure*,
and was flagged as "not actioned, needs a human decision" at the time — before the amendment was
drafted and before the current joint-gain figure exists at all. The record supports the sequence.

---

## 4. Every result this project has produced is exploratory

**No number recorded in this repository before 2026-07-30 may be used as confirmatory evidence.**
Not one. Three independent reasons, any of which alone is sufficient:

1. **All of it is on the validation split**, which is now a declared selection surface (item 2).
2. **All of it predates the B-22 / B-23 corrections.** Per [F-18](findings_log.md#f-18) the arms were
   minimising different objectives, and the asymmetry **inflated joint gain**.
3. **None of it has an uncertainty estimate**, because the seed axis was inert (item 1).

Specifically superseded as confirmatory evidence, and retained only as the exploratory record:

| Result | What it was | Why it is exploratory only |
| --- | --- | --- |
| [F-10](findings_log.md#f-10), [F-13](findings_log.md#f-13) | First screening grids | Predate the F-16 fixes; already superseded by F-17 |
| [F-14](findings_log.md#f-14) | The 410M/160M **joint-gain sign flip** | 160M half already void per F-17; both halves predate B-22/B-23; validation split; no error bars |
| [F-17](findings_log.md#f-17) | Screening re-run, S5 joint gain **+1.03 pp** | Gain column already retracted by F-18; validation split; no error bars |
| The **410M confirmation** table in [protocol_freeze.md](protocol_freeze.md#confirmed-on-410m-) | moderate/aggressive eligibility at 410M | Same three reasons |

**The budget freeze itself survives** and is *not* reopened by this amendment. Sequential retention has
been stable across every version of the code — ≈80% at 30% + W8 and ≈57% at 30% + W4 — because none of
the faults touched the sequential arm's first stage or the eligibility rule. The two frozen budgets
stand. What is retracted is every *comparison between arms*, not the choice of where to compare them.

**Consequence for the paper.** Exploratory and confirmatory results must appear in **separate
sections**, never in one table and never in one trend line, with the exploratory section carrying an
explicit statement that those results were used to select the compression budgets and must not be read
as confirmatory estimates. Absolute values legitimately differ between validation and test because the
text and the number of windows differ. `scripts/summarise_screening.py` already refuses to span
evaluation windows and `METHOD_VERSION` already refuses to mix records produced by different
algorithms; both now serve this separation mechanically rather than only by convention.

---

## 5. The amended protocol

### 5.1 Replicates and statistics (amends §5.5, §6.3)

The run-seed axis is **withdrawn**. It is replaced by **paired calibration replicates**: R independent
calibration draws, where replicate *r* is used by **every arm** in that comparison — sequential, Q→P
and joint alike.

```yaml
calibration_replicates: [1729, 2718, 3141, 5772, 8111]
```

- **R = 5** is the adopted minimum; **R = 8** if the compute budget permits (see §6 — the cost
  multiplier is substantial and this choice is not free).
- Within a replicate, all arms use byte-identical calibration data. §3.11 requires identical
  calibration *between arms within a comparison*, not across repeats, so the fairness invariant is
  preserved — and pairing is what §6.3 asks for anyway.
- Test windows are fixed and identical for every method and every draw.
- These are **paired calibration replicates**. They must never be described as random training seeds,
  because nothing is trained and the run seed is inert.

**Reporting.** For each model and budget, report **all R replicate-level joint gains individually**,
then the mean, the median, the min–max range, and the number of draws positive in sign.

**What must not be claimed.** With R = 5, formal significance is not available and will not be
asserted:

- A paired *t*-test is inappropriate: normality cannot be assessed at R = 5.
- A sign test is **too weak to reach conventional thresholds**: with all five draws sharing a sign the
  exact two-sided probability under a fair-sign null is 2 / 2⁵ = **0.0625**. Six identical signs would
  give 2 / 2⁶ = **0.03125**. So five unanimous replicates cannot clear p < 0.05 *even in the best
  case*, which is a fact about the design and not about the data.
- At R ≥ 8 a paired permutation or sign test becomes informative.

**This is therefore an effect-size study, and will be written as one.** Consistency and magnitude,
reported transparently at replicate level, with no significance language manufactured from five
observations.

**§6.3 as amended:**

> Joint gain is reported over paired calibration replicates. A practically meaningful result must meet
> the pre-declared threshold of ≥ 1.0 percentage point retention, retain the same direction in most
> calibration draws, and be interpreted alongside the paired uncertainty interval. All replicate-level
> values are reported.

This is **a loosening of a pre-registered rule**, and is declared as such: the original rule was
stricter on its face but unmeasurable in fact, since its binding clause evaluated to zero. Replacing an
unmeasurable criterion with a measurable weaker one is a correction; it is also a reduction in
pre-registered strength, and the paper must say so rather than present the amended rule as the original
one.

**Additionally: a paired block bootstrap over evaluation windows.** For each test window *i*, form
`dᵢ = NLL_sequential,i − NLL_joint,i` and resample **complete windows**, never individual tokens —
neighbouring tokens are statistically dependent, so a token-level bootstrap would understate the
interval. Both arms must always use the same resampled window indices.

The two uncertainty sources measure different things and neither replaces the other:

| Source | What it quantifies |
| --- | --- |
| Paired calibration replicates | Sensitivity of the compressed model to which calibration data it saw |
| Paired block bootstrap over windows | Uncertainty from the finite evaluation corpus |

### 5.2 Two evaluation configurations (amends §4.1)

**Exploratory configuration** — `calibration_split: train`, `eval_split: validation`. Used for
implementation debugging, the budget screening re-run, evaluating P→Q against Q→P, selecting the
sequential order, the S6 control, the external anchors, and all mechanism diagnostics.

**Confirmatory configuration** — `calibration_split: train`, `eval_split: test`. Used for the final
Pythia scale comparison, the headline tables, the headline joint-gain plot, and the final downstream
task evaluation.

Held **identical** between the two unless separately justified in writing: tokenizer, sequence length,
stride, preprocessing, label shifting, module coverage, calibration sample count. The primary study
uses the test split **at the already-frozen sequence length** — changing the split and the context
length together would make it impossible to attribute any movement to either.

### 5.3 Sequential ordering (enforces §3.6, §6.1)

Both orders run, and **the winner is selected on validation, never on test.**

1. On the **validation** split, run P→Q and Q→P at every model and budget, using the same calibration
   replicates for both.
2. Compare mean validation **excess NLL**.
3. Select the better order per (model, budget) and **freeze it before any test evaluation**.
4. On the **test** split, run only the frozen winning order against joint.

```yaml
selected_sequential_order:
  pythia-160m:  {moderate: p_to_q, aggressive: TBD}
  pythia-410m:  {moderate: TBD,    aggressive: TBD}
  pythia-1b:    {moderate: TBD,    aggressive: TBD}
```

The winning order may legitimately differ by model and by budget; where it does, that is recorded and
reported.

**Why validation selection rather than a max over test results.** Taking the better *test* score per
cell would make the baseline pessimistically biased against joint **and** would use the test set for
method selection — spending the confirmatory split on a choice. Selecting on validation gives joint the
hardest sequential competitor while leaving the test comparison untouched: validation chooses the
method, test estimates its performance. Both orders may additionally be shown on test in an appendix,
but the primary comparison uses the validation-selected order.

### 5.4 S6 — secondary quality-matched mechanistic control (addition)

**Purpose, and the whole of it:** distinguish a joint effect caused by *low-bit quantisation* from one
caused merely by *severe quality degradation*.

| Setting | Sparsity | Precision | Approx. validation retention |
| --- | --- | --- | --- |
| Aggressive primary | 30% | W4 | ≈57% |
| S6 control | 40% | W8 | ≈54% |

Two different recipes at nearly the same quality. If joint helps at 30% + W4 but not at 40% + W8
despite comparable degradation, the mechanism is **precision-specific** — which is what F-05 predicts,
since the joint arm's mask diverges 8.86% at W4 against 0.46% at W8. If joint helps at both, the cause
is compression severity, a substantially weaker claim.

**Reduced design — S6 is not added to the full five-arm sweep:**

```yaml
models:     [pythia-160m, pythia-410m]
budget:     40% + W8
methods:    [selected sequential order, joint]
replicates: 3 paired calibration draws
```

**12 compressed runs.** Omitted deliberately: dense (already measured), pruning-only,
quantisation-only, CPU latency, downstream tasks, and checkpoint-size analysis unless essentially free.

For **Pythia-1B**, run only the cheap layer-level diagnostics — mask divergence, quantisation-scale
divergence, reconstruction loss, accepted/rejected joint updates. The full 1B S6 comparison runs only
if the smaller-model control produces a genuinely useful distinction.

**Label, to be used verbatim:** *secondary, validation-selected, quality-matched mechanistic control.*
Never confirmatory. It must not be used to modify the primary budgets after test results exist.

### 5.5 External correctness anchors (addition)

Two distinct things need validating and no single check covers both: **Wanda-style mask selection**, and
**SparseGPT/GPTQ-style error compensation and reconstruction**.

**(a) Wanda initial-mask agreement — cheapest, run first.** Our saliency rule is exactly the Wanda
criterion, `S_ij = |W_ij| · ‖X_j‖₂`, applied per output row. With formula, comparison group, model,
activations, module coverage and target sparsity all matched, the masks should be **virtually or
exactly identical**; any divergence should be confined to deterministic tie handling.

```
Pythia-160M · 30% sparsity · identical calibration draw and count · identical modules
no reconstruction · no quantisation
compare: mask overlap · differing positions · per-layer sparsity · module coverage · activation norms
```

A disagreement beyond a tiny tie-related fraction means one implementation differs in activation
collection, grouping, ordering or module selection — all of which we have already had bugs in
(B-19 was exactly a grouping fault).

**(b) SparseGPT pruning-only — highest information from a single run.** If only one full external run
is affordable, choose SparseGPT over Wanda: Wanda validates the pruning criterion but **not** our
reconstruction, whereas SparseGPT exercises the Hessian / error-compensation path, which is the more
intricate and more bug-prone half.

```
Pythia-160M · 30% unstructured · FP32 survivors · same calibration data and module coverage
compare: dense ppl · pruning-only ppl · per-layer reconstruction loss · achieved sparsity
         layer ordering · calibration fingerprint
```

A maintained Transformers-compatible `SparseGPTModifier` targeting linear modules may adapt to
GPT-NeoX/Pythia more easily than the original OPT/BLOOM scripts.

**(c) GPTQ W4 quantisation-only — optional.** Pythia-160M, 0% pruning, W4, same calibration data,
sequence length and evaluation set.

**Passing criteria are debugging alarms, not statistical acceptance tests.** Bit-for-bit agreement must
**not** be required — the algorithms are not identical. Pre-declared thresholds:

| Check | Alarm threshold |
| --- | --- |
| Dense perplexity | agree within ≈0.1% |
| Wanda initial masks | almost exactly identical |
| Module counts, target sparsity | must match **exactly** |
| Pruning-only retention | same general range |
| Retention gap | > ≈3 pp, or 5–10% relative perplexity → investigate |
| Per-layer reconstruction loss | must not be systematically much worse than the reference |

**Why a matched anchor rather than a published table.** There appear to be **no published numbers** for
our exact combination — Pythia-160M, 30% unstructured, W4 weight-only, our calibration size, our
512-token window. The major reference implementations report OPT, BLOOM and LLaMA-family results. So
running a matched anchor ourselves is more defensible than comparing against an unmatched paper figure,
and it is the only thing that settles whether ≈57% retention is plausible or signals a remaining
implementation gap.

---

## 6. Cost, stated before committing rather than discovered later

The confirmatory design is **3 models × 2 budgets × 2 arms × R replicates**, plus the validation-stage
ordering selection, plus S6. At R = 5 that is **60 confirmatory compressed runs**. Extrapolating from
the measured screening rate (13 cells ≈ 2 h at 160M, so ≈9 min per 160M cell) and scaling by parameter
count, the confirmatory stage alone is on the order of **30 hours of compute**, before the Q→P
validation stage, the S6 control and the anchors.

**R = 8 multiplies the confirmatory stage by 1.6.** That is the real trade-off behind the R = 5 versus
R = 8 choice: R = 8 is what makes a permutation or sign test informative at all (§5.1), and it costs
roughly 18 additional hours. **This is a scheduling decision, not a methodological one, and it is
recorded here so that it is made deliberately.** R = 5 is adopted as the floor; going to 8 is a
compute-availability call and may be taken per model — for example 8 at 160M and 410M, 5 at 1B — since
nothing in §5.1 requires R to be constant across scales, provided the actual R is reported per cell.

---

## 7. Execution order

Strictly sequential. Each step exists to avoid spending compute on a pipeline the next step might
invalidate.

1. ✅ **Write and commit this amendment.**
2. ✅ **Central implementation corrections** — the nine fixes; suite at 804, lint clean.
3. ⬜ **External correctness anchors** — Wanda mask agreement, then SparseGPT pruning-only.
4. ⬜ **Create the five fixed calibration draws and fingerprint them.**
5. ⬜ **Re-run validation screening** on the corrected implementation.
6. ⬜ **Run both sequential orders on validation.**
7. ⬜ **Freeze the winning order** per model and budget.
8. ⬜ **Run the reduced S6 mechanistic control.**
9. ⬜ **Freeze the entire confirmatory configuration.**
10. ⬜ **Run test evaluation once, with no further tuning.**

**Step 3 precedes step 5.** This reverses the previously planned order and is deliberate: if the anchor
disagrees, the screening grid would have been measuring a pipeline we do not trust, and its two hours
would be spent twice.

---

## 8. The closing commitment

**No further protocol decision will be changed after the test split has been examined.**

Steps 1–9 above are the complete set of permitted decisions. Once step 10 runs, the protocol is closed:
budgets, arms, sequential ordering, replicate count, metrics, evaluation window and reporting structure
are all fixed by then, and the test result is reported **as it comes out**. If the test result
disagrees with the exploratory findings, that disagreement is the finding and it will be reported as
such. If the joint gain fails the ≥ 1.0 pp threshold, that is reported as a null result — the study is
designed to be able to return one, and §6.3 forbids adjusting the threshold to reach a positive.

Any subsequent amendment must be a new dated document, must state what it changes and why the change
does not depend on a test result, and must declare every result produced before it as superseded. That
is a deliberately high bar. It should be difficult to change the protocol again from here.

---

## Related

- [protocol_freeze.md](protocol_freeze.md) — the frozen values this amends
- [findings_log.md](findings_log.md) — every measurement, with its conditions; F-15 and F-18 are the
  evidence for items 1 and 3
- [validity_threats.md](validity_threats.md) — what could still make the results wrong
- [research_plan.pdf](research_plan.pdf) — authoritative source; §5.5, §6.3, §4.1, §3.6, §6.1
