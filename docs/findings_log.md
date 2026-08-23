# Findings log

Every measurement this project has produced, with the conditions that produced it.

**Why this file exists.** [STATUS.md](STATUS.md) says where the work stands and is rewritten every
session. [validity_threats.md](validity_threats.md) says what could make the results wrong. This file
is the **append-only record of what was actually measured** — so that when the paper is written, every
number in it can be traced to a configuration, a commit, and a machine, and nothing has to be
reconstructed from memory or re-derived under different conditions.

**Rules for this file:**

- **Append, do not rewrite.** If a number is superseded, add the new one and mark the old one
  superseded with the reason. A deleted measurement is a measurement that will be re-argued later.
- **Every number carries its conditions.** Model, revision, budget, calibration set, evaluation
  window, machine. A perplexity without its evaluation window is not a result.
- **Record rejected approaches with their numbers.** "We tried X and it was worse" is a reportable
  ablation, and without the numbers it becomes an unsupported assertion in the Limitations section.
- **Mark provenance honestly.** Synthetic-layer measurements and real-model measurements are labelled
  as such, because one of them has already been wrong once (see [F-04](#f-04)).

---

## 1. Environment of record

Every number below comes from this machine unless stated otherwise. §4.7 forbids mixing machines in
one table, and §2.7 requires the runtime frozen for the study's duration.

**Machine policy amended 2026-08-01.** Compression, activation capture and quality evaluation may
run on any CUDA machine; only **deployment measurements** are bound to this host. Any finding below
produced elsewhere states its machine explicitly — that is the whole point of this section. A
comparison still may not span machines, enforced by `exists_valid` (B-33).

| | |
| --- | --- |
| Machine | HP Omen — **the designated benchmark host** |
| CPU | 13th Gen Intel Core i7-13620H · 6 P-cores + 4 E-cores · 10 physical / 16 logical |
| RAM | 13.7 GiB |
| GPU | NVIDIA GeForce RTX 4050 Laptop · **6.0 GiB** (6141 MiB) · sm_89 (Ada) |
| NVIDIA driver | 592.82 |
| OS | Windows 11 Home 10.0.26200 |
| Power profile | High performance `ad8e16f4-0e1d-4811-9f3b-165752347277` · processor throttle min **and** max 100% · AC sleep and hibernate disabled |
| Benchmark threads | **4** — inside the 6 P-core budget, so no repetition lands on an E-core or an SMT sibling |
| Python | 3.11.9 |
| torch | **2.13.0+cu126** |
| transformers | 5.14.1 |
| datasets | 5.0.0 |
| numpy | 2.4.6 |
| Quantised engine | **`onednn`** — the *only* engine this build supports (see [F-02](#f-02)) |
| Smart App Control | **disabled** 2026-07-28, machine rebooted (see [F-01](#f-01)) |

**Do not upgrade torch mid-study.** `torch.ao.quantization` and the `qint8`/`quint8`/`qint32` dtypes
are both deprecated in favour of `torchao`. They work on the pinned version; §2.7 pins the environment
for the study's duration, so a deprecated-but-working path is reproducible. Moving off the pin would
change the quantisation backend under the results.

### Model revisions (pinned 2026-07-28)

Resolved with `HfApi().model_info(repo_id, revision="main").sha`. All **standard** Pythia — no
`-deduped` anywhere, which §2.7 requires and which is checkable at a glance from this table.

| Config | Repository | Commit SHA |
| --- | --- | --- |
| `pythia_160m.yaml` | `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` |
| `pythia_410m.yaml` | `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| `pythia_1b.yaml` | `EleutherAI/pythia-1b` | `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` |
| `pythia_1_4b.yaml` | `EleutherAI/pythia-1.4b` | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` |
| `qwen2_5_0_5b.yaml` | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |

### Data of record

`Salesforce/wikitext` / `wikitext-2-raw-v1`, tokenised with the pinned pythia-160m tokeniser.

| Split | Tokens | Blocks | Fingerprint |
| --- | --- | --- | --- |
| train | 2,431,839 | 9,499 × 256 | `798e41edc78ea923` |
| validation | 252,468 | 986 × 256 | `b96c6c1be84cad97` |

| Set | Detail | Fingerprint |
| --- | --- | --- |
| Evaluation loader (pilot) | 64 sequences × 256 tokens | `8b17abdeb30e252b` |
| Calibration (pilot) | 16 sequences + 3 held-out, seed 1234 | indices `20bf57e6b08ed60d`, tokens `4914adc5531d4aad` |
| Calibration (128-sample probe) | 128 sequences | `60fc1307e7c7e0ac` |

### ⚠ Two evaluation windows exist. Never mix them.

| Window | Used by | Sequences × tokens | Total tokens | Dense reference |
| --- | --- | --- | --- | --- |
| **Pilot** | §3 exploratory table, [F-07](#f-07), [F-08](#f-08) | 64 × 256 | 16,320 | **34.77** |
| **Screening** | [F-10](#f-10) | 493 × 512 (whole validation split) | 252,416 | **36.97** |

Retention is computed against the dense run *in the same window*, so a retention figure is meaningful
only within its own window and a perplexity is meaningless without one.
`scripts/summarise_screening.py` refuses to print a table spanning both.

Neither window is the protocol published papers use (full test set at a 2048-token context), so
**none** of these numbers is directly comparable with the literature. They are internally comparable,
which is what the arm comparison needs.

---

## 2. Findings

### F-44 — Joint's advantage reverses under equal recovery: 8/8 replicates, p = 0.0078 {#f-44}

**2026-08-23 · `45f288d` · POST-HOC EXPLORATORY · validation split · NOT CONFIRMATORY.** Cannot
alter, replace or reinterpret [F-37](#f-37). Config
`configs/experiments/recovery_ablation_160m_w4_r8.yaml`; 16 records in `outputs/recovery_ablation/`
under the `_r8` id. Run entirely after the [B-54](#4-bugs-found-that-would-have-invalidated-results)
fix, so the arms compress against the 128-sequence calibration set and recover on a **verified**
disjoint slice.

Pythia-160M, 30% + W4, dense perplexity 36.9741, **eight paired calibration replicates**, masks
frozen, W4 fake quantisation live, identical 50-step / 204,800-token recovery at lr 1e-5 — the
setting [F-43](#f-43) established as non-destructive and near-peak.

#### The result

| rep | seq before | seq after | joint before | joint after | gain before | gain after |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 56.6554 | 70.8858 | 57.7347 | 70.0553 | +1.0794 | −0.8305 |
| 1 | 56.8942 | 72.5509 | 58.5460 | 69.7678 | +1.6518 | −2.7831 |
| 2 | 55.7244 | 69.5589 | 58.0615 | 68.9582 | +2.3371 | −0.6008 |
| 3 | 57.0642 | 70.6031 | 57.5933 | 69.6081 | +0.5291 | −0.9950 |
| 4 | 56.9150 | 70.2684 | 57.1636 | 68.1052 | +0.2486 | −2.1633 |
| 5 | 57.1998 | 71.2669 | 59.2641 | 70.4450 | +2.0643 | −0.8219 |
| 6 | 56.2618 | 71.3271 | 58.4089 | 69.8208 | +2.1471 | −1.5063 |
| 7 | 56.3095 | 69.7457 | 58.0564 | 68.6932 | +1.7468 | −1.0525 |
| **mean** | 56.6280 | **70.7759** | 58.1036 | **69.4317** | **+1.4755** | **−1.3442** |
| sd | | 0.8227 | | 0.5744 | 0.7748 | 0.7619 |

**Both columns are unanimous.** Joint leads in **8/8** replicates before recovery and trails in
**8/8** after. A two-sided exact sign test on the reversal gives **p = 0.0078**, the floor reachable
at R=8.

**Sequential improved more, in 8/8 replicates** — **+14.1478 pp** against joint's **+11.3281 pp**.
That is the mechanism of the reversal: not that joint degrades, but that it gains less from the same
recovery.

#### Three things worth separating

**1. The reversal is the strongest sign-consistent effect this project has measured — and it points
against joint.** Its magnitude (1.3442 pp) and unanimity (8/8) would satisfy the two criteria §6.3
sets, where joint's *own* advantage at this scale and budget on test reached +1.0120 pp at 7/8 and
failed. **This is an illustrative comparison, not a §6.3 verdict**: §6.3 is pre-registered for the
test split and the primary comparison, and applying its thresholds to a post-hoc validation ablation
does not confer its authority. Recorded because the contrast is the honest way to convey the size of
what was found, not to claim a pass.

**2. Recovery is worth far more than the choice of arm.** Fifty steps and 204,800 tokens — about
0.0007 of Pythia-160M's pretraining — move retention from ~56.6% to ~70.8%, a **+14 pp** gain. The
joint-versus-sequential difference at the same budget is ~1.5 pp. So at this scale the compression
*method* is roughly an order of magnitude less important than whether any recovery is done at all.
That reframes the study's question rather than answering it, and it belongs in the discussion.

**3. The before-recovery column independently reproduces [F-27](#f-27).** On the three draws the two
share, at the same budget and split, a month apart and through a different driver:

| rep | F-27 | this run |
| --- | --- | --- |
| 0 | +1.08 | **+1.0794** |
| 1 | +1.65 | **+1.6518** |
| 2 | +2.34 | **+2.3371** |
| mean (n=3) | +1.69 | **+1.6894** |

This is the reproduction check [B-54](#4-bugs-found-that-would-have-invalidated-results) had silently
destroyed — pre-fix, replicate 0 read +1.4536 against F-27's +1.08 — and it is stronger evidence for
the fix than any comparison to F-37, because F-27 shares this run's split *and* its draws.

#### What may NOT be claimed

- **Not confirmatory, and it cannot touch [F-37](#f-37).** Post-hoc, validation split, invented after
  the confirmatory result was known. F-37 remains the study's answer to its primary question.
- **One scale, one budget.** 160M at 30% + W4 only. Nothing here says whether the reversal holds at
  410M or 1B, or at W8 where [F-05](#f-05) says the mechanism is inert to begin with.
- **One recovery configuration.** 50 steps at lr 1e-5. [F-43](#f-43) showed the *magnitude* of the
  gap moves with step count, though its sign did not over 50–200 steps.
- **Do not pool with [F-42](#f-42) or [F-43](#f-43).** Different learning rates and step counts, so
  those draws are not exchangeable with these. This run's p = 0.0078 stands on its own eight draws.
- **The recovery slice is train-split data.** Retention is measured on validation, disjoint from
  both, so the improvements are genuine generalisation — but a different recovery corpus could give
  a different magnitude.

#### The claim this licenses

**At 160M and 4 bits, joint's layerwise advantage does not survive equal end-to-end recovery, and
reverses: sequential is the better starting point when any recovery is available.** Sign-consistent
across eight paired draws at p = 0.0078, exploratory.

This closes the last standing defence of joint compression in this study. [F-37](#f-37) found the
direct advantage too small to matter and not growing with scale; the remaining argument was that
joint might be a better *initialisation*. On this evidence it is a worse one.

#### Guards and provenance

| Guard | Result |
| --- | --- |
| `mask_sparsity` before vs after | **0.2996961805555556**, identical in all 16 cells |
| `fake_quant_forward_calls` | **9,600** in all 16 — W4 live throughout, and equal across arms |
| `assert_budgets_match` | passed in all 8 replicates — 50 × 2 × 4 × 512 = 204,800 tokens |
| Recovery/calibration disjointness | **verified at runtime**, not asserted — the run exits non-zero on any overlap (B-54 fix) |
| `git_commit` | **0 nulls in 16 records**, against F-43's 1 — the [B-53](#4-bugs-found-that-would-have-invalidated-results) timeout fix holding |
| Failures | 0 |

Both arms of every replicate ran on one host, under one `method_version`, with CPU evaluation
throughout, so the absolute retentions are directly comparable to [F-42](#f-42) and [F-43](#f-43).

### F-43 — A recovery phase that works: F-42's instrument is exonerated, and joint's advantage still does not survive {#f-43}

> ⚠️ **CORRECTED 2026-08-23 — [B-54](#4-bugs-found-that-would-have-invalidated-results).** Both arms
> in this finding were compressed against the **recovery slice** (1600 sequences) rather than the
> 128-sequence calibration set, so the "Recovery slice — disjoint from calibration" guard row below
> is **wrong as implemented**: the arms were recovered on exactly the data they were calibrated on.
> Both arms still saw byte-identical data, so **the gap, the trajectory and the inversion all stand
> as internal comparisons**. What does not stand is comparability of the *absolute* before-recovery
> retentions to [F-37](#f-37), which calibrates on 128 sequences. B-54 also argues the bug's bias
> ran *against* the reported inversion, so this result is more likely understated than inflated.

**2026-08-23 · sequential at `b680e09`, joint at `e9bf9af` · POST-HOC EXPLORATORY · validation split
· ONE PAIRED DRAW · NOT CONFIRMATORY.** Cannot alter, replace or reinterpret
[F-37](#f-37). Config `configs/experiments/recovery_ablation_160m_w4_gentle.yaml`; records in
`outputs/recovery_ablation/` under the `_gentle` id.

[F-42](#f-42) degraded both arms, so it could not test durability and could not separate
"overfitting to the slice" from "learning rate too large". This changes **one** variable —
lr 5e-5 → **1e-5** — holding steps, tokens, optimiser, schedule, clipping, precision, seed, budget
and data at F-42's values, and adds mid-recovery evaluation every 50 steps.

#### The trajectory

Pythia-160M, 30% + W4, dense perplexity 36.9741, masks frozen, W4 fake quantisation live.

| step | sequential | joint | gap |
| --- | --- | --- | --- |
| 0 | 56.1181 | 57.5629 | **+1.4448** |
| 50 | **68.5175** | **67.4093** | −1.1082 |
| 100 | 64.9433 | 63.8432 | −1.1000 |
| 150 | 63.5475 | 62.5161 | −1.0314 |
| 200 | 63.3258 | 62.2882 | −1.0376 |
| **after (CPU)** | **63.3266** | **62.2890** | **−1.0376** |

Three things follow, in decreasing order of how much weight they can carry.

**1. A non-destructive recovery phase exists — F-42's failure was the learning rate, not the
concept.** At lr 1e-5 recovery *adds* quality: **+7.2086 pp** (sequential) and **+4.7262 pp**
(joint), against F-42's −2.9100 and −4.6048 pp at 5e-5. The prerequisite question the gentle probe
was built to ask is answered **yes**, and F-42's suspicion of its own instrument was correct.

**2. Two hundred steps badly overshoots, for both arms.** Both peak at step 50 — +12.3994 pp
(sequential) and +9.8464 pp (joint) — then decline in parallel, giving back 5.19 and 5.12 pp by step
200. **The optimum is at or before step 50, and this run cannot locate it more precisely**: with
probes only every 50 steps, the true peak could be at 50, 25 or 10. Any follow-up needs finer early
probes.

**3. The gap inverts by step 50 and then barely moves.** Joint starts +1.4448 pp ahead and ends
**1.0376 pp behind**, and the offset sits between −1.03 and −1.11 pp at all four checkpoints. So in
this draw, joint's advantage does not survive a recovery phase that *helps* — which is the fair
version of the test F-42 could not run.

#### What this does to F-42's reading

F-42 recorded two candidate readings and could not separate them. This narrows them.

F-42 found joint degrading **more** under a destructive phase (−4.6048 against −2.9100 pp). This
finds joint improving **less** under a beneficial one (+4.7262 against +7.2086 pp). **Two opposite
perturbation regimes, the same direction**: the joint solution responds worse to end-to-end gradient
recovery. That is no longer attributable to an over-strong probe, because one of the two regimes is
not over-strong.

So F-42's "the joint solution is more fragile to global gradient perturbation" survives and
generalises to "the joint solution responds worse to global gradient recovery, beneficial or
destructive". Its stronger sibling — *joint's advantage is not durable under fair recovery* — now has
**one** draw of direct support where before it had none.

#### What may NOT be claimed

- **This is one paired draw. It is not an effect size**, and −1.0376 pp must not be quoted as one.
- **The four checkpoints are not four replicates.** They are autocorrelated points on a single
  trajectory from a single draw. Their agreement shows the offset is stable *within* this run; it is
  not evidence it would recur in another.
- **Do not pool this with F-42 into one sign test.** F-42's three draws ran at 5e-5 and this one at
  1e-5; they are not exchangeable, so 4/4 in the same direction is not a p = 0.125 result. The
  honest statement is *two regimes, consistent direction, one draw in the regime that matters.*
- **[F-37](#f-37) is untouched.** Post-hoc, exploratory, validation-only.

#### A practical implication, flagged as a hypothesis

If end-to-end recovery is available after compression, **sequential appears to be the better
starting point** — it both starts lower and ends higher. That would weaken the case for joint beyond
what F-37 already reports, since F-37's negative verdict is about magnitude while this would be about
sign. **From one draw this is a hypothesis, not a finding**, and it is exactly the claim that needs
R=8 before it goes in a paper.

#### Guards and provenance

| Guard | Result |
| --- | --- |
| `mask_sparsity` before vs after | **0.2996961805555556, identical in both arms** ([B-46](#4-bugs-found-that-would-have-invalidated-results) per-row quantisation) |
| `fake_quant_forward_calls` | **62,208 in both arms** — W4 live throughout; the count exceeds F-42's 38,400 because the probes add forward passes, equally in both arms |
| `assert_budgets_match` | passed — 200 × 2 × 4 × 512 = 819,200 tokens in both arms, **restored from the record across the resume** so the gate still fired |
| Recovery slice | disjoint from calibration by construction, 1600 candidates, overlap 0 |
| Mean / final loss | sequential 3.4980 / 3.4393; joint 3.5199 / 3.4673 |
| GPU probe vs CPU evaluation | step-200 probe against the baked CPU number: **0.0008 pp** (sequential), **0.0008 pp** (joint) — also confirms `bake_recovery_modules` preserves the effective weight |

⚠️ **The two arms were produced at different commits** — sequential at `b680e09`, joint at
`e9bf9af`, because the run was paused and resumed. The diff between them touches only
`scripts/run_recovery_ablation.py` resume bookkeeping and its tests; **no compression, recovery or
evaluation code changed**, so the pair is numerically comparable. Recorded rather than glossed
because a comparison spanning two commits is the kind of thing that has to be checkable.

⚠️ **The joint record carries `git_commit: null`.** Diagnosed rather than shrugged at, and the cause
matters beyond this run — see [B-53](#4-bugs-found-that-would-have-invalidated-results). The joint
arm ran at `e9bf9af`.

### F-42 — The recovery ablation closes the gap, but it degraded both arms, so it does not answer the question it was built to ask {#f-42}

> ⚠️ **CORRECTED 2026-08-23 — [B-54](#4-bugs-found-that-would-have-invalidated-results).** As in
> [F-43](#f-43), both arms were compressed against the **recovery slice** rather than the
> 128-sequence calibration set, so the "disjoint from calibration by construction" guard row below
> is **wrong as implemented**. Both arms saw byte-identical data, so every within-run comparison
> here stands. **One claim below is withdrawn:** that the +1.2772 pp before-recovery gain being
> "consistent with F-37's +1.0120 pp on test" shows the compression half reproduces. The two used
> different calibration sets (1600 sequences against 128), so that agreement was never apples to
> apples and cannot be cited as a reproduction check.

**2026-08-22 · `0ddb139` · POST-HOC EXPLORATORY · validation split · NOT CONFIRMATORY.** Cannot
alter, replace or reinterpret [F-37](#f-37). Config
`configs/experiments/recovery_ablation_160m_w4.yaml`; records in `outputs/recovery_ablation/`
(a separate tree, invisible to `audit_confirmatory_run.py` and to the confirmatory resume logic).

Pythia-160M at revision `50f5173d`, the frozen headline budget (30% unstructured + W4 symmetric
per-output-channel), dense perplexity **36.9741**, 3 paired calibration replicates, 6 cells,
**4.26 h** of recovery compute. Both arms then received an **identical** short global recovery
phase: 200 AdamW steps, **819,200 tokens**, lr 5e-5, linear warmup + cosine decay, seed 1234,
pruning mask frozen, W4 fake quantisation live through a straight-through estimator.

#### What happened

| rep | seq before | seq after | joint before | joint after | gain before | gain after |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 56.1181 | 53.7044 | 57.5629 | 53.6680 | **+1.4448** | −0.0364 |
| 1 | 57.0458 | 53.4488 | 58.2798 | 53.1176 | **+1.2340** | −0.3312 |
| 2 | 56.3043 | 53.5849 | 57.4573 | 52.7000 | **+1.1530** | −0.8849 |
| **mean** | 56.4894 | **53.5794** | 57.7667 | **53.1619** | **+1.2772** | **−0.4175** |
| sd | | 0.1279 | | 0.4855 | 0.1507 | 0.4308 |

**The joint gain went from +1.2772 pp (3/3 positive) to −0.4175 pp (0/3 positive)**, a shift of
−1.6947 pp, and it shrank in **3/3** replicates. Literally, that is the third pre-registered
outcome: *the gap closes*.

**But recovery degraded both arms, which is the finding that governs how the rest may be read:**

| | mean change | sd | direction |
| --- | --- | --- | --- |
| sequential | **−2.9100 pp** | 0.6143 | worse in 3/3 |
| joint | **−4.6048 pp** | 0.6473 | worse in 3/3 |

A phase called *recovery* that costs both arms 2.9 and 4.6 pp of retention did not recover
anything. The pre-registered reading of "the gap closes" — *joint's initialisation is not a durable
advantage under equal global recovery* — presumes a recovery phase that at least holds quality
steady. This one did not, so the premise it rests on is not satisfied.

#### The instrument, diagnosed

Three things say the probe is mis-specified rather than revealing:

1. **Training loss fell while validation quality fell.** Mean loss over the run was 3.14–3.28 and
   final loss 2.89–2.95 in every cell, so the objective was being minimised throughout, while
   evaluated retention dropped. The phase moved weights toward the 819k-token recovery slice at the
   expense of general language-model quality. At this budget that is consistent with either
   overfitting to the slice or a learning rate too large for the compressed initialisation, and
   **these six cells cannot distinguish those two.**
2. **Four distinct starting points converge on a narrow band.** Pre-recovery retention spans
   56.12–58.28; post-recovery it sits at 53.5794 ± 0.1279 (sequential) and 53.1619 ± 0.4855
   (joint). The perturbation is large enough to overwrite where each arm started.
3. **The direction of the degradation is systematic, not noise.** Joint degraded *more* than
   sequential in **3/3** replicates and ends with ~4× the spread.

Point 3 is the one that stops this being purely an instrument failure. If the perturbation merely
erased the initialisation, the arms would land together with no consistent ordering; instead joint
lands **consistently below** sequential (0/3 positive) and less stably. **That is a real, consistent
directional signal that the joint solution is more fragile to global gradient perturbation** — but
"more fragile under an over-strong probe" is a weaker and different claim than "its advantage is not
durable under fair recovery", and only the first is supported here.

#### What this does and does not license

- **Supported:** before recovery, joint leads sequential by **+1.2772 pp** at 160M/W4 on
  validation, 3/3 positive. ~~Consistent with [F-37](#f-37)'s **+1.0120 pp** on test, so the
  compression half of the pipeline reproduces.~~ **Withdrawn per
  [B-54](#4-bugs-found-that-would-have-invalidated-results):** the two figures come from different
  calibration sets, so their agreement is not a reproduction check.
- **Supported:** under this specific 200-step lr-5e-5 phase, joint's advantage does not survive,
  and joint degrades more than sequential in 3/3 replicates.
- **NOT supported:** that joint's advantage is not durable under recovery in general. That requires
  a recovery phase which does not itself destroy 3–5 pp of retention.
- **NOT supported:** any statistical claim. Three paired draws reach at best **p = 0.25** on a
  two-sided sign test. This is an effect size and a direction.
- **The [F-38](#f-38) hypothesis that motivated the ablation did NOT occur.** F-38 found the local
  objective improving where the global one does not, raising the possibility that the joint solution
  holds structure the layerwise objective cannot translate into quality — which would have shown up
  as the *gap growing*. It did not grow in any replicate. This is evidence **against** that reading,
  at this scale and budget, with the instrument caveat above attached.
- **[F-37](#f-37) is untouched.** This is post-hoc, exploratory, validation-only, and F-37 already
  reports that no cell meets the §6.3 bar. The ablation neither rescues joint nor further damages
  it.

#### Provenance and guards

Every fairness guard fired clean, which is why the numbers above are comparable at all:

| Guard | Result |
| --- | --- |
| `mask_sparsity` before vs after | **0.2996961805555556, identical in all 6 cells** — no regrowth, no drift |
| `fake_quant_forward_calls` | **38,400 in all 6 cells** — W4 stayed live; recovery never silently became FP32 |
| `assert_budgets_match` | passed — 200 steps × 2 × 4 × 512 in both arms |
| Recovery data | **disjoint from calibration by construction** — all 128 calibration indices excluded, 1600 candidate sequences, overlap 0 |
| Identical between arms | data, order, budget, optimiser, schedule, precision, clipping, device, seed |

The realised sparsity **0.299696** is the per-output-row integer quantisation from
[B-46](#4-bugs-found-that-would-have-invalidated-results), not a budget miss.

**Recovery data was made disjoint from calibration deliberately**, and the choice is worth
recording: both arms are *fitted* on the calibration sequences, so recovering on the same
sequences would partly be re-fitting on seen data and would flatter whichever arm had underfitted
them.

⚠️ **Provenance gap in these records.** The ablation writer does not stamp `git_commit`,
`method_version` or `hardware`, unlike `ExperimentTracker`. The run was made at **`0ddb139`** on the
Omen with GPU compression and GPU evaluation, recorded here because the records do not carry it.
Fixed in the writer for any future run; these six are not re-run to backfill a field.

#### If this question is worth pursuing

A gentler probe first — lr 1e-5, or 50 steps, or an early stop monitored on held-out data — chosen
to produce a phase that **improves or holds** both arms. Only then does "does joint's advantage
survive recovery?" have a fair test. Until such a phase exists, the durability question is **open**,
not answered negatively. Sizing note: at 3 draws nothing statistical is reachable, so a serious
attempt needs R=8 and roughly 11 h of recovery compute at this budget.

### F-41 — External validation on Qwen2.5-0.5B: the W4/W8 structure transfers, the verdict does not change {#f-41}

**Date:** 2026-08-14 · **65 cells, 16/16 pairs, 0 failures, R = 8 at both budgets, test split.**
**Exploratory. This leg cannot alter [F-37](#f-37)** and is not a fourth scale point. Report:
[`results/evidence/qwen_external_validation.txt`](../results/evidence/qwen_external_validation.txt),
regenerable with `python scripts/report_confirmatory.py --models qwen2.5-0.5b`.

#### The result

| Budget | R | Mean gain | sd | Positive | Raw *p* | Holm *p* | §6.3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| moderate, 30% + W8 | 8 | **−0.0321 pp** | 0.022 | 1/8 | 0.0703 | 0.1406 | **NO** |
| aggressive, 30% + W4 | 8 | **+0.4213 pp** | 0.413 | 7/8 | 0.0703 | 0.1406 | **NO** |

**Neither cell meets §6.3**, and neither reaches significance before or after correction. The W4
cell fails on **both** criteria — 0.58 pp short of the 1.0 pp bar *and* not sign-consistent, with
rep4 at −0.4407 pp.

#### What transfers, and what that is worth

**The W4/W8 structure reproduces on a second model family.** W4 positive (+0.42 pp, 7/8), W8
negative (−0.03 pp, 1/8). That is the one structural claim which survived the freeze on Pythia
([F-37](#f-37) §6), and it now holds across two families, two tokenisers and two training corpora.

Across both families the picture is consistent:

| | 160M | 410M | 1B | **Qwen 0.5B** |
| --- | --- | --- | --- | --- |
| 30% + W4 | +1.0120 | +0.9348 | +0.1316 | **+0.4213** |
| 30% + W8 | +0.0381 | +0.0289 | −0.1794 | **−0.0321** |

**Every W4 cell is positive; every W8 cell is near zero or negative.** Two W8 cells carry a
small positive mean (+0.0381 at 160M, +0.0289 at 410M) but neither is sign-consistent, so
neither is distinguishable from zero -- "at or below zero" would overstate it. **And no cell
in either family meets the practical-importance bar.**

**Qwen is not a scale point and must not be plotted as one.** Different tokeniser, vocabulary
(151,936) and corpus mean absolute perplexity is incomparable; only retention ratios and the sign
transfer. Its 358M targeted parameters happen to sit between 410M and 1B, which makes the
temptation to interpolate obvious and wrong.

#### The validation estimate overstated the test result, again

| | Validation (1 draw) | Test (R=8) | Movement |
| --- | --- | --- | --- |
| W4 | +0.6216 pp | **+0.4213 pp** | **−0.20 pp** |
| W8 | −0.0144 pp | **−0.0321 pp** | −0.02 pp |

**Third time this project has seen a validation figure fail to hold on test** — 160M fell
+1.69 → +1.01, 410M rose +0.39 → +0.93 ([F-37](#f-37)), and now Qwen falls +0.62 → +0.42. The
direction varies; the unreliability does not. It is the clearest available argument that the
two-split design earned its cost.

**A partial reading would also have overstated it.** At 5 of 8 draws this cell read ~+0.5 pp with
5/5 positive; the last three draws included rep4's −0.4407 pp, which removed both the magnitude and
the sign consistency. Logged at the time as *"not to be read as final"* — and it was not.

#### Conditions

| | |
| --- | --- |
| Revision | `060db6499f32faf8b98477b0a26969ef7d8b9987`, single value across all 65 records |
| Host / split / device | single host · all `test` · all CPU evaluation |
| `method_version` | **4**, uniform |
| Null `git_commit` | **0** — cleaner provenance than the primary, which has two ([F-37](#f-37)) |
| Frozen order | **P→Q at both budgets** ([F-40](#f-40)); no `sequential_qp` cell exists on this leg, so [B-50](#4-bugs-found-that-would-have-invalidated-results)'s dropped-pair mode cannot arise |
| Mean retention | pruning 95.11% · quantisation 99.87% (W8) / 79.56% (W4) · sequential 76.94% (W4) · joint 77.36% (W4) |

#### What the paper may say from this

**May claim:** the W4-specific structure of the primary result appears in a second model family, at
a magnitude that also fails the pre-registered practical bar; and that the joint advantage at 8 bits
is absent or slightly negative in both families.

**May not claim:** that Qwen is a scale point, that its perplexity is comparable to Pythia's, that
anything here is statistically significant (nothing is, corrected or not), or that this leg
strengthens the primary result. It is external *validity* evidence for a structural pattern, not
additional evidence for an effect size.

---

### F-40 — Qwen2.5-0.5B sequential order frozen: P→Q at both budgets {#f-40}

**Date:** 2026-08-11 · **Validation split, one calibration draw, seven cells, all `success`.**
Exploratory and outside the frozen primary result. Config:
`configs/experiments/qwen_order_selection.yaml`. Dense validation perplexity **17.7758**.

#### The measurement

| Budget | P→Q | Q→P | Margin (Q→P − P→Q) | Frozen | Basis |
| --- | --- | --- | --- | --- | --- |
| moderate, 30% + W8 | **95.2854%** | 95.1916% | **−0.0937 pp** | **P→Q** | pre-declared **fallback** |
| aggressive, 30% + W4 | **77.3993%** | 76.0485% | **−1.3507 pp** | **P→Q** | **measured** |

#### Why the two budgets are frozen on different grounds

**Aggressive is measured.** 1.351 pp is far outside single-draw noise, and it is the same direction
and rough magnitude as every Pythia W4 cell (+4.26, +6.82, +2.15 pp). The mechanism recorded in
`protocol.py` predicts it: Q→P reuses dense-fitted scales against a post-pruning distribution, which
is nearly free at 8 bits and punishing at 4. **The staged rule in the config — one draw first, more
only if the orders fail to separate — takes its measured branch here.**

**Moderate is a fallback, not a measurement.** 0.094 pp is well inside noise; one draw cannot
separate two orders at a near-lossless precision. **Additional draws were deliberately not run**,
and the reasoning is worth recording because it could look like a shortcut:

> P→Q is *both* the single-draw winner *and* the pre-declared fallback for an indistinguishable
> comparison. No replicate count can change what gets frozen — only the confidence with which the
> indistinguishability is stated. Four more W8 cells (~2 h) could flip the sign of a 0.094 pp margin
> without flipping the decision.

This is [F-28](#f-28) repeating on a new family: there, five paired draws at W8 produced a mean
margin of 0.18 pp with an inconsistent sign, and the same fallback fired. The cost of that
thoroughness bought a sentence, not a different baseline. **Recorded as ARBITRARY**, exactly as the
Pythia W8 cells are, and it carries the same caveat as [limitations §7](limitations.md): the order
uncertainty at W8 is comparable to the W8 effect itself.

#### Screening joint gain — selection surface, NOT a result

| Budget | Best-of sequential | Joint | Gain |
| --- | --- | --- | --- |
| moderate, W8 | 95.2854% | 95.2710% | **−0.0144 pp** |
| aggressive, W4 | 77.3993% | 78.0208% | **+0.6216 pp** |

**The W4/W8 structure transfers to a different model family.** W8 is inert-to-slightly-negative and
W4 is positive, which is the one structural claim that looked the same before and after the freeze
on Pythia ([F-37](#f-37) §6). One draw on a selection surface, so it is a screening signal and
nothing more — but it is the signal the confirmatory grid will either replicate or not.

**Qwen tolerates the recipe far better than the small Pythias:** 77.4% retention at 30% + W4 against
55.8% (160M) and 57.6% (410M). Its screening gain, +0.62 pp, sits below both. That is what the
headroom explanation in [limitations §6](limitations.md) predicts — a baseline that loses less leaves
less for a better layer solution to recover — and it makes the confirmatory outcome predictable in
advance rather than after, which is the honest order.

#### What was frozen, mechanically

`FROZEN_SEQUENTIAL_ORDER` now carries `("qwen2.5-0.5b", "aggressive")` and
`("qwen2.5-0.5b", "moderate")`, both `SEQUENTIAL` (P→Q), with the margins recorded in
`FROZEN_ORDER_EVIDENCE`. `qwen_validation.yaml` previously refused to build a plan and now builds
**65 executable cells, 16 pairs**. Two tests were updated: one asserted exactly six frozen cells,
the other used Qwen as its example of an *unfrozen* cell. Both assumptions were true until this
finding and are now stale; the guards are preserved with genuinely unfrozen cells.

---

### F-39 — Qwen2.5-0.5B external-validation leg: setup verified, arms run correctly {#f-39}

**Date:** 2026-08-11 · **Exploratory, validation split, and outside the frozen primary result.** This
leg cannot alter [F-37](#f-37); it exists to test whether the Pythia finding is a property of
transformer compression or of Pythia specifically.

#### Checkpoint and adapter

| | |
| --- | --- |
| Revision | `060db6499f32faf8b98477b0a26969ef7d8b9987`, matching the config pin; verified on disk |
| Architecture | `Qwen2ForCausalLM`, **24 blocks**, hidden 896, GQA 14 heads / **2 KV heads**, gated MLP, **tied embeddings** |
| Targeted modules | **168** = 24 × 7 (separate q/k/v, plus gate/up/down), against Pythia's 4 per block |
| Targeted parameters | **357,826,560** — 99.98% of non-embedding |
| Exclusions | `embed_tokens`, `lm_head` — **0 targeted**, verified empirically |

**The exclusions are load-bearing here in a way they are not for Pythia.** Qwen ties `lm_head` to
`model.embed_tokens`, so targeting the head would silently prune the input embedding and the arm
would stop matching Pythia's coverage. Both patterns are excluded in the model config and the
experiment config, and the selection confirms neither is targeted.

**Scale placement worth noting for the write-up:** 358M targeted parameters with **24 blocks** —
the same depth as pythia-410m. Qwen therefore sits between 410M and 1B on the parameter axis
*without* the depth confound that makes pythia-1b the shallow outlier ([F-38](#f-38)).

#### Dense baseline (validation split)

**Perplexity 17.7758** over 261,632 tokens (512 × 512), CPU. Generation diagnostics healthy —
repetition 0.15, distinct-token ratio 0.49. CPU benchmark: median **594.94 ms**, 215.1 tok/s at 4
threads, batch 1, seq 128.

**This perplexity is not comparable to any Pythia number.** Different tokeniser, vocabulary (151,936
against ~50,000) and training corpus. Only retention ratios and the sign of the joint gain transfer.

#### Budget realisation at 30% + W4 — and the B-46 arithmetic flips direction

| | Qwen2.5-0.5B | Pythia |
| --- | --- | --- |
| Mask sparsity | **0.300146** (*above* target) | 0.2997–0.2999 (*below*) |
| Numeric zero fraction | 0.338618 | 0.3171–0.3217 |
| Effective bits per weight | **4.0272** | 4.0117–4.0312 |
| Modules converted | 168 | 48–96 |
| Reload verified | **True**, max logit difference **0** | True |
| Joint updates | **168 accepted, 0 rejected** | 310–384 accepted |

**The realised sparsity lands *above* target here, where on Pythia it landed below**, and that is the
same integer arithmetic reported in [B-46](#4-bugs-found-that-would-have-invalidated-results) seen
from the other side: rows are 896 wide, `round(896 × 0.3) = 269`, and `269/896 = 0.300223`. Pythia's
768-wide rows gave `230/768 = 0.299479`. The B-46 fix guards only the shortfall, so both directions
pass — which is the first evidence that the fix generalises beyond the width it was written for.

**Packing is bit-exact on a new architecture**: the independent reload reproduced logits with a
maximum difference of **0**, on a model with grouped-query attention and a gated MLP that the packing
path had never seen.

#### Operational note

The first attempt failed on a transient `RemoteProtocolError` fetching the calibration corpus, even
though the raw dataset was already cached. Re-run with `HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1` and
it proceeded. **Use offline mode for the long grid** — a network refresh failure mid-grid would drop
cells under `continue_on_error`.

#### Status

Order selection (both sequential orders plus joint, one draw, validation) is running. **No order is
frozen for this model yet, and `qwen_validation.yaml` refuses to build a test-split plan until one
is** — verified: it raises `ProtocolError` naming the missing cell.

---

### F-38 — The mechanism does not weaken with scale; its translation into quality does {#f-38}

**Date:** 2026-08-10 · **Computed entirely from the committed confirmatory records — nothing was
re-run.** Regenerate with `python scripts/report_diagnostics.py`; committed output at
[`results/evidence/diagnostics_report.txt`](../results/evidence/diagnostics_report.txt).

These are the mechanism diagnostics behind [F-37](#f-37). The headline says *whether* joint wins;
these say *where* the difference is created and where it is lost.

#### 1. Mask disagreement is W4-specific, and flat across scale

Fraction of mask positions the joint refinement moves away from the sequential choice
(`joint_trace[].mask_divergence`, mean over layers then replicates):

| Scale | 30% + W8 | 30% + W4 | max layer at W4 |
| --- | --- | --- | --- |
| 160M | 0.0036% | **3.6896%** | 11.21% |
| 410M | 0.0009% | **2.9241%** | 9.63% |
| 1B | 0.0019% | **3.4063%** | 10.16% |

**Three orders of magnitude between W8 and W4**, confirming [F-05](#f-05)'s prediction at all three
scales on the test split rather than on six hand-picked 160M layers. And it is **flat with scale** —
1B diverges as much as 160M.

#### 2. The layerwise joint advantage is also flat across scale

| Scale | W8 layer gain | W4 layer gain | accepted | rejected |
| --- | --- | --- | --- | --- |
| 160M | 0.0120% | **1.5295%** | 384 | 0 |
| 410M | 0.0103% | **1.3898%** | 760 | 8 |
| 1B | 0.0097% | **1.4914%** | 310 | 10 |

The incumbent guard rejects more often at W8, which is the guard working: at 8 bits the joint
proposal usually is not better, so the layer keeps its sequential value.

#### 3. Final layer objective, matched layer by layer — joint wins locally at every scale

| Scale | W8 | W4 | layers joint wins at W4 |
| --- | --- | --- | --- |
| 160M | +0.0039% | **+2.1647%** | 367 / 384 |
| 410M | −0.4911% | **+2.2826%** | 752 / 768 |
| 1B | +0.1668% | **+2.3214%** | 317 / 320 |

> **A correction worth recording, because the first version of this table was wrong.**
> `relative_improvement` is **not comparable across arms** and must never be used for one. Each arm
> divides by its own naive baseline and the arms do not share one: Q→P quantises first, so at
> 1B/moderate its `naive_loss` is **9,548** where the joint arm's is **1,635,000** — same layer,
> references three orders of magnitude apart. Averaging that ratio produced an apparent **−6921%
> "solver efficiency"** for sequential, which is an artefact of the denominator. `final_loss` is the
> comparable quantity and is what the table uses.

#### The central mechanistic finding

**At W4 the local mechanism is undiminished at 1B — mask divergence 3.41%, layer gain 1.49%, layer
objective advantage +2.32%, all at or above the 160M values — while the end-to-end gain collapses
from +1.01 pp to +0.13 pp.** The joint step keeps finding better layer solutions at scale; those
solutions stop translating into model-level quality.

That is a dissociation, not an explanation, and this log does not have the evidence to close it.
Two candidates, neither tested:

- **Depth.** See the confound below. Layerwise error compensation is applied block by block, and 1B
  has *fewer* blocks than 410M.
- **Headroom.** The 1B aggressive cell retains 89.5% against 160M's 55.8%. A baseline that already
  loses little leaves little for a better layer solution to recover, so the same local advantage
  buys less end-to-end.

#### ⚠️ The scale axis is confounded with depth, and 1B is the shallow point

| Model | Blocks | Targeted parameters | Modules |
| --- | --- | --- | --- |
| pythia-160m | **12** | 84,934,656 | 48 |
| pythia-410m | **24** | 301,989,888 | 96 |
| pythia-1b | **16** | 805,306,368 | 64 |

**Pythia-1B is shallower than Pythia-410M** — 16 blocks against 24 — while being 2.7× wider in
targeted parameters. Depth is therefore **not monotone across the sweep**: 12 → 24 → 16.

This matters because the method is layerwise. Activations are captured through the already-compressed
prefix, so reconstruction error compounds with depth, and the number of blocks is the number of
opportunities for a joint step to help. **The drop in joint gain at 1B coincides with a drop in
depth**, and this design cannot separate the two. It is a property of the Pythia suite, not a choice
made here — but it was not accounted for when the scale axis was defined as targeted parameters
(§2.6), and it is a live alternative explanation for the only large movement in the trend.

#### 4. The retention metric is not doing the work

Per-token NLL is additive where retention is a ratio; if they disagreed on sign the headline would
be a metric artefact. They agree in **all six cells**:

| Scale | Budget | Mean NLL advantage (nats/token) | Positive | Retention pp | Agree |
| --- | --- | --- | --- | --- | --- |
| 160M | W8 | +0.000479 | 5/8 | +0.0381 | yes |
| 160M | W4 | +0.017990 | 7/8 | +1.0120 | yes |
| 410M | W8 | +0.000380 | 5/8 | +0.0289 | yes |
| 410M | W4 | +0.016100 | 8/8 | +0.9348 | yes |
| 1B | W8 | −0.001861 | 0/5 | −0.1794 | yes |
| 1B | W4 | +0.001466 | 4/5 | +0.1316 | yes |

#### 5. The budgets are matched, and it is now checkable rather than asserted

Within every cell all arms share target sparsity, mask sparsity, effective bits, module count and
targeted-parameter count. §3.11's matched-conditions requirement holds on the test split:

| Scale | Mask sparsity | Effective bits, W8 | Effective bits, W4 | Modules | Targeted params |
| --- | --- | --- | --- | --- | --- |
| 160M | 0.2997 | 8.0312 | 4.0312 | 48 | 84,934,656 |
| 410M | 0.2999 | 8.0234 | 4.0234 | 96 | 301,989,888 |
| 1B | 0.2999 | 8.0117 | 4.0117 | 64 | 805,306,368 |

Mask sparsity sits just below 0.30 because the per-row prune count is an integer — the arithmetic of
[B-46](#4-bugs-found-that-would-have-invalidated-results). Effective bits exceed the nominal width by
the fp32 scale overhead, which shrinks with width because the group count per weight falls.

**Realised zero fraction is higher than mask sparsity at W4 only** — 0.3171–0.3217 against 0.2997 —
because 4-bit rounding sends surviving weights to zero. The quantisation-only arm shows this cleanly:
**0.2183 / 0.2165 / 0.2321** of weights become zero at W4 with **no pruning at all**. That is a large
uncontrolled sparsity the W4 comparison carries in both arms, and it is why `mask_sparsity` rather
than the zero count is the budget of record.

---

### F-37 — THE CONFIRMATORY RESULT: no cell meets the pre-registered practical-importance bar {#f-37}

**Date:** 2026-08-10 · **A1 step 10, run once on the held-out test split, no tuning afterwards**

This is the result the study exists to produce. Everything above it is exploratory by A1's own
declaration; this is the only entry on the **test** split.

#### Conditions

| | |
| --- | --- |
| Split | **test** (all 171 records; the validation split is a declared selection surface and appears nowhere here) |
| Replicates | **R = 8** at 160M and 410M, **R = 5** at 1B, as frozen |
| Evaluation | **CPU**, 512 sequences × 512 tokens |
| Deployment benchmark | **CPU**, this host only |
| Machine | HP Omen, single host — verified identical `host_key` across all 171 records |
| `method_version` | **4**, uniform |
| Commits | the run spanned `b19b98a` … `ecaf8c7`; all intervening commits are documentation, scripts, or the two numerically inert guard fixes ([B-45](#4-bugs-found-that-would-have-invalidated-results), [B-46](#4-bugs-found-that-would-have-invalidated-results)). `METHOD_VERSION` never moved, which is the guard that compression behaviour did not change |
| Gate | `scripts/audit_confirmatory_run.py` — **171/171 cells valid, 42/42 pairs complete, 0 failures, 0 stale, 0 missing** |
| Baseline | the **frozen** sequential order per cell, not best-of-both — A1 §3 froze it on validation before test, so only that order was run. Q→P at 1B/moderate, P→Q everywhere else |
| Full report | [`results/evidence/confirmatory_report.txt`](../results/evidence/confirmatory_report.txt), regenerable with `python scripts/report_confirmatory.py` |

#### The headline

**No cell is practically important under §6.3**, which requires mean gain **≥ 1.0 pp *and* a
consistent sign across every paired replicate.**

| Scale | Budget | R | Mean gain | sd | Positive | Sign-test p | ≥1.0 pp | Sign-consistent | **§6.3** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 160M | 30% + W4 | 8 | **+1.0120 pp** | 0.781 | 7/8 | 0.0703 | ✅ | ❌ | **NO** |
| 410M | 30% + W4 | 8 | **+0.9348 pp** | 0.623 | **8/8** | **0.0078** | ❌ | ✅ | **NO** |
| 1B | 30% + W4 | 5 | +0.1316 pp | 0.186 | 4/5 | 0.3750 | ❌ | ❌ | **NO** |
| 160M | 30% + W8 | 8 | +0.0381 pp | 0.085 | 5/8 | 0.7266 | ❌ | ❌ | NO |
| 410M | 30% + W8 | 8 | +0.0289 pp | 0.052 | 5/8 | 0.7266 | ❌ | ❌ | NO |
| 1B | 30% + W8 | 5 | **−0.1794 pp** | 0.073 | **0/5** | 0.0625 | ❌ | ❌ | NO |

**The two W4 cells fail on opposite criteria, and both by a hair.** 160M clears the effect-size bar
and has one negative replicate. 410M is unanimous — the only cell in the study reaching
significance, p = 0.0078 — and lands **0.065 pp short** of the threshold. Recorded explicitly
because that is exactly the configuration in which a threshold gets quietly renegotiated, and §6.3
was pre-registered to prevent it. **The bar is not moved.**

#### The significance must be corrected for multiple comparisons, and it barely survives

Six cells were examined and the largest is being reported, which is precisely the inflation
[validity_threats.md](validity_threats.md) warned about before any result existed. Corrected:

| Cell | Raw p | Holm–Bonferroni adjusted |
| --- | --- | --- |
| **410M / W4** | **0.0078** | **0.0469** ← the only cell below 0.05 |
| 1B / W8 | 0.0625 | 0.3125 |
| 160M / W4 | 0.0703 | 0.3125 |
| 1B / W4 | 0.3750 | 1.0000 |
| 160M / W8 | 0.7266 | 1.0000 |
| 410M / W8 | 0.7266 | 1.0000 |

**The study has exactly one significant cell, and after correction it clears 0.05 by 0.0031.** The
paper must quote **0.0469**, not 0.0078, and must not say "significant at 410M" without the
correction attached.

**This compounds with the §6.3 near-miss on the same cell**: marginally significant after correction
*and* 0.065 pp short of the practical bar. Two independent criteria, both barely missed or barely
met, on one cell out of six. The honest summary is that the 410M effect is **real but small and
fragile** — not that it is established.

#### Every replicate, individually (A1 §5.1)

Retention is `100 × dense_ppl / compressed_ppl`. The CI is a paired block bootstrap over whole
evaluation windows, 10 000 resamples, one index draw applied to both arms, on per-token NLL
advantage.

**pythia-160m / aggressive (30% + W4), order P→Q**

| rep | sequential % | joint % | gain pp | 95% CI (nats/token) |
| --- | --- | --- | --- | --- |
| 0 | 56.4467 | 57.4827 | +1.0359 | [+0.01267, +0.02376] excludes 0 |
| 1 | 55.9025 | 56.3163 | +0.4138 | [+0.00169, +0.01295] excludes 0 |
| 2 | 55.1392 | 57.2509 | +2.1117 | [+0.03222, +0.04315] excludes 0 |
| 3 | 56.7151 | 56.4003 | **−0.3148** | [−0.01107, +0.00012] includes 0 |
| 4 | 55.8483 | 56.6375 | +0.7893 | [+0.00745, +0.02037] excludes 0 |
| 5 | 56.2995 | 57.7733 | +1.4739 | [+0.02025, +0.03154] excludes 0 |
| 6 | 54.6818 | 56.4863 | +1.8046 | [+0.02672, +0.03838] excludes 0 |
| 7 | 55.4459 | 56.2275 | +0.7816 | [+0.00833, +0.01955] excludes 0 |

**pythia-410m / aggressive (30% + W4), order P→Q** — the strongest cell in the study

| rep | sequential % | joint % | gain pp | 95% CI (nats/token) |
| --- | --- | --- | --- | --- |
| 0 | 57.6153 | 59.0904 | +1.4751 | [+0.02016, +0.03071] excludes 0 |
| 1 | 57.6845 | 58.2020 | +0.5174 | [+0.00351, +0.01434] excludes 0 |
| 2 | 57.5442 | 58.8484 | +1.3042 | [+0.01676, +0.02813] excludes 0 |
| 3 | 57.3335 | 58.4299 | +1.0964 | [+0.01351, +0.02442] excludes 0 |
| 4 | 57.0057 | 58.9802 | +1.9745 | [+0.02869, +0.03938] excludes 0 |
| 5 | 58.0092 | 58.4586 | +0.4495 | [+0.00170, +0.01377] excludes 0 |
| 6 | 57.2530 | 57.7330 | +0.4800 | [+0.00286, +0.01383] excludes 0 |
| 7 | 58.0401 | 58.2216 | +0.1814 | [−0.00226, +0.00843] includes 0 |

**pythia-1b / aggressive (30% + W4), order P→Q**

| rep | sequential % | joint % | gain pp | 95% CI (nats/token) |
| --- | --- | --- | --- | --- |
| 0 | 89.7111 | 90.0441 | +0.3330 | [+0.00183, +0.00553] excludes 0 |
| 1 | 89.6524 | 89.7436 | +0.0913 | [−0.00063, +0.00266] includes 0 |
| 2 | 89.4832 | 89.7806 | +0.2974 | [+0.00163, +0.00497] excludes 0 |
| 3 | 89.7740 | 89.6558 | −0.1182 | [−0.00332, +0.00053] includes 0 |
| 4 | 89.6521 | 89.7064 | +0.0543 | [−0.00106, +0.00223] includes 0 |

**pythia-1b / moderate (30% + W8), order Q→P** — joint is reliably *worse*

| rep | sequential % | joint % | gain pp | 95% CI (nats/token) |
| --- | --- | --- | --- | --- |
| 0 | 96.5362 | 96.3887 | −0.1475 | [−0.00187, −0.00120] excludes 0 |
| 1 | 96.4681 | 96.3535 | −0.1146 | [−0.00156, −0.00083] excludes 0 |
| 2 | 96.4702 | 96.2333 | −0.2369 | [−0.00283, −0.00209] excludes 0 |
| 3 | 96.5359 | 96.4137 | −0.1222 | [−0.00161, −0.00092] excludes 0 |
| 4 | 96.5866 | 96.3105 | −0.2761 | [−0.00323, −0.00250] excludes 0 |

All five intervals exclude zero on the negative side. This is a **small, reliable disadvantage**,
not noise.

The two W8 control cells at 160M and 410M are in
[`results/evidence/confirmatory_report.txt`](../results/evidence/confirmatory_report.txt) in the
same form; both straddle zero (5/8 positive, p = 0.7266).

#### Neither exploratory point estimate replicated

| Cell | Exploratory (validation, R=3) | Confirmatory (test, R=8) | Movement |
| --- | --- | --- | --- |
| 160M / W4 | +1.69 pp, 3/3, "robust" ([F-27](#f-27)) | **+1.0120 pp, 7/8** | **down 0.68 pp**, lost unanimity |
| 410M / W4 | +0.39 pp, 2/3, "indistinguishable from zero" ([F-27](#f-27)) | **+0.9348 pp, 8/8, p = 0.0078** | **up 0.54 pp**, gained unanimity |
| 1B / W4 | +0.20 pp, 3/3 ([F-32](#f-32)) | +0.1316 pp, 4/5 | roughly held |

**The 410M reversal is the most consequential.** The scale point the exploratory work wrote off as
null is the one cell that reaches significance. The two larger estimates moved *toward each other*,
which is the signature of regression from a selection surface — and it substantially weakens the
160M→410M half of the "shrinks with scale" story.

#### The scale trend, restated honestly

| Budget | 160M | 410M | 1B |
| --- | --- | --- | --- |
| 30% + W4 | +1.0120 | +0.9348 | +0.1316 |
| 30% + W8 | +0.0381 | +0.0289 | −0.1794 |

**The motivating hypothesis is NOT SUPPORTED, and that is the strongest wording the evidence
carries.** The study asked whether joint pays off *more* at scale. The observed direction is the
**opposite** — it pays off less, on the test split, at both budgets. But **the cross-scale decline is
not statistically established**, and three things stop it being called a refutation:

1. **The shape is not the monotone decline [F-32](#f-32) reported.** 160M and 410M are within
   0.08 pp of each other; effectively all of the movement sits in the 410M→1B step. Two points at
   one level and one below them is a weaker basis for a trend than three ordered points.
2. **No test compares the scales.** Every p-value here is within-cell. The differences *between*
   cells carry no significance test at all, and with three points none is available.
3. **The one large step is confounded with depth.** pythia-1b has **16 blocks against
   pythia-410m's 24** ([F-38](#f-38)) — depth is not monotone across the sweep — and the method is
   layerwise. The decline at 1B and the drop in depth cannot be separated by this design.

The defensible claim is: *the advantage did not increase with scale, and the observed direction was
opposite, but the decline is not established.*

**The W4/W8 split survives confirmation.** Every W8 cell is near zero or negative; every W4 cell is above
it. That is consistent with [F-05](#f-05)'s mechanism prediction (8.86% mask divergence at W4
against 0.46% at W8) and with the [F-33](#f-33) control, and it is the one structural claim that
looks the same before and after the freeze.

#### What the paper may claim, and may not

**May claim:**

- On the test split, at 30% sparsity and 4-bit, joint compression gives a **small positive gain over
  the frozen sequential baseline at 160M and 410M** — +1.01 pp and +0.93 pp — with the 410M cell
  **unanimous across 8 paired calibration replicates (exact two-sided sign test p = 0.0078)**.
- **No cell meets the pre-registered practical-importance threshold** of ≥1.0 pp with a consistent
  sign, so the study reports **no practically important joint gain**.
- At 8 bits the mechanism produces **nothing** at 160M and 410M and a **small reliable
  disadvantage** at 1B.
- The joint advantage **does not grow with scale**; it is flat from 160M to 410M and falls at 1B.

**May not claim:**

- That joint compression is practically important at any scale. The bar was pre-registered and is
  not met anywhere.
- A scaling law, or a monotone decline, from three points — and less so now that two of them
  coincide.
- That the 1B cells carry a significance claim in either direction: at R = 5 the best attainable
  two-sided p is 0.0625, so 1B/moderate's unanimous 0/5 is the **strongest possible** outcome at
  that R and still does not reach 0.05. That is a design limit, not a finding.

#### Limitations specific to this run

- **R = 5 at 1B cannot reach significance at any effect size.** Accepted at freeze time to fit the
  schedule; it means the 1B leg contributes effect sizes and sign consistency only.
- **Only the frozen sequential order was run**, so the confirmatory gain is measured against that
  order rather than best-of-both. Where the frozen choice was recorded as arbitrary — the W8 cells,
  per [F-28](#f-28) — the baseline could differ by up to the order margin measured there (~0.18 pp
  at 160M/W8), which is larger than the W8 gains themselves. **The W8 cells therefore carry an
  uncertainty comparable to their effect**, and no W8 conclusion should rest on the sign alone.
- **Two of 171 records carry a null `git_commit`** — `pythia-410m_pruning_aggressive_s30_b32_rep0`
  and `pythia-1b_joint_aggressive_s30_b4_rep3`. Their configs match their siblings field for field
  (modulo the intended `calibration_replicate`) and both carry `method_version 4`, and their
  timestamps fall between documentation-only commits, so the code that produced them is bounded.
  Provenance is nonetheless weaker for those two than for the other 169, and one of them is a joint
  cell inside a reported pair.
- **The 1B leg required the sweep process to be recycled** roughly every three cells: it accumulates
  ~4 GiB of commit per 1B compression cell and never releases it (commit free fell 20.24 → 1.03 GiB
  over five cells). Recycling is numerically inert — `skip_existing` re-runs nothing and each record
  is written whole — but it is the reason the run spans many process lifetimes rather than one.

#### Reproduction

```bash
python scripts/audit_confirmatory_run.py     # gate: must print AUDIT PASSED
python scripts/report_confirmatory.py        # every number above
python scripts/export_evidence.py --check    # the committed evidence set is current
```

---

### F-36 — The CPU timing pilot caught an unfrozen 1B offload setting before confirmation {#f-36}

**Date:** 2026-08-05

**Commit:** `0f05b9e` (`confirmatory-freeze-v2`)

**Config:** `configs/experiments/confirmatory_timing_pilot.yaml`

**Surface:** validation only, 493 × 512; quality values deliberately not used or reported

**Machine:** HP Omen, CPU evaluation and benchmark; RTX 4050 compression

The two-cell prelaunch timing pilot completed successfully:

| Cell | Total | Relevant decomposition |
| --- | ---: | --- |
| Pythia-1B dense | **25.12 min** | CPU quality 23.98 min; CPU benchmark 0.62 min |
| Pythia-1B joint, 30% + W4, replicate 0 | **121.40 min** | compression **65.67 min**; checkpoint save/reload/hash 0.42 min; CPU quality 55.13 min |

The compressed checkpoint independently reloaded with **0.0 maximum logit difference**, occupied
**1.145 GiB**, and recorded SHA-256
`6ea850eac1d66caf7c127b9c95577a331eb8cdac8b2166f2eff79bdf5d9ee665`.

The 121-minute figure is **not** the cost to extrapolate across the confirmatory grid. The record
resolved `compression.reconstruction.offload_blocks: false`. The 1B screening config explicitly
sets it true, and F-31 measured that path at 4 min 34 s without spilling. The main confirmatory
config never carried the setting, so its manifest froze the default false path. In this pilot the
joint apply stage alone took 65 min 24 s: **14.3× the verified offloaded compression time**.

This is B-44. It was caught before any test-split result existed. Confirmation remains blocked until
the main config explicitly freezes `offload_blocks: true`, the manifest checks it, and a new freeze
is recorded. Because F-31 proved the resident and offloaded paths bit-identical, this is an
operational correction rather than a scientific-condition change. The validation quality numbers
from this pilot remain excluded from evidence and from budget/order decisions.

**Runtime consequence.** The dense CPU measurement replaces the old extrapolation with 25.12 min.
The compressed measurement separates two real costs — roughly 55 min for the packed W4 CPU quality
path and an accidental 65.67 min for non-offloaded compression — but does not yet support a new
whole-grid total. Re-run only the joint timing cell after freezing offload; do not multiply 121.40
minutes by every compressed arm.

**Follow-up, 2026-08-05, commit `e0c06ac` (`confirmatory-freeze-v3`).** Amendment A3 pinned
`offload_blocks: true`, made both manifest generation and resume validation enforce it, and rebuilt a
clean valid manifest. The corrected joint-only timing cell then completed successfully:

| | Non-offloaded v2 pilot | Corrected offloaded v3 pilot |
| --- | ---: | ---: |
| joint apply | 65.40 min | **11.55 min** |
| full compression | 65.67 min | **11.70 min** |
| checkpoint verification | 0.42 min | **0.37 min** |
| CPU quality | 55.13 min | **40.98 min** |
| total | 121.40 min | **53.51 min** |

The independently reloaded artefact again had maximum logit difference **0.0** and the same SHA-256,
which is direct run-level confirmation that the residency correction did not change the checkpoint.
B-44 is closed before test evaluation.

The revised planning estimate is approximately **65.3 hours** before retries: 65 executable 160M
cells at the measured 7.5 min/cell (8.1 h), 65 executable 410M cells at 19.5 min/cell (21.1 h), plus
one 25.12-minute dense 1B cell and 40 compressed 1B cells conservatively costed at the corrected
joint/W4 total (36.1 h). This is an operational estimate, not a coverage change: the logical grid is
still 210 slots and the executable manifest still contains 171 records.

### F-01 — Smart App Control silently broke the environment 30 minutes after install {#f-01}

*2026-07-28 · environment*

Windows Smart App Control (`VerifiedAndReputablePolicyState = 1`) enforces user-mode code integrity
and blocks unsigned native extensions. The full test suite passed **502 tests** immediately after
install; torch stopped importing roughly thirty minutes later as the policy caught up with the
newly written binaries.

Blocked, from `Microsoft-Windows-CodeIntegrity/Operational` event 3077: `shm.dll` and
`torch_cuda.dll` (torch), `ccalendar` / `interval` / `lib` `.pyd` (pandas), `_regex` (a transformers
dependency).

**Resolved** by disabling SAC and rebooting. Verified: state `0`, enforcement `0`, and torch, pandas,
regex and datasets all import. Suite runtime fell from ~108 s to ~32 s, because torch was no longer
retrying blocked loads.

**Carry into the paper:** nothing. **Carry into reproduction instructions:** yes — a green test run is
not evidence the environment is stable on a SAC-enabled Windows machine. Re-check the policy state
before trusting a result. Disabling SAC is irreversible without reinstalling Windows.

### F-02 — The quantisation engine is `onednn`, not `x86` {#f-02}

*2026-07-28 · would have failed at conversion, after the compute was spent*

Probed rather than read from documentation:

```
torch.backends.quantized.supported_engines  ->  ['onednn']
  engine = 'x86'      -> RuntimeError: quantized engine X86 is not supported
  engine = 'fbgemm'   -> RuntimeError: quantized engine FBGEMM is not supported
  engine = 'qnnpack'  -> RuntimeError: quantized engine QNNPACK is not supported
  engine = 'onednn'   -> ok
```

The shipped configs said `backend: x86`, the name every PyTorch tutorial uses. INT8 *is* functional
under `onednn`: `quantize_dynamic` on an `nn.Linear` produces a module whose stored weight dtype is
genuinely `torch.qint8`.

**Guarded by** a `requires_torch` test asserting the shipped backend against the installed torch, so a
future upgrade that renames engines fails a test rather than a run.

### F-03 — The solver had to change for the study to be feasible at all {#f-03}

*2026-07-28 · real Pythia layer shapes, full-rank Gram, RTX 4050*

The per-output-channel exact solve (damped ALS) cannot scale: at `in_features = 8192` one row is a
4096-wide system and there are 8192 rows. Batching does not help — the batched form alone would need
~1 PB. Replaced with an error-compensated column sweep over a Cholesky factor of `H⁻¹`:
`O(in³ + out·in²)` against ALS's `O(out·|S|³)`.

| Layer | Shape | Sweep | ALS |
| --- | --- | --- | --- |
| 160m `attention.dense` | 768×768 | **0.46 s** | 4.65 s |
| 160m `mlp.dense_4h_to_h` | 768×3072 | **1.27 s** | 21.08 s |
| 410m `mlp.dense_4h_to_h` | 1024×4096 | **1.59 s** | not feasible |
| **1b `mlp.dense_4h_to_h`** | **2048×8192** | **4.12 s** | not feasible |

Peak GPU memory 2.55 GiB, inside the 6.0 GiB budget.

**Honest trade-off, for the Methods section:** ALS reaches a *better* objective where both run —
16.6% vs 8.9% improvement over naive rounding at 768×768, and 21.9% vs 11.9% at 768×3072 — because it
solves the survivors exactly rather than greedily. The sweep is used because it is the only one that
scales, and **one solver must be used for every layer and arm in a results table**. ALS is retained as
a reference implementation and both are parametrised over the Phase 5 exit tests.

### F-04 — A synthetic layer gave a wrong answer that a real layer corrected {#f-04}

*2026-07-28 · methodological note worth keeping*

The first measurement of whether the joint mask responds to quantisation used a synthetic layer
(random weights, correlated random activations). It reported **zero** mask divergence at W4. Six real
Pythia-160M layers report **8.86%**.

Real weights have heavier tails and real activations have outlier channels, both of which let rounding
reorder the saliency. The synthetic layer was too well-behaved to show the effect.

**Carry forward:** synthetic layers are adequate for *correctness* (exact sparsity, lossless packing,
objective decreases) and unreliable for *effect sizes*. Every effect size in this file is from real
layers, and labelled.

### F-05 — The joint mechanism is weak by construction, and it is precision-dependent {#f-05}

*2026-07-28 · six real Pythia-160M layers (0/5/11, attention and MLP), real calibration set, 50% sparsity*

§3.8 defines a method as joint if (a) the mask is scored under quantised weights and (b) the scales
are re-estimated after the mask moves. Measured:

| Bits | Joint vs sequential mask differs | Channels whose max-abs scale moves on refit |
| --- | --- | --- |
| W8 | **0.46%** | 0.2% |
| W4 | **8.86%** | 0.2% |
| W3 | 15.42% | 0.2% |
| W2 | 45.54% | 0.2% |

**Mechanism (a) is live at W4 and effectively inert at W8.** At W8 quantisation error is small enough
that the saliency ranking survives intact, so a joint arm's mask is nearly identical to the sequential
arm's and any gain there can only come from reconstruction ordering.

**Mechanism (b) is inert at every width.** A symmetric per-channel scale is `max|W_row| / qmax`, and
saliency pruning removes the *smallest* entries, so each row's maximum almost always survives.

> **Correction on the record.** This was first stated as *provably* inert. It is not. Because the
> saliency is activation-*weighted*, a row's largest weight can be pruned when it sits on a low-energy
> input column, and it does — 1.3% of channels in layer 11's MLP. The claim is empirical, not
> algebraic.

**Consequences for the paper:** W4 carries the headline comparison. **W8 should be read as a control**,
where a near-zero joint gain is the expected outcome rather than a failure. Reporting a W8 null as a
scale-independent finding about pipeline design would be a misreading. And W2 must **not** be adopted
to chase a larger effect — selecting a precision because it produces a positive result is what §6.3
forbids.

### F-06 — Two principled fixes for F-05, both measured, both rejected {#f-06}

*2026-07-28 · same six real layers · layer-objective joint gain, positive = joint better than sequential*

| Configuration | W8 | W4 |
| --- | --- | --- |
| max-abs scales + activation-weighted magnitude (**default**) | −0.49% | **+1.12%** |
| + error-minimising clipping scale search | −1.51% | −0.99% |
| + quantisation-aware keep-benefit scoring | −11.83% | **−16.15%** |

**Error-minimising clipping scale search** does everything it promises in isolation: it cuts *naive*
quantisation error by **12.8% at W4** (38.5% at W3, 68.1% at W2; optimal clip ratio α = 0.81 at W4,
1.00 at W8), and it makes scale re-estimation genuinely mask-dependent — **70.0% of channels move
their grid on refit at W4, against 0.2% for max-abs.** But the layer gets *worse* after
reconstruction. The two objectives are different: clipping saturates outliers, and a saturated weight
cannot be repaired by error compensation.

**Keep-benefit scoring** `B_ij = ‖X_j‖²[W_ij² − (W_ij − Q(W_ij))²]` fails for an analytic reason. For
round-to-nearest symmetric quantisation the score is bounded below by zero — if `|W| < s/2` then
`Q(W) = 0`, both error terms equal `W²`, and `B = 0` exactly; otherwise `|W − Q(W)| ≤ s/2 ≤ |W|`. And
above the step size `(W − Q(W))²` is nearly independent of `W`, leaving `B ≈ ‖X_j‖²·W_ij²` minus a
near-constant — a monotone transform of activation-weighted magnitude. So it largely *reproduces* the
ranking it was meant to improve, and where it deviates it favours weights that happen to sit near a
grid point, which says nothing about importance.

Both are retained behind `compression.reconstruction.scale_search` and `.keep_benefit_saliency`,
defaulting off. **These are reportable ablations**, not dead ends: "the obvious quantisation-aware
criterion is worse than magnitude, and here is why" belongs in the paper.

**Open direction:** a criterion consistent with error compensation needs the inverse-Hessian term
rather than a diagonal approximation. A clipping search evaluated against the *post*-reconstruction
objective would be the principled version of the scale fix, at the cost of one full sweep per
candidate ratio.

### F-07 — The mask comparison group cost 6.7× perplexity {#f-07}

*2026-07-28 · real Pythia-160M, end to end · **the largest single quality finding so far***

The first compressed run retained only 15% of dense perplexity. Isolating the two techniques found the
cause in one step:

| Arm | Perplexity | Retention |
| --- | --- | --- |
| Dense | 34.77 | 100% |
| **Quantisation only (W8)** | **34.85** | **99.8%** — essentially lossless |
| **Pruning only (50%)** | **233.94** | 15% |

Quantisation was never the problem. **All the damage was pruning**, and specifically the comparison
group.

**Mechanism.** Activation-weighted saliency multiplies every weight in an input column by that
column's norm. Ranked across the whole tensor, a low-energy column scores low *everywhere* and is
pruned out **entirely** — deleting an input feature rather than thinning it. Ranking within each
output channel makes every row keep its own top-k, so no column can be removed wholesale. §3.10
permits either, so this is a default change inside the frozen protocol, not a protocol change.

| Configuration | Perplexity |
| --- | --- |
| Pruning 50%, tensor-wide ranking | 233.94 |
| Pruning 50%, **per-output ranking** | **124.32** |
| Joint 50% + W8, tensor-wide | 231.96 |
| Joint 50% + W8, **per-output** | **122.51** |

Two controls confirm the rest of the stack is sound rather than merely less broken:

- **Reconstruction does real work.** Mask only, no reconstruction: **209.21**. With the sweep:
  **124.32** — a 41% improvement end to end, not just on the layer objective.
- **Calibration size is not a factor.** 8× more calibration data (16 → 128 sequences, fingerprint
  `60fc1307e7c7e0ac`) moved perplexity from 231.96 to **227.08**, under 3%.

### F-08 — Degradation curve on Pythia-160M, and what it means for budget selection {#f-08}

*2026-07-28 · joint arm, per-output masks, W8, pilot evaluation window, one seed*

| Budget | Perplexity | Retention |
| --- | --- | --- |
| Dense | 34.77 | 100% |
| W8 only, no pruning | 34.85 | 99.8% |
| **30% + W8** | **42.43** | **82%** |
| 40% + W8 | 60.46 | 58% |
| 50% + W8 | 122.51 | 28% |

§5.3 requires budgets that are "technically stable, measurably but non-catastrophically degraded". At
160M that is **30%, not 50%.** The screening grid's S1 (30% + W8) looks right; S2–S4 (50% + W8,
50% + W4, 70% + W4) look likely to be catastrophic at this scale.

That is itself **scale-relevant** — larger models should tolerate more sparsity — and it is a
hypothesis for Phase 7 to test, not a conclusion. **One seed, one model, one evaluation window.**

**Unresolved and explicitly not chased down:** retention at 50% is below published one-shot pruning
results on comparably sized models. Part of the gap is protocol (64 sequences at 256 tokens vs a full
test set at 2048), part may be Pythia-160M having little redundancy to give. This does **not**
invalidate the design — the study measures *differences between arms at matched budgets*, not absolute
quality — but the paper must not present these absolute numbers as comparable to the literature.

### F-09 — Phase 6 verification run {#f-09}

*2026-07-28 · joint arm, 50% + W8, real Pythia-160M, full pipeline config → run record*

| | |
| --- | --- |
| Target modules | 48 |
| Targeted parameters | **84,934,656** (the §2.6 scale x-axis; total model 162.3M) |
| Measured sparsity | **0.5000019779911747** against a 0.5 target |
| Effective bits per weight | **8.03125** |
| Storage efficiency | **0.89** |
| Reconstruction vs naive rounding | mean **+40.94%**, min +27.32%, max +65.30% across 48 layers |
| Total local steps | 192 |
| Compression wall-clock | 116.3 s |

This is the evidence that the pipeline is sound independently of the quality question in F-08: the
budget is exactly hit, the precision is real, and every layer's objective improves.

### F-10 — Phase 7 screening: only one of the four planned budgets is usable at 160M {#f-10}

*2026-07-28 · Pythia-160M `50f5173d` · **493 sequences × 512 tokens** (the whole WikiText-2 validation
split, 252,416 tokens) · calibration 128 sequences · seed 1234 · per-output masks, sweep solver, one
local step, 4 joint iterations · dense reference **36.97***

The §5.3 grid, sequential and joint at every budget. Evidence table also written to
`outputs/tables/screening_summary.md` by `scripts/summarise_screening.py`.

| Budget | Sparsity | Bits | Sequential ppl | Joint ppl | Seq ret. | Joint ret. | Joint gain (pp) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 `s1_30_w8` | 30% | W8 | 45.97 | 45.93 | **80.4%** | **80.5%** | +0.06 | **ELIGIBLE** |
| S2 `s2_50_w8` | 50% | W8 | 161.46 | 163.85 | 22.9% | 22.6% | −0.33 | CATASTROPHIC |
| S3 `s3_50_w4` | 50% | W4 | 250.25 | 256.74 | 14.8% | 14.4% | −0.37 | CATASTROPHIC |
| S4 `s4_70_w4` | 70% | W4 | 4663.88 | 4802.72 | 0.8% | 0.8% | −0.02 | CATASTROPHIC |

Thresholds applied: "measurably degraded" below 99% retention, "catastrophic" below 50%. Neither bound
is a number in the plan, so both are stated in the tool's output rather than hidden inside a verdict.

**The finding is that §5.3's grid does not contain two usable budgets at this scale.** Only S1
survives. S2 at 22.9% retention and S3 at 14.8% are broken models, and S4 at 0.8% — perplexity 4664
against a dense 36.97 — is a collapsed one. This supersedes the assumption behind the shipped
`main_scale_sweep.yaml`, whose moderate/aggressive pair was 50% + W8 and 70% + W4: **both of those are
catastrophic at 160M.**

Consistent with [F-08](#f-08), which found the same shape at the pilot window, and with much less
compute. The absolute numbers differ because the window differs (dense 36.97 here versus 34.77 at
64 × 256) and are not comparable between the two; the ordering and the verdicts are.

**Joint did not beat sequential at any budget.** The gains are +0.06, −0.33, −0.37 and −0.02
percentage points of retention. **No sign here is interpretable.** §5.5 gives screening one seed and
§6.3 requires a gain to exceed the seed spread before it counts, and the seed spread is unmeasured.
What can be said is that nothing in this grid contradicts [F-05](#f-05)'s prediction that the joint
mechanism is close to inert at W8.

Cost: 9 cells, ~9 minutes each, ~80 minutes wall-clock on the Omen.

**Consequence: the two budgets §5.3 requires cannot both be chosen from this grid.** Options are set
out in [STATUS.md](STATUS.md); the choice is a human one and is not recorded here until made.

### F-11 — Two config traps found while setting screening up {#f-11}

*2026-07-28 · neither affected the F-10 numbers, both would have later*

**The evaluation window was not the one the config asked for.** `screening.yaml` set
`data.max_eval_samples: 256`; the run evaluated 493. Two keys cap the evaluation set and
`evaluation.max_samples` is the one the evaluation path honours — it was inherited as 512 from an
included config, and the validation split holds only 493 blocks at 512 tokens. Every shipped config
happened to keep the two equal, so the trap had never fired.

Harmless here: all nine cells used the same window, so the comparison is internally valid, and 493 is
the *whole* split rather than a subsample. The config now states 512 and explains why.

**The real hazard was adjacent, and worse.** `data.max_eval_samples` is not a duplicate — when
calibration is drawn from the *same* split as evaluation, it is the size of the prefix reserved for
evaluation, and calibration comes only from beyond it. So an `evaluation.max_samples` larger than
`data.max_eval_samples` on a shared split evaluates on sequences the calibration set was drawn from,
which §4.1 forbids. It would inflate every arm's score equally, so no comparison would look wrong.

`ExperimentConfig.__post_init__` now rejects that combination, and only that combination — an earlier
attempt required the two keys to be equal always, which was wrong because their defaults differ and it
would have forbidden the arrangement every shipped config uses. Three tests cover it.

### F-12 - The arms ran on unequal optimisation budgets, and the guard that existed was never called {#f-12}

*2026-07-29 - affects every joint-gain number produced before this date*

**The joint arm received twice the solver budget of the sequential arm, in every cell of the screening
grid.** Recorded totals: joint **192** local steps per run, sequential **96**.

The cause is arithmetic. The joint arm calls the solver once per outer iteration and the default was
`joint_iterations = 4`; the sequential pipeline calls it twice (mask then reconstruct, then quantise
then reconstruct). 4 x 48 layers = 192 against 2 x 48 = 96.

This violates §3.11 directly, whose critical fairness point is that a score obtained with more
optimisation cannot be attributed to the method.

**The root cause is worse than the arithmetic.** `assert_matched_plans` was written specifically to
catch this, is exported from the package, and is covered by tests - and **nothing called it during a
real run**. The guard existed and was never wired in. A second contributing split: `LayerPlan` and
`ReconstructionConfig` each carried their own `joint_iterations` default, and they had diverged, so
changing the config default left the plan default - the one that actually runs - untouched.

**What it does and does not invalidate.** Joint received *more* compute and still did not win at any
budget, so the direction of the observation is robust: equalising can only move results against joint.
But the magnitudes were not attributable, and the comparison was not protocol-compliant. Budget
*eligibility* is unaffected, being driven by the sequential arm, and at S1 the two arms agree to
within 0.2 pp.

**Fixed:**

- `LayerPlan.reconstruction_passes(arm)` - one source of truth for solver calls per arm, next to the
  driver that makes them.
- `assert_arms_can_be_matched(plan, arms)` - a **pre-flight** check wired into `run_sweep`, so an
  unfair grid fails before spending hours rather than after. Confirmed firing in the run log.
- `joint_iterations` default **4 to 2** in *both* places, pinned equal by a test.
- `scripts/summarise_screening.py` now reads the recorded totals and marks any row whose arms differ
  as **gain NOT usable**, separately from the budget's eligibility - an unmatched note must not hide
  a catastrophic verdict.
- 10 new tests, including one tying the predicted pass count to what the driver actually spends.

**Re-measured at matched budgets** (96 passes each) for the three eligible budgets. The other three
were left as they were: catastrophic on retention regardless, with their gain marked unusable.

| Budget | Sequential | Joint | Joint gain, K=4 (unfair) | Joint gain, K=2 (matched) |
| --- | --- | --- | --- | --- |
| S1 30% + W8 | 45.97 | **45.90** | +0.06 pp | **+0.12 pp** |
| S5 30% + W4 | 66.03 | **71.87** | -5.46 pp | **-4.55 pp** |
| S6 40% + W8 | 67.10 | **67.06** | -0.06 pp | **+0.03 pp** |

**A side observation worth following up.** At S5, cutting the joint arm from four alternations to two
*improved* it - 73.17 to 71.87. More alternation made the result worse. That is the opposite of what an
alternating optimiser converging on a better solution would do, and suggests the loop wanders rather
than converges at W4. Not chased down; it bears on whether the joint arm as specified does what §3.7
intends.

### F-13 - Screening round 2: three eligible budgets, and the W4 cell is the interesting one {#f-13}

*2026-07-29 - Pythia-160M `50f5173d` - 493 x 512 window - dense **36.97** - one seed - matched budgets
for the eligible rows*

Extends [F-10](#f-10) with the two candidates the original grid never tested. Evidence table in
`outputs/tables/screening_summary.md`.

| Budget | Sparsity | Bits | Sequential ppl | Joint ppl | Seq ret. | Joint ret. | Joint gain | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1** | 30% | W8 | 45.97 | 45.90 | **80.4%** | **80.6%** | +0.12 pp | **ELIGIBLE** |
| S2 | 50% | W8 | 161.46 | 163.85 | 22.9% | 22.6% | *unusable* | catastrophic |
| S3 | 50% | W4 | 250.25 | 256.74 | 14.8% | 14.4% | *unusable* | catastrophic |
| S4 | 70% | W4 | 4663.88 | 4802.72 | 0.8% | 0.8% | *unusable* | catastrophic |
| **S5** | 30% | W4 | 66.03 | 71.87 | **56.0%** | 51.4% | **-4.55 pp** | **ELIGIBLE** |
| **S6** | 40% | W8 | 67.10 | 67.06 | **55.1%** | 55.1% | +0.03 pp | **ELIGIBLE** |

**Three budgets are eligible**, so the budget axis survives - S5 was the cell worth adding.

**S5 is the only budget where the two arms measurably differ**, and joint is the *worse* one, by
4.55 pp of retention. Every other eligible budget is a tie to within 0.12 pp. That is consistent with
[F-05](#f-05): W4 is the only regime where the joint mechanism is live, and at W4 it appears to
*hurt*. One seed, so a hypothesis rather than a result - but "the mechanism is live and it costs
quality" would be a more interesting and more awkward finding than "the mechanism is inert".

### F-14 - 410M confirmation: the budgets hold, and the joint gain changes sign with scale {#f-14}

*2026-07-29 - Pythia-410M `9879c9b5` - 493 x 512 window - dense **22.17** - one seed - matched solver
budgets (192 passes both arms, 96 target modules, verified from the records)*

The §5.3 confirmation that the frozen budgets still satisfy the selection rule one scale up.

| Budget | Sparsity | Bits | Sequential ppl | Joint ppl | Seq ret. | Joint ret. | Joint gain | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| moderate | 30% | W8 | 29.08 | 29.09 | **76.2%** | 76.2% | -0.03 pp | **ELIGIBLE** |
| aggressive | 30% | W4 | 39.51 | **38.03** | **56.1%** | **58.3%** | **+2.18 pp** | **ELIGIBLE** |

**Both budgets confirmed.** The freeze holds at 410M and §5.3's pre-1B requirement is satisfied.

A pleasing property worth noting: the aggressive budget produces almost identical *sequential*
retention at both scales - 56.0% at 160M against 56.1% at 410M. For a variable that is supposed to be
held constant across scales, that is the behaviour one would hope for.

**The result that matters, and the reason to be careful about it.** Putting the two scales together:

| Budget | 160M joint gain | 410M joint gain |
| --- | --- | --- |
| 30% + W8 (control) | +0.12 pp | -0.03 pp |
| **30% + W4** | **-4.55 pp** | **+2.18 pp** |

At W8 the gain is ~0 at both scales, which is exactly what [F-05](#f-05) predicts: the mechanism is
near-inert there, so the control behaves as a control should. At W4 - the only regime where the
mechanism is live - **the gain changes sign between 160M and 410M**, from clearly negative to clearly
positive.

That is the shape of the study's primary research question. It is also precisely the result we would
*want* to see, which is a reason for more scepticism rather than less. Specifically:

- **Two points are not a trend.** The plan says so itself about three.
- **There is no uncertainty estimate on either number**, and per [F-15](#f-15) the planned mechanism
  for producing one does not work.
- +2.18 pp exceeds the pre-registered practical-importance threshold of 1.0 pp, but that threshold also
  requires consistency across three seeds *and* exceeding the seed spread. Neither clause can currently
  be evaluated.
- The 160M and 410M numbers come from different dense references (36.97 and 22.17), which is correct -
  retention is always against a model's own dense run - but it means the comparison is between two
  ratios, not two perplexities.

**Nothing here may be reported as a scale finding yet.** What it does justify is prioritising the
uncertainty question before spending 1B compute, because the entire headline now rests on whether
+/-2 pp is inside or outside noise.

### F-15 - The run seed is inert, so the planned error bars do not exist {#f-15}

*2026-07-29 - a design problem, found by checking rather than by failure*

**Two runs of the same cell with different run seeds produce bit-identical results.** Pythia-160M,
30% + W4, pilot window:

```
runtime.seed = 1234  ->  perplexity 65.1548
runtime.seed = 2345  ->  perplexity 65.1548
```

Identical to four decimal places. Confirmed by inspection too:

- Calibration indices derive from `data.calibration_seed`, which the code comments explicitly describe
  as *"independent of the run seed, so every arm at a given scale calibrates on the same sequences"*.
- Nothing in the solver, the mask construction, or the layerwise driver is stochastic - no sampling, no
  shuffling, no dropout. The column sweep is a deterministic pass and `topk` is deterministic.

So the pipeline is fully deterministic given a fixed calibration draw, and **the run seed cannot change
the compressed model.**

**Why this matters more than it looks.** §5.5 prescribes three confirmatory seeds for the central
comparison, and §6.3's practical-importance rule requires a joint gain to exceed *the seed spread*.
Under this method:

- three confirmatory seeds would produce three **identical** numbers,
- the seed spread would be exactly **zero**,
- so the "exceeds the seed spread" clause becomes vacuous - any nonzero gain passes it trivially,
- and the paper would have **no error bars at all**, while appearing to have followed a three-seed
  protocol.

This is the same root cause as the superseded method definition: the seed policy was written for the
*original* full-model quantisation-aware-training design, where training is stochastic and seeds
genuinely produce variance. It does not transfer to deterministic post-training reconstruction. Nothing
was done wrong in following it; it simply does not measure anything here.

**The variance that does exist is in the calibration draw.** Different calibration sequences give
different Gram matrices, hence different masks, different scales, and a genuinely different compressed
model. That is what published post-training-quantisation work varies to obtain error bars.

Note the fairness requirement is preserved under such a change: §3.11 requires identical calibration
*between arms within a comparison*, not across repeats. So repeat r would use calibration draw r for
**both** arms - which also gives §6.3 the paired comparisons it asks for.

**Not actioned.** Replacing the seed axis with a calibration-draw axis is a change to §5.5's frozen
seed policy, and §6.3 forbids revisiting protocol choices after seeing results. The 410M sign flip in
[F-14](#f-14) is a result, and it is exactly what makes this decision urgent and delicate: the change
must be justified on the *mechanism* - seeds provably do nothing - and not on wanting error bars around
a number we like.

### F-16 - External review found three real bugs; fixing them improved quality by 9.7% {#f-16}

*2026-07-29 - every claim verified against the code before being accepted*

An external review of the repository raised eight technical issues. All eight were checked against the
source and **all eight were correct**. Three were algorithmic and are fixed here; the rest are recorded
below as outstanding.

Combined effect on Pythia-160M at 30% + W4, pilot window: perplexity **65.15 -> 58.85**, a 9.7%
improvement. All three fixes are quality-neutral-or-better by construction, so the direction is not a
surprise; the magnitude is.

#### (a) The joint outer loop had no acceptance test

Every iteration replaced the working state unconditionally. The solver has an accept-only-if-better
rule for a *fixed* mask, but nothing protected the outer loop across mask *changes*, so an iteration
choosing a worse mask discarded a better feasible solution the loop had already found.

This explains the unexplained anomaly in [F-12](#f-12): four alternations scoring worse than two
(73.17 against 71.87). An alternating optimiser going backwards as it runs longer is wandering, not
converging.

Now tracks an incumbent and accepts only on improvement, measured against the original dense target.
**Accept rates, six real Pythia-160M layers at 30% sparsity, K=4:**

| Bits | Proposals | Accepted | Rate | Mean gain when accepted |
| --- | --- | --- | --- | --- |
| W8 | 18 | 8 | 44.4% | 0.012% |
| W4 | 18 | 13 | **72.2%** | **1.353%** |
| W3 | 18 | 7 | 38.9% | 2.964% |
| W2 | 18 | 13 | 72.2% | 3.457% |

The guard has not neutered the mechanism: the final mask still differs from the sequential arm's in 5
or 6 of 6 layers at every width, so the arm remains joint by §3.8's definition. And the benefit when a
proposal wins rises monotonically with how aggressive the quantisation is, which is what
[F-05](#f-05) predicts. Between 28% and 61% of proposals were rejected -- every one of which the old
loop accepted and was degraded by.

#### (b) Conversion re-quantised the model it was supposed to pack

`pack_linear` recomputed max-abs scales from the reconstructed weight instead of reusing the grid the
solver worked on. The sweep can move a row's maximum, so the refitted grid need not be the same grid --
conversion would round a second time and ship a different model from the one evaluated for quality.
`verify_packing` was written to catch exactly this and **was never called**, the same failure mode as
[F-12](#f-12)'s unwired fairness assertion.

Passing the scales alone turned out to be insufficient. Re-deriving codes from an already-quantised
weight is **not idempotent in floating point**: a value sitting on a rounding boundary flips. Measured
deviation 3.7e-03, with the weight itself on-grid to 4.8e-07 (fp32 epsilon). So the driver now carries
the integer **codes** as well and canonicalises once per layer, so the stored weight is exactly
`codes x scales`. Conversion packs those codes -- a pure re-encoding, exact by construction rather
than by tolerance. Verification now runs by default inside `pack_linear`.

#### (c) Reconstruction was blockwise, not layerwise

The driver captured activations **once per block** and compressed every module in that block against
them. So an MLP down-projection was fitted against activations the *dense* up-projection produced --
inputs that never occur once the up-projection is compressed. The code comment admitted it: *"a
block's activations are captured once and used for all of its layers."* The docstring claimed
layerwise.

Fixed with per-architecture **dependency groups**, recapturing between groups:

| Architecture | Residual | Groups per block |
| --- | --- | --- |
| Pythia (GPT-NeoX) | parallel | **2** - {QKV, h_to_4h} then {attn.dense, 4h_to_h} |
| Qwen2 | sequential | **4** - {q,k,v}, {o_proj}, {gate,up}, {down} |

Pythia's parallel residual means attention and MLP entry projections both read a layernorm of the same
block input, so they are independent; Qwen2's sequential residual makes the MLP depend on the whole
attention sub-block. Cost is one extra forward pass over the calibration set per group beyond the
first. A targeted module belonging to no group now raises rather than being compressed against stale
activations.

#### (d) A fourth bug, found while measuring the fix

The dense-reference lookup matched on **model and seed only**, not on the evaluation window. So a
pilot-window run (64 x 256) was silently normalised against the screening-window dense baseline
(493 x 512) -- 34.77 against 36.97 on the same model. It produced a plausible retention figure from
two incomparable numbers. Now checks the sequence length and rejects a baseline that evaluated more
sequences than this run's cap.

#### Still outstanding from the same review

| Item | Why it matters |
| --- | --- |
| **W8 latency is not native INT8** | Every quantised layer runs through `PackedLinear`, whose forward unpacks, dequantises and calls a dense FP32 matmul. A W8 latency number today measures unpacking, not oneDNN INT8. **Do not publish W8 latency until a native runtime path exists.** |
| Resumability | `ExperimentTracker.exists()` returns true whenever the JSON file exists, regardless of `status`, config hash, code revision or dataset fingerprint. |
| CSV duplicates | Rows are appended, not upserted, so re-running a cell adds a second row for the same experiment id even though its JSON is overwritten. |
| `scale_trend()` unimplemented | Needed, and tested, *before* the main sweep -- otherwise a missing record field is discovered after the compute is spent. |
| Excess NLL as the primary metric | Perplexity is exponential, so a fixed perplexity gap means different things at different baselines. `delta NLL` is additive and comparable across scales. |
| Targeted parameters on the scale axis | The sweep populates `parameter_count` from the registry's *total*; §2.6 wants targeted non-embedding parameters, which the layerwise report already records. |
| Test-split separation | Budgets were selected after observing **validation** results, so reusing validation for the headline scale study is selection bias. Final results should move to the WikiText-2 **test** split. |
| Calibration replicates | See [F-15](#f-15). Also proposed: paired block bootstrap over evaluation windows, not over tokens, because neighbouring tokens are dependent. |
| Sequential baseline policy | `method_definition.md` promises best-of {P->Q, Q->P}; the main sweep runs only P->Q. Pick one and make the documents and the grid agree. |
| S6 as an auxiliary control | 40% + W8 has almost the same retention as 30% + W4, so running it across scales would separate "compression severity" from "low-bit quantisation changes the mask". |
| External sanity anchor | One limited comparison against a reference Wanda / SparseGPT / GPTQ implementation at matched settings, to confirm the custom solver sits in a plausible quality range. This is the check that would settle [review question 7.4](review_brief.md). |

**Every screening number now predates a method change.** [F-10](#f-10), [F-13](#f-13) and
[F-14](#f-14) were produced by the pre-fix pipeline and must be re-run before the budgets are treated
as frozen on current code.

### F-17 - Screening re-run on the corrected pipeline. The -4.55 pp result was a bug. {#f-17}

*2026-07-29 - Pythia-160M `50f5173d` - 493 x 512 window - dense **36.97** - one seed - matched solver
budgets (96 passes both arms, verified) - **supersedes [F-10](#f-10) and [F-13](#f-13)***

> 🔴 **Superseded by [F-23](#f-23); the joint-gain column below was already retracted by [F-18](#f-18).** Two further faults were found
> in the comparison itself, and the one with a direction inflated joint gain. The perplexities are
> retained as the record of what that version of the code produced; the **gain** must not be quoted.

Every earlier screening number was produced before the three algorithmic fixes in
[F-16](#f-16). All records were deleted and the full grid re-run.

| Budget | Sparsity | Bits | Sequential | Joint | Seq ret. | Joint ret. | Joint gain | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1** | 30% | W8 | 46.06 | 46.08 | **80.3%** | 80.2% | -0.03 pp | **ELIGIBLE** |
| S2 | 50% | W8 | 175.41 | 173.34 | 21.1% | 21.3% | +0.25 pp | catastrophic |
| S3 | 50% | W4 | 259.64 | 255.50 | 14.2% | 14.5% | +0.23 pp | catastrophic |
| S4 | 70% | W4 | 5238.87 | 5847.58 | 0.7% | 0.6% | -0.07 pp | catastrophic |
| **S5** | 30% | W4 | 64.70 | **63.56** | **57.1%** | **58.2%** | **+1.03 pp** | **ELIGIBLE** |
| **S6** | 40% | W8 | 68.12 | 68.04 | **54.3%** | 54.3% | +0.06 pp | **ELIGIBLE** |

**The frozen budgets survive.** The same three budgets are eligible and the same two are frozen, so
the [protocol freeze](protocol_freeze.md) stands unchanged. That is worth stating plainly: the method
fixes did not move the budget decision.

#### The headline correction

**S5's joint gain moved from -4.55 pp to +1.03 pp.** The negative result was an artefact of the
missing incumbent guard: the joint arm was discarding good solutions, and once it stops, joint comes
out marginally *ahead* rather than 4.55 points behind.

Two consequences, and the second is the important one.

**The 160M half of [F-14](#f-14)'s sign flip is void.** That finding's headline was joint negative at
160M and positive at 410M. The 160M negative was a bug, so "the joint gain changes sign with scale" is
**no longer supported** by the 160M data. If the 410M re-run also comes out positive, the story becomes
"joint is marginally ahead at 4-bit at both scales" -- far less dramatic, and correspondingly less
likely to be an artefact.

**+1.03 pp sits exactly on the pre-registered threshold** of >= 1.0 pp, which is uncomfortably close to
read anything from. And per [F-15](#f-15) there are still no error bars, because the run seed is inert.
So this is "consistent with a small positive effect at W4", not evidence of one.

#### Other movements worth noting

The harsh budgets got *worse* under the corrected pipeline: S2 sequential 161.46 -> 175.41, S4
4663.88 -> 5238.87. That is the expected direction from the dependency-group recapture. Each layer is
now fitted against activations carrying the accumulated error of earlier layers in its own block, and
at severe compression that error is larger. The old numbers flattered the harsh budgets by fitting
every layer against artificially clean inputs.

The mild budgets moved the other way or not at all: S5 sequential 66.03 -> 64.70, S1 45.97 -> 46.06.

#### A fifth bug, found while checking this grid

`mask_sparsity` and `zero_code_fraction` read **0.0000** in every record. The fields were added to
`LayerResult` but never to `to_dict()`, so the whole point of the separation -- reporting the pruning
budget apart from quantisation-induced zeros -- never reached the records. Fixed, with a test that the
three quantities reach the record and that they add up.

The distinction is real and visible now: at W4 the numeric sparsity exceeds the mask sparsity by
roughly 1.8 percentage points (0.3176 against 0.30 at S5), all of it survivors that rounding collapsed
to zero. **The pruning budget must be verified against `mask_sparsity`.**

---

### F-35 - Downstream tasks: the harness anchors to published values, and no arm difference is resolvable {#f-35}

*2026-08-04 - Pythia-160M `50f5173d`, 410M `dd47b0e`, 1B `f73d7dcc` - HellaSwag / PIQA / ARC-Easy,
**full tasks**, no subsampling - lm-eval **0.4.12**, task versions all 1.0 - **GPU** evaluation
(declared; see below) - aggressive budget 30% + W4 - one calibration draw - 9 evaluations, ~2 h 15 m -
`configs/experiments/downstream.yaml` - gap A4 / §4.3*

Closes the second of the two §-required gaps. 27 rows, `complete: true`.

#### The anchor: dense scores reproduce published Pythia values

This is the check that the harness is measuring what it claims, and it is the reason to trust
anything below.

| Model | Task | Ours (dense) | Published | Diff |
| --- | --- | --- | --- | --- |
| 160M | hellaswag | 0.2838 | ~0.285 | −0.12 pp |
| 160M | piqa | 0.6230 | ~0.620 | +0.30 pp |
| 160M | arc_easy | 0.4364 | ~0.435 | +0.14 pp |
| 410M | hellaswag | 0.3372 | ~0.337 | +0.02 pp |
| 410M | piqa | 0.6670 | ~0.668 | −0.10 pp |
| 410M | arc_easy | 0.5189 | ~0.517 | +0.19 pp |
| 1B | hellaswag | 0.3778 | ~0.377 | +0.08 pp |
| 1B | piqa | 0.7073 | ~0.707 | +0.03 pp |
| 1B | arc_easy | 0.5699 | ~0.569 | +0.09 pp |

**Worst deviation 0.30 pp across nine cells.** Comparability with published work is the entire reason
the harness was pinned rather than reimplemented (§2.7 freeze table), and this is that decision paying
off.

**It also resolves the smoke-run anomaly.** The 200-sample prefill run gave HellaSwag 0.3900 against a
published ~0.29, which was flagged as too high to shrug at. The full-task value is **0.2838**. Cause:
`--limit` takes the **first N examples, not a random sample**, so a 200-example prefix is biased. Not a
scoring defect, and worth remembering before anyone quotes a `--limit` number.

#### Compression costs real downstream accuracy, and it is measurable

Accuracy retention against each model's own dense score:

| Model | hellaswag | piqa | arc_easy |
| --- | --- | --- | --- |
| 160M | 0.993 / 0.992 | 0.988 / 0.978 | 0.966 / 0.934 |
| 410M | 0.940 / 0.944 | 0.970 / 0.975 | 0.900 / 0.897 |
| 1B | 0.965 / 0.962 | 0.964 / 0.970 | 0.935 / 0.934 |

*sequential / joint.* **ARC-Easy is the most sensitive task** -- down to 0.897 at 410M -- and HellaSwag
the least. Every arm at every scale is **demonstrably above chance** (interval clears the floor), so no
compressed model is broken; the budget degrades them measurably and non-catastrophically, consistent
with §5.3.

#### No joint-versus-sequential difference is resolvable, at any scale

| Model | hellaswag | piqa | arc_easy |
| --- | --- | --- | --- |
| 160M | −0.01 pp (0.02σ) | −0.60 pp (0.37σ) | −1.39 pp (**0.97σ**) |
| 410M | +0.14 pp (0.21σ) | +0.33 pp (0.21σ) | −0.17 pp (0.12σ) |
| 1B | −0.12 pp (0.18σ) | +0.44 pp (0.28σ) | −0.08 pp (0.06σ) |

σ is of the *difference*, from the two arms' own standard errors. **Not one of nine cells reaches 1σ**,
let alone 2. The honest statement: **these tasks at these sample sizes cannot distinguish the arms.**

That is a *bounded* null, not evidence of no effect. ARC-Easy at 160M would need roughly 4x the items
to resolve a 1.39 pp difference, and the tasks are fixed-size.

#### The endpoints disagree at 160M, and that belongs in the paper

At 160M aggressive, perplexity says **joint wins by +1.69 pp** ([F-27](#f-27)); downstream says joint is
*behind* on all three tasks (−0.01, −0.60, −1.39 pp).

Both can be true. Perplexity is average log-likelihood over natural text; multiple-choice accuracy is
an **argmax over candidate continuations**. A method can lower average NLL while degrading the
*ranking* between a correct completion and a plausible distractor -- those are different functionals of
the same distribution.

**What may not be done with this:** the downstream sign must not be used to argue against the
perplexity result, nor the reverse. Every downstream difference here is under 1σ on one draw, so it is
not evidence of anything directional. What is reportable is that **the perplexity advantage did not
transfer to a measurable downstream advantage**, which is a weaker and more defensible claim than
either endpoint alone.

#### Two limitations that are properties of the design, not of the run

* **160M HellaSwag has almost no headroom.** Dense scores 0.2838 against a 0.25 floor -- 3.4 pp of
  range. A task cannot show compression damage it has no room to show, and retention of 0.993 there
  reflects the ceiling rather than robustness.
* **One calibration draw**, per the policy declared in `downstream.yaml` before the run. Downstream
  results are **descriptive secondary endpoints**; no formal joint-superiority claim is made from them,
  and the seed-era downstream importance rule is withdrawn rather than amended
  ([protocol_freeze.md](protocol_freeze.md)). The harness standard error quantifies **task-item
  sampling only** -- it says nothing about calibration-draw variance, which [F-26](#f-26) measured at a
  1.47 pp swing on perplexity.

#### Conditions worth keeping

**GPU-evaluated, declared not assumed.** ~53,000 forward passes per model makes CPU ~150 h against
~2 h 15 m here. §4.6 binds *deployment* measurements to CPU because those are properties of the
machine; a multiple-choice accuracy is a property of the weights and the data. `benchmark.device`
is untouched, and the rationale is written into the record itself.

**Every row carries its provenance** -- commit, model revision, `METHOD_VERSION`, budget, sparsity,
bits, resolved sequential order, calibration draw and fingerprint, targeted parameter count, task
version, task split, timestamp, status.

**The sequential arm resolved its frozen order** rather than assuming P→Q ([B-42](#f-35)). At the
aggressive budget it is P→Q at all three scales, so the resolution changed nothing here -- but it
logged its evidence per cell, which is what makes that checkable.

**Timing, after the B-41 fix:** 160M 3:48-4:33, 410M 9:05-9:16, 1B 22:59-24:26 per evaluation.
Compressed cells now run at dense speed; 410M/joint was **3 h 37 m** before the fix.

---

### F-34 - Prefill and decode separated, and 30% unstructured sparsity buys no CPU latency {#f-34}

*2026-08-01/04 - Pythia-160M `50f5173d`, 410M `dd47b0e`, 1B `f73d7dcc` - **CPU**, 4 threads, batch 1
- 5 warm-up + 30 measured runs per cell x 2 rotation rounds = **60 samples per cell** - benchmark
host (i7-13620H) - gap A5 / §4.7 - `configs/experiments/prefill_decode.yaml`*

All three scales. Compression ran on GPU with block offload; every **measurement** is CPU.

| Model | Arm | Prefill @128 | @512 | Decode @128 | @512 |
| --- | --- | --- | --- | --- | --- |
| 160M | dense | 156.45 (5.2) | 593.23 (23.4) | 21.78 (1.9) | 29.68 (4.6) |
| 160M | pruning 30% | 155.43 (6.3) | 600.04 (19.7) | 23.58 (3.3) | 26.28 (1.6) |
| 410M | dense | 452.96 (51.3) | 1811.84 (125.7) | 58.61 (5.6) | 70.82 (16.2) |
| 410M | pruning 30% | 445.01 (63.3) | 1764.69 (41.5) | 54.21 (2.1) | 73.46 (9.7) |
| 1B | dense | 1061.36 (88.6) | 4223.90 (329.8) | 102.72 (9.8) | 125.36 (7.4) |
| 1B | pruning 30% | 1053.59 (36.8) | 4098.86 (46.1) | 99.85 (2.0) | 120.40 (1.8) |

Medians in ms, IQR in parentheses.

#### The split behaves as the plan says it should, at every scale

Prompt-length scaling, dense, for a **4x** longer prompt:

| Model | Prefill | Decode |
| --- | --- | --- |
| 160M | **3.79x** | 1.36x |
| 410M | **4.00x** | 1.21x |
| 1B | **3.98x** | 1.22x |

**Prefill scales linearly with prompt length; decode is nearly flat.** That is exactly the
compute-bound versus bandwidth-bound distinction §4.7 asks to be made visible, it holds at all three
scales, and a single blended generation latency would have hidden it. Decode is also **8-34x cheaper
per token than a prefill**, which is why a long generation is dominated by it.

It also **validates the implementation**. Had the decode callable silently re-run the prompt -- the
failure the module exists to prevent -- decode would have tracked prefill's 4x. It does not, at any
scale, so the cache is genuinely primed and the timed region is genuinely one token.

#### 30% unstructured sparsity does not deliver the speedup its compression ratio suggests

The honest reading needs the right comparison. Against **zero** the sign leans towards pruning:
9 of 12 cells are faster, by 1-3%. But 9/12 reaches only **p = 0.15** on an exact sign test, every
gap is **inside the IQR** -- 1B prefill @512 differs by 125 ms against a dense IQR of 330 ms -- and
at 160M the direction is 2/4, i.e. absent where the noise is lowest.

Against **what the compression would predict** the answer is unambiguous. Removing 30% of the
targeted weights predicts roughly a 30% reduction if those weights were being skipped. The measured
effect is 1-3%, an order of magnitude short.

**An unstructured mask stores zeros, and a dense BLAS kernel multiplies by them anyway.** Skipping
them needs a sparse kernel or a structured pattern (2:4 / 4:8) the hardware can exploit. So:

* the **compression-ratio** and **checkpoint-size** results stand;
* the **latency** result is a null **at the one sparsity measured**. RQ4 asks for a
  sparsity-versus-latency *curve*; this is a single point (30%) against dense, at three scales.
  Calling it a flat curve would claim more than one point can support -- what is established is
  that 30% unstructured pruning did not produce a commensurate CPU speedup.

**Do not report the compression ratio as though it implied a speedup.** The mask primitives already
support 2:4 and 4:8 (`compression/masks.py`), so a structured variant is the obvious follow-up if a
latency claim is wanted -- but it is a different experiment, and choosing it *after* seeing this null
would need declaring as such.

#### One measurement-quality note worth carrying

The **pruned arm's IQR is consistently tighter** than the dense arm's -- 46 ms against 330 ms at 1B
prefill @512, 41 against 126 at 410M. Both arms ran the same protocol under rotation, so this is
not an artefact of ordering. Most likely the dense arm simply drew more of the machine's background
noise across these particular rounds. It is a reason to prefer the **median and IQR over the mean
and std** for anything reported here, not evidence about the arms: a mean would have been dragged
around by whatever produced those tails.

#### Conditions that make these numbers comparable

* **CPU-only, on the benchmark host.** A deployment measurement (§4.6), and one results table never
  spans two machines.
* **FP32 arms only.** Per decision D1 a packed W4/W8 layer dequantises on every forward, so timing
  it would measure the unpacking kernel. The record carries the exclusion as a field rather than
  omitting it silently.
* **Model-order rotation**, arms rebuilt inside each round, so thermal drift is spread across arms
  rather than loaded onto whichever ran first.
* **Median and IQR**, not mean and std: latency is bounded below with a long right tail, and one
  scheduler preemption moves the mean while leaving the median alone.

#### Conditions and limits

* **CPU-only, on the benchmark host.** A deployment measurement (§4.6); one results table never
  spans two machines.
* **FP32 arms only.** Per decision D1 a packed W4/W8 layer dequantises on every forward, so timing
  it would measure the unpacking kernel. The record carries the exclusion as a field.
* **Model-order rotation**, arms rebuilt inside each round, so thermal drift is spread rather than
  loaded onto whichever ran first.
* **Two rotation rounds**, 60 samples per cell. Enough for the prefill/decode structure, which is a
  4x effect. **Not** enough to resolve the 1-3% dense-versus-pruned question, and that is stated as
  a bound rather than a null: what is established is that the effect is far smaller than the
  compression ratio predicts, not that it is exactly zero.
* Only **one sparsity** (30%, the frozen budgets' value). A real sparsity-versus-latency *curve*
  would need several, and on this evidence it would be flat.

---

### F-33 - The S6 control: the mechanism is precision-specific where there is a mechanism at all {#f-33}

*2026-08-01 - Pythia-160M `50f5173d` and Pythia-410M `dd47b0e` - 493 x 512 **validation** window -
**CPU evaluation** - `METHOD_VERSION = 4` - 3 paired calibration draws (replicates 0-2, the same draws
as [F-27](#f-27)) - 12 cells, ~2 h - `configs/experiments/s6_control.yaml`*

**Label, to be used verbatim in the write-up:** *secondary, validation-selected, quality-matched
mechanistic control.* Never confirmatory (Amendment A1 §5.4).

#### The question

Is the joint effect caused by low-bit **quantisation**, or merely by severe **quality degradation**?
Two recipes at nearly the same quality answer it:

| | Sparsity | Precision | 160M retention | 410M retention |
| --- | --- | --- | --- | --- |
| **S5** aggressive primary | 30% | **W4** | 56.7% | 58.6% |
| **S6** control | 40% | **W8** | 54.2% | 56.8% |

#### The S6 gain is nothing, at both scales

| Model | rep0 | rep1 | rep2 | mean | sd | positive |
| --- | --- | --- | --- | --- | --- | --- |
| 160M | −0.23 | +0.05 | −0.08 | **−0.09 pp** | 0.14 | 1/3 |
| 410M | +0.03 | +0.26 | −0.04 | **+0.09 pp** | 0.16 | 2/3 |

#### The comparison the control exists to make, draw by draw

Because the draws are paired, S5 and S6 can be differenced *within* a draw rather than compared as
two means:

| Model | rep0 | rep1 | rep2 | mean difference | positive |
| --- | --- | --- | --- | --- | --- |
| **160M** | +1.31 | +1.60 | +2.42 | **+1.78 pp** | **3/3** |
| 410M | +0.65 | −0.76 | +1.01 | +0.30 pp | 2/3 |

**At 160M the discrimination is clean and unanimous.** Matched quality, two recipes, and the W4 one
shows a joint gain of +1.69 pp while the W8 one shows −0.09 pp. Every draw agrees. That is what A1
§5.4 commissioned this control for, and it **supports a precision-specific mechanism over a
compression-severity effect** — exactly what [F-05](#f-05) predicts from 8.86% mask divergence at W4
against 0.46% at W8.

**At 410M the control is uninformative, and that is not a failure of the control.** The primary itself
shows no reliable effect at 410M (+0.39 pp, 2/3, sign inconsistent — [F-27](#f-27)), so there is
nothing there to attribute to a cause. A control can only discriminate where an effect exists. Read the
410M row as "no effect to explain", not as "the explanation failed".

**So the honest statement is narrower than "the mechanism is precision-specific":** it is
*precision-specific at the one scale where the mechanism is measurable at all.*

#### The baseline here flatters joint, which makes the null stronger

§6.1 requires joint gain against best-of {P→Q, Q→P}, and this control ran **P→Q only**, as A1 §5.4
specifies. That is the [B-30](#f-24) fault by construction — but the direction is what matters: at W8
the orders are indistinguishable and Q→P is slightly *ahead* on the mean ([F-28](#f-28)), so omitting
it makes the sequential competitor **weaker** and flatters joint.

The result is a null measured against a flattering baseline, which is a stronger null than the
arithmetic alone. The pre-committed clause in the config — *if this control shows a positive gain, run
Q→P before believing it* — does not fire.

#### A cross-check that came free

The first analysis pass accidentally included F-23's original S6 cells, which use the superseded
`_seed1234` naming. Their retention is **54.43 / 54.20%**, identical to this run's replicate 0 to two
decimals. Replicate 0 uses `DEFAULT_SEED`, so that cell *should* reproduce — and it does, across the
seed→replicate rename, the solver rewrite, the capture refactor and the offload work.

#### Limits

* Three draws reach p = 0.25 at best on an exact sign test. **No significance claim.**
* Validation split, and secondary by label. This may never be used to modify the primary budgets.
* CPU evaluation throughout, deliberately: the dense records it normalises against are CPU-evaluated,
  and retention is a ratio whose halves must share a device (B-37).
* **Pythia-1B was not run**, per A1 §5.4 — the full 1B S6 comparison is conditional on this control
  producing a useful distinction. It did at 160M, so a 1B S6 is now defensible if wanted; it was not
  run here because A1 scopes 1B to cheap layer-level diagnostics only.

---

### F-32 - Pythia-1B: both budgets hold, the orders split, and the joint gain keeps shrinking {#f-32}

*2026-08-01 - Pythia-1B `f73d7dcc` - 493 x 512 **validation** window - **GPU evaluation** (`cuda:0`) -
`METHOD_VERSION = 4` - `offload_blocks: true` - 3 paired calibration draws (replicates 0-2, the same
draws used at 160M and 410M in [F-27](#f-27)) - 19 cells, **2.9 h** - `configs/experiments/screening_1b.yaml`*

The third scale point, and the first that could be run at all -- see [F-31](#f-31). Dense perplexity
**17.9432**.

#### Budget confirmation (§5.3): both hold

| Budget | Retention, best-of sequential | Verdict |
| --- | --- | --- |
| moderate 30% + W8 | **96.34%** | eligible, but *barely* degraded |
| aggressive 30% + W4 | **89.46%** | **eligible** -- measurably, non-catastrophically |

**This satisfies §5.3 at all three scales**, which was the pre-1B requirement. Across the sweep the
aggressive budget retains 56.66% / 58.56% / **89.46%** at 160M / 410M / 1B: larger models tolerate the
same recipe far better, and the third point is a long way above the other two.

**The moderate budget is now nearly inert at 1B.** 96.3% retention leaves almost no headroom for any
arm to differ in. That is acceptable *because it is the control* -- F-05 predicts the joint mechanism
is inert at 8 bits and a null is the expected reading -- but a reader should not mistake the tiny W8
numbers below for a measurement of anything other than "nothing happens here".

#### Sequential order selection, and the orders split by budget

| Budget | rep0 | rep1 | rep2 | mean margin | Frozen |
| --- | --- | --- | --- | --- | --- |
| moderate, Q->P − P->Q | +0.12 | +0.05 | +0.13 | **+0.10 pp**, 3/3 | **Q->P** |
| aggressive, Q->P − P->Q | −2.32 | −2.24 | −1.88 | **−2.15 pp**, 3/3 | **P->Q** |

**W4 goes to P->Q, consistent in sign, as predicted before the run** -- it won by 4.26 pp at 160M and
6.82 pp at 410M. The margin narrows with scale (6.82 -> 2.15 at 410M -> 1B) but the direction is
stable across three scales and nine draws.

**W8 goes to Q->P, and this differs from the smaller scales.** At 160M and 410M the sign varied and
[F-28](#f-28) froze P->Q as the *pre-declared arbitrary fallback*. Here the sign is consistent, so the
rule fixed in the config before the run -- consistent sign, freeze that order -- selects **Q->P**. A1
§3 freezes the order per (model, budget), so a different order at a different scale is the design
working rather than a contradiction.

**Two caveats on the W8 freeze, both material:** the margin is **0.10 pp**, and three unanimous draws
reach only p = 0.25 on an exact sign test. This is consistent-in-sign, not significant, and it is on
the control budget where nothing is expected to happen anyway.

#### The joint gain, against best-of sequential (§6.1)

| Budget | rep0 | rep1 | rep2 | mean | sd | positive |
| --- | --- | --- | --- | --- | --- | --- |
| moderate 30% + W8 | −0.11 | −0.06 | −0.16 | **−0.11 pp** | 0.05 | **0/3** |
| aggressive 30% + W4 | +0.00 | +0.15 | +0.45 | **+0.20 pp** | 0.23 | 3/3 |

**The W8 control is cleanly negative at every draw**, which is the *correct* sign for an inert
mechanism measured against best-of: Q->P beats joint, so joint loses. Three scales now agree that
8 bits produces nothing.

**The W4 gain is +0.20 pp -- far below the pre-registered ≥1.0 pp practical-importance bar** (§6.3),
and one of the three draws is +0.00 to two decimals.

#### The scale trend, now on three points

| Scale | Joint gain, 30% + W4 | Draws |
| --- | --- | --- |
| **160M** | **+1.69 pp** | 3/3 above the ≥1.0 pp bar |
| **410M** | +0.39 pp | 2/3 positive, sign inconsistent |
| **1B** | **+0.20 pp** | 3/3 positive, all below the bar |

**Monotone decline across all three scale points.** [F-27](#f-27) drew that conclusion from two points;
the third agrees, and it was predicted in `screening_1b.yaml` **before the run** that a 1B gain at or
above 410M's would put the trend in doubt. It came in below.

**This runs against the study's motivating hypothesis.** The question was whether joint pays off *more*
as models grow. On three scale points it pays off **less**, and by 1B it is not practically important
at any draw.

#### A mechanism observation worth carrying

At the aggressive budget the two arms are *converging* as scale grows. At rep0 the joint and P->Q
perplexities are **20.0301 and 20.0311** -- a difference of 0.001, which is why that draw's gain reads
+0.00. The joint arm is also markedly more stable across draws (20.0301 / 20.0304 / 19.9807) than the
sequential arm (20.0311 / 20.0640 / 20.0802).

That is consistent with F-05's account: the joint mechanism acts through *mask divergence under
quantisation*, and a larger model with more redundancy has fewer weights whose keep/prune decision the
quantisation grid can flip. Speculative, and stated as such -- it would need the mask-divergence
measurement of F-05 repeated at 1B to be more than a story.

#### What this is not

* **Not confirmatory.** Validation split, which A1 §5.2 declares a selection surface, and this run
  *is* the selection.
* **Not significant.** Three draws reach p = 0.25 at best.
* **GPU-evaluated**, unlike the CPU-evaluated 160M and 410M records. Within a cell every arm and the
  dense reference share a device, so retention and joint gain are internally consistent; the drift is
  8.3e-06 relative ([F-29](#f-29)) against margins of 0.1-2 pp. **But the cross-scale table above does
  mix devices**, and that is declared here rather than left to be discovered. A1 steps 9-10 put all
  three scales back on CPU.

---

### F-31 - Per-block GPU offload, and the mask flip that made the first attempt wrong {#f-31}

*2026-08-01 - Pythia-160M `50f5173d` - 493 x 512 validation window - `METHOD_VERSION = 4`, **not
bumped** - suite 977 passing*

Offload holds **one decoder block** on the card at a time instead of the whole model. It is the
change [F-29](#f-29) left outstanding, and the one that decides whether Pythia-1B is runnable at
all: with the model resident, 1B peaks at **6.31 GiB on a 6.00 GiB card** and completes only by
spilling to host memory at 7x the solve time.

#### The first implementation was wrong, and the reproduction gate is what caught it

Captured on the host, on the reasoning that aborting at block 0 means only the embedding runs, and
an embedding lookup is a gather -- no arithmetic, so bit-identical on either device.

**That reasoning was incomplete.** GPT-NeoX also computes the **rotary `cos`/`sin`** during that
forward and passes them into every block as replay context. CPU and CUDA trigonometry disagree in
the last bits.

| | Required | First attempt | Difference |
| --- | --- | --- | --- |
| sequential | 65.261 | **65.666** | +0.405 |
| joint | 64.041 | **63.028** | −1.013 |
| joint gain | +1.08 pp | **+2.35 pp** | more than double |

`scripts/verify_block_offload.py` localised it exactly. In block 0, `query_key_value` and both MLP
projections matched **bit-for-bit**; only **`attention.dense`** differed, by **2.25 absolute** --
and then every later block with it.

That pattern names the cause. `attention.dense` is the only module in the block whose input is the
attention output, so it is the only one that sees the perturbed rotary embeddings. **A mask is a
discrete function of saliency**, so last-bit noise flipped a near-tie, and a flipped mask position
is not a small numerical difference -- it is a kept weight becoming a pruned one.

This is [F-19](#f-19)'s observation biting for real. There, 4 positions in 85 million flipped
between float32 and float64 norms and it was harmless. Here one flip in an early block propagated
through every block downstream.

**The direction is worth recording: it flattered the joint arm**, more than doubling the gain. That
is now B-14, B-17, B-22, B-23, B-30 and B-34 -- six faults, six in the same direction, none the
other way.

#### The fix took two attempts, and the second one is the interesting failure

**Attempt one: move the whole model to the device for the capture, then pull the blocks straight
back off.** Correct -- 0 of 148 parameters disagreed at 160M -- and reasoned as affordable because
no Gram factorisation is live during capture, so the 6.31 GiB peak that made 1B unrunnable was the
model and those temporaries *coexisting*, not the model alone.

**It died at 1B with `CUDA error: out of memory`** -- at the exact step offload exists to make
possible. 3.77 GiB of weights, ~0.5 GiB of cached hidden states and the forward's own activations
do not fit in the ~4.95 GiB actually free on a 6.00 GiB card. The reasoning about Gram temporaries
was right and still insufficient: it accounted for what was *absent* and not for what was present.

**Attempt two: move only the modules outside the decoder blocks** -- the embedding and the rotary
tables are what the capture needs; 3.77 GiB of decoder weights is what it must not drag along.
`_move_outside_blocks` walks the model with `recurse=False` per module, because moving a parent
would take its block children with it.

The second attempt is also what makes the 160M saving real rather than cosmetic:

| Capture strategy | 160M offloaded peak | Equivalent? | 1B |
| --- | --- | --- | --- |
| whole model to device | 0.89 GiB | ✅ bit-identical | **out of memory** |
| **pre-block modules only** | **0.60 GiB** | ✅ bit-identical | see below |

#### Verification

Weight-level equivalence, real Pythia-160M, same calibration draw, offloaded against resident:

| Arm | Parameters | Disagreeing | Worst difference | Resident peak | Offloaded peak |
| --- | --- | --- | --- | --- | --- |
| sequential | 148 | **0** | **0.000e+00** | 1.25 GiB | **0.60 GiB** |
| joint | 148 | **0** | **0.000e+00** | 1.25 GiB | **0.60 GiB** |

Compared with `torch.equal`, not `allclose`. A tolerance would hide exactly the failure above: a
flipped mask position is a large difference in one weight, not a small one everywhere.

Full-cell gate, the authoritative one:

| Arm | Required ([F-23](#f-23)) | Measured | |
| --- | --- | --- | --- |
| sequential | 65.261 | **65.2614** | retention 56.66% |
| joint | 64.041 | **64.0413** | retention 57.73% |

**Exact to four decimals.** `METHOD_VERSION` not bumped, so the existing records stay valid.

#### Pythia-1B now fits, and by a wide margin

**This is the measurement the change exists for.** Compression only, aggressive budget, same
calibration draw as a real cell:

| | Resident ([F-29](#f-29)) | **Offloaded** |
| --- | --- | --- |
| Peak, tensors allocated | — | **3.34 GiB** |
| Peak, allocator reserved | — | **4.29 GiB** |
| Peak, device level | **6.31 GiB on a 6.00 GiB card** | ~5.0 GiB observed, not instrumented |
| Outcome | completed only by spilling to host, **7x** the solve time | **no spill, 4 m 34 s** |

**Report both torch numbers and do not mix them.** `max_memory_allocated` is what the tensors need;
`max_memory_reserved` is what the caching allocator holds on the device. F-29's 6.31 GiB was a
*device-level* figure, so **reserved** is what it compares against -- quoting the 3.34 GiB against
it would overstate the headroom by a gigabyte.

**Against §5.2's 5.1 GiB ceiling (85% of 6.0), be careful.** Reserved at 4.29 GiB clears it with
0.8 GiB to spare. Device-level occupancy is reserved *plus* the CUDA context and whatever else
holds VRAM, which on this machine measured **1.04 GiB** with nothing allocated -- a laptop with a
display attached, so it is desktop plus context rather than a clean constant. A single `nvidia-smi`
sample mid-run read **5.05 GiB** total. So:

* **1B runs, comfortably, and that is settled** -- no spill, and 4 m 34 s against a resident path
  that took 7x longer on the widest layer alone.
* **The margin for 1.4B is thinner than 4.29 against 5.1 makes it look**, because the baseline is
  not free. The §5.2 go/no-go must be *measured* on 1.4B, not extrapolated from this.

A device-level peak was not instrumented; the 5.05 GiB is a spot reading, not a maximum, and is
recorded as such.

**52% at 160M is not the benefit and should not be quoted as it.** At that size the whole model is
0.65 GiB, so nothing was ever at risk of not fitting.

#### Also fixed on the way (B-35)

The first real run died at `convert` with *"found at least two devices, cuda:0 and cpu"* -- **after
every second of the compression was spent**. `LayerwiseReport.grids_by_module` holds the codes and
scales captured while a block is resident, so under offload they came back as CUDA tensors
describing weights that were now on the host. Invisible while everything lived on one device. The
grids now travel back with their block, and the unit test asserts device as well as value -- the
first version compared only parameters and sailed straight past it.

---

### F-30 - The machine policy was stricter than the protocol, and nothing in the code enforced it {#f-30}

*2026-08-01 - no measurement; an audit and a policy amendment - suite 974 passing*

The second author could not run anything: every entry document said **"the HP Omen is the only
machine that runs code"**. Audited what that rule was actually protecting.

#### Finding 1 — no code binds to this machine

Searched the whole tree for host-specific gates. There are **none**. What exists is:

| Enforcement | Where | Correct? |
| --- | --- | --- |
| `benchmark.device` must be CPU | `config.py`, `benchmarking/cpu.py` | **yes** — keep |
| `check_evaluation_device` warns off CPU | `evaluation/common.py` | **yes** — a warning is right, only *reported* numbers must be CPU |
| Backend asserted against the installed torch | `test_repository.py` | **yes**, and it is platform-adaptive already |
| Mixed-CPU warning at plot time | `generate_plots.py` | **too loose where it matters, too noisy where it does not** — see below |

`resolve_device`, `get_hardware_info` and the loaders are all portable. **The blocker was entirely
documentation.**

#### Finding 2 — the protocol never asked for one machine per *project*

`benchmarking_protocol.md` ("One machine per results table") and `methodology.md` ("one machine per
results table") both state the weaker, correct rule. CLAUDE.md, STATUS.md, `protocol_freeze.md` and
the partner handoff had hardened it into one machine per project. That was operational shorthand
from when one person had one GPU box, and it cost a collaborator all of their throughput for no
scientific gain.

**Amended to three tiers:** anywhere (tests, lint, docs, analysis) · any CUDA machine (compression,
capture, quality) · the designated benchmark host (latency, throughput, peak memory, checkpoint
size).

**Why tier 2 is portable.** Compression and perplexity are determined by the weights and the data;
across hosts they move only by floating-point reduction order. That is the ~1e-5 already measured
twice here — CPU against GPU at 8.3e-06 ([F-29](#f-29)) and CPU thread configuration at ~1e-5
([F-23](#f-23)) — against the ~1e-2 effects this study measures. A latency has no such property: it
*is* a property of the machine, and no correction makes two hosts comparable.

#### Finding 3 — relaxing the policy exposed a real bug (B-33)

`exists_valid` compared the evaluation device (B-32) but **not the machine**. Harmless while one
host existed; the moment two hosts can write into one `outputs/metrics/`, `skip_existing` would
reuse the other machine's record and put two hosts inside a single comparison — the same
unmatched-condition class as B-32, and equally invisible.

Fixed with `hardware.host_key`, built **only from fields every record already carried**
(`system`, `cpu_model`, `cpu_count_logical`, `cuda_device_names`), so the guard computes
retroactively and **invalidated no existing record**. A record predating those fields reports
`"unknown"` and is not invalidated — fail-open on absence, fail-safe on mismatch. Three tests pin
all three cases.

#### Finding 4 — the plot-time guard was backwards

It warned whenever *any* record set spanned two CPUs, including compression-only records where the
machine does not affect the number. Once two people share the work that fires constantly, and a
warning that fires when nothing is wrong stops being read. Meanwhile the case that genuinely makes a
table wrong — deployment measurements from two hosts — was only a warning.

Now: deployment-bearing records spanning hosts is an **error and a non-zero exit**; compression-only
records spanning hosts is an `INFO` line saying so.

#### One consequence for anyone verifying a refactor elsewhere

**The exact reproduction gates in this log are host-specific.** 65.261 / 64.041 at 160M and
37.851 / 37.415 at 410M came from the Omen; a different GPU runs different cuBLAS kernels, the Gram
differs in its last bits, and a near-tie in the saliency ranking can flip —
[F-19](#f-19) found 4 positions in 85 million flipping between float32 and float64 norms alone.

The portable substitute is a **host-local before/after baseline**: run the cell on your machine
before the change and after it, and require bit-identical. For isolating a refactor that is
*stronger* than an absolute target, because it holds the machine constant. The absolute gates stay,
as a final check on the benchmark host.

---

### F-29 - Two speedups, both verified numerically neutral: 2.7x on compression, 22.5x on evaluation {#f-29}

*2026-07-31 - Pythia-160M `50f5173d` and Pythia-410M `dd47b0e` - 493 x 512 validation window -
`METHOD_VERSION = 4`, **not bumped** - full rationale in
[capture_refactor_rationale.md](capture_refactor_rationale.md)*

Started as an attempt to make Pythia-1B runnable at all. Produced a ~7x speedup on exploratory cells as
a by-product, and neither change moves a number.

#### Where the time actually went

Decomposed from the stage markers in a real 160M joint cell, rather than guessed:

| Stage | Time | Share |
| --- | --- | --- |
| load model | 2 s | — |
| compress | 62 s | 14% |
| measure checkpoint | <1 s | — |
| **evaluate quality (CPU)** | **377 s** | **86%** |

**The compression this project has spent weeks optimising was 14% of a cell.** Perplexity evaluation on
CPU was the other 86%, and nobody had looked.

#### Speedup 1 — block-sequential activation capture (2.7x on compression)

`_compress_group` captured activations by running the **entire model** forward, once per dependency
group. Blocks after the current one had their outputs discarded; blocks before it were recomputed from
the embedding for every group. That is `O(blocks x groups)` full-model forwards where `O(blocks)`
single-block forwards suffice.

Replaced with the standard approach: capture block 0's inputs once, then replay **one block at a time**
over cached hidden states, advancing the cache after each block.

| | Before | After |
| --- | --- | --- |
| Compression stage, 160M | ~170 s | **62 s** |
| Block-forwards per 1B cell | ~512 | **~48** |
| Model resident on the capture device | whole model | **one block** |

**Verified bit-identical before the driver was touched.**
`scripts/verify_block_sequential_capture.py` compared the Gram both ways on all 48 targeted modules of
real Pythia-160M: **worst relative error 0.000e+00**, on the Gram and on the column norms. Not close --
exactly zero. The two strategies compute the same quantity, which is why the extra work bought nothing.

Two correctness details, both of which would have been silent failures:

* **Blocks with no targeted modules still advance the cache.** The old code skipped them, which was
  harmless when every capture re-ran the whole model. Skipping one now would leave the next block
  replaying inputs from the wrong depth -- a wrong answer, not an error.
* **`use_cache` is disabled for the duration**, in a `try/finally`. A live key/value cache accumulates
  across the repeated single-block replays and would change what the solver is fitted to. This is
  [B-28](#f-22) exactly, found first in the external SparseGPT driver.

#### Speedup 2 — GPU evaluation for exploratory runs (22.5x on evaluation)

`evaluation.device` has always been a config field, and `check_evaluation_device` **warns rather than
errors** off CPU: *"Exploratory evaluation on GPU is fine, but any number reported in the write-up must
be produced on CPU."* Nothing was using it.

| | CPU | GPU |
| --- | --- | --- |
| Perplexity, 160M dense | 36.974099 | 36.974405 |
| Time, 493 x 512 window | **345.6 s** | **15.4 s** |
| Relative difference | — | **8.3e-06** |
| Worst per-window difference | — | 2.1e-05 |

**The 8.3e-06 drift is the same magnitude as the CPU thread-configuration sensitivity already recorded
in [F-23](#f-23)** (36.9741 against 36.9744 on identical data). Floating-point reduction order, three
orders of magnitude below the ~1e-2 effects this study measures.

#### Combined effect

| | Before | After |
| --- | --- | --- |
| Compression | ~170 s | 62 s |
| Evaluation | ~377 s | 15 s |
| **Exploratory 160M cell** | **~9.3 min** | **~1.3 min** |

**~7x.** The 13-cell screening grid goes from 2 h 08 m to roughly 20 minutes, which makes
eight-replicate exploratory work cheap rather than a day's commitment.

**The confirmatory test-split run keeps CPU evaluation and its ~38 hours.** That is the rule and it is
not being touched.

#### The guard that makes GPU evaluation safe to adopt

`evaluation_device` was already recorded per run, but `exists_valid` did not compare it. So switching to
GPU evaluation would have let `skip_existing` reuse the ~50 existing CPU records inside a GPU grid,
**silently mixing devices within a single comparison** -- the unmatched-condition class of error §3.11
exists to prevent, small enough to change no conclusion and invisible without the check.

`exists_valid` now compares it, so a device change invalidates stale records loudly. `cuda` and
`cuda:0` are treated as the same backend.

#### The gates, all passed

Gram equivalence is necessary but not sufficient, so the full reproduction gates were run:

| Gate | Requirement | Result |
| --- | --- | --- |
| Gram equivalence, 48 modules | bit-identical | ✅ **0.000e+00** |
| Unit suite | green | ✅ **966 passing** |
| **160M cell** | reproduce [F-23](#f-23) | ✅ **65.261 / 64.041**, gain **+1.08 pp** |
| Exact-optimum anchor | reproduce [F-20](#f-20) | ✅ 0.6409, every module identical |
| **410M cell** | reproduce [F-25](#f-25) | ✅ **37.851 / 37.415** |
| `METHOD_VERSION` | bump if anything moved | **not needed — nothing moved** |

**No `METHOD_VERSION` bump, so the ~50 existing records stay valid and nothing needs recomputing.** That
was the main risk of touching `layerwise.py`, and it did not materialise.

Worth being precise about one thing: **the anchors do not exercise the refactored path.** Both anchor
scripts capture activations with their own hooks and call `sweep_reconstruct` directly, never entering
`compress_model_layerwise`. Their passing confirms the solver is untouched -- which it is -- and says
nothing about the capture change. **The 160M and 410M cell reproductions are the meaningful gates.**

#### Other techniques checked and rejected

Measured or reasoned per stage, not guessed:

| Technique | Why not |
| --- | --- |
| Checkpoint measurement | <1 s. Nothing to gain. |
| Agreement + generation diagnostics | ~30 s of the 377 s stage. Marginal. |
| Larger CPU evaluation batch | Pointless once GPU evaluation makes the stage 15 s |
| Smaller `max_eval_samples` for screening | Breaks comparability with existing records, for a stage GPU already fixes |
| `torch.compile` / SDPA attention | Changes numerics, for a stage that is no longer the bottleneck |
| Smaller `block_size` | Does not even reduce memory: peak 6.31 / 6.37 / 6.37 GiB at 128 / 64 / 32, because the Gram factorisation dominates |

#### A documentation correction found on the way

`sweep_reconstruct`'s docstring claims `block_size` is *"purely a memory/throughput knob; it does not
change the result."* Three block sizes gave three distinct losses -- 1.983672e+07, 1.983672e+07,
1.983673e+07 -- differing at ~5e-7 relative. Negligible in effect, smaller than the thread sensitivity
in F-23, but the guarantee as written is false and should read "does not meaningfully change the
result."

#### Still outstanding

**Per-block GPU offload**, which is what actually unblocks 1B. The refactor makes it small -- the block
loop now owns the forward, so moving one block to the device is a contained addition -- but it is a
separate change and will be verified the same way. Measured 1B peak with the model resident was
**6.31 GiB on a 6.00 GiB card**, completing only by spilling to host memory at 7x the solve time.

---

### F-28 - The W8 sequential orders are indistinguishable. P→Q is frozen by the pre-declared rule {#f-28}

*2026-07-31 - Pythia-160M `50f5173d` - 493 x 512 **validation** window, dense **36.9741** - five paired
calibration draws - `METHOD_VERSION = 4` - **1 h 42 m**, 11 cells - **resolves the contested W8 freeze
in [F-24](#f-24) / [F-25](#f-25)***

[F-24](#f-24) froze **Q→P** as the moderate-budget order on a **+0.43 pp margin from one draw**.
[F-25](#f-25) then found the direction *reversed* at 410M. This settles it across draws.

| Draw | P→Q | Q→P | Margin (Q→P − P→Q) |
| --- | --- | --- | --- |
| rep0 | 80.20% | 80.63% | **+0.43 pp** |
| rep1 | 80.44% | 80.35% | **−0.09 pp** |
| rep2 | 80.10% | 80.31% | +0.21 pp |
| rep3 | 80.33% | 80.55% | +0.22 pp |
| rep4 | 80.20% | 80.34% | +0.14 pp |
| | | **mean** | **+0.18 pp**, sd 0.19, SE 0.08 |

**Q→P ahead in 4 of 5 draws. Sign not consistent. Sign-test p = 0.375. Mean / sd = 0.97.**

#### The pre-declared rule applies, and its second branch fires

`order_selection_w8_replicates.yaml` fixed the decision *before* any of this was measured:

> Q→P ahead in all five → the freeze stands, and now on evidence. **The sign varying → the two orders
> are indistinguishable at W8. Freeze P→Q, the pre-registered primary order (§3.6), and record that the
> choice is arbitrary. Do not pick the winner of a coin toss and report a joint gain against it.**

The sign varies. **W8 is therefore frozen at P→Q**, and the choice is recorded as arbitrary rather than
as a measured preference.

Note what this is *not*: it is not "P→Q is better." It is "the two are not distinguishable, so the
pre-registered primary order is used." Q→P is ahead on the mean. Choosing it anyway would mean picking a
+0.18 pp winner out of noise and then reporting a joint gain against it — and that choice would flip the
sign of the moderate budget's headline, which is precisely why the rule existed.

#### The consequence for the moderate budget's joint gain

| Baseline | Moderate joint gain |
| --- | --- |
| P→Q — **now frozen** | **+0.07 pp** |
| Q→P — F-24's contested choice | −0.36 pp |

So the moderate budget's gain is **+0.07 pp**: a clean null, which is what [F-05](#f-05) predicts for a
mechanism that is inert at 8 bits. The −0.36 pp figure F-24 derived is withdrawn along with the Q→P
freeze it rested on.

#### An expectation of mine that was wrong twice, in both directions

Worth recording because it shows how easily a plausible variance argument misleads.

**First I expected the orders to be indistinguishable**, reasoning from the aggressive budget where
draws move retention by 0.63–0.78 pp — far more than 0.43 pp.

**Then, seeing P→Q's five draws span only 0.34 pp with sd 0.13, I reversed** and said the 0.43 pp margin
was "over three standard deviations, which could be a genuine difference."

**Both were wrong, for the same reason.** The relevant spread is not each arm's, it is the **paired
margin's** — sd 0.19 pp, from which the mean of +0.18 pp sits 0.97 sd away. Each arm is tight *and* the
margin is still noise, because the arms do not move together: a draw changes which mask each order
picks, and they respond differently by construction. [F-26](#f-26) found the same thing at W4, where the
paired difference was noisier than either arm.

**The rule was fixed in advance, so being wrong twice changed nothing.** That is the entire argument for
pre-declaring decision rules.

#### Why W8 noise is small and the margin still is not resolvable

W8 quantisation is near-lossless ([F-07](#f-07): 99.8% retention W8-only), so there is little damage for
a calibration draw to modulate — hence each arm's sd of 0.13 pp against 0.63+ at W4. But the *difference*
between two orderings at W8 is also tiny, because with an almost-lossless quantiser it barely matters
which operation runs first. **Small signal and small noise, in roughly equal measure.** No affordable
replicate count fixes that: at sd 0.19 pp, resolving a +0.18 pp margin at p < 0.05 would need R ≈ 12,
spent on the *control* budget to settle a question that does not affect the headline.

#### What is unaffected

The **aggressive** budget, which carries the study. P→Q wins there by **+4.26 pp at 160M and +6.82 pp at
410M** — margins twenty to thirty times this one, in the same direction at both scales. That freeze
stands on evidence, and the [F-27](#f-27) headline is measured against it.

---

### F-27 - 160M replicates cleanly. The effect is real there, and it does shrink with scale {#f-27}

*2026-07-31 - Pythia-160M `50f5173d` and Pythia-410M `dd47b0e` - 493 x 512 **validation** window -
three paired calibration draws at each scale, the **same** draws (replicates 0-2) -
`METHOD_VERSION = 4` - 160M leg 1 h 01 m, 7 cells - **restores the qualitative conclusion of
[F-25](#f-25) that [F-26](#f-26) had put in doubt***

[F-26](#f-26) retracted the 410M point estimate and the scale claim built on it. This replicates the
**160M** cell on the same three draws, which is what the claim needed to stand on.

| Draw | 160M gain | 410M gain |
| --- | --- | --- |
| rep0 | **+1.08 pp** | +0.68 pp |
| rep1 | **+1.65 pp** | −0.50 pp |
| rep2 | **+2.34 pp** | +0.98 pp |
| **mean** | **+1.69 pp** | **+0.39 pp** |
| sd | 0.63 | 0.78 |
| positive draws | **3 / 3** | 2 / 3 |
| mean / sd | **2.68** | 0.50 |

#### The 160M effect is robust, and the original figure understated it

**All three draws are positive and all three exceed the pre-registered ≥ 1.0 pp bar.** The mean is
**+1.69 pp** at 2.68 standard deviations from zero.

**[F-23](#f-23)'s +1.08 pp turns out to have been the *lowest* of the three draws.** The single-draw
figure that looked "uncomfortably close to the threshold" was in fact the pessimistic end of the
distribution, not a lucky high reading. That is the opposite of the direction every prior fault in this
project ran, and it was not the outcome expected when this run was queued.

#### The scale conclusion survives, on better evidence than it had

| | 160M | 410M |
| --- | --- | --- |
| mean gain | +1.69 pp | +0.39 pp |
| **difference** | **+1.30 pp** | |

And it holds **draw by draw**, which matters more than the difference of means because the same
calibration draws were used at both scales:

| Draw | 160M − 410M |
| --- | --- |
| rep0 | +0.40 pp |
| rep1 | +2.15 pp |
| rep2 | +1.36 pp |
| **mean** | **+1.30 pp**, sd 0.88, **3 / 3 positive** |

So [F-25](#f-25)'s *conclusion* — the joint gain shrinks with scale — is supported. Its *numbers* were
wrong in both directions: 160M understated (+1.08 against +1.69) and 410M overstated (+0.68 against
+0.39). **The retraction of the point estimates stands; the direction they pointed does not need
retracting.**

#### What may and may not be claimed

**May:** at 30% + W4 on Pythia-160M the joint arm beats best-of-sequential by roughly 1.7 pp of
retention, consistently in sign across three calibration draws and above the pre-registered threshold in
every one. At Pythia-410M the same comparison is indistinguishable from zero. The gain is smaller at the
larger scale, consistently across the three paired draws.

**May not:** any significance claim. Three draws cannot support one — a sign test on three unanimous
observations reaches only p = 0.25. This is a **consistent-in-sign effect-size result**, exactly as
[A1 §5.1](protocol_amendment_a1.md) says the study must report, and it is on the **validation** split,
which A1 §4 declares a selection surface. It is not confirmatory.

**Also may not:** a scaling law. Two scale points. The third (1B) is not downloaded.

#### The confirmatory stage is now worth running

The decision rule was written into the config before the run. This is the first outcome:

| Predicted outcome | Consequence |
| --- | --- |
| **stable and positive** ✅ | a real effect at small scale — the ~38 h confirmatory stage is worth spending |
| straddles zero | reframe as a bounded null |
| stable at ~0.4 pp | R ≈ 30 needed; report a bounded null |

At 160M's sd of 0.63 pp, R=8 gives a standard error of **0.22 pp**, so a +1.69 pp effect sits ~7.6
standard errors from zero — comfortably detectable. At 410M's sd of 0.78 the standard error is 0.28 pp
and a +0.39 pp effect sits at 1.4, which is *not* detectable. **That asymmetry is itself informative:**
R=8 is sufficient to confirm the 160M effect and sufficient to establish that the 410M effect is small,
which together is exactly what the scale question needs.

#### Two methodological lessons worth keeping

**Two draws systematically understate the spread.** It happened three times in this session. At 410M
reps 0-1 sat 0.10 pp apart before rep2 landed 0.40 pp away; on 160M's sequential arm reps 0-1 sat
0.23 pp apart before rep2 landed 1.17 pp away; and an intermediate claim in F-26 that a competing figure
was "five standard deviations out" was built on the first of those and had to be withdrawn. **No spread
estimate should be quoted from two observations.**

**Reproduction held throughout.** Replicate 0 reproduced [F-23](#f-23) exactly at 160M — sequential
65.261, joint 64.041 — and [F-25](#f-25) exactly at 410M. Every retraction in this session was about
*inference from too few draws*, never about the pipeline.

---

### F-26 - The 410M joint gain changes SIGN between calibration draws {#f-26}

> 📌 **Partly superseded by [F-27](#f-27).** The 410M measurement below stands. Its implication that
> the scale conclusion was unsupportable does **not**: replicating 160M on the same three draws
> gives +1.69 pp there against +0.39 pp here, with 160M ahead in all three paired draws.

*2026-07-31 - Pythia-410M `dd47b0e` - 493 x 512 **validation** window, dense **22.166** - three paired
calibration draws - `METHOD_VERSION = 4` - **3 h 28 m**, 7 cells - **retracts the 410M headline of
[F-25](#f-25)***

Run to resolve a cross-machine disagreement. It resolved something more important instead.

| Draw | Sequential | Joint | **Joint gain** | Excess NLL advantage |
| --- | --- | --- | --- | --- |
| rep0 | 58.56% | 59.24% | **+0.68 pp** | +0.0116 |
| rep1 | 58.46% | 57.96% | **−0.50 pp** | −0.0085 |
| rep2 | 58.06% | 59.04% | **+0.98 pp** | +0.0167 |
| | | **mean** | **+0.39 pp** | +0.0066 |
| | | **sd (n=3)** | **0.78 pp** | 0.0130 |

**The mean sits 0.50 standard deviations from zero. Two draws of three are positive. The sign is not
consistent.**

#### What this retracts

[F-25](#f-25) reported **+0.68 pp at 410M** and concluded the joint gain *shrinks with scale*
(+1.08 pp → +0.68 pp). That +0.68 pp is now visible as **rep0 alone** — one draw of a distribution
whose spread is 1.47 pp wide and straddles zero.

**The 410M point estimate is withdrawn, and with it the scale claim.** On three draws the honest
statement is: *at 410M the joint gain is indistinguishable from zero, and no comparison with 160M can
be made until 160M is replicated too.* F-25's other results stand — the budget confirmation, the W4
order, the W8 null — because those rest on gaps of 6.82 pp and larger, far outside this variance.

**160M's +1.08 pp is now equally suspect.** It is also a single draw. Nothing yet says its variance is
smaller.

#### Pairing did not rescue it, and that was the surprise

The paired design exists because a calibration draw that hurts one arm should hurt the other, so the
*difference* is expected to be far more stable than either arm. That reasoning was stated in this log
before the measurement and **it is wrong here**:

| Quantity | Spread across the three draws |
| --- | --- |
| Sequential retention | 0.50 pp |
| Joint retention | 1.28 pp |
| **Paired difference** | **1.47 pp** |

The difference is *noisier than either arm*, not less. At rep1 the arms moved in opposite directions —
sequential landed mid-range while joint fell to its minimum. So the draw does not apply a common shift
that cancels; it changes *which mask each arm picks*, and the two arms respond to it differently by
construction, because the mask is what distinguishes them.

**Pairing still helps** — it removes the dense-baseline and window variance, and it is what §3.11
requires — but it must not be assumed to cancel calibration noise in the difference. It does not.

#### What it means for the confirmatory design

With sd ≈ 0.78 pp per draw:

| R | Standard error of the mean gain |
| --- | --- |
| 3 | 0.45 pp |
| 5 | 0.35 pp |
| **8** | **0.28 pp** |

A ~1 pp effect at R=8 sits about 3.6 standard errors from zero, which is detectable. At R=3 it is 2.2,
and the mean measured here (+0.39 pp) is under one. **So the R=8 decision is vindicated as necessary
and roughly sufficient** — it was chosen on the arithmetic of sign tests, and it turns out to be about
right on the empirical variance too, which was not guaranteed.

**A caveat on the caveat:** sd from n=3 is a crude estimate. The real spread could be materially larger,
in which case R=8 would not be enough. The eight-draw confirmatory run measures its own variance and
must be allowed to say so.

#### On the cross-machine disagreement

The partner's Colab figure was **+1.96 pp**, against our observed range of −0.50 to +0.98.

Earlier in this session that gap was described as roughly five standard deviations outside our
measurements, on the strength of two draws that happened to agree to 0.10 pp. **That framing was wrong**
and is withdrawn: the third draw widened the spread fivefold, and "their number is implausible because
ours is stable" is not an argument available when ours is not stable.

What survives is narrower and unaffected by any of this:

* their run is **not reproducible** — source recorded as `aec5099-dirty`, uncommitted changes, and the
  commit absent from the history their audit could see;
* it was produced on **Colab**, and at the time the machine policy forbade that outright.

> **Amended 2026-08-01.** The second bullet no longer holds as stated: the policy now permits
> compression and quality evaluation on any CUDA machine, and only deployment measurements are
> host-bound. **The first bullet is the one that mattered all along** and is untouched — a
> `-dirty` tree 22 commits behind `main` is unusable wherever it runs. Recording the amendment
> rather than deleting the bullet, because "we rejected it partly on a rule we later relaxed" is
> exactly the kind of thing a reader is entitled to check.

Their +1.96 pp remains above everything measured here, but it is now one unreproducible draw from a cell
known to swing by 1.47 pp — not an anomaly demanding explanation. **Both figures are single draws from a
noisy cell. Neither should be quoted.**

#### The reproduction check passed

Replicate 0 reproduced [F-25](#f-25) **exactly** — sequential 37.851, joint 37.415, to three decimals,
in a different session. The pipeline is deterministic and our records are trustworthy. The problem was
never the code; it is that one draw of a 1.47 pp-wide distribution was reported as a point estimate.

---

### F-25 - 410M: budgets confirmed, W4 order confirmed, and the joint gain SHRINKS with scale {#f-25}

> 🔴 **The §5 scale headline below is RETRACTED by [F-26](#f-26).** The +0.68 pp figure is one draw
> of a distribution spanning −0.50 to +0.98 pp. Sections 1–4 stand; the scale claim does not.

*2026-07-30 - Pythia-410M `dd47b0e` - 493 x 512 **validation** window, dense **22.166** - 128
calibration sequences from train, one draw - `METHOD_VERSION = 4` - **3 h 18 m**, 7 cells -
**supersedes [F-14](#f-14)***

One grid answering three questions: do the frozen budgets hold at 410M, which sequential order wins
there, and does the joint gain change with scale.

| Budget | Arm | Perplexity | Retention | Excess NLL |
| --- | --- | --- | --- | --- |
| **moderate** 30% + W8 | sequential P→Q | 29.102 | 76.17% | 0.2722 |
| | sequential Q→P | 29.117 | 76.13% | 0.2728 |
| | joint | **29.100** | 76.17% | 0.2722 |
| **aggressive** 30% + W4 | sequential **P→Q** | 37.851 | **58.56%** | 0.5351 |
| | sequential Q→P | 42.843 | 51.74% | 0.6590 |
| | **joint** | **37.415** | **59.24%** | **0.5235** |

#### 1. Both frozen budgets are confirmed at 410M

76.17% and 58.56% sequential retention — both inside §5.3's measurable-but-not-catastrophic band, well
above the 50% floor and far below the 99% ceiling. **§5.3's pre-1B requirement is satisfied.**

#### 2. The W4 order is confirmed, decisively, at both scales

| Budget | 160M | 410M |
| --- | --- | --- |
| **W4 aggressive** | P→Q by **+4.26 pp** | P→Q by **+6.82 pp** |
| W8 moderate | Q→P by +0.43 pp | **P→Q by +0.04 pp** |

**P→Q is unambiguously right at W4, and the margin grows with scale.** On the additive scale the Q→P
penalty is 0.078 nats at 160M and 0.124 at 410M. That is consistent with the mechanism proposed in
[F-24](#f-24): Q→P reuses the dense-fitted scales without refitting — nearly free at W8 where
quantisation is almost lossless, punishing at W4 where a coarse grid is badly matched to the
post-pruning distribution, and worse at scale because a larger model has more channels whose
distributions shift.

#### 🔴 3. The W8 order freeze is CONTESTED and must not stand as F-24 recorded it

**The W8 direction flips between scales, and both margins are noise.** 160M gave Q→P by 0.43 pp; 410M
gives P→Q by 0.04 pp — 0.0006 nats. At 410M all three arms sit within **0.017 perplexity** of one
another.

[F-24](#f-24) froze Q→P at W8 on the 160M single-draw margin, and this contradicts it. The consequence
is not cosmetic: the moderate budget's joint gain is **+0.07 pp against P→Q** and **−0.36 pp against
Q→P**, so the *sign* of that budget's headline depends on which order is frozen.

`order_selection_w8_replicates.yaml` re-checks it across five paired calibration draws. That config
pre-declared the rule before any of this was seen: *if the sign varies, the orders are indistinguishable
at W8 — freeze **P→Q**, the pre-registered primary order (§3.6), and record that the choice is
arbitrary. Do not pick the winner of a coin toss and report a gain against it.* The sign has now varied
across **scales**; the replicate run tests whether it also varies across draws.

**Provisional status: W8 order contested, P→Q the pre-declared fallback.** Not finalised until the
replicate evidence lands.

#### 4. The W8 control gives a clean null at both scales

| Scale | Joint gain at W8 |
| --- | --- |
| 160M | +0.07 pp |
| 410M | **+0.00 pp** |

This matters more than it looks. [F-05](#f-05) predicted the joint mechanism is inert at 8 bits (0.46%
mask divergence against 8.86% at W4), and two independent scales now agree it produces nothing there.
**The same pipeline yields zero when the mechanism is switched off**, which is what makes a non-zero
result at W4 hard to dismiss as pipeline noise.

#### 🔵 5. The headline: the joint gain SHRINKS with scale

Against best-of-sequential, which is P→Q at both scales:

| Scale | Sequential | Joint | **Joint gain** | Excess NLL advantage |
| --- | --- | --- | --- | --- |
| **160M** | 56.66% | 57.74% | **+1.08 pp** | **+0.0189 nats** |
| **410M** | 58.56% | 59.24% | **+0.68 pp** | **+0.0116 nats** |
| | | | ratio **0.63** | ratio **0.61** |

**Joint wins at both scales, but by roughly 40% less at the larger one.** The two metrics agree on the
ratio to within 0.02 despite being different functional forms — retention is exponential, excess NLL
additive — which is at least internally coherent rather than an artefact of one scale.

**This runs against the study's motivating hypothesis.** The question was whether joint compression pays
off *more* as models grow. On this evidence it pays off **less**.

**And at 410M it no longer clears the pre-registered bar.** §6.3 sets practical importance at
**≥ 1.0 pp retention**. 160M's +1.08 pp passes by 0.08; 410M's +0.68 pp **fails**. So the effect is
positive at both scales but practically important at only the smaller one.

#### What this is not

**Not a trend.** Two scale points, one calibration draw each, on the validation split. The research plan
already states that *three* points cannot fit a scaling law; two cannot support one either. What this
establishes is a **direction**, and a direction that two independent metrics agree on.

**Not confirmatory.** Validation is a declared selection surface ([A1 §4](protocol_amendment_a1.md)),
and there is no uncertainty estimate. A +0.68 pp gain with no error bar cannot be distinguished from
+1.08 pp with no error bar; the ratio of 0.63 could be entirely calibration noise. That is precisely
what the eight paired replicates on the test split exist to settle.

**Pythia-1B is the point that matters.** With three points a direction becomes checkable rather than
merely stated. 1B is not downloaded yet.

#### Incidental: 410M is damaged more at W8, less at W4

| Budget | 160M excess NLL | 410M excess NLL |
| --- | --- | --- |
| moderate 30% + W8 | 0.221 | **0.272** |
| aggressive 30% + W4 | **0.568** | 0.535 |

The sign of the scale effect on *sequential* damage differs by budget. At W4, where damage is large, the
larger model tolerates it better — the expected redundancy story. At W8 the damage is small in absolute
terms and runs the other way, which one draw cannot separate from noise. Recorded because an earlier
reading of the moderate cell alone suggested "410M is damaged more", and the aggressive cell reverses
it.

---

### F-24 - The winning sequential order differs by budget, and the headline survives best-of {#f-24}

> 🔴 **The W8 freeze below is WITHDRAWN — resolved by [F-28](#f-28).** Across five paired draws the
> orders are indistinguishable (mean +0.18 pp, sd 0.19, 4/5, p = 0.375), so the pre-declared
> fallback applies and W8 is frozen at **P→Q**. The moderate joint gain is therefore **+0.07 pp**,
> not the −0.36 pp derived here. The W4 freeze is confirmed and strengthened.

*2026-07-30 - Pythia-160M `50f5173d` - 493 x 512 **validation** window, dense **36.9741** - 128
calibration sequences from train, one draw - `METHOD_VERSION = 4` - **42 min**, 5 cells -
**closes [A1](protocol_amendment_a1.md) step 6***

`method_definition.md` (plan §3.6, §6.1) has always required joint gain to be measured against
**best-of {P→Q, Q→P}**, with the winning order recorded. The sweep only ever ran P→Q. Q→P had never
been run end to end in this project until now.

| Budget | Arm | Perplexity | Retention |
| --- | --- | --- | --- |
| **moderate** 30% + W8 | sequential **P→Q** | 46.101 | 80.20% |
| | sequential **Q→P** | **45.856** | **80.63%** ← best |
| | joint | 46.064 | 80.27% |
| **aggressive** 30% + W4 | sequential **P→Q** | **65.261** | **56.66%** ← best |
| | sequential **Q→P** | 70.557 | 52.40% |
| | joint | **64.041** | **57.73%** |

#### The winning order genuinely differs by budget

| Budget | Winner | Margin over the loser |
| --- | --- | --- |
| moderate 30% + W8 | **Q→P** | +0.43 pp |
| aggressive 30% + W4 | **P→Q** | +4.26 pp |

A1 §5.3 anticipated this explicitly — "the winning order may legitimately differ by model and by
budget" — but it is worth noting that it *does*, because a single global choice would have been wrong
at one of the two budgets.

#### Consequence 1: the moderate budget's joint gain flips sign

| Baseline | Joint gain at moderate |
| --- | --- |
| P→Q only, as [F-23](#f-23) reported it | **+0.07 pp** |
| **best-of-sequential** | **−0.36 pp** |

Q→P beats both P→Q *and* joint at W8. So the small positive gain at the control budget becomes a small
negative one once the baseline is the one the documents always required.

This is exactly the omission §3.6 existed to prevent, and it ran the direction the rest of this
project's bugs ran: **not running Q→P was flattering joint.** It is also, notably, *more* consistent
with [F-05](#f-05) than +0.07 was — F-05 predicts the joint mechanism is inert at W8, and a mechanism
that is inert should not produce a positive gain.

#### Consequence 2: the headline is unchanged

**At the aggressive budget P→Q was already the stronger order, so the +1.08 pp joint gain stands
exactly as [F-23](#f-23) measured it.** Best-of-sequential does not erode it. Q→P is 4.26 pp *behind*
P→Q there, so it never becomes the baseline.

That is the more important half of this finding. The headline effect had already survived a solver
rewrite (F-23); it has now also survived being measured against the stronger of two baselines.

#### Why Q→P wins at W8 and loses badly at W4

A coherent mechanistic reading, offered as interpretation rather than as measurement:

* **At W8 quantisation is nearly lossless** ([F-07](#f-07): W8-only retention 99.8%). Quantising first
  costs almost nothing, and the mask is then chosen on weights that already sit on their final grid —
  so pruning reconstruction optimises the deployed representation directly rather than an FP32 proxy.
* **At W4 the order is punishing.** Q→P commits to a coarse grid before pruning, and by construction it
  **reuses the dense-fitted scales without refitting** — that non-refit is what keeps it a *sequential*
  arm rather than a joint one. Those dense-fitted scales are badly matched to the post-pruning weight
  distribution at 4 bits, and unlike joint it never gets to revise them.

This also predicts the asymmetry in the margins: +0.43 pp at W8 against −4.26 pp at W4. The penalty for
a frozen grid grows as the grid gets coarser.

#### Two consistency checks that passed

**Both P→Q cells reproduced [F-23](#f-23) exactly** — 46.101 and 65.261, to three decimals, under
different budget labels (`moderate`/`aggressive` against `s1_30_w8`/`s5_30_w4`) and from a different
config file. The budget-label plumbing changes nothing, which is worth having verified rather than
assumed.

Dense also reproduced at **36.974**, matching F-23 rather than the F-22 anchor's 36.9744 — consistent
with the thread-configuration sensitivity F-23 recorded, since both runs went through the same runner.

#### What to freeze

| Model | Budget | Frozen sequential order |
| --- | --- | --- |
| pythia-160m | moderate | **Q→P** |
| pythia-160m | aggressive | **P→Q** |

**Still outstanding before the confirmatory stage:** the same selection at 410M and 1B. Pythia-1B is
not downloaded yet. One draw per cell was enough here because both margins (+0.43 pp, +4.26 pp) exceed
anything a single draw's noise plausibly explains — the aggressive margin by a wide factor. Had they
landed within noise, replicates would have been added before freezing.

---

### F-23 - Screening on anchored code. The frozen budgets hold, and S6 answers the mechanism question {#f-23}

*2026-07-30 - Pythia-160M `50f5173d` - 493 x 512 **validation** window, dense **36.9741** - 128
calibration sequences from train, one draw - `METHOD_VERSION = 4` - **2 h 08 m**, 13 cells -
**supersedes [F-17](#f-17)***

The first screening grid produced by code that has passed three independent correctness anchors
([F-19](#f-19), [F-20](#f-20), [F-22](#f-22)).

| Budget | Sparsity | Bits | Seq ppl | Joint ppl | Seq ret. | Joint ret. | **Joint gain** | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1** | 30% | W8 | 46.10 | 46.06 | **80.2%** | 80.3% | **+0.07 pp** | **ELIGIBLE** |
| S2 | 50% | W8 | 177.76 | 175.48 | 20.8% | 21.1% | +0.27 pp | catastrophic |
| S3 | 50% | W4 | 254.53 | 240.99 | 14.5% | 15.3% | +0.82 pp | catastrophic |
| S4 | 70% | W4 | 5041.61 | 5895.00 | 0.7% | 0.6% | −0.11 pp | catastrophic |
| **S5** | 30% | W4 | 65.26 | **64.04** | **56.7%** | **57.7%** | **+1.08 pp** | **ELIGIBLE** |
| **S6** | 40% | W8 | 67.93 | 68.22 | **54.4%** | 54.2% | **−0.23 pp** | **ELIGIBLE** |

#### The frozen budgets survive a third time

The same three budgets are eligible and the same two stay frozen, so
[protocol_freeze.md](protocol_freeze.md) is unchanged. Sequential retention across every version of
this code:

| Budget | F-14 | F-17 | **F-23** |
| --- | --- | --- | --- |
| S1 30% + W8 | 80.4% | 80.3% | **80.2%** |
| S5 30% + W4 | 56.0% | 57.1% | **56.7%** |

Within half a percentage point across three rewrites of the solver. None of the retracted bugs touched
the sequential arm's first stage, and this is what that looks like.

#### 🔵 S6 answers the question A1 planned a separate experiment for

**S5 and S6 are quality-matched by two different recipes, and only the low-bit one shows a joint gain.**

| | Sparsity | Bits | Seq retention | Joint gain |
| --- | --- | --- | --- | --- |
| **S5** | 30% | **W4** | 56.7% | **+1.08 pp** |
| **S6** | 40% | **W8** | 54.4% | **−0.23 pp** |

Sequential retention differs by 2.3 pp — comparable damage from very different settings — yet the joint
gain differs by 1.31 pp and changes sign. Among all three *eligible* budgets the pattern is clean: both
W8 budgets give gains indistinguishable from zero (+0.07, −0.23) and the only W4 budget gives +1.08.

This is exactly the discrimination [A1 §5.4](protocol_amendment_a1.md) commissioned a 12-run control to
make: **it supports a precision-specific mechanism rather than a compression-severity effect.** It is
what [F-05](#f-05) predicts from mask divergence of 8.86% at W4 against 0.46% at W8.

**It arrived free, as a by-product of screening**, which is worth noting because A1 flagged S6 as the
weakest of its five decisions on the "would this have been justified before seeing results" test. It no
longer costs anything at 160M. The 410M half and the paired replicates are still outstanding.

**Do not over-read it.** One draw, one model, validation split. The catastrophic budgets do *not*
support the same pattern — S2 is W8 with +0.27 and S4 is W4 with −0.11 — but at 21% and 0.6% retention
those models are rubble and the differences carry no information.

#### The headline figure, and a prediction that failed

**S5 joint gain is +1.08 pp, against the retracted +1.03 pp.**

The prediction on record before this run was that it would come out *below* +1.03, since the B-22
correction removed a bias that flattered joint. It did not move. That prediction was wrong.

What the stability does buy: the figure survived a rewrite that changed the reconstruction objective,
added an acceptance guard, fixed activation grouping, and altered the packing path. **Two very
different code versions landing on +1.03 and +1.08 is genuine evidence the effect is not a bug
artefact** — which could not be said of either retracted number.

What it does not buy:

- it still sits **barely over** the pre-registered ≥ 1.0 pp threshold, the same discomfort F-17 recorded;
- **no uncertainty estimate** — one calibration draw, by design for screening;
- **validation split**, which A1 §4 declares a selection surface.

One consideration that cuts *toward* joint: [F-21](#f-21) measured the solver as handling the joint mask
**less** well (efficiency 0.5631 against 0.6409), so solver slack works against joint here. If anything
+1.08 understates.

#### Two incidental observations, both worth keeping

**Equal wall-clock corroborates the matched solver budgets.** Every compressed cell took ~9.3 minutes,
joint and sequential alike. §3.11 requires matched optimisation budgets and B-14 was a violation of it;
equal wall-clock is independent evidence the fix holds in practice and not only in the step counter.
One outlier: sequential S3 took 20.6 min against ~9.3 for everything else, cause unknown, result
consistent with its neighbours.

**Dense perplexity is not bit-reproducible across thread configurations.** This run measured dense at
**36.9741**; the [F-22](#f-22) anchor measured **36.9744** on the identical model and identical data an
hour earlier. A relative difference of 8e-6, from floating-point reduction order under a different CPU
thread count. Negligible in itself — but dense perplexity is the *denominator of every retention
figure*, so retention is reproducible to about four significant figures rather than exactly. Worth
stating before someone treats a 0.01 pp retention difference as signal.

---

### F-22 - External SparseGPT comparison: our absolute numbers are credible {#f-22}

*2026-07-30 - Pythia-160M `50f5173d` - 493 x 512 **validation** window, dense **36.9744** - 128
calibration sequences from train, fingerprint `b0e766b25fdd6536` - 30% sparsity, pruning-only -
reference is `IST-DASLab/sparsegpt` `SparseGPT.fasterprune`, **unmodified** - **closes A1 §5.5b2***

The only check that speaks to *absolute* quality rather than internal consistency. Matched on model,
revision, calibration draw, module coverage and evaluation loader; the compression algorithm is the
only unmatched variable.

| Arm | Perplexity | Retention |
| --- | --- | --- |
| Dense | **36.9744** | 100% |
| **Ours**, per-output-row mask + sweep | **45.6644** | **80.97%** |
| **Ours**, *tensor-wide* mask + sweep | **59.9617** | **61.66%** |
| **Reference SparseGPT**, unmodified | **66.0355** | **55.99%** |

All at measured sparsity 0.2997-0.3000.

#### We beat the canonical implementation by 25 points, and that was a reason for suspicion

The raw gap is **+24.98 pp retention** in our favour, **-30.85%** relative perplexity -- eight times the
A1 alarm threshold. A result that flattering against a well-cited reference is far more likely to be a
misconfiguration than a genuine win, so it was chased rather than reported.

**Predicted direction was wrong.** [F-20](#f-20) found our sweep captures only 0.6409 of the achievable
objective gain, so the expectation on record before running this was that SparseGPT would come out
*ahead*. It did not, by a wide margin, and in the direction that most needed scepticism.

#### The cause, read from their source and then measured

`fasterprune` selects its mask like this:

```python
tmp = W1**2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
mask1 = tmp <= thresh
```

`W1` is `(out_features, 128)` and `tmp.flatten()` thresholds **jointly across all output rows** within
each 128-column block. That is a tensor-wide comparison group. Ours is **per-output-row** -- and
[F-07](#f-07) had already measured that exact difference as worth 6.7x perplexity on this model, because
tensor-wide ranking lets a low-energy input column be deleted in every row at once.

**So the control was to run our own pipeline with their comparison group.** Result:

| | Retention gap | Share of the 24.98 pp |
| --- | --- | --- |
| Explained by the comparison group | **19.31 pp** | **77.3%** |
| Residual after matching it | 5.67 pp | 22.7% |

Residual relative perplexity is **-9.20%**, just inside A1's 10% alarm band.

**Three-quarters of the gap is the mask comparison group.** Not our reconstruction, and not a defect in
either implementation -- theirs is their *documented default*, and SparseGPT is published on models like
OPT-175B where a 160M model's lack of redundancy is not a factor. This is an external corroboration of
[F-07](#f-07) from a direction we did not choose.

#### The residual 5.67 pp, and what it is not

Not separable from this run. The remaining candidates:

* **the criterion.** Theirs is `w^2 / [H^-1]^2_jj`; ours is Wanda's `|w| * ||X_j||_2`. The Wanda paper
  reports its criterion matching or beating SparseGPT at moderate sparsity, so a residual in this
  direction is consistent with published work rather than surprising.
* **the pool shape.** Their threshold is per 128-column block, ours is over the whole tensor -- so their
  mask is *more* constrained (exactly 30% per block), which should if anything help them.
* **the reconstruction.** Cannot be isolated here, because `fasterprune` chooses its mask internally and
  cannot be handed an external one without editing it, which would forfeit running it unmodified.

#### What this settles

**Our absolute retention is not implausibly high.** ~81% at 30% pruning-only is what a per-output-row
Wanda mask plus error compensation produces, it reproduces bit-for-bit across runs, and when matched on
comparison group it sits within ~9% relative perplexity of the canonical SparseGPT. The open question
from [F-20](#f-20) -- "is ~57% retention plausible or does it indicate a remaining implementation gap?"
-- is answered: **plausible.**

**What it does not settle:** whether our *reconstruction* specifically is competitive. That would need
their sweep driven by our mask, which their code does not permit without modification.

#### Three faults in the driver, all of which would have produced numbers rather than errors

Recorded because the pattern is the point. Only the second one crashed.

| Fault | Would have looked like |
| --- | --- |
| Live KV cache across block replays (B-28) | a slightly different algorithm |
| `DatasetSummary.token_fingerprint` -- wrong attribute | *(crashed; the harmless kind)* |
| `blocks[0] = Catcher(...)` on a **copy** (B-29) | SparseGPT run on **zero** calibration data |

The third is the instructive one. Every published driver uses that swap trick, because in their repos
`model.decoder.layers` is the live `nn.ModuleList`. Our `get_decoder_blocks` returns `list(current)` --
a copy -- so the assignment rebound a throwaway list while the model went on calling the real block.
**The only reason it surfaced is an empty-capture guard added an hour earlier for unrelated reasons.**
Without it, SparseGPT would have pruned against nothing and the number would have been reported.

---

### F-21 - Solver slack IS arm-dependent, but it never inverted a mask ranking {#f-21}

*2026-07-30 - Pythia-160M `50f5173d` - calibration fingerprint `b0e766b25fdd6536` - 30% sparsity, joint
mask scored at W4 - 12 modules x 8 rows = **96 rows** solved exactly - **partially answers the
[F-20](#f-20) confound***

| Quantity | Result |
| --- | --- |
| Mask divergence between arms | **7.8% - 12.6%** per module (consistent with [F-05](#f-05)'s 8.86%) |
| Mean solver efficiency, **sequential** mask | **0.6409** |
| Mean solver efficiency, **joint** mask | **0.5631** |
| **Efficiency gap (joint − sequential)** | **−0.0778** |
| **Rows where the solver misranks the masks** | **0 of 96 (0.0%)** |

#### The reassuring half

**The solver never got the direction wrong.** On all 96 rows, whenever the sweep said one mask gave a
lower objective, the exact optimum agreed. The sweep does not invert mask quality, so the *sign* of a
measured arm difference is not a solver artefact.

#### The confound is confirmed real

**Solver efficiency differs systematically by mask shape: 0.6409 versus 0.5631, a 7.8 percentage-point
gap.** So the [F-20](#f-20) worry was not hypothetical -- slack *is* arm-dependent. The mechanism is the
expected one: the joint mask is a different shape, different shapes give different `H[S,S]` conditioning,
and conditioning determines how much a one-pass sweep recovers.

The gap is also not uniform in sign across depth. Layers 0 and 4 favour the sequential mask by 0.06-0.33;
layer 8 favours the **joint** mask by 0.10-0.20. So this is not a constant offset that would cancel in a
difference -- it varies by layer and could partially cancel or partially accumulate depending on the
model.

#### 🔴 What this measurement does NOT settle, and why

**The design cannot isolate the effect on the headline joint gain, and I first described it as though it
could.** Two reasons, and the second is a hard limit rather than a fixable oversight:

1. **Reconstruction ran pruning-only for both arms**, deliberately, to isolate the mask's effect on
   solver efficiency from quantisation. But the joint mask is *selected under a quantised grid* (D3), so
   scoring it on a pruning-only objective disadvantages it by construction. Both `sweep_advantage` and
   `optimal_advantage` came out **negative on all 96 rows** -- the sequential mask wins the pruning-only
   objective unanimously, under both the sweep and the exact optimum. That is expected and says nothing
   about the arms' real comparison.

2. **No exact optimum exists for the quantised problem.** The closed-form masked minimiser
   `(H_SS)^-1 H_S,: w` solves a *continuous* least-squares problem. With weights constrained to a
   discrete grid the problem is an integer program with no closed form, so the anchor's reference -- the
   thing that makes it a lower bound at all -- is unavailable in the regime the study actually reports.

A metric-naming error worth recording because it nearly propagated: the aggregate ratio was originally
called `attributable_joint_benefit` and printed as "1.0 = entirely real, 0.0 = entirely solver". On this
run it read **+0.6425**, which invites "64% of the joint gain is real" -- when in fact every row's
advantage was negative and the number is the ratio of two *disadvantages*, meaning the sweep
**overstates the joint mask's penalty by about 1.56x**. Renamed `advantage_fidelity`, documented for
both sign cases, and now printed next to the sign of the advantage it is a ratio of.

#### Where this leaves the confound

| Question | Status |
| --- | --- |
| Does solver slack differ between arms? | **Yes, 7.8 pp, and it varies in sign by depth** |
| Can the solver invert which mask is better? | **No -- 0 of 96 rows** |
| Does that change the sign of the reported joint gain? | **No**, on this evidence |
| Does it change the *magnitude* of the reported joint gain? | **Unknown, and not answerable this way** |

So a ~1 pp joint gain cannot be dismissed as a solver artefact in *direction*, which is the part that
matters most for the research question. Its *magnitude* remains subject to a 7.8 pp efficiency
difference whose net effect on end-to-end perplexity is unmeasured. **This belongs in the paper's
limitations either way**, stated as: the reconstruction solver is approximate, its approximation quality
differs measurably between the two arms' masks, and the study reports a difference that the solver
provably ranks correctly but may not scale correctly.

---

### F-20 - The sweep is correct, and captures 64% of the achievable gain {#f-20}

*2026-07-30 - Pythia-160M `50f5173d` - 128 calibration sequences x 512 tokens from **train**,
fingerprint `b0e766b25fdd6536` - 30% sparsity, pruning-only - 12 modules x 8 rows = **96 rows** solved
exactly in float64*

The second Amendment A1 anchor (§5.5b1). **Both hard invariants hold.**

| Quantity | Result |
| --- | --- |
| Rows compared | **96** (12 modules, 4 module types, 3 depths) |
| Rows scoring **below** the provable optimum | **0** - required, since it is impossible |
| Rows ending **worse than naive** masking | **0** - required, the accept-only-if-better guard works |
| Mean improvement over naive | **+38.65%** of the naive objective |
| **Mean efficiency** | **0.6409** of the achievable gain |
| Worst-row efficiency | 0.3927 |

#### What was checked, and why it is stronger than a SparseGPT port

For a fixed mask the objective `sum_o (w-w_hat)^T H (w-w_hat)` has a closed-form minimiser per output
row, `w_hat_S = (H_SS)^-1 H_S,: w`. The anchor solves that exactly, in float64, with no damping, and
compares our sweep against it.

That beats reimplementing SparseGPT, and the reason is worth stating: **SparseGPT's contribution is
speed, not a different objective.** Comparing our sweep to another approximation would only establish
that two approximations agree. Comparing it to the exact optimum measures how much ours actually gives
up -- and gives a genuine **lower bound**, so a result below it would prove a defect rather than
suggest one.

The objective is **separable across output rows**, which is what makes this tractable: each sampled row
is a complete test of that row, not an approximation of the layer. A full-layer exact solve would cost
`out_features * |S|^3`, the very cost the sweep exists to avoid.

#### 🟡 The 64% is a finding in its own right, and it has a consequence

Our one-pass sweep leaves roughly **36% of the achievable objective improvement unclaimed** versus the
exact per-row optimum. That is not a defect -- a single pass giving up some of the optimum is the
documented trade for making wide layers tractable, and it is why the sweep was chosen over ALS (D2).
It is also remarkably **consistent**, which is what makes it worth recording:

| Module type | Mean efficiency across depths |
| --- | --- |
| `attention.query_key_value` | 0.613, 0.612, 0.662 |
| `attention.dense` | 0.570, 0.611, 0.609 |
| `mlp.dense_h_to_4h` | 0.612, 0.627, 0.686 |
| `mlp.dense_4h_to_h` | 0.674, 0.721, 0.695 |

Range 0.57-0.72 across every module type and depth sampled. So this is a systematic property of the
solver, not noise and not one bad layer.

**The consequence, which is a validity threat and not just a curiosity.** The joint-versus-sequential
difference this study exists to measure has been around **1 pp of retention**. The solver is leaving
36% of the achievable objective gain on the table. If that slack is **not identical between the two
arms** -- and there is no reason to assume it is, because the arms produce different masks, different
masks give different `H_SS` conditioning, and conditioning is exactly what determines how well a
one-pass sweep does -- then part or all of the measured arm difference could be **solver slack rather
than the mask mechanism**.

This is directly measurable with the tool that produced this finding: run the anchor separately on the
sequential and joint masks and compare efficiency. **Not yet done.** Until it is, a small joint gain
cannot be cleanly attributed to the joint mechanism. Recorded in
[validity_threats.md](validity_threats.md#solver-slack-may-exceed-the-effect-being-measured).

#### What this does *not* establish

Whether our absolute quality is competitive. This says the solver optimises what it claims to optimise;
it says nothing about whether ~57% retention at 30% + W4 is in line with published work. That still
needs an external run with comparable numbers -- A1 §5.5(b2), still open. **Passing this anchor must
not be read as closing the external-comparison question.**

#### A sampling fault in the anchor's first run

The first version sampled 6 modules by striding `len(names) // 6`. A GPT-NeoX block contributes four
target modules in a fixed order, so on a 48-module model that stride is exactly 8 and returned
`attention.query_key_value` **six times** -- never touching an MLP projection, which are the widest
layers and the ones where a one-pass sweep has the most to compensate for. The verdict looked fine and
covered a quarter of the model.

Replaced with stratified sampling across module types and depths, which is what produced the table
above. It also now warns when the requested count cannot be spread evenly, rather than silently
returning fewer. Both behaviours are pinned by tests, including one that asserts the old stride would
have failed.

---

### F-19 - The mask is confirmed correct against an independent implementation {#f-19}

*2026-07-30 - Pythia-160M `50f5173d` - 128 calibration sequences x 512 tokens from **train**,
fingerprint `b0e766b25fdd6536` - 30% sparsity, per-output comparison group - GPU capture -
**the first external check this project has ever passed***

The first of Amendment A1's correctness anchors (§5.5a). **It passes.**

| Quantity | Result |
| --- | --- |
| Modules compared | **48** (84,934,656 weights) |
| Column norms agree | **yes** - worst relative difference **6.0e-07** |
| Masks agree, on matched norms | **yes** - **0 differing positions** |
| Worst per-module overlap | **1.000000** |
| Precision-sensitive positions | 4 (informational, explained below) |

Our saliency rule *is* the Wanda criterion, so an independent implementation on matched inputs must
produce the same mask. It does, exactly, across every targeted weight in the model.

**Why this is worth more than a passing test.** The two paths differ where faults would hide. Ours
derives `||X_j||_2` from the streamed Gram as `sqrt(diag(X^T X))`; the reference sums column squares
directly and never forms a Gram, so a streaming fault cannot cancel out. Ours computes an exact prune
count and scatters; the reference sorts and takes top-k per row. Agreement across two different routes
is evidence the criterion, the norm accumulation, the module selection and the comparison group are
all right.

**What it does not cover.** Reconstruction. The error-compensated column sweep is the more intricate
half and the half where B-22 and B-23 lived; it needs the SparseGPT anchor (§5.5b), which has not run.
So this closes the mask question and leaves the reconstruction question open.

#### The 4 disagreements, and why they are not a defect

The first run reported 4 differing positions out of 84,934,656 and a verdict of INVESTIGATE. Chased to
the ground rather than dismissed:

| Module | Disputed pair, float64 score | float32 gap | float32 eps at that magnitude |
| --- | --- | --- | --- |
| `layers.10.mlp.dense_4h_to_h` | both `7.898050546646e-01` | 1.79e-07 | 9.40e-08 |
| `layers.11.attention.query_key_value` | both `1.520126152039e+01` | 4.77e-06 | 1.81e-06 |

Two modules, one row each, two positions each, and **both arms pruned the same count** - the signature
of a swap, not a systematic divergence. Both pairs **tie exactly in float64** and differ by 2-3 ULPs in
float32, so which weight survives is decided by arithmetic rather than by importance.

The decisive test was to rebuild our mask from the *reference's* float64 norms: **the disagreement went
to zero.** Same norms in, identical mask out. So the selection logic is provably identical and the
divergence is entirely attributable to our float32 Gram versus the reference's float64 accumulation.

**The bug was in the anchor, not in the pipeline.** The original verdict asked whether each
disagreement sat on a score exactly equal to our prune threshold. That test is wrong in principle: the
tie exists in float64, and float32 has already broken it by the time the question is asked. It caught
one of the two positions per module and called the other a fault.

Fixed by separating the two questions, which is the design it should have had from the start:

- the **strict** comparison feeds both selectors the same float64 norms, so a disagreement can only be
  a selection defect - and ties cannot excuse anything, because identical inputs must give identical
  output;
- **norm precision** is compared separately, and its effect on the mask is reported as
  `precision_sensitive_positions` - which deliberately does **not** gate the verdict, because failing
  an anchor on arithmetic teaches us to ignore it.

4 precision-sensitive positions in 85 million is worth recording rather than hiding: it says the
ranking is essentially reproducible across arithmetic. A large count would have meant the mask was not
stable to precision, which *would* have mattered.

#### Two faults in the anchor itself, found by running it

Both now regression-tested, and both worth noting because they are the kind that make a diagnostic
quietly useless rather than obviously broken:

| Fault | What it did |
| --- | --- |
| Accumulator buffer on CPU, activations on CUDA | Crashed on the first real run. Now reduces on the activation's device and moves only the length-`in_features` result, keeping float64 arithmetic off a consumer GPU |
| Relative tolerance divided by near-zero norms | A column the calibration barely excites has a norm near zero, so float32 noise became an enormous "relative error". Denominator now floored at a fraction of the largest norm; exactly-dead columns are still counted separately, so a disagreement about *which* columns are dead is not hidden |

A design rule the package enforces with a test: `anchors/` must not import the code it validates.
A reference that calls our saliency function proves only that the call succeeded.

---

### F-18 - F-17's joint-gain column is retracted. Three figures, three retractions. {#f-18}

*2026-07-29 - no measurement - **retracts the joint-gain column of [F-17](#f-17)***

A third review round found two more algorithmic faults, both in the comparison itself rather than in
either arm. F-17's perplexities were produced by code carrying them, so **its joint-gain column is
withdrawn and there is no number in its place.** The grid was re-started on corrected code and
deliberately stopped part-way, so no partial records exist and nothing has to be untangled later.

#### The two faults

**B-22 - the arms were minimising different objectives.** Each arm reconstructed against its own
intermediate weight rather than against the dense weight. Sequential fitted its quantisation step to
its *pruned* reconstruction; joint fitted to dense throughout. So the two arms were not two solutions
to one problem -- they were solutions to two different problems, and their losses were not on a common
scale. `solve` no longer accepts a target; the objective is structurally the dense weight for every
arm. What legitimately still differs between arms is where the quantisation *grid* comes from, which
is the actual method distinction.

**B-23 - acceptance compared a different model from the one kept.** The joint arm scored a proposal
before canonicalising it, then stored the canonicalised version. In fp32 those are not the same
tensor, so the incumbent guard was choosing between quantities it had not measured consistently.
`solve` now returns a `SolvedLayer` carrying weight, codes, scales and loss together, so acceptance,
the recorded objective, evaluation and packing all refer to one object.

#### Direction of the bias, which is the part that matters

B-22 was **not symmetric.** Fitting sequential's second stage to its own intermediate gives it an
easier target than the dense weight while measuring it against dense, so it is penalised at
measurement time; joint was already fitting dense and is not. **The bug disadvantaged sequential and
inflated joint gain.** F-17's +1.03 pp is therefore an upper bound on the truth, not an estimate of
it -- and it was already sitting exactly on the >= 1.0 pp pre-registered threshold.

#### The pattern is now the finding

| Reported | Budget | Retracted because |
| --- | --- | --- |
| **-4.55 pp** | 30% + W4 | Joint outer loop had no acceptance test (B-17) |
| **+1.03 pp** | 30% + W4 | Arms minimised different objectives (B-22, B-23) |
| *pending* | 30% + W4 | Not yet measured on corrected code |

**Every bug found so far has pointed the same way: flattering the joint arm.** Four separate faults
(B-14 unequal solver budgets, B-17 the missing guard, B-22 unequal objectives, B-23 compare-before-
canonicalise) and not one of them favoured sequential. That is not a coincidence to be explained away
-- joint is the arm with more moving parts, so it has more places for an unearned advantage to hide,
and it is the arm the study hopes to find an effect for. **This belongs in the paper's limitations
whatever the final number is**, and it is a standing reason to treat the next figure as provisional
until an independent implementation agrees with it.

#### What is *not* retracted

The budget decision. Sequential retention has been stable across every version of the code -- around
80% at 30% + W8 and around 57% at 30% + W4 -- because none of these bugs touched the sequential arm's
first stage or the eligibility rule. The frozen pair is therefore expected to hold. **Expected, not
verified:** the re-run confirms it or it does not.

Also not retracted: [F-07](#f-07) (the mask comparison group, 6.7x), [F-05](#f-05) (W4 is the only
regime where the mask mechanism is live), and [F-15](#f-15) (the run seed is inert). None depends on
the arm comparison.

---

## 3. All end-to-end perplexities in one table

> **The confirmatory numbers are NOT in this table.** They are on the **test** split at a different
> evaluation window (512 × 512) and live in [F-37](#f-37) and
> [`results/evidence/confirmatory_report.txt`](../results/evidence/confirmatory_report.txt). Do not
> read them beside the pilot rows below. The test-split dense baselines, for reference:
> **pythia-160m 35.8575 · pythia-410m 21.3231 · pythia-1b 17.2564.**

**Pilot window only.** Pythia-160M at `50f5173d`, `Salesforce/wikitext` validation, **64 sequences ×
256 tokens (16,320 tokens)**, seed 1234, calibration `20bf57e6b08ed60d`, CPU evaluation, one seed each.
The Phase 7 screening numbers are in [F-10](#f-10) and belong to a different window — do not read the
two tables side by side.

| # | Arm | Sparsity | Bits | Comparison group | Reconstruction | Perplexity | Retention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | dense | — | 32 | — | — | **34.77** | 100% |
| 2 | quantisation only | 0% | 8 | — | sweep | **34.85** | 99.8% |
| 3 | pruning only | 50% | 32 | tensor | sweep | **233.94** | 15% |
| 4 | pruning only | 50% | 32 | **output** | sweep | **124.32** | 28% |
| 5 | pruning only | 50% | 32 | **output** | **none** | **209.21** | 17% |
| 6 | joint | 50% | 8 | tensor | sweep | **231.96** | 15% |
| 7 | joint | 50% | 8 | tensor | sweep, 128-sample calibration | **227.08** | 15% |
| 8 | joint | 50% | 8 | **output** | sweep | **122.51** | 28% |
| 9 | joint | 40% | 8 | **output** | sweep | **60.46** | 58% |
| 10 | joint | 30% | 8 | **output** | sweep | **42.43** | 82% |

Rows 3, 6 and 7 are **superseded** by 4 and 8 — they used the tensor-wide comparison group, which
F-07 shows was costing 6.7×. Retained because they are the evidence for F-07.

Rows 2 and 5 are **diagnostic controls**, not experimental arms: row 2 isolates precision damage, row
5 isolates the mask from reconstruction.

---

## 4. Bugs found that would have invalidated results

Each of these produced plausible-looking output rather than an error, which is why they are worth
recording.

| # | Bug | What it would have done |
| --- | --- | --- |
| B-54 | **The recovery ablation compressed its arms against the RECOVERY slice instead of the calibration set, inverting the disjointness its own design claimed** ([F-42](#f-42), [F-43](#f-43)) | `scripts/run_recovery_ablation.py` called `arm.set_calibration(list(recovery_batches), fingerprint=calibration_fingerprint)` — passing the recovery data where the calibration batches belong, while stamping the record with the *calibration* set's fingerprint. `ExperimentRunner._attach_calibration` had always done it correctly, building batches from `calibration.loader`; this script diverged from the reference and nothing compared them. **Three consequences, in increasing order of how much they matter.** (i) The record's `calibration_fingerprint` described 128 sequences that were never used to calibrate anything. (ii) `recovery.max_steps` silently coupled to *compression*, because the slice is sized `max_steps × batch × accumulation` — so F-42 and F-43 compressed against 1600 sequences and the R=8 leg at 50 steps against 400, making their before-recovery numbers non-comparable across configs. (iii) **The design's headline property was exactly inverted.** The code comment states the arms are fitted on the calibration sequences, so recovering on those tokens "would partly be re-fitting on seen data" — and then the implementation fitted them on the recovery sequences, making the recovery phase re-fit on seen data *entirely*. The log line "DISJOINT from the 128 calibration sequences (overlap 0 by construction)" was true about the wrong pair of sets. **How it was caught, and how it could have run to completion unnoticed:** the R=8 leg's replicate 0 reported a before-recovery joint gain of **+1.4536 pp** where F-43's replicate 0 — same draw, same budget, same frozen order, compression finishing before any recovery begins — had reported **+1.4448 pp**. A quantity that must be bit-identical differed by 0.0088 pp. Nothing failed; both runs completed, every fairness guard passed, and both numbers looked entirely reasonable in isolation. **What survives:** both arms always saw byte-identical data in identical order, so every *within-run* comparison stays fair and the joint gains of [F-42](#f-42) and [F-43](#f-43) stand as internal comparisons. **What does not:** F-42's claim that its +1.2772 pp before-recovery gain is "consistent with [F-37](#f-37)'s +1.0120 pp on test, so the compression half of the pipeline reproduces" is weaker than stated — the two used different calibration sets (1600 sequences against 128), so the agreement was never apples to apples; and F-43's guard row asserting the recovery slice was disjoint from calibration is **wrong as implemented**. **Direction of the bias, argued not measured:** [F-21](#f-21) found solver efficiency 0.6409 under the sequential mask against 0.5631 under the joint mask, so joint leaves more of its calibration objective unclaimed and recovering on that same data should help **joint more**. The runs observed joint improving **less**. So the bug's expected bias runs *against* the reported inversion, which is if anything understated by it — the sixth time a fault here ran in the flattering direction and the second time one ran against it. **[F-37](#f-37) and [F-41](#f-41) are untouched:** they run through `ExperimentRunner`, which was never wrong. **FIXED 2026-08-23**: calibration batches are now built from `calibration.loader` exactly as the runner does, and the script **fails closed** with a non-zero exit if the recovery indices intersect the calibration indices, so the disjointness is checked rather than asserted in a comment. The R=8 records produced under the bug are moved to `outputs/recovery_ablation/quarantine_b54/` rather than deleted. **Worth carrying:** 35 tests guarded this ablation — mask freezing, fake quantisation, budget matching, split, tags, output tree, resume — and not one asserted what the arms were calibrated *on*. The suite was guarding the recovery phase and ignoring the compression that precedes it |
| B-53 | **A record's `git_commit` can be silently null, and this is the mechanism behind the two null commits in the confirmatory run** ([F-43](#f-43)) | Found on the gentle recovery probe: the joint arm's record carries `git_commit: null` while the sequential arm's carries `b680e09`, on the same host, same script, ~3 hours apart. `get_git_commit` shells out to `git rev-parse HEAD` with a **5-second timeout** and returns `None` on `OSError` or any `subprocess.SubprocessError` — which includes `TimeoutExpired` — logging the reason at **DEBUG**. Every run in this project logs at INFO, so **the failure left no trace whatsoever**: a null looked like a field nobody had populated rather than an event that happened and was swallowed. Five seconds is not generous on a host running a 200-step GPU training loop with a CPU-bound evaluation alongside, which is exactly the condition under which the confirmatory grid produced **its** two null commits — [F-37](#f-37) records them as an unexplained limitation, and this is very likely the explanation. **No number is affected**: the commit is provenance, not input, and the run's code is identifiable from the surrounding records and the log timestamps. But it defeats the one rule the reproducibility policy rests on — *run from a clean tree at a committed SHA* — because a record that cannot name its SHA cannot be checked against that rule, and the project's single unusable result came from precisely a provenance failure (`aec5099-dirty`). **FIXED 2026-08-23**: both git subprocess timeouts raised 5 s → 30 s, and both failure paths now log at **WARNING** rather than DEBUG, so a null commit is loud at the level runs actually use. Deliberately *not* made fatal — aborting a 78-minute compression cell because `git` was slow would trade a provenance gap for a lost measurement, which is the worse failure. The affected records are **not** re-run to backfill the field; the SHA is recorded in [F-43](#f-43) instead, and the two confirmatory nulls stay as they are because step 10 runs once. Worth carrying: this is the third fault in this family (**[B-50](#4-bugs-found-that-would-have-invalidated-results)**, **[B-51](#4-bugs-found-that-would-have-invalidated-results)**) where a silent-loss path left the *count* of records healthy and something inside them missing or wrong |
| B-52 | **The exported joint-gain table paired arms ACROSS evaluation splits**, putting a wrong number in a committed artefact ([F-41](#f-41)) | `_joint_gain_rows` keyed on `(model, budget, replicate)` and not on the split, and the two splits produce the same experiment ids by construction -- so nothing looked wrong. At `qwen2.5-0.5b/moderate/rep0` the **validation** Q→P record was selected as best-of against the **test** joint record, exporting **−0.2143 pp** where the frozen-order test gain is **−0.0357 pp**. Same root cause as [B-51](#4-bugs-found-that-would-have-invalidated-results), seen from the analysis side rather than the write side. The aggressive rep0 row was correct only by luck: P→Q happened to beat the stray validation Q→P value, so best-of picked the right record while still exporting a contaminated `sequential_qp_retention` column. **The reports were never wrong** -- `report_confirmatory.py` pairs independently and always filtered on split -- so F-37 and F-41 stand, but `joint_gains.csv` is the file a reviewer recomputes from, and it disagreed with them. Fixed by adding **`eval_split` and `dataset_fingerprint`** to the pairing key and exporting both, and by taking the baseline on the test split from the **frozen order** rather than best-of: A1 §3 freezes the order before test, so only one order exists there, and maximising over whatever else happens to be present is precisely how a validation record leaked in. Best-of is retained on validation, where both orders genuinely ran. A cell with no frozen order, or missing its frozen arm, now emits **nothing** rather than substituting another order. Three regression tests, one pinned to the exact numbers. After the fix every CSV cell mean matches its report to four decimals |
| B-51 | **A record id does not encode the evaluation split, so a test-split run silently OVERWRITES the validation record of the same cell** ([F-40](#f-40)) | Caught on the Qwen leg, and **the blast radius is larger than first recorded**: the test grid overwrote **five** validation records, not one -- the dense baseline, **both `sequential` (P→Q) records**, and **both `joint` records**. Only the two `sequential_qp` records survived on disk, because no test-split cell shares their id (the frozen order is P→Q at both budgets, so `sequential_qp` was never re-run). `qwen2.5-0.5b_dense_moderate_s00_b32_rep0.json` held the **validation** dense baseline (ppl 17.7758) used as the retention reference for order selection. The test grid planned the same cell, `exists_valid` correctly refused to reuse a validation record for a test run -- and then wrote the test result **to the same filename**, destroying the validation one (now ppl 17.0962, split `test`). **[F-40](#f-40) survives only by luck:** the dense smoke happened to run under a distinct `experiment.id`, so a copy exists as `qwen_smoke_dense__...`, and every arm record stores the retention it computed at run time, so the reported figures remain checkable. Had neither been true, F-40 would cite retention numbers whose denominator no longer existed. **The class is the dangerous part**: `exists_valid` gates *reuse* on the split but nothing gates *overwrite*, so the guard that prevents reading the wrong record does not prevent destroying the right one. Same family as [B-45](#4-bugs-found-that-would-have-invalidated-results) (split confusion) and [B-50](#4-bugs-found-that-would-have-invalidated-results) (silent loss that leaves the count looking healthy). **FIXED 2026-08-14**, once the Qwen grid was no longer mid-flight. `ExperimentTracker.save` now ARCHIVES an existing record to `<id>__split-<old>.json` when the incoming record carries a different `eval_split`, rather than overwriting it. Renaming rather than refusing, deliberately: refusing would block the legitimate case -- running the same grid on the other split, which is the entire two-split design -- and an error there would strand a completed cell. The archive keeps its `.json` suffix in the same directory, so `load_all` still finds it and every split-aware consumer still filters correctly; `_load_dense_reference` in particular recovers the reference this bug used to delete. Two tests: one reproduces the exact Qwen failure and asserts BOTH splits survive, the other asserts a same-split re-run still overwrites, so the archive path cannot fire on the ordinary case and litter the directory. **The destroyed records are RECOVERED from git history, not re-run.** The evidence set committed at **`2832914`** -- the [F-40](#f-40) commit, made *before* the test grid ran -- contains all eight Qwen validation rows with a sha256 per source record. `scripts/recover_qwen_order_selection.py` extracts them to `results/evidence/qwen_order_selection.csv` with a provenance file naming the source commit, the per-record hashes, which records were destroyed and which survive; `--check` verifies the artefact still matches history. The recovered values reproduce F-40 exactly (P→Q 95.2854 against Q→P 95.1916 at W8; 77.3993 against 76.0485 at W4). **Order selection was deliberately NOT re-run**: the test results are known now, and repeating a selection step after seeing the outcome it feeds is post-hoc by definition and would invalidate the freeze it exists to document. The dense record is additionally still on disk under its smoke experiment id (`qwen_smoke_dense__...`, ppl 17.7758); no file is manufactured under the canonical name, which would create a record whose filename and internal id disagree. The Pythia primary was never affected: its 171 test and 54 validation records coexist under distinct experiment ids |
| B-50 | **`scale_trend` dropped every Q→P cell**, so 5 of 42 pairs vanished from figures and record-level analysis -- and they were the pairs *against* joint ([F-37](#f-37)) | The record-level pairing filtered on `SEQUENTIAL` and `JOINT` and never learned about `SEQUENTIAL_QP`. `find_comparison_pairs` **was** fixed for this ([B-42](#4-bugs-found-that-would-have-invalidated-results)) and carries a comment explaining why; this function was missed, so the same fault survived in a second place. The consequence: the one cell whose frozen order is Q→P -- **pythia-1b/moderate** -- was silently excluded from every trend and every figure built from records. `generate_plots.py` reported "**37 comparable pairs of 37**", which reads as complete: the denominator was computed from the same filtered set, so nothing looked missing. **The direction is what makes it serious.** The dropped cell is the *only* one where joint is consistently worse than sequential (−0.179 pp, 0/5, every bootstrap interval excluding zero), so omitting it removed the strongest evidence against joint -- the **seventh** fault in this project to run in joint's favour, and none has ever run the other way. **Caught while generating the paper figures**, before any figure was published; [F-37](#f-37) and [F-38](#f-38) are unaffected because `report_confirmatory.py` and `report_diagnostics.py` pair on their own and always accepted both orders. Fixed by treating whichever order was frozen as the comparator, with two regression tests. After the fix: **42 comparable pairs of 42** |
| B-49 | **Per-cell deployment measurements are not comparable across arms**, and read as a 40% speedup from sparsity ([F-37](#f-37)) | The sweep benchmarks every cell inside its own run, so each latency is taken at whatever moment that cell happened to execute. The confirmatory grid spanned **six days**, across which the host saw commit exhaustion, roughly a dozen process recycles ([B-48](#4-bugs-found-that-would-have-invalidated-results)) and repeated Modern Standby ([B-47](#4-bugs-found-that-would-have-invalidated-results)). Comparing two such numbers compares machine states, not models. **Caught while building the paper tables:** pythia-1b dense read **1041 ms** against pythia-1b pruning **630 ms**, an apparent **40% speedup from masking weights** -- impossible, because pruned weights stay FP32 and dense in storage so the GEMM does identical work, and flatly contradicting [F-34](#f-34)'s finding that 30% unstructured sparsity buys no CPU latency. The cause is the dates: the dense figure was measured **2026-08-05** and all ten pruning figures **2026-08-07**, where they are tightly self-consistent (627-650 ms, IQR 10-33 ms). Both benchmarks used identical settings -- 4 threads, seq 128, batch 1, 30 runs, 5 warmup -- so the difference is not configuration. **Nothing published is wrong**, because [F-34](#f-34) is a dedicated §4.7 study with model-order rotation, which exists precisely to control this drift, and it is the authoritative latency result. The risk was forward-looking: a paper table built from these incidental numbers would have claimed a large sparsity speedup the study elsewhere reports as absent. `build_paper_tables.py` now prints the measurement **date** in every latency row and refuses the comparison in the caption |
| B-48 | **The runner does not release memory between cells**, and at 1B that exhausts the commit limit ([F-37](#f-37)) | Measured on the confirmatory 1B leg: commit free fell **20.24 → 1.03 GiB over five `sequential` cells**, about **4 GiB per cell**, and stopping the process returned all of it at once — so it is the sweep accumulating, not system pressure. Pruning cells leak less (~1.6 GiB) than the arms that pack, which is consistent with `convert` plus `_verify_saved_artefact` loading a **second** full model to reload the checkpoint. **The consequence is the dangerous part:** `continue_on_error` is on, so the resulting `MemoryError` would not stop the run — it would *drop a cell and continue*, and a dropped `sequential` or `joint` cell silently removes an entire comparison. That is [B-46](#4-bugs-found-that-would-have-invalidated-results)'s failure shape again, arriving by a different route. **Worked around, not fixed:** the sweep was recycled whenever commit free fell below a threshold, preferentially within the first minutes of a cell so the loss is bounded by one in-flight cell (`skip_existing` re-runs nothing and each record is written whole, so recycling is numerically inert — the 171 records were produced across roughly a dozen process lifetimes). **No number is affected.** But any future long grid must either fix the release or supervise the process, and the naive fix — restarting on a threshold alone — thrashes once the recycle interval approaches the cell duration, which at 1B it did (~55 min against ~54 min). **Fixed 2026-08-11, after the confirmatory run and therefore without touching it:** `scripts/run_scale_sweep.py --isolate-cells` runs every cell in a child process, so memory is released at the boundary by construction rather than by hunting every retention inside the runner. Overhead is ~4 s per cell for interpreter start and config load, negligible against a ~50 min 1B cell. `--only-cell` drives the children and is also the way to re-run a single failed cell by hand. **Use it for any future long grid**; the completed primary experiment is untouched |
| B-47 | The benchmark host **enters Modern Standby mid-run**, and `duration_seconds` counts the suspended time as compute | STATUS recorded the power profile as "High performance, no downclocking, never sleeps". Measured rather than trusted: `powercfg` reports `STANDBYIDLE` on **AC** = 0x0e10 = **3600 s**, not 0, and the System log shows six Modern Standby exits on 2026-08-05 alone. Modern Standby is S0 low-power idle, so it logs Kernel-Power **506/507**, *not* the classic 42 that a sleep check greps for -- which is why the sleep history looked clean while the machine had been suspending all day. A busy CPU does not prevent it; a process must assert `SetThreadExecutionState`, and this one does not. **A first reading of this -- that standby explains the "intermittent stall" -- was published here on 2026-08-06 and is now WITHDRAWN as unsupported.** The correlation looked decisive: both slow cells of 2026-08-05 contain a standby window, 43.6 min containing 37.6 leaving **6.0 min** of work against a 6.2 min norm, and 52.5 containing 44.7 leaving **7.8** against 7.8, arithmetic closing to the minute on two cells. **It does not replicate.** On 2026-08-06 an 83-minute standby window (14:46:36-16:09:38, same `506 Idle Timeout` -> `507` structure, no power-source change in between) contained **four** 410M cells that ran at completely normal speed -- quality stages 15.7-15.8 min against a 15.8 norm, benchmark stages 16-17 s. A Win32 process is not frozen by Modern Standby as a rule, which is what those four cells show. So the stall mechanism is **unknown again**, the withdrawal of the [B-36](#f-33)/[B-41](#f-35) attribution is itself only provisional, and two matching cells were simply not enough evidence for a causal claim I stated as settled. Recorded in full because the error is instructive: an exact-looking arithmetic fit on n=2 persuaded me to close an open question, and the next four observations falsified it. **Consequence for quality numbers: none** -- suspension is not arithmetic, and the cells either side reproduce. **Consequence for §4.6 deployment measurements: real** -- a latency or throughput benchmark that spans a standby entry records the suspended interval inside its timing, and while the median survives one such sample the p95 and IQR need not. **Modern Standby entry could not be prevented, and on the evidence it does not need to be. Stop trying.** Three methods were applied and all three failed to stop entry: `standby-timeout-ac 0` (applied 10:59 on 2026-08-06, host suspended twice afterwards, both `Reason: Idle Timeout`, verifiably on AC with `STANDBYIDLE` reading 0); then `monitor-timeout-ac 0` plus `UNATTENDSLEEP 0`; then `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)` held by `scripts/keep_host_awake.py` for the whole run -- the host entered standby again at 09:36 on 2026-08-07 with the request verifiably held by a live process. That flag blocks classic **S3** sleep; it does not block **S0** low-power idle, which is a different state, and no user-mode setting reliably does. **It also does not matter.** Every cell measured across a standby window ran at normal speed: four 410M cells through the 83-minute window of 2026-08-06 (quality 15.7-15.8 min against a 15.8 norm), and two more through the 13.6-minute window of 2026-08-07 at 15.2 and 14.4 min against a **median of 15.0 over all 16** 410M sequential cells -- the cell containing the suspend was the fastest of the sixteen. A Win32 process keeps running through Modern Standby. The setting changes above are harmless and were left in place, but they bought nothing, and the ~65 cells run since have shown no stall at all. **Audited, and the result is clean:** all 73 suspend windows since 2026-07-14 -- the full span of the System log, which predates every deployment record -- were paired 506/507 and 42/107 and intersected against the 24 records carrying deployment data. Exactly one cell overlaps, `pythia-160m_pruning_aggressive_s30_b32_rep3`, and its suspend fell wholly inside the **quality evaluation** stage (17:02:56-17:45:56) while its `benchmark (CPU)` stage ran 17:45:56-17:46:02, after the wake. **No deployment figure in this project is contaminated.** The exposure is structurally small: the benchmark stage is ~6 s against a 6-40 min cell, so it is ~1% of the window -- which is luck rather than design, and is the reason the check was worth running rather than assuming. One methodological note from the audit itself: a record's `timestamp` is the cell **start**, not the end; reading it as the end inverts every window by one cell length and manufactures false overlaps |
| B-46 | The reload guard demanded a sparsity the mask arithmetic cannot reach | `_verify_reloaded` compared the reloaded zero fraction against the **nominal** `target_sparsity` with a `1e-6` tolerance. But `build_mask_from_scores` prunes `round(in_features * sparsity)` weights **per output row** -- an integer count -- so the realised fraction is quantised to multiples of `1/in_features` and lands up to `0.5/in_features` either side of target. pythia-160m's `attention.query_key_value` is 768 wide: `round(768 x 0.30) = 230`, realising `230/768 = 0.299479`, short of 0.30 by **5.2e-04**, or 520x the tolerance. The guard therefore rejected masks that were exactly right, and the comment above it asserted the opposite of the truth ("the mask budget is a floor, not a target"). **Caught live on the confirmatory run: every `sequential` cell failed at `measure checkpoint`.** It hits `sequential` and `joint` only -- pruning-only stays FP32 so never enters `load_packed_model`, and quantisation-only has `target = 0` so the check is skipped -- which is to say it removed **both arms of every comparison** while leaving the controls green, so a casual read of the record count would have looked healthy. Fixed by deriving the allowance from the row width the manifest already records, keeping it three orders of magnitude below any real serialisation loss. **Numerically inert:** it only decides whether a verification raises, so the 36 records already written stay valid and `METHOD_VERSION` is not bumped |
| B-45 | The dense reference was chosen **before** the fingerprint that guards it exists | `_load_dense_reference` filtered candidates with `_window_mismatch`, whose corpus check reads `self._eval_fingerprint` -- an attribute set *during* evaluation, while the dense reference is loaded *before* evaluation to be passed into it. The expected fingerprint was therefore always `None` there, that check short-circuited, and the remaining window checks passed because validation and test share identical 493x512 shapes. A test-split cell silently accepted a **validation-split** dense record and the error surfaced later in `add_quality`. Caught live on the confirmatory run: **17 of the first 20 test-split records failed**, and with `continue_on_error` on it would have spent ~78 h producing dense records and nothing else. Fixed by filtering on `config.data.eval_split`, which every record carries and which *is* available at lookup time. Note against myself: [B-37](#f-33) added the **device** to this same guard earlier and did not notice the fingerprint half was already dead -- adding a check beside a broken one is not the same as checking it |
| B-44 | The confirmatory config never enabled the verified 1B per-block offload path ([F-36](#f-36)) | `offload_blocks` defaults false. The prelaunch pilot therefore spent 65.67 min in joint compression instead of F-31's 4 min 34 s offloaded measurement, a 14.3× regression that would add many hours and risk shared-memory spill across the 1B grid. The paths are bit-identical, so quality is not changed; feasibility and the frozen runtime are. Caught before test evaluation. |
| B-01 | `method_definition.md` specified full-model QAT while the plan specifies layerwise PTQ | The normative document described a method nobody was building; decision D3 was a direct casualty |
| B-02 | D3 recommended ranking masks on FP32 shadow weights | Would have failed §3.8's definition of joint — the joint arm would not have been joint |
| B-03 | `backend: x86` in shipped configs | Conversion fails *after* the compression compute is spent ([F-02](#f-02)) |
| B-04 | `dataset: wikitext` | `datasets` 5.x rejects the bare alias with an opaque `HfUriError` about an internal `hf://` path; fails after the model is resident |
| B-05 | `storage_efficiency` assumed every parameter reached the target bit width | Read 0.41 with a "2.4× larger than its budget allows" warning on an artefact that was exactly as small as the method can make it. Embeddings are excluded by design (§2.6) and are nearly half of a 160M model |
| B-06 | `.gitignore` named only specific subdirectories | Run output at `outputs/<experiment_id>/` was untracked **and unignored** — one `git add -A` from being committed, against an explicit hard rule |
| B-07 | `plan_from_config` read `quantisation.bits` directly | The **pruning-only arm** was handed a bit width and `convert` packed it — silently quantising the one FP32 arm, which is the arm that answers RQ4 |
| B-08 | `tiny_causal_lm` is session-scoped and compression is destructive | Driver tests leaked compressed weights into every later test in the suite, including other files |
| B-09 | Tensor-wide mask comparison group | 6.7× perplexity ([F-07](#f-07)) |
| B-10 | Packed metadata returned from `get_extra_state` as a dict | `save_pretrained` walks the state dict expecting tensors; the first full run crashed at save |
| B-11 | `evaluation.max_samples` may exceed `data.max_eval_samples` on a shared split | Evaluates on sequences the calibration set was drawn from, violating §4.1. Inflates every arm equally, so no comparison looks wrong ([F-11](#f-11)) |
| B-12 | Four identical dense cells planned per four-budget grid | Wasted compute plus near-duplicate records §10.4 asks the audit to reject |
| B-14 | Joint arm ran on 2x the sequential arm's solver budget; the fairness guard was never called ([F-12](#f-12)) | Every joint-gain number before 2026-07-29 was non-attributable under §3.11 |
| B-15 | `LayerPlan` and `ReconstructionConfig` carried separate `joint_iterations` defaults | Changing the config default silently left the plan default in place, which is what runs |
| B-17 | Joint outer loop accepted every proposal, including worse ones ([F-16](#f-16)) | An alternating optimiser that discards a better solution it already found; explains four iterations scoring worse than two |
| B-18 | Conversion refit the quantisation grid instead of reusing the solver's | Packed a different model from the one evaluated; `verify_packing` existed and was never called |
| B-19 | Activations captured once per block, not per dependency group | MLP down-projections fitted against inputs the dense up-projection produced -- blockwise reconstruction described as layerwise |
| B-21 | `mask_sparsity` and `zero_code_fraction` were never serialised ([F-17](#f-17)) | Both read 0.0000 in every record, so the pruning budget could only be checked against the conflated numeric sparsity |
| B-20 | Dense-reference lookup ignored the evaluation window | Normalised a 64x256 run against a 493x512 baseline and reported the resulting ratio as retention |
| B-16 | Three-seed confirmatory protocol produces three identical numbers ([F-15](#f-15)) | The paper would report a seed spread of zero as though the protocol had been followed, and §6.3's practical-importance rule would be vacuous |
| B-22 | Every arm reconstructed against its own intermediate weight, not the dense weight ([F-18](#f-18)) | The arms solved two different problems and their losses were not on a common scale; the asymmetry penalised sequential and **inflated joint gain** |
| B-23 | Joint acceptance compared pre-canonicalisation weights, then stored the canonicalised ones ([F-18](#f-18)) | The incumbent guard chose between quantities it had not measured consistently, and the packed artefact was not the object that won |
| B-24 | An OpenMP deadlock mitigation pinned **inter-op threads process-wide** at three entry points | Inter-op can be set only once per process, and `set_cpu_threads` only *logs* the failure to re-set it — so `CpuBenchmark.prepare()` would request the frozen 4 threads, silently run at 1, and record `requested_interop_threads: 4`. The mismatch guard checked intra-op only. Hits the pruning-only arm, which under D1 is the sole route to RQ4 |
| B-25 | The Wanda anchor's own tie test asked whether a disagreement sat on a score equal to our float32 prune threshold ([F-19](#f-19)) | The tie exists in float64 and float32 has already broken it, so the test called 2 of 4 precision-driven swaps genuine faults and returned INVESTIGATE on a pipeline that was correct |
| B-26 | The reconstruction anchor sampled modules by a plain stride ([F-20](#f-20)) | `48 // 6 == 8` and a block has four target modules in fixed order, so all six samples were `attention.query_key_value` and no MLP projection was ever checked -- a confident PASS over a quarter of the model |
| B-27 | The arm-slack metric was named `attributable_joint_benefit` and printed as "1.0 = entirely real" ([F-21](#f-21)) | On a run where every row's advantage was negative it read +0.6425 and invited "64% of the joint gain is real", when it was the ratio of two disadvantages meaning the sweep overstates joint's penalty by 1.56x |
| B-28 | Reference SparseGPT driver replayed blocks with `use_cache=True` and a live `Cache` ([F-22](#f-22)) | The cache would accumulate across replays, growing the key/value length and silently changing the activations SparseGPT fits to. Caught before the reference stage ran |
| B-29 | `blocks[0] = Catcher(...)` mutated a **copy**, because `get_decoder_blocks` returns `list(current)` ([F-22](#f-22)) | The model kept calling the real block, so zero calibration inputs were captured; SparseGPT would have pruned against nothing and still produced a perplexity. Only an empty-capture guard turned it into an error |
| B-30 | Joint gain was measured against P→Q only, though §3.6 and §6.1 always required best-of {P→Q, Q→P} ([F-24](#f-24)) | Q→P beats P→Q *and* joint at 30% + W8, so the moderate budget's joint gain was reported as +0.07 pp when against the required baseline it is −0.36 pp. Another omission that flattered joint |
| B-31 | A single-draw joint gain was reported as a point estimate, and a scale trend built on two of them ([F-26](#f-26)) | The 410M cell swings from −0.50 to +0.98 pp across calibration draws, so +0.68 pp was luck. The paired difference is *noisier* than either arm, contradicting the stated expectation that pairing would cancel draw noise |
| B-32 | `exists_valid` did not compare the evaluation device ([F-29](#f-29)) | Switching exploratory runs to GPU evaluation would have let `skip_existing` reuse CPU-evaluated records inside a GPU grid, mixing devices within one comparison at ~1e-5 -- too small to change a conclusion, invisible without the check |
| B-33 | `exists_valid` did not compare the **machine** ([F-30](#f-30)) | Same class as B-32, and it went live the moment the machine policy allowed a second host to run compression: two hosts writing into one `outputs/metrics/` would have let `skip_existing` pull the other machine's record into a comparison. `host_key` is built from fields every record already carried, so the guard added no recompute |
| B-34 | Block-offload captured block-0 inputs on the **host** ([F-31](#f-31)) | GPT-NeoX computes the rotary `cos`/`sin` in that forward and passes them into every block; CPU and CUDA trigonometry disagree in the last bits, which flipped a near-tie in the saliency ranking. One mask position in block 0 moved `attention.dense` by **2.25 absolute** and cascaded through every later block, for a **1.6% perplexity change that more than doubled the joint gain** (+2.35 pp against +1.08). Caught by the F-23 reproduction gate; would have been invisible to any tolerance-based check |
| B-34b | The first fix moved the **whole model** to the device for the capture ([F-31](#f-31)) | Numerically correct, and it fitted at 160M and 410M. It then hit `CUDA error: out of memory` at 1B -- the one model offload exists for. Reasoning that no Gram factorisation is live during capture accounted for what was absent and not for what was present: 3.77 GiB of weights, ~0.5 GiB of cached hidden states, and the forward's own activations |
| B-35 | Recorded quantisation grids did not follow their block back to the host ([F-31](#f-31)) | `grids_by_module` is captured while a block is resident, so under offload it held CUDA codes describing host weights. `convert` then died with a device mismatch **after the whole compression was spent** -- an artefact-stage failure caused by a compression-stage bug |
| B-43 | "Above chance" was a bare arithmetic comparison ([F-35](#f-35)) | 0.2501 on a four-choice task counted as above chance while being statistically indistinguishable from it, so a report could have said "the model still performs the task" on the strength of a meaningless margin. Replaced by a three-way verdict over a +/- 2 stderr interval, plus `unknown` when the harness reports no stderr. Descriptive only -- the primary downstream comparison is retention against dense, so no claim depends on the threshold |
| B-42 | The **confirmatory** sweep did not resolve the frozen sequential order ([F-35](#f-35)) | `main_scale_sweep.yaml` listed `sequential` for every cell and encoded no order at all, and `run_downstream.py` mapped every sequential arm to P→Q. At pythia-1b/moderate the frozen order is **Q→P** ([F-32](#f-32)), so both would have used the *weaker* baseline and inflated the joint gain -- [B-30](#f-24) recurring, in the one run that cannot be redone. Fixed with a machine-readable table in `scale_aware_compression.protocol`; an unfrozen cell **raises** rather than defaulting, because defaulting to the §3.6 primary is what makes the fault look safe |
| B-41 | The allocator cache was not released between compression and **downstream** evaluation ([F-35](#f-35)) | Third appearance of one mechanism, and the largest: F-29 measured it at 7x, B-36 at 4x, this at **24x**. 410M/joint took 3 h 37 m against 10 m 44 s for 410M/dense, at the *same* instantaneous rate (151-159 against 181-186 it/s) -- so it was stalling, not running slowly, and Windows was serving the shortfall from shared system memory. I had fixed the runner for this and not this script. Accuracies unaffected: memory placement is not arithmetic, and the re-run reproduced 160M/joint's values exactly. Also fixed: the script wrote only at the END, so the 3 h 37 m stall would have discarded five completed evaluations |
| B-40 | The exact sign test counted **ties as negatives** ([F-35](#f-35)) | `positive_count` counted `gain > 0` while the denominator stayed the full replicate count, so an exact zero biased the p-value. Verified **latent** before fixing -- no gain in any committed record is exactly zero, and the smallest recorded 1B gain is +0.0044 pp, which *rounds* to +0.00 in a two-decimal table but is genuinely positive -- so no published figure moved. Size of the error pinned as a test: three positives and one tie gave p = 0.625 and now gives p = 0.25. Ties use **exact equality, no tolerance**, because every candidate tolerance is now visible in the results and choosing one would be selecting an analysis parameter after seeing the data (§6.3) |
| B-39 | `summarise_screening.py` reported the **first** matching cell per (budget, arm) | With replicates in `outputs/metrics/` it silently picked one arbitrary draw and printed it as the number for that budget -- the automated form of the fault [B-31](#f-26) retracted a headline for, and worse because no human chose it. Its own output even asserted "one calibration draw". It now **refuses**, the same way it already refused on mixed evaluation windows, and points at `metrics.replicates.summarise_replicates`, which reports mean, sd and n. Verified: it refuses on the current record set (5 draws at moderate, 3 at aggressive and S6) and still summarises the genuinely single-draw F-23 budgets |
| B-38 | `find_comparison_pairs` keyed on (model, budget, seed) and **not the replicate** | A1 gives every replicate the same run seed, so all R replicates collided on one key and the dict kept only the last. A 3-replicate grid with 6 sequential and 6 joint cells reported **2** pairs; `main_scale_sweep` reported 6 against 42. The symmetric-difference warning could not catch it, because the dropped cells were never distinct keys. **No published number moved** -- the record-level `_pair_key` already included the replicate, and the findings were computed through that path -- but it under-reported how many comparisons a grid would yield, on the line a person reads before committing 38 hours. There were no tests for this function at all; there are six now |
| B-37 | The dense reference was matched on window and corpus but not on **device** ([F-33](#f-33)) | Retention is a ratio, so both halves must come from one device. Harmless until GPU evaluation was wired in; after that a GPU-evaluated compressed run could normalise against a CPU-evaluated dense record, putting the ~1e-5 device drift *inside* the retention figure where it cannot even be declared, rather than across tables where it can |
| B-36 | The allocator's cache was not released between the compression and evaluation stages ([F-32](#f-32)) | Introduced by wiring GPU evaluation. PyTorch's caching allocator holds freed memory, so one stage's peak stayed reserved while the next asked for its own -- and **on Windows the driver satisfies the shortfall from shared system memory instead of raising**. No error, no warning, a plausible perplexity, and a 1B cell at **32 min instead of 8m46s**: compression 21 min against a 4m34s standalone measurement, evaluation 11 min. This is the same silent-spill mechanism [F-29](#f-29) measured at 7x on the widest layer, and it is the clearest argument on record for why deployment measurements are CPU-only -- a latency taken under these conditions would have looked entirely normal |
| B-13 | Sweep cells inherited the base config's model revision | Every cell pinned to the *first* model's SHA — fails to load, or silently loads the wrong weights if the SHA exists in both repos |

Two of these were **masked by tests that should have caught them**: B-07 (the test disabled
quantisation in its fixture) and B-09 (the synthetic layer was too well-behaved). Both crutches have
been removed.

---

## 5. Decisions taken, with the evidence

Full reasoning in [protocol_freeze.md](protocol_freeze.md). Summarised here with what settled each.

| # | Decision | Settled as | Settled by |
| --- | --- | --- | --- |
| D1 | CPU quantisation backend | PyTorch native INT8, engine `onednn`. W4 for quality and size only, never latency. RQ4 answered from the pruning-only arm, whose weights stay FP32 | §3.12 + §10.1 already separate size from latency; probe in [F-02](#f-02) |
| D2 | Reconstruction solver | Error-compensated column sweep; damped ALS retained as reference | Feasibility measurement in [F-03](#f-03) |
| D3 | Mask scoring | Activation-weighted magnitude on **quantised** weights in the joint arm | §3.7 and §3.8 require it; overrode the previous recommendation |
| — | Mask comparison group | **Per-output** | Measurement in [F-07](#f-07) |
| — | Scale rule | **max-abs**, clipping search rejected | Measurement in [F-06](#f-06) |
| — | Pythia variant | standard, never deduped | §2.7 |
| — | Downstream evaluator | `lm-evaluation-harness`, pinned | §4.3 requires HellaSwag / PIQA / ARC-Easy; reimplementing risks silent scoring differences |
| — | Practical-importance rule | ≥ 1.0 pp retention, consistent in sign across all three confirmatory seeds, exceeding the seed spread | §6.3 requires it predefined; set before any compressed result existed |

---

## 6. What the paper may and may not claim from this log

> **Rewritten 2026-08-10, after [F-37](#f-37).** The previous version of this section was written
> when the log held one model and no confirmatory comparison; it said "anything about scale" and
> "that joint beats sequential, or does not" were both off-limits. The confirmatory run has since
> been executed and both are now answerable. The old text is superseded rather than deleted — it is
> in the git history at `ecaf8c7` — because it governed nothing that was published.

**The primary claim, from [F-37](#f-37) — the only test-split evidence:**

- **No cell meets the pre-registered §6.3 practical-importance bar** (≥ 1.0 pp mean gain *and* a
  consistent sign across every paired replicate). The study's answer to its own primary question is
  therefore **negative on practical importance**.
- Joint compression gives a **small positive gain at 4 bits** over the frozen sequential baseline:
  **+1.01 pp at 160M** (7/8 replicates positive, p = 0.0703) and **+0.93 pp at 410M** (**8/8**,
  raw p = 0.0078, **Holm–Bonferroni-adjusted p = 0.0469 over the six cells examined**). The 410M
  cell is the only statistically significant result in the study, and it survives correction by
  0.0031. **Always quote the adjusted value.**
- At **8 bits** the mechanism yields nothing at 160M and 410M (+0.04, +0.03 pp, sign inconsistent)
  and a **small reliable disadvantage at 1B** (−0.18 pp, 0/5, every bootstrap interval excluding
  zero on the negative side).
- **The W4/W8 structure holds in a second model family.** Qwen2.5-0.5B, run under the same protocol
  at R = 8, gives **+0.4213 pp at W4** (7/8) and **−0.0321 pp at W8** (1/8)
  ([F-41](#f-41)). Neither cell meets §6.3 and neither is significant, so this is evidence for the
  **structural** claim only — W4-specific, small, sub-threshold — not for any effect size.
- **The joint advantage does not grow with scale.** It is flat from 160M to 410M (+1.01, +0.93) and
  falls at 1B (+0.13). The study's motivating hypothesis — that joint compression would pay off
  more as models grow — is therefore **not supported**. It must **not** be described as refuted:
  no test compares the scales, and the 410M→1B step is confounded with depth ([F-38](#f-38)).

**May also claim, with the conditions attached:**

- The pipeline hits its budgets exactly and the precision is real, verified on a converted, reloaded
  artefact ([F-09](#f-09)).
- Reconstruction improves the layer objective on every layer measured, and improves end-to-end
  perplexity by 41% over mask-only ([F-07](#f-07)).
- The mask is **bit-identical** to an independent Wanda implementation over 84,934,656 weights
  ([F-19](#f-19)); the reconstruction sweep never falls below the provable optimum of its own
  objective ([F-20](#f-20)); and absolute retention is credible against unmodified SparseGPT once
  the comparison group is matched ([F-22](#f-22)).
- Weight-only INT8 quantisation is essentially free at this scale (99.8% retention).
- The comparison group is a first-order design choice for activation-weighted pruning, worth 6.7×
  perplexity at 50% sparsity on a 160M model ([F-07](#f-07)).
- Two quantisation-aware refinements — clipping scale search and keep-benefit scoring — measurably
  *hurt* under error-compensating reconstruction, with a mechanism for why ([F-06](#f-06)).
- 30% unstructured sparsity buys **no** CPU latency at any of the three scales ([F-34](#f-34)), and
  no downstream-task difference between arms is resolvable at this evaluation size
  ([F-35](#f-35)).

**May not claim:**

- **That joint compression is practically important at any scale.** The threshold was pre-registered
  and is not met anywhere. In particular the 410M cell's +0.9348 pp must not be rounded, restated
  as "≈1 pp", or compared against a relaxed bar: it fails §6.3 by 0.065 pp and the paper must say so.
- **"Significant at 410M" without the multiple-comparison correction.** Six cells were examined; the
  raw p = 0.0078 becomes **0.0469** under Holm–Bonferroni, which clears 0.05 by 0.0031. Quoting the
  raw value while reporting the largest of six cells is the inflation
  [validity_threats.md](validity_threats.md) warned about before any result existed.
- **A scaling law, or a monotone decline.** Three points cannot fit one, and two of the three now
  coincide within 0.08 pp. [F-32](#f-32)'s "monotone decline" reading was exploratory and does not
  survive the test split.
- **Any significance claim at 1B.** At R = 5 the best attainable two-sided sign-test p is 0.0625, so
  even a unanimous 1B cell cannot reach 0.05. This is a design limit accepted at freeze time.
- **That the exploratory point estimates were reliable.** Neither replicated: 160M fell from +1.69
  to +1.01 and 410M rose from +0.39 to +0.93 ([F-27](#f-27) vs [F-37](#f-37)). Any narrative built
  on the exploratory numbers is superseded.
- **That Qwen2.5-0.5B is a scale point, or that its perplexity is comparable to Pythia's.**
  Different tokeniser, vocabulary and corpus. Its 358M targeted parameters fall between pythia-410m
  and pythia-1b, so plotting it on the scale axis would look natural and be wrong
  ([F-41](#f-41)).
- **That the external leg strengthens the primary result.** Nothing in it is significant, and no
  cell meets §6.3. It corroborates a structural pattern; it adds no evidence for an effect size.
- **Anything about absolute quality against published results at the exploratory evaluation window.**
  The screening numbers used 64 sequences at 256 tokens; the confirmatory run used 512 × 512, and
  only the latter is comparable across cells here.
- **Anything with uncertainty from the §3 pilot table.** Every end-to-end number in
  [§3](#3-all-end-to-end-perplexities-in-one-table) is a **single seed** at a single model and a
  single window, and none of those rows is a confirmatory run. Uncertainty exists only for the
  test-split cells in [F-37](#f-37), where it comes from R paired calibration replicates plus a
  paired block bootstrap over evaluation windows — never from the pilot rows.
- **Any latency claim beyond [F-34](#f-34)**, which measured one sparsity (30%) at three scales
  under the §4.7 protocol. RQ4's *curve* needs several sparsities and was not run.

**Must disclose:**

- **Every implementation fault found in this project flattered the joint arm** — B-14, B-17, B-22,
  B-23, B-30, B-34, six in a row, none in the other direction. The final effect is small and was
  reached by removing biases that all pointed the same way; that belongs in Limitations regardless
  of where the number landed.
- **§6.3 was amended by A1 before the confirmatory run**, and the amendment *loosened* a
  pre-registered rule whose original binding clause was unmeasurable. The paper must state that the
  bar it failed is already the weaker of the two.
- The two records with null `git_commit`, and the R = 5 / frozen-order limitations, per
  [F-37](#f-37).

---

## 7. Reproduction

```bash
# Environment (Omen only; see §1 for the pinned versions)
.venv\Scripts\python.exe -m pytest -q          # offline; trust the count printed by this checkout
.venv\Scripts\ruff.exe check . && .venv\Scripts\ruff.exe format --check .

# Data of record
python scripts/prepare_data.py --config configs/experiments/pilot.yaml

# The rows of §3, by number
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml            # 1
python scripts/run_quantisation.py  --config configs/experiments/pilot.yaml             # 2
python scripts/run_pruning.py       --config configs/experiments/pilot.yaml             # 4
python scripts/run_joint.py         --config configs/experiments/pilot.yaml             # 8

# Controls and variants used above
--override compression.reconstruction.comparison_group=tensor        # rows 3, 6
--override compression.reconstruction.solver=als \
--override compression.reconstruction.local_steps=0                  # row 5, mask only
--override data.calibration_samples=128                              # row 7
--override compression.pruning.sparsity=0.3                          # row 10

# Rejected ablations of F-06
--override compression.reconstruction.scale_search=true
--override compression.reconstruction.keep_benefit_saliency=true
```

The historical run-ID collision is fixed. Record filenames now preserve the evaluation split, so a
test run cannot overwrite the validation record it depends on (B-51). Use the shipped sweep configs
and the canonical reproduction commands in [reproducibility.md](reproducibility.md); do not reconstruct
the confirmatory run from the older pilot overrides above.

---

## 8. Related

- [research_plan.pdf](research_plan.pdf) — the authoritative source
- [STATUS.md](STATUS.md) — where the work stands now
- [protocol_freeze.md](protocol_freeze.md) — the frozen decisions and the environment record
- [validity_threats.md](validity_threats.md) — what could still make the results wrong
- [method_definition.md](method_definition.md) — what the arms are
- [implementation_plan.md](implementation_plan.md) — build phases and exit tests
