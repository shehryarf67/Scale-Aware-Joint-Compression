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

| | |
| --- | --- |
| Machine | HP Omen — **the only machine that runs code** |
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

---

## 3. All end-to-end perplexities in one table

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
| B-16 | Three-seed confirmatory protocol produces three identical numbers ([F-15](#f-15)) | The paper would report a seed spread of zero as though the protocol had been followed, and §6.3's practical-importance rule would be vacuous |
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

**May claim, with the conditions attached:**

- The pipeline hits its budgets exactly and the precision is real, verified on a converted, reloaded
  artefact ([F-09](#f-09)).
- Reconstruction improves the layer objective on every layer measured, and improves end-to-end
  perplexity by 41% over mask-only ([F-07](#f-07)).
- Weight-only INT8 quantisation is essentially free at this scale (99.8% retention).
- The comparison group is a first-order design choice for activation-weighted pruning, worth 6.7×
  perplexity at 50% sparsity on a 160M model.
- Two quantisation-aware refinements — clipping scale search and keep-benefit scoring — measurably
  *hurt* under error-compensating reconstruction, with a mechanism for why.

**May not claim:**

- Anything about absolute quality relative to published results. The evaluation window is 64
  sequences at 256 tokens, not a full test set at 2048 ([§1](#1-environment-of-record)).
- Anything about scale. Every number here is Pythia-160M. One model is not a trend.
- Anything with uncertainty. **Every end-to-end number in §3 is a single seed.** §5.5 requires three
  confirmatory seeds for the central comparison, and none of these are confirmatory runs.
- That joint beats sequential, or does not. No matched joint-vs-sequential comparison at a frozen
  budget has been run yet — that is Phase 7 onward. The +1.12% in [F-06](#f-06) is a *layer*
  objective on six layers, not a model-level result.
- Any latency claim. No benchmark in this log was collected under the §4.7 protocol (20–30
  repetitions, prefill/decode split, model-order rotation).

---

## 7. Reproduction

```bash
# Environment (Omen only; see §1 for the pinned versions)
.venv\Scripts\python.exe -m pytest -q          # 740 passing, offline, ~37 s
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

Run IDs currently collide across arms — `experiment.id` is `pilot` for every arm, so a compressed run
overwrites the dense record it needs for retention. Pass `--override experiment.id=<name>` until §5.6's
convention (`<family>_<size>_<method>_<sparsity>_<bits>_<seed>`) is implemented.

---

## 8. Related

- [research_plan.pdf](research_plan.pdf) — the authoritative source
- [STATUS.md](STATUS.md) — where the work stands now
- [protocol_freeze.md](protocol_freeze.md) — the frozen decisions and the environment record
- [validity_threats.md](validity_threats.md) — what could still make the results wrong
- [method_definition.md](method_definition.md) — what the arms are
- [implementation_plan.md](implementation_plan.md) — build phases and exit tests
