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

> 🔴 **The joint-gain column below is retracted — see [F-18](#f-18).** Two further faults were found
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
