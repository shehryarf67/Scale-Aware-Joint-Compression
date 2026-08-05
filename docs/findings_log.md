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

### F-31 — Provisional Pythia-1B validation values were produced in Colab, but the Task 2 run is not yet admissible {#f-31}

*2026-08-01 — Pythia-1B, revision `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` — WikiText-2 validation, 493 × 512 — exploratory GPU evaluation on Tesla T4 — Python 3.12.13, torch 2.11.0+cu128 — repository commit `7ecaa28fb91c273890de54dce32d9ffa46244039` with uncommitted notebook changes*

The Colab notebook produced the following provisional quality values from one calibration draw:

| Budget | Arm | Perplexity | Retention |
| --- | --- | ---: | ---: |
| Moderate, 30% + W8 | dense | 17.9432 | 100.00% |
| Moderate, 30% + W8 | sequential P→Q | 18.6302 | 96.31% |
| Moderate, 30% + W8 | sequential Q→P | 18.6199 | 96.37% |
| Moderate, 30% + W8 | joint | 18.6305 | 96.31% |
| Aggressive, 30% + W4 | sequential P→Q | 20.0938 | 89.30% |
| Aggressive, 30% + W4 | sequential Q→P | 20.5073 | 87.50% |
| Aggressive, 30% + W4 | joint | 19.9903 | 89.76% |

The provisional order signal is Q→P for moderate W8 and P→Q for aggressive W4. However, the run
must not be used to freeze either order: the notebook restored `layerwise.py` from Git before the
sweep, later patched `activations.py` and `runner.py` in place, and the focused post-run tests failed
during collection (`capture_activations` / `LayerwiseReport` import errors). The recorded run also
used an uncommitted state and GPU quality evaluation, which is allowed for exploration but must be
reported as such. Rerun Task 2 from the clean committed Task 1 state, verify the full test suite and
offload gates, and then record the trusted 1B order decision.


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
