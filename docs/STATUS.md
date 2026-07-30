# Project status

**Last updated:** 2026-07-29 · second session on the HP Omen · **Phases 0, 5 and 6 complete**;
**Phase 7 (budget screening) paused with no valid results** — two rounds of external review
invalidated every number, the fixes are in, the grid needs re-running from scratch

> Read this first. It is the handoff between sessions and between machines. If it looks stale,
> check `git log` — the truth is the commit history, this file is a summary of it.
>
> **This file = where we are now.** For the durable roadmap (all ten phases, exit tests, the
> testing plan) see [implementation_plan.md](implementation_plan.md). For the frozen decisions and
> the environment record, [protocol_freeze.md](protocol_freeze.md). For **every measurement this
> project has produced, with the conditions that produced it**, see
> [findings_log.md](findings_log.md) — that is what the paper gets written from. For the
> authoritative source, [research_plan.pdf](research_plan.pdf).

---

## ✅ Resolved: the Smart App Control blocker

Smart App Control was enforcing user-mode code integrity and blocking the unsigned native
extensions the stack depends on — `shm.dll` and `torch_cuda.dll` (torch), three pandas `.pyd`
files, and `_regex` (a transformers dependency). It appeared *progressively*: the suite passed 502
tests right after install, then torch stopped importing about thirty minutes later as the policy
caught up with the newly written binaries.

**Turned off on 2026-07-28 and the machine rebooted.** Verified afterwards:
`VerifiedAndReputablePolicyState = 0`, user-mode code integrity enforcement `0`, and `torch`,
`pandas`, `regex` and `datasets` all import. The suite now runs in ~32 s rather than ~108 s,
because torch is no longer retrying blocked loads.

This is **irreversible** — Smart App Control cannot be re-enabled without reinstalling Windows.
Consequence worth remembering: this machine no longer screens unsigned executables, so be
deliberate about what gets installed on it.

If the environment ever appears to break again, re-check that state value first — a green test run
is not by itself evidence the environment is stable.

---

## Where we are

Infrastructure, the Phase 0 decisions, the compression primitives and the layerwise driver are all
done. **Every arm runs from a config to a run record on real Pythia-160M.**

| | State |
| --- | --- |
| Tests | **804 passing** in ~40 s, offline |
| Lint / format | `ruff check .` and `ruff format --check .` both clean |
| CI | `.github/workflows/ci.yml` — lint, format, tests on push/PR to `main` |
| Environment | verified end to end: torch 2.13.0+cu126, CUDA available, sm_89 |
| Runnable today | **all five arms** plus dense, config to run record, on a real model |
| Not yet done | budget screening (Phase 7), downstream tasks (A4), prefill/decode split (A5) |

### What works

- **Config system** — YAML with `include:` composition, typed dataclasses, validation at load.
- **Model registry** — 5 models, offline lookup, safe loader that validates before downloading.
- **Data pipeline** — tokenise → chunk → cache → fingerprint; calibration draw from a fixed seed
  with a held-out subset for the overfitting check.
- **Evaluation** — perplexity, dense-vs-compressed agreement, generation diagnostics.
- **CPU benchmark** — pinned threads, warm-up, repeated runs, median/p95/IQR-ready statistics.
- **Run records** — JSON + CSV, with git commit, hardware, software versions, and `status`.
- **`ExperimentRunner.run`** — complete for every arm, including calibration injection.

### Sanity check that the evaluation is correct

An untrained model with a 259-token vocabulary scored **perplexity 257.18**. A near-uniform model
should score near |V|, so this confirms the loss is computed over the right axis. A shift or shape
bug lands nowhere near it. Dense retention computes as exactly 100% (it is its own reference).

---

## ✅ Settled this session: the method family and the three open decisions

The repo carried **two generations of plan** that contradicted each other. The older markdown
specified full-model quantisation-aware fine-tuning; `research_plan.pdf` §3.1 specifies **layerwise
post-training reconstruction**, which is what actually fits on a 6 GB laptop GPU. The markdown was
still labelled "the spec", so it was quietly overriding the plan on real choices — D3 below was a
direct casualty.

Fixed:

- [method_definition.md](method_definition.md) **rewritten** against §3.1–3.12 — layerwise
  objective, the five arms through one shared solver, `local steps` as the fairness unit, Q→P
  reverse ablation, best-of-sequential. The fine-tuning material survives only as the optional
  recovery ablation.
- [protocol_freeze.md](protocol_freeze.md) **created** — the §2.7 freeze table, the environment
  record, and the three decisions below with the plan sections they follow from.
- Six new doc-guard tests, so the superseded design cannot creep back silently.

| # | Decision | Settled as |
| --- | --- | --- |
| **D1** | CPU quantisation backend | PyTorch native CPU **INT8**, engine **`onednn`**, is the sole latency backend. W4 keeps quality + size, never appears in a latency table. **RQ4 survives** — the sparsity→latency curve comes free from the pruning-only arm, whose weights stay FP32. |
| **D2** | Reconstruction solver depth | **Damped ALS first**, Hessian sweep as a later drop-in behind the same interface. §3.3 makes second-order optional, not expected. `H = XᵀX` accumulated from the start regardless. Memory is *not* the constraint: the worst-case layer Hessian is 256 MiB. |
| **D3** | Mask scoring rule | **Activation-weighted magnitude, scored on the quantised weights** in the joint arm — `S_ij = \|Q_b(W_ij)\| · ‖X_j‖₂`. This **overrides** the old Option B recommendation, which would have failed §3.8's definition of joint. |

Full reasoning for each is in
[protocol_freeze.md](protocol_freeze.md#the-three-decisions-that-were-open).

---

## ✅ Phase 5 — single-layer compression primitives: **done and passing**

Every primitive is implemented at the **tensor level** and validated against a synthetic layer, as
the plan's §10.2 checklist prescribes. 93 new tests.

| Primitive | Where | Status |
| --- | --- | --- |
| Activation capture — streamed `H = XᵀX`, `‖X_j‖₂`, relative damping, forward hook | `compression/activations.py` | done |
| Saliency — `S_ij = \|W_ij\| · ‖X_j‖₂` | `compression/pruning.py` | done |
| Mask — unstructured + 2:4 / 4:8, **exact** realised sparsity | `compression/masks.py` | done |
| Quantiser — symmetric, per-tensor / channel / group, W8 + W4 (+ W2) | `compression/quantisation.py` | done |
| Packing — int2/4/8 storage, bit-exact unpack, effective-bits accounting | `compression/quantisation.py` | done |
| Reconstruct — damped ALS on `(H + λI)`, per **D2** | `compression/reconstruct.py` | done |

**Exit criteria, all asserted:**

- reconstruction strictly reduces `‖Y − Ŷ‖²_F` versus naive rounding — both pruning-only and
  combined with W4
- realised sparsity is exact, including when scores tie (which activation weighting makes routine,
  since a dead input column zeroes a whole column of scores)
- quantised weights take ≤ 2^b distinct values per group, across all three granularities
- pack → unpack is bit-exact, tested at sizes that are *not* multiples of the lane count, because
  padding is where that breaks

Two design points worth knowing, both load-bearing:

- **The solve redistributes pruned mass.** The right-hand side is `H w` over the *full* dense row,
  so survivors absorb what the pruned weights were contributing. That error compensation is most of
  what reconstruction buys, and there is a test that fails if it is dropped.
- **The refinement loop only accepts improvements.** Projecting onto a discrete grid is not
  guaranteed to reduce a quadratic objective, so an unguarded loop can finish worse than it
  started. Naive rounding is iterate zero and nothing replaces it unless it measurably wins — which
  is what makes "reconstruction improved the layer" safe to report.

### Known limitation, deliberately left

`solve_masked_rows` does one dense solve per output channel, roughly `out_features × |S|³`. Correct
and fine for validation and the small end of the sweep, but it **will not scale** to
`in_features = 8192` as written. Phase 6 needs either mask-grouping (rows sharing a keep-set solved
together) or the Hessian column sweep D2 defers. Chosen simple-and-obviously-correct first, and
flagged in the docstring rather than discovered later.

---

## 🔴 Characterised in Phase 6: the joint mechanism is weak, and W4 must carry the comparison

**Read [validity_threats.md](validity_threats.md#the-joint-mechanism-may-be-inert-at-moderate-precision)
before running any screening.** Investigated on **six real Pythia-160M layers** with the real
calibration set, not the synthetic layer the first pass used.

| Bits | Joint vs sequential mask differs | Max-abs scale moves on refit | Layer-objective joint gain |
| --- | --- | --- | --- |
| W8 | **0.46%** | 0.2% | −0.49% |
| W4 | **8.86%** | 0.2% | **+1.12%** |

- **The mask mechanism is live at W4, inert at W8.** My synthetic measurement said *zero* divergence
  at W4 and was wrong — real weights have heavier tails and real activations have outlier channels.
- **The scale mechanism is inert at every width.** Pruning removes the smallest weights so each row's
  maximum survives. Correction to an earlier claim: this is empirical, not provable — activation
  weighting *can* prune a row's largest weight, and does at 1.3% of channels in layer 11's MLP.

**Two candidate fixes were implemented, measured, and rejected**, both because they made the layer
objective worse:

| Configuration | W8 | W4 |
| --- | --- | --- |
| max-abs + magnitude (**default**) | −0.49% | **+1.12%** |
| + clipping scale search | −1.51% | −0.99% |
| + keep-benefit scoring | −11.83% | **−16.15%** |

The clipping search cuts *naive* quantisation error by 12.8% and does make scale re-estimation live
(70% of channels move, vs 0.2%) — but clipping saturates outliers, and a saturated weight cannot be
repaired by error compensation, so the post-reconstruction result degrades. Keep-benefit scoring is
worse and analytically so: it reduces to a monotone transform of activation-weighted magnitude plus a
near-constant, so it mostly reproduces the magnitude ranking and where it deviates it favours weights
that happen to sit near a grid point.

Both are retained as declared ablations behind `compression.reconstruction.scale_search` and
`.keep_benefit_saliency`, defaulting off.

**Consequences for the experimental design:**

- W4 carries the headline comparison; W8 is a **control**, where a near-zero gain is the expected
  result rather than a failure.
- Do **not** move to W2 to chase a larger effect — the mechanism is more active there, but selecting
  a precision because it yields a positive result is what §6.3 forbids.
- A criterion that respects error compensation needs the inverse-Hessian term, not a diagonal
  approximation. Research direction, not a config change.

Every number above is pinned by tests in `tests/test_layerwise.py`.

---

## ✅ Phase 6 — complete, and verified on a real model

All five arms run through one shared driver, from a YAML config to a run record. Verified end to end
on **real Pythia-160M** with the real WikiText calibration set:

| | |
| --- | --- |
| Target modules | 48 · **84,934,656** targeted parameters (the §2.6 scale x-axis) |
| Measured sparsity | **0.5000** against a 0.5 target |
| Effective bits/weight | **8.03** — real 8-bit plus scale overhead |
| Storage efficiency | **0.89** |
| Reconstruction | **+40.9%** mean objective improvement over naive rounding (27%–65%) |
| Dense perplexity | 34.77 |
| Joint 50% + W8 perplexity | 231.96 → retention **15.0%** |

Delivered this session:

- `compression/arms.py` — the five arms as thin declarations over the driver, plus
  `plan_from_config` so every arm derives its budget from one function. `prepare` refuses to run if
  the arm and the config's method disagree, because the budget comes from the config.
- `compression/packed.py` — `PackedLinear` holding int2/4/8 codes plus fp32 scales and **no mask
  buffer** (a byte-per-weight mask at 4 bits would be twice the size of the weights it describes).
  Scheme metadata is an int64 tensor rather than `get_extra_state`, so a packed model saves through
  the same `save_pretrained` path as the dense baseline — which is what makes their checkpoint sizes
  comparable.
- `SEQUENTIAL_QP` registered; all five methods now runnable. The older fine-tuning compressors stay
  importable but unregistered, so one cannot be run by accident.
- `ExperimentRunner` draws the calibration set once and injects it, so every arm at a budget sees
  byte-identical data.

### ✅ Resolved: the mask comparison group was costing 6.7x perplexity

The 15% retention was traced by isolating the arms. **Quantisation was never the problem** — W8 alone
is essentially lossless. All the damage was pruning, and specifically the *comparison group*.

Activation-weighted saliency multiplies every weight in an input column by that column's norm, so
ranked tensor-wide a low-energy column scores low **everywhere** and gets deleted entirely — removing
an input feature rather than thinning it. Per-output ranking makes each row keep its own top-k, so no
column can go wholesale. §3.10 permits either; **per-output is now the default.**

| Arm | Perplexity | Retention |
| --- | --- | --- |
| Dense | 34.77 | 100% |
| Quantisation only (W8) | 34.85 | **99.8%** |
| Pruning 50%, tensor-wide | 233.94 | 15% |
| Pruning 50%, **per-output** | **124.32** | 28% |
| Joint 50% + W8, **per-output** | **122.51** | 28% |

Two checks confirm the rest of the stack is sound: reconstruction buys a real 41% end to end
(mask-only 209.21 → 124.32), and calibration size is irrelevant (8× more data moved it under 3%).

Also fixed a bug found on the way: `plan_from_config` read `quantisation.bits` directly, so the
**pruning-only arm** was handed a bit width and `convert` packed it — silently quantising the one
FP32 arm, the arm that answers RQ4. Now derived from `effective_bits`. The test that should have
caught it was disabling quantisation in its fixture and masking it; that crutch is gone.

### 🟡 Remaining, and it shapes Phase 7 rather than blocking it

Retention at 50% on 160M is still ~28%, below published one-shot results. Partly protocol (64
sequences at a 256-token window, not the full test set at 2048) and partly a 160M model having little
redundancy to give. **Not chased to the ground.** It does not invalidate the design — the study
measures differences between arms at matched budgets, not absolute quality — but it does decide budget
selection:

| Joint budget | Perplexity | Retention |
| --- | --- | --- |
| **30% + W8** | **42.43** | **82%** |
| 40% + W8 | 60.46 | 58% |
| 50% + W8 | 122.51 | 28% |

§5.3 wants budgets "measurably but non-catastrophically" degraded. At 160M that is **30%**, not 50%.
S1 looks right; S2–S4 look likely catastrophic at this scale — itself scale-relevant, since larger
models should tolerate more. One seed, one model; screening decides.

Two smaller things noticed in the same runs:

- **Run IDs collide across arms.** `experiment.id` is `pilot` for every arm, so a compressed run
  overwrites the dense record it needs for retention. §5.6's convention
  (`<family>_<size>_<method>_<sparsity>_<bits>_<seed>`) fixes it; currently on the A-list.
- **`.gitignore` only covered named subdirectories.** A run writes to `outputs/<experiment_id>/`,
  which nothing matched — so the first real run left `outputs/pilot/` untracked *and unignored*, one
  `git add -A` from being committed. Closed with catch-all rules, and now enforced by tests that
  check the git index rather than the working tree (the old test asserted `outputs/` was empty, which
  fails on the one machine that is supposed to fill it).

---

## 🟡 Phase 7 — paused with NO valid results. Resume here.

### The state in one line

**Every screening number this project has produced has been retracted.** Two rounds of external review
found bugs that invalidated them, all the fixes are applied and pushed, and `outputs/metrics/` is
deliberately empty. Nothing is known about joint gain right now.

### 📋 Read [Protocol Amendment A1](protocol_amendment_a1.md) before running anything

Adopted **2026-07-30**, after the five protocol decisions below were put to an external reviewer and
settled. A1 is now the governing document for how the remaining experiments run, and it changes the
execution order.

**The one thing to know:** A1 declares **every result this project has produced so far exploratory** —
all of it is on the validation split, all of it predates the B-22/B-23 corrections, and none of it has
an uncertainty estimate. The frozen budgets are *not* reopened.

**The screening re-run is no longer the next step.** A1 §7 puts the **external correctness anchors
first**: if Wanda and SparseGPT disagree with us, the screening grid would have spent two hours
measuring a pipeline we do not trust.

### ✅ First anchor passed — the mask is confirmed correct

**[F-19](findings_log.md#f-19).** An independent Wanda implementation, sharing no code with ours,
produces **exactly our mask** — 0 differing positions across **48 modules and 84,934,656 weights**, on
matched norms. Column norms agree to **6.0e-07** despite ours accumulating the Gram in float32 and the
reference summing in float64.

Four positions in 85 million flip between float32 and float64 norms. Chased down rather than dismissed:
both disputed pairs **tie exactly in float64** and sit 2–3 ULPs apart in float32, so the choice is
arbitrary. Rebuilding our mask from the reference's norms drops the disagreement to zero, which proves
the selection logic is identical. The original INVESTIGATE verdict was **a bug in the anchor**
(B-25), not in the pipeline.

**This closes the mask question.**

### ✅ Second anchor passed — and it found something that matters more

**[F-20](findings_log.md#f-20).** The reconstruction sweep was checked against the **provable optimum**
of its own objective — for a fixed mask, `ŵ_S = (H_SS)⁻¹H_{S,:}w` is the exact minimiser, solved in
float64 over 96 rows spanning 4 module types and 3 depths. Both hard invariants hold: **0 rows below the
optimum** (impossible, so it would prove a defect) and **0 rows worse than naive masking** (the
accept-only-if-better guard works).

This is stronger than porting SparseGPT, because SparseGPT's contribution is *speed*, not a different
objective — a second approximation would only show two approximations agree.

### ✅ The slack question, measured — the sign is safe

**[F-21](findings_log.md#f-21).** Arm-dependent slack is **real**: solver efficiency is 0.6409 under the
sequential mask against **0.5631** under the joint mask, a 7.8 pp gap that varies in sign by depth.

But the decisive check came out clean: **across all 96 rows the solver never inverted which mask was
better.** Whenever the sweep preferred a mask, the exact optimum agreed. So the *direction* of a measured
joint gain is not a solver artefact — and the gap runs *against* joint, meaning the mechanism would be
understated rather than flattered. Notable, since every previous fault ran the other way.

**What stays open:** the effect on *magnitude*. It cannot be measured this way, because no exact optimum
exists for the quantised problem — the closed-form minimiser solves a continuous least-squares problem,
and a discrete grid makes it an integer program. Recorded as a limitation with wording ready for the
paper.

### ✅ Third anchor: our absolute numbers are credible

**[F-22](findings_log.md#f-22).** `IST-DASLab/sparsegpt`'s `fasterprune`, **unmodified**, matched on
model, revision, calibration draw, coverage and evaluation loader:

| Arm | Perplexity | Retention |
| --- | --- | --- |
| Dense | 36.9744 | 100% |
| **Ours**, per-output-row | **45.6644** | **80.97%** |
| Ours, *tensor-wide* (their group) | 59.9617 | 61.66% |
| Reference SparseGPT | 66.0355 | 55.99% |

**We came out 25 pp ahead, which was a reason for suspicion rather than celebration** — and the
prediction on record beforehand was the opposite direction, since [F-20](findings_log.md#f-20) had found
our sweep only captures 0.64 of the achievable gain.

Reading their source found the cause: `fasterprune` thresholds `tmp.flatten()` over a whole
`(out_features × 128)` block, i.e. a **tensor-wide comparison group**, where ours is per-output-row —
the difference [F-07](findings_log.md#f-07) already measured at 6.7× on this model. Running our own
pipeline with *their* group settles it: **77.3% of the 24.98 pp gap is the comparison group**, leaving a
5.67 pp residual (−9.20% relative perplexity, inside A1's 10% band).

**So ~81% retention at 30% pruning-only is plausible, not inflated** — which closes the question F-20
left open. It does *not* establish that our reconstruction specifically is competitive; `fasterprune`
picks its own mask internally and cannot be handed ours without editing it.

The reference checkout lives at `c:/Users/shehr/sajc_external/` — **outside the repo**, so third-party
code never enters the git index.

<details>
<summary>The original framing of this blocker, kept for the record</summary>

**The blocker as first stated:**

The same anchor measured that our sweep captures only **0.6409 of the achievable objective gain**,
consistently (0.57–0.72 across every module type and depth). So **~36% of the available improvement is
left unclaimed.**

That is not a defect — it is the documented trade that makes wide layers tractable. But set it against
a joint-versus-sequential difference of about **1 pp of retention** and the problem is plain:

> **If solver slack differs between the arms, the measured joint gain may be solver slack rather than
> the mask mechanism.** And it is *expected* to differ — the arms produce different masks, different
> masks give different `H_SS` conditioning, and conditioning determines how much a one-pass sweep
> recovers.

Settled by `scripts/run_arm_slack_anchor.py` — see above. Full reasoning and the residual risk in
[validity_threats.md](validity_threats.md#solver-slack-may-exceed-the-effect-being-measured).

</details>

**Still open regardless:** whether ~57% retention is *competitive*. F-20 shows the solver optimises what
it claims to; it says nothing about absolute quality against published work (A1 §5.5b2).

### The order to work in (A1 §7)

| | Step | State |
| --- | --- | --- |
| 1 | Write and commit Amendment A1 | ✅ done |
| 2 | Central implementation corrections — the nine fixes | ✅ done |
| 3a | **Wanda mask-agreement anchor** | ✅ **PASSES** — [F-19](findings_log.md#f-19) |
| 3b1 | **Exact-optimum reconstruction anchor** | ✅ **PASSES** — [F-20](findings_log.md#f-20) |
| 3b2 | Arm-dependent solver slack | ✅ **measured** — [F-21](findings_log.md#f-21); sign is safe, magnitude open |
| 3b3 | External SparseGPT comparison | ✅ **done** — [F-22](findings_log.md#f-22); our numbers are credible |
| **4** | **Create and fingerprint the calibration draws** | ⬜ **next** |
| 4 | Create and fingerprint the five fixed calibration draws | ⬜ |
| 5 | Re-run validation screening on the corrected implementation | ⬜ |
| 6 | Run both sequential orders (P→Q, Q→P) on validation | ⬜ |
| 7 | Freeze the winning order per (model, budget) | ⬜ |
| 8 | Run the reduced S6 mechanistic control (12 runs) | ⬜ |
| 9 | Freeze the entire confirmatory configuration | ⬜ |
| 10 | Run test evaluation **once**, with no further tuning | ⬜ |

Step 5 is still the command below, and it still writes to the exploratory (validation) configuration:

```bash
python scripts/run_scale_sweep.py --config configs/experiments/screening.yaml     # 13 cells, ~2 h
python scripts/summarise_screening.py --model pythia-160m \
    --budgets s1_30_w8,s2_50_w8,s3_50_w4,s4_70_w4,s5_30_w4,s6_40_w8
```

**Confirmatory cost, now decided.** Replicates are **R=8 at 160M and 410M, R=5 at 1B** — roughly
**38 hours**, against 31 for flat R=5 and 50 for flat R=8. The reason for the split is that the
statistical constraint is a cliff, not a slope: at R=5 the *best possible* result (every replicate
agreeing) is p = 0.0625, so no significance claim exists at any effect size, while R=8 reaches 0.008.
The extra hours are spent only where they buy that transition, on the two models carrying most of the
scale-trend evidence. **R must be reported per cell** — A1 §5.1 makes that a hard requirement.

### Why the numbers were retracted, in order

| Figure | Budget | Cause of retraction |
| --- | --- | --- |
| **−4.55 pp** | 30% + W4 | Joint outer loop had no acceptance test, so it discarded better solutions it had already found |
| **+1.03 pp** | 30% + W4 | The arms minimised **different objectives** — sequential targeted its own intermediate, joint targeted dense |
| *unknown* | — | Not yet measured on corrected code |

**Every bug pointed the same way: flattering the joint arm.** That belongs in the paper's limitations
regardless of where the number lands, and it is a reason to hold the next figure loosely too. Full
detail in [findings_log.md](findings_log.md) F-16 and F-17.

### What the budgets were, before retraction

The frozen pair in [protocol_freeze.md](protocol_freeze.md) is **moderate 30% + W8** and
**aggressive 30% + W4**. Both survived the first re-run, and the *sequential* arm's retention has been
stable across every version of the code (≈80% and ≈57%), so the budget choice is unlikely to move. It
is the joint-versus-sequential difference that has been unstable, not the budgets.

### Nine fixes applied since the last valid run

All pushed. Suite at **804 passing**, lint and format clean.

| Fix | What it was |
| --- | --- |
| Common reconstruction objective | Arms minimised three different objectives; the direction inflated joint gain |
| Canonicalise before measuring | Joint accepted a proposal on one weight and packed a different one |
| Joint incumbent guard | Outer loop accepted every proposal, including worse ones |
| Dependency-group recapture | Activations captured once per block, so reconstruction was blockwise, not layerwise |
| Packing reuses solver codes | Conversion refit the grid and re-quantised; `verify_packing` existed and was never called |
| Mask sparsity as the budget | `measured_sparsity` conflated pruned weights with rounding zeros (~1.8 pp at W4) |
| Latency gate | Quantised arms were benchmarked through a dequantising path, so timings measured unpacking |
| Independent reload | Manifest plus `load_packed_model`; the checkpoint could not previously be loaded on its own |
| `scale_trend` + `METHOD_VERSION` | Analysis entry point was a stub; nothing detected records produced by different code |

### ✅ The five protocol decisions — all settled by Amendment A1

Put to an external reviewer and settled on **2026-07-30**. Full reasoning and the exact designs are in
[protocol_amendment_a1.md](protocol_amendment_a1.md); summarised here.

**Independently corroborated.** A partner working separately reached the *same* conclusions on items 1,
2 and 5 — calibration replicates in place of run seeds, validation-for-screening with test reserved for
final reporting, and Wanda/SparseGPT as matched sanity references that do not replace the primary
comparison. Two reviewers converging on the same three corrections without conferring is a stronger
signal than either alone, and it is why those three are treated as settled rather than provisional.

| # | Decision | Settled as | Status |
| --- | --- | --- | --- |
| 1 | Run seeds | **Withdrawn.** Paired calibration replicates, **R=8 at 160M/410M, R=5 at 1B** (decided 2026-07-30); §6.3 reworded; R reported per cell | necessary correction |
| 2 | Evaluation split | **Validation for selection, test for confirmation.** Two configs, everything else held identical | necessary correction |
| 3 | Sequential ordering | **Both orders run; winner selected on validation**, frozen per cell before test | enforcement, not a change |
| 4 | S6 (40% + W8) | **Secondary control only.** 2 models × 2 arms × 3 draws = 12 runs; 1B gets diagnostics only | optional addition |
| 5 | External anchor | **Wanda mask agreement first, then SparseGPT pruning-only.** Alarm thresholds, not acceptance tests | validation, runs first |

Two points from A1 worth carrying in your head:

- **The §6.3 amendment is a *loosening* of a pre-registered rule**, and the paper must say so. The
  original rule was stricter on its face but unmeasurable in fact, because its binding clause evaluated
  to zero. Replacing an unmeasurable criterion with a weaker measurable one is a correction *and* a
  reduction in pre-registered strength.
- **S6 is the weakest of the five** on the "would this have been justified before seeing results" test —
  it became interesting because we saw 40% + W8 land near 30% + W4, which is a result. It is defensible
  because it tests a mechanism rather than a headline, and A1 records its provenance rather than
  presenting it as pre-planned.

[review_brief.md](review_brief.md) is written for an outside reader and states these as the open
questions they were before A1.

---

## Then: Phase 8 — the full scale sweep

The driver and all five arms are done, tested, and wired into `ExperimentRunner`.

Done this session:

- **Phase 2 adapters** — `select_compressible_modules` (adapter-gated *and* substring-gated, raising
  `EmptySelectionError` rather than returning the dense model), `get_decoder_blocks`,
  `get_weight_tensors`, `get_linear_modules`, `describe_architecture`,
  `count_targeted_parameters` (A6).
- **A2 config** — a `reconstruction` section (`solver`, `local_steps`, `joint_iterations`,
  `damping`, `block_size`, `activation_order`), with `local_steps` as the fairness unit.
- **`CompressionMethod.SEQUENTIAL_QP`** and the `ReconstructionSolver` / `SaliencyRule` enums (A3).
  `SEQUENTIAL` remains P→Q, the primary order, so no existing config changed meaning.
- **`compression/layerwise.py`** — depth-order block iteration, activations captured *through the
  already-compressed prefix*, all five arms as call-order variations on one solver,
  `assert_matched_plans` enforcing §3.11.

All three items that were outstanding here are now closed:

1. **Registered** — `COMPRESSOR_REGISTRY` maps all five methods to the layerwise arms in
   `compression/arms.py`. The older `Pruner` / `Quantiser` / `SequentialCompressor` /
   `JointCompressor` classes are *deliberately* left unregistered: they implement the superseded
   full-model fine-tuning design, and their stage methods still raise. Importable, unrunnable.
2. **`convert`** — real int2/4/8 packing into `PackedLinear`, reusing the solver's own codes rather
   than re-quantising, with `verify_packing` on the way out and a manifest so the artefact reloads
   independently. `is_converted` and `storage_efficiency` now mean something.
3. **The scale rule** — decided as D3 and recorded in
   [protocol_freeze.md](protocol_freeze.md#the-three-decisions-that-were-open).

What gates Phase 8 is therefore not code. It is the screening re-run and the five protocol decisions
above. Full detail and exit tests for every phase:
[implementation_plan.md](implementation_plan.md#phases).

---

## 🔴 Still open

Reduced to the items that genuinely cannot be settled yet. Tracked in
[protocol_freeze.md](protocol_freeze.md#still-open).

- **Calibration indices, token count, sequence length** — frozen by config once `prepare_data.py`
  has run for real. The WikiText load path has still never been executed anywhere.
- **The two final budgets** — output of Phase 7 screening on 160M/410M. §5.3 requires them frozen
  before 1B.
- **W4 latency via `torchao`** — deferred to Phase 6. Would lift D1's "no W4 latency row"
  limitation if one 4-bit CPU path can serve both arms. Needs measuring, not assuming.
- **1.4B go/no-go** — §5.2 needs peak VRAM under ~85% of 6.0 GiB, a **5.1 GiB ceiling**. Tight.
  Decide after Phase 5 profiling.

Settled since the last revision, no longer open: the backend (**`onednn`** — see below), the
solver, the mask scoring rule, the Pythia variant (**standard**), lm-eval-harness (**yes, pinned**),
the practical-importance threshold (**≥ 1.0 pp retention, consistent in sign across all three
confirmatory seeds, exceeding the seed spread**), Smart App Control, the power profile (**High
performance**, no downclocking, never sleeps), the benchmark thread count (**4**, inside the P-core
budget), and all five **model revision SHAs**.

### One correction found by probing rather than reading docs

D1 originally froze the latency backend as `x86`, the name every PyTorch tutorial uses. On the
pinned torch 2.13.0+cu126 `supported_engines` is **`['onednn']` only** — `x86`, `fbgemm` and
`qnnpack` all raise "quantized engine is not supported". The shipped configs said `x86`, so
conversion would have failed *after* the compression compute was spent. Corrected everywhere, and a
`requires_torch` test now asserts the shipped backend against the installed torch so a future
upgrade that renames engines fails a test instead of a run.

Also recorded: `torch.ao.quantization` and the `qint8`/`quint8`/`qint32` dtypes are both deprecated
in favour of `torchao`. Neither blocks the study — §2.7 pins the environment for its duration — but
**the torch version must not be upgraded mid-study**.

---

## Deferred reconciliation (plan vs scaffold)

Known gaps, not yet implemented. Roughly in priority order.

| # | Gap | Size |
| --- | --- | --- |
| A1 | Layerwise reconstruction: `activations.py`, `reconstruct.py`, `layerwise.py` | large |
| A2 | Config: `reconstruction` section; `local_steps` replaces `max_steps` as the fairness unit | medium |
| A3 | Reverse sequential **Q→P**, and joint gain vs **best-of** {P→Q, Q→P} (plan §3.6, §6.1) | medium |
| A4 | Downstream tasks — HellaSwag, PIQA, ARC-Easy are **required** (§4.3) | medium |
| A5 | Prefill vs decode timed **separately**, at 128 and 512 prompt lengths; IQR; model-order rotation (§4.7) | small–medium |
| A6 | **Targeted non-embedding parameter count** as the scale x-axis (§2.6) — currently uses total | small |
| A7 | Seed policy: 1 for screening/first pass, 3 confirmatory only (§5.5). Current sweep runs 3 everywhere → ~40% wasted compute | small |
| A8 | Budget **screening** stage S1–S4 on 160M, freeze 2 budgets before 1B (§5.3) | small |
| A9 | Record fields: per-layer reconstruction loss, compression time, peak GPU memory, effective bits, tokenizer revision | small |

A3 and D3 interact: `CompressionMethod` gains `SEQUENTIAL_QP`, and joint gain must be computed
against best-of-sequential with the winning order recorded (§6.1).

---

## Environment notes

Full record in [protocol_freeze.md](protocol_freeze.md#environment). Summary:

- **Omen is the only machine that runs code.** `outputs/`, `results/`, and `data/` are git-ignored
  and exist only there.
- CPU i7-13620H (10P/16L) · 13.7 GiB RAM · **RTX 4050 Laptop, 6.0 GiB VRAM**, sm_89 · driver 592.82
- Python 3.11.9 · torch **2.13.0+cu126** · transformers 5.14.1 · datasets 5.0.0 · numpy 2.4.6
- Installed from scratch this session: the Omen had **no Python at all** (`python` was the Microsoft
  Store stub, and there was no uv or conda).
- **13.7 GiB system RAM is worth watching.** CPU-only evaluation of Pythia-1.4B in FP32 is ~5.6 GiB
  of weights before activations, and the benchmark is CPU-bound by design.
- The real **WikiText load path has still never been executed** — every data test stubs the corpus.
  Treat the first `prepare_data.py` run as its first real test, and note `datasets` is currently
  blocked independently of torch.

---

## Immediate checklist for the Omen

- [x] `git clone`
- [x] Python 3.11 + venv; CUDA torch (`cu126`); `pip install -e . -r requirements-dev.txt`
- [x] `pytest` → 502 passing (before the SAC blocks landed)
- [x] Record GPU and VRAM for §5.2 — **RTX 4050, 6.0 GiB**
- [x] Settle the three open decisions → [protocol_freeze.md](protocol_freeze.md)
- [x] Resolve the Smart App Control blocker — **off, rebooted, verified**
- [x] Re-run `pytest` → **511 passing in ~32 s**, stably green
- [x] Power profile → **High performance**; thread count pinned at **4**
- [x] Pin model revision SHAs in all five model configs
- [x] Probe the real quantisation backend → **`onednn`**, not `x86`
- [x] **Phase 5 — single-layer compression primitives**, all exit criteria asserted
- [x] `python scripts/download_models.py --models pythia-160m` — and 410M
- [x] `python scripts/prepare_data.py` — the WikiText path has now run for real
- [x] `python scripts/run_dense_baseline.py` — first real record, on a real model
- [x] **Phase 6 — the five arms through one shared layerwise driver**, verified end to end
- [x] Two rounds of external code review applied — nine fixes, suite at 804
- [x] Settle the five protocol decisions → **[Amendment A1](protocol_amendment_a1.md)**, adopted
- [ ] **External anchors: Wanda mask agreement, then SparseGPT pruning-only** ← **next** (A1 §7 step 3)
- [ ] Five fixed calibration draws, fingerprinted
- [ ] Re-run the 160M validation screening (~2 h, 13 cells)
- [ ] Both sequential orders on validation; freeze the winner per cell
- [ ] Reduced S6 control (12 runs)
- [ ] Freeze the confirmatory config, then test evaluation **once** — no tuning after that
