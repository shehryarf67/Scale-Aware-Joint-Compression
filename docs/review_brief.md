# External review brief

**A self-contained account of what this study does, what has been decided, what has been measured, and
which of those decisions most deserve challenging.**

Written to be read by someone with **no access to the repository**. Everything needed to judge the
work is here. If you are reviewing this: the last section, [Questions we want
challenged](#7-questions-we-most-want-challenged), is the part we care about most.

Status as of **2026-07-29**. Phases 0, 5, 6 complete; Phase 7 (budget screening) complete on
Pythia-160M with confirmation on Pythia-410M in progress. 759 automated tests passing.

---

## 1. What the study asks

> **Does model scale change whether *joint* pruning-and-quantisation beats *sequential*
> pruning-then-quantisation?**

Two ways to compress a language model:

- **Pruning** — set a fraction of weights to zero.
- **Quantisation** — store the remaining weights at reduced precision (8 or 4 bits instead of 32).

Standard practice applies them **in sequence**. The alternative is to optimise for both **at once**, so
the pruning decision is aware of the quantisation grid. Published evidence for joint methods is almost
always reported at a single model size, which leaves the practical question open: if you have a model
of a given size and a fixed compute budget, is the joint pipeline worth building?

**Model suite:** Pythia 160M → 410M → 1B (1.4B optional), plus Qwen2.5-0.5B as external validation.
Pythia because the suite holds data order, tokeniser and recipe fixed across sizes, which makes scale a
clean variable.

**Primary outcome:** *joint gain* = joint perplexity retention − sequential perplexity retention, at a
matched compression budget. Retention is measured against each model's **own** dense baseline, because
absolute perplexity falls with scale for reasons unrelated to compression.

**Hardware constraint that shapes everything:** one laptop. RTX 4050 with **6 GB VRAM**, 13.7 GB system
RAM. All deployment measurements (latency, memory, checkpoint size) are **CPU-only** by design; the GPU
is used for compression only.

---

## 2. The method, as actually implemented

Full-model fine-tuning will not fit at 1B on 6 GB, so the method is **layerwise post-training
reconstruction**. For each targeted linear layer, calibration activations `X` are captured and the
compressed weights chosen to minimise the layer's *output* error:

```
minimise  ‖ X·Wᵀ − X·(M ∘ Q_b(W))ᵀ ‖²_F
   over   masks M with sparsity s, and quantised weights Q_b at bit width b
```

Because `nn.Linear` computes `Y = X·Wᵀ`, this expands to a sum over output channels of a quadratic form
in the Gram matrix `H = XᵀX`. The objective therefore needs only `H`, never the activations themselves
— which is what keeps memory at one `(in_features, in_features)` buffer per layer regardless of how
much calibration data is used.

### Components

| Component | Choice |
| --- | --- |
| **Saliency** (what to prune) | Activation-weighted magnitude, `S_ij = \|W_ij\| · ‖X_j‖₂` (the Wanda criterion) |
| **Comparison group** | **Per output row** — each row keeps its own top-k. See [F-07](#f-07) |
| **Quantiser** | Symmetric, weight-only, per-output-channel, scale = `max\|W_row\| / (2^(b−1)−1)` |
| **Solver** | Error-compensated column sweep over a Cholesky factor of `H⁻¹` (GPTQ/SparseGPT-style), with activation-order column permutation |
| **Layer order** | Depth order, activations captured **through the already-compressed prefix** |
| **Excluded from compression** | Token embeddings, LM head, LayerNorms, all biases |

### The five arms

```
pruning-only     mask → reconstruct                                   (weights stay FP32)
quant-only       quantise → reconstruct
sequential P→Q   mask → reconstruct → quantise → reconstruct          (2 solver calls)
sequential Q→P   quantise → reconstruct → mask → reconstruct          (reverse-order ablation)
joint            repeat K=2 times:
                   fake-quantise the survivors
                   rescore saliency UNDER the quantised weights
                   update the mask at target sparsity
                   re-estimate quantisation scales on the survivors
                   reconstruct
                 freeze M and Q                                        (2 solver calls)
```

Every arm calls the **same** solver and differs only in call order. That is deliberate: it makes "is
this actually joint?" checkable in code rather than by inspection, and a regression test fails if the
joint arm ever degenerates into "prune fully, then plain PTQ."

### Excluding embeddings, and why it matters

Embeddings are a large fraction of a 160M model and a small fraction of a 1.4B one. Including them
would make the *effective* compression budget vary with scale, so a scale trend in joint gain could not
be separated from a scale trend in how much of each model was being compressed. Consequence: the
study's scale x-axis is **targeted non-embedding parameter count**, not total parameters, and reported
compression ratios are over compressible parameters only.

---

## 3. Findings, with numbers

Two evaluation windows exist and **must never be mixed** — retention is always against a dense run in
the same window.

| Window | Sequences × tokens | Dense perplexity (Pythia-160M) |
| --- | --- | --- |
| Pilot | 64 × 256 | 34.77 |
| Screening | 493 × 512 (whole WikiText-2 validation split) | 36.97 |

Neither is the protocol published papers use (full test set at 2048-token context), so **none of these
numbers is directly comparable with the literature.** They are internally comparable, which is what the
arm comparison needs.

### F-1 · Quantisation is nearly free; pruning does all the damage

Pilot window, Pythia-160M, 50% sparsity:

| Arm | Perplexity | Retention |
| --- | --- | --- |
| Dense | 34.77 | 100% |
| **8-bit only, no pruning** | **34.85** | **99.8%** |
| **50% pruning only, FP32** | **233.94** | 15% |

Weight-only INT8 costs essentially nothing at this scale. All degradation comes from sparsity.

### F-07 · The mask comparison group was worth 6.7× perplexity {#f-07}

The largest single quality finding. Our first compressed run retained 15%.

Activation-weighted saliency multiplies every weight in an input column by that column's activation
norm. Ranked across the **whole tensor**, a low-energy column scores low *everywhere* and is pruned out
entirely — deleting an input feature rather than thinning it. Ranking **within each output row** makes
every row keep its own top-k, so no column can be removed wholesale.

| Configuration | Perplexity |
| --- | --- |
| Pruning 50%, tensor-wide ranking | 233.94 |
| Pruning 50%, **per-output ranking** | **124.32** |
| Joint 50% + W8, tensor-wide | 231.96 |
| Joint 50% + W8, **per-output** | **122.51** |

Two controls confirm the rest of the stack is sound rather than merely less broken:

- **Reconstruction does real work.** Mask only, no reconstruction: 209.21. With the sweep: 124.32 — a
  41% improvement end to end.
- **Calibration size is nearly irrelevant.** 8× more calibration data (16 → 128 sequences) moved
  perplexity by under 3%.

### F-05 · The joint mechanism is precision-dependent, and near-inert at 8 bits {#f-05}

Measured on **six real Pythia-160M layers** (layers 0, 5, 11; attention and MLP) with the real
calibration set, at 50% sparsity.

| Bits | Joint vs sequential mask differs | Channels whose max-abs scale moves when refitted on survivors |
| --- | --- | --- |
| W8 | **0.46%** | 0.2% |
| W4 | **8.86%** | 0.2% |
| W3 | 15.42% | 0.2% |
| W2 | 45.54% | 0.2% |

**Mechanism 1 (mask scored under quantisation) is live at W4 and essentially inert at W8.** At 8 bits
quantisation error is small enough that the saliency ranking survives intact, so the joint arm picks
almost the same mask as the sequential arm — and any difference between them cannot be coming from the
mechanism under study.

**Mechanism 2 (scales re-estimated after mask changes) is inert at every width.** A symmetric
per-channel scale is `max|W_row|`, and pruning removes the *smallest* weights, so each row's maximum
almost always survives.

Two honesty notes:

- An earlier version of this measurement used a **synthetic** layer and reported *zero* mask divergence
  at W4. Real weights have heavier tails and real activations have outlier channels. The synthetic
  layer was too well-behaved. All effect sizes quoted here are from real layers.
- We first claimed the scale invariance was *provable*. It is not — because the saliency is
  activation-*weighted*, a row's largest weight **can** be pruned if it sits on a low-energy input
  column, and it does at 1.3% of channels in layer 11's MLP. The claim is empirical.

### F-06 · Two principled fixes for F-05, both measured, both rejected

Layer-objective joint gain across the same six real layers. Positive = joint reconstructs better.

| Configuration | W8 | W4 |
| --- | --- | --- |
| max-abs scales + activation-weighted magnitude (**kept**) | −0.49% | **+1.12%** |
| + error-minimising clipping scale search | −1.51% | −0.99% |
| + quantisation-aware keep-benefit scoring | −11.83% | **−16.15%** |

**Clipping scale search** does everything it promises in isolation. It cuts *naive* quantisation error
by 12.8% at W4 (optimal clip ratio α = 0.81; α = 1.00 at W8, i.e. no clipping wanted), and it makes
scale re-estimation genuinely mask-dependent — **70% of channels move their grid on refit, against
0.2% for max-abs.** But the layer gets *worse after reconstruction*. Our reading: clipping saturates
outliers, and a saturated weight cannot be repaired by error compensation. It optimises the wrong
objective.

**Keep-benefit scoring** `B_ij = ‖X_j‖²[W_ij² − (W_ij − Q(W_ij))²]` is worse, and we believe
analytically so. For round-to-nearest symmetric quantisation the score is **bounded below by zero**: if
`|W| < s/2` then `Q(W) = 0`, both error terms equal `W²`, and `B = 0` exactly; otherwise
`|W − Q(W)| ≤ s/2 ≤ |W|`. And above the step size `(W − Q(W))²` is nearly independent of `W`, leaving
`B ≈ ‖X_j‖²·W_ij²` minus a near-constant — a **monotone transform of activation-weighted magnitude**.
So it largely reproduces the ranking it was meant to improve, and where it deviates it favours weights
that happen to sit near a grid point, which says nothing about importance.

Both are retained in the code as switchable ablations, defaulting off.

### F-12 · The arms ran on unequal optimisation budgets {#f-12}

**Found 2026-07-29. It invalidated the magnitude of every joint-gain number produced before that date.**

The joint arm received **192** local steps per run; the sequential arm **96**. The joint arm calls the
solver once per outer iteration and the default was `K = 4`; the sequential pipeline calls it twice.
4 × 48 layers against 2 × 48.

The root cause is worse than the arithmetic: an assertion written specifically to catch this existed,
was exported, and was covered by tests — and **was never called during a real run**. A second
contributing split: two classes each carried their own default for `K` and they had diverged, so
changing one left the other (the one that actually runs) untouched.

**What it does not invalidate:** the direction. Joint had *more* compute and still did not win at any
budget, so equalising can only move results against joint.

Fixed by: a single source of truth for solver calls per arm; a **pre-flight** check wired into the
sweep runner so an unfair grid fails before hours of compute; `K` default 4 → 2 in both places, pinned
equal by a test; and the summary tool now reads recorded budgets and marks a row's gain unusable when
the arms differ. Ten new tests.

Re-measured at matched budgets:

| Budget | Joint gain, K=4 (unfair) | Joint gain, K=2 (matched) |
| --- | --- | --- |
| 30% + W8 | +0.06 pp | **+0.12 pp** |
| 30% + W4 | −5.46 pp | **−4.55 pp** |
| 40% + W8 | −0.06 pp | **+0.03 pp** |

### F-13 · Budget screening on Pythia-160M {#f-13}

Screening window (493 × 512), dense 36.97, one seed, matched solver budgets for the eligible rows.
Thresholds applied: "measurably degraded" below 99% retention, "catastrophic" below 50%. Neither bound
is specified as a number in the plan; both are stated in the tool's output rather than hidden.

| Budget | Sparsity | Bits | Sequential | Joint | Seq ret. | Joint ret. | Joint gain | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **S1** | 30% | W8 | 45.97 | 45.90 | **80.4%** | **80.6%** | +0.12 pp | **ELIGIBLE** |
| S2 | 50% | W8 | 161.46 | 163.85 | 22.9% | 22.6% | *unusable* | catastrophic |
| S3 | 50% | W4 | 250.25 | 256.74 | 14.8% | 14.4% | *unusable* | catastrophic |
| S4 | 70% | W4 | 4663.88 | 4802.72 | 0.8% | 0.8% | *unusable* | catastrophic |
| **S5** | 30% | W4 | 66.03 | 71.87 | **56.0%** | 51.4% | **−4.55 pp** | **ELIGIBLE** |
| **S6** | 40% | W8 | 67.10 | 67.06 | **55.1%** | 55.1% | +0.03 pp | **ELIGIBLE** |

**The plan's own proposed budgets do not work at this scale.** S1–S4 was the grid the research plan
specified, and only S1 survived. The moderate/aggressive pair previously written into the sweep configs
was 50% + W8 and 70% + W4 — **both catastrophic on the smallest model.** Since the budget is a
controlled variable across scales, the smallest model sets the ceiling for all three. S5 and S6 were
added afterwards to find a usable second budget.

**S5 is the only eligible budget where the arms measurably differ, and joint is the worse one** by
4.55 percentage points of retention. Every other eligible budget is a tie to within 0.12 pp.

**A side observation:** at S5, cutting the joint arm from four alternations to two *improved* it
(73.17 → 71.87). More alternation making the result worse is the opposite of what a converging
alternating optimiser should do.

---

## 4. Decisions taken, and what settled each

| Decision | Settled as | Settled by |
| --- | --- | --- |
| **CPU quantisation backend** | PyTorch native INT8, engine `onednn`. W4 contributes quality and size only, never latency | A packed 4-bit CPU linear would measure the dequantisation kernel, not the compression. The plan already separates size from latency and treats a null latency result as valid |
| **Reconstruction solver** | Error-compensated column sweep; damped ALS retained as a reference implementation | Feasibility. The per-output-channel exact solve needs ~1 PB at `in_features = 8192`. The sweep does the study's widest layer in **4.1 s** at 2.5 GB peak GPU |
| **Mask scoring** | Activation-weighted magnitude, computed on the **quantised** weights in the joint arm | The plan defines "joint" as the mask being evaluated under quantised weights, and lists the alternative as explicitly *not* qualifying |
| **Mask comparison group** | **Per output row** | Measurement: 6.7× perplexity ([F-07](#f-07)) |
| **Scale rule** | max-abs; clipping search rejected | Measurement ([F-06](#f-06)) |
| **Joint iterations K** | **2**, matching the sequential pipeline's two solver calls | Fairness ([F-12](#f-12)) |
| **Compression budgets** | **Moderate 30% + W8; aggressive 30% + W4** | Screening ([F-13](#f-13)) |
| **Practical-importance rule** | ≥ 1.0 pp retention, consistent in sign across all three confirmatory seeds, exceeding the seed spread | Set **before** any compressed result existed, as the plan requires |

### The budget freeze, in detail — the decision we are least certain about

Three budgets were eligible: S1 (30% + W8, 80.4%), S5 (30% + W4, 56.0%), S6 (40% + W8, 55.1%). No pair
gives everything:

- **S5 + S6** — rejected. Both sit at ~55% retention, so there is no severity contrast.
- **S1 + S6** — rejected. Would vary *sparsity* and keep both budgets on the benchmarkable INT8 path,
  which is attractive for the deployment half of the study. But it contains **no 4-bit condition at
  all**, and [F-05](#f-05) measured the mechanism as near-inert at 8 bits (0.46% mask divergence). A
  sweep with two 8-bit budgets would be structurally incapable of detecting the effect the study exists
  to measure, and would produce a confident null that was an artefact of the design.
- **S1 + S5** — **chosen.** Varies precision, keeps the one live-mechanism regime, separates 80.4% from
  56.0% retention.

**The cost, stated plainly: both budgets prune 30%, so sparsity never varies across the frozen pair.**
The sparsity-versus-latency question therefore cannot be answered from these budgets. It has to come
from benchmark-only runs of the pruning-only arm at several sparsities — cheap, because that arm stays
FP32 and a latency measurement does not need the full quality evaluation, but it must be actively
scheduled because it is not part of any budget cell.

---

## 5. Bugs found that produced believable numbers rather than errors

Recorded because several were caught only by deliberately looking, and two were **masked by tests that
should have caught them**.

| Bug | What it would have done |
| --- | --- |
| Tensor-wide mask comparison group | 6.7× perplexity ([F-07](#f-07)) |
| Joint arm on 2× the solver budget; guard never invoked | Every joint-gain number non-attributable ([F-12](#f-12)) |
| Two classes with separate defaults for `K` | Changing one silently left the other — the one that runs |
| Pruning-only arm handed a bit width; conversion packed it | Silently quantised the one FP32 arm, the arm that answers the latency question. **Masked by a test whose fixture disabled quantisation** |
| Synthetic layer used for an effect size | Reported *zero* mask divergence at W4 where real layers show 8.86%. **Masked by the test layer being too well-behaved** |
| Size metric compared against an unachievable target | Every compressed artefact read as "2.4× larger than its budget allows" when it was exactly as small as the method permits — because embeddings are excluded by design |
| Evaluation could exceed the calibration-reserved prefix on a shared split | Model evaluated on text the calibration set was drawn from. Inflates every arm equally, so no comparison looks wrong |
| Run outputs neither tracked nor ignored by git | One `git add -A` from committing run artefacts |
| Four identical dense cells planned per four-budget grid | Wasted compute and near-duplicate records |
| Sweep cells inherited the base config's model revision | Every cell pinned to the *first* model's commit — wrong weights, silently, if the hash exists in both repos |

---

## 6. What we explicitly do **not** claim

- **Nothing about absolute quality relative to published results.** Neither evaluation window matches
  the literature's protocol.
- **Nothing about scale.** Every number above is Pythia-160M. The 410M confirmation is in progress. One
  model is not a trend.
- **Nothing with uncertainty attached.** Every number is a **single seed**. The plan requires three
  confirmatory seeds for the central comparison, and none of these are confirmatory runs.
- **That joint beats sequential, or does not.** No three-seed matched comparison at a frozen budget has
  been run yet.
- **Any latency claim.** No benchmark here was collected under the full protocol (20–30 repetitions,
  prefill/decode split, model-order rotation).

---

## 7. Questions we most want challenged

Ordered by how much a wrong answer would cost us.

### 7.1 Is joint being *worse* than sequential at 4 bits plausible, or does it indicate a bug?

At 30% + W4 with matched solver budgets, joint retention is **51.4%** against sequential's **56.0%** —
4.55 pp worse. At 8 bits the two are tied to within 0.12 pp. This is the single result we are least
comfortable with.

Arguments that it is real: 4 bits is the only regime where the mask actually changes, so it is the only
place a difference of any sign could appear. And the joint arm re-derives its mask from weights that
have already been perturbed, which could plausibly be a worse basis for the decision than the clean
dense weights the sequential arm uses.

Arguments that it is suspicious: an alternating optimiser on a shared objective should not do worse than
a single pass of the same solver, and **cutting the alternations from four to two improved it** (73.17 →
71.86). That looks like divergence rather than convergence.

**What we would like challenged:** is there a known failure mode in alternating mask/quantiser schemes
where additional alternations degrade the result? Should the reconstruction target be re-anchored each
iteration (it is currently always the original dense output, deliberately)? Is our accept-only-if-better
guard operating at the wrong granularity — per layer rather than per iteration?

### 7.2 Is the frozen budget pair defensible?

Both budgets prune 30% and differ only in precision (8-bit vs 4-bit). We chose that over varying
sparsity because 4 bits is the only regime where the joint mechanism is measurably live.

**What we would like challenged:** is "the mechanism is inert at 8 bits, so a study with only 8-bit
budgets cannot detect the effect" the right way to reason about budget selection? Or does insisting on
4-bit amount to choosing the condition most likely to produce a difference, which would be a form of
selection bias? We think not — the choice was made on a *mechanism* measurement rather than on outcome
measurements — but it is close enough to the line that we want an outside view.

### 7.3 Is ~56% retention at 30% sparsity + 4-bit plausible for a 160M model?

Published one-shot results on comparable-scale models degrade considerably less. We have ruled out:
calibration size (8× more moved it under 3%), the comparison group (fixed, worth 6.7×), and
reconstruction being broken (it buys a real 41% end to end).

Remaining hypotheses we have **not** tested: the evaluation window (493 × 512 rather than a full test
set at 2048-token context) inflating perplexity; Pythia-160M genuinely having little redundancy; or our
sweep implementation being weaker than a reference GPTQ/SparseGPT.

**What we would like challenged:** what retention *should* a 160M model show at 30% unstructured
sparsity plus 4-bit weight-only quantisation with layerwise reconstruction? If the answer is "much
better than 56%", we have a remaining implementation problem and should find it before Phase 8.

### 7.4 Is matching *solver calls* the right fairness unit?

The plan requires "equal total local optimisation steps and approximately equal objective evaluations."
With a single-pass deterministic sweep, `local_steps` controls nothing, so we matched the number of
solver calls: 2 for both arms.

The alternative the plan permits is padding the sequential arm to the joint arm's larger budget. We
rejected it because, with a deterministic sweep, extra passes on an already-converged sequential result
are near-idempotent — "matched on paper, not in compute."

**What we would like challenged:** is that reasoning right? Does K=2 give the joint arm enough
alternation to be a fair representative of "joint compression", or have we handicapped it to satisfy a
fairness constraint?

### 7.5 Is our analysis of why keep-benefit scoring fails correct?

We claim `B_ij = ‖X_j‖²[W_ij² − (W_ij − Q(W_ij))²]` is non-negative and reduces to a monotone transform
of activation-weighted magnitude minus a near-constant, and therefore mostly reproduces the ranking it
was meant to improve. Measured effect: −16.15% layer-objective joint gain at W4.

**What we would like challenged:** is the algebra right, and is there a formulation of the same idea
that avoids the collapse? A criterion consistent with error compensation would presumably need the
inverse-Hessian term rather than the diagonal approximation we used.

### 7.6 How should JSQ relate to this study?

JSQ (Guo et al., ICML 2024, *Compressing Large Language Models by Joint Sparsification and
Quantization*) states its motivation as: *"sparsification tends to preserve outliers that are harmful to
quantization."* That is **exactly** our [F-05](#f-05) finding — pruning removes the smallest weights so
each row's maximum survives, which is why scale re-estimation is inert.

Its two fixes are a sparsity metric bridging the two techniques, and a **search-based activation editor**
removing useless activation outliers. The second attacks the problem at a point we have not tried: our
failed clipping search edited *weights*; JSQ edits *activations*.

Two caveats we see. JSQ quantises activations as well as weights, and its headline compression figure
depends on that; our study is deliberately weight-only. And our study uses standard baseline methods on
purpose, so that a measured difference is attributable to **pipeline ordering** rather than to a clever
criterion — adopting JSQ wholesale would change the research question.

**What we would like challenged:** should the activation editor be implemented as a declared ablation,
cited as related work that independently corroborates F-05, or treated as grounds to restructure the
study?

---

## 8. Reproduction

Environment: Windows 11, Python 3.11.9, torch 2.13.0+cu126, transformers 5.14.1, datasets 5.0.0. All
five model checkpoints pinned to commit SHAs. Data: `Salesforce/wikitext` / `wikitext-2-raw-v1`,
tokenised with the pinned Pythia tokeniser, split fingerprints recorded per run.

```bash
pytest -q                                      # 759 passing, offline, ~40 s
ruff check . && ruff format --check .

python scripts/prepare_data.py     --config configs/experiments/pilot.yaml
python scripts/run_scale_sweep.py  --config configs/experiments/screening.yaml
python scripts/summarise_screening.py --model pythia-160m \
    --budgets s1_30_w8,s2_50_w8,s3_50_w4,s4_70_w4,s5_30_w4,s6_40_w8
```

Every measurement in this brief, with the exact conditions that produced it, is in
`docs/findings_log.md`. The frozen decisions are in `docs/protocol_freeze.md`. What could still make the
results wrong is in `docs/validity_threats.md`.
