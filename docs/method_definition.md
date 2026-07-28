# Method Definition

This document fixes exactly what "sequential" and "joint" mean in this study. It exists so that
the arms cannot drift apart during implementation, and so that a reader can tell precisely what was
compared.

Every decision that was once marked **DECISION REQUIRED** here is now settled in
[protocol_freeze.md](protocol_freeze.md). That file is the record of what was chosen and why; this
one is the specification of the method itself.

> **Revision note (2026-07-28).** This document previously specified full-model
> quantisation-aware fine-tuning — a `Trainer`, global optimiser steps, gradual sparsity ramps,
> mask freezing. Research plan §3.1 selects **layerwise post-training reconstruction** instead,
> because full-model fine-tuning will not fit at 1B–1.4B on a 6 GB laptop GPU. The two are not
> variations of each other: the unit of optimisation is *local steps per layer*, not global
> optimiser steps. This document has been rewritten to match the plan. The fine-tuning material
> survives only as the optional ablation described under
> [Optional recovery ablation](#optional-recovery-ablation).

## Scope

This study compares **one specific sequential implementation** against **one specific joint
implementation**. It does not claim to represent pruning or quantisation in general.

Concretely, the comparison is:

- **Sequential (P→Q):** mask → reconstruct → quantise → reconstruct.
- **Sequential (Q→P):** quantise → mask → reconstruct. Run at one representative budget as a
  reverse-order ablation, so the joint arm is not compared only against a weak ordering (§3.6).
- **Joint:** alternating layerwise co-optimisation of the mask and the quantised representation,
  described in full under [Joint pipeline](#joint-pipeline).

Every arm runs through the **same** layerwise solver and differs only in the order in which that
solver is called. That is what makes §3.8 ("what qualifies as joint") checkable in code rather
than by inspection.

What follows from that scope:

- A result here is evidence about *these pipelines*, not about "joint compression" as a family. A
  different joint method — learned masks, a differentiable sparsity budget, a second-order
  criterion — could behave differently at any scale.
- The pruning and quantisation methods are deliberately **standard baselines**, not
  state-of-the-art. That is the point: a stronger base method would raise every arm, and the
  quantity of interest is the *difference* between pipelines. A study built on a bespoke criterion
  could not separate "joint helps" from "our criterion helps".
- No claim is made that any arm is the best available way to compress these models.
- Nothing in the design presumes the joint arm wins. A null or negative joint gain is a valid,
  reportable outcome, and given the seed variance involved it is a likely one at the moderate
  budget.

## The layerwise objective

For each targeted linear layer, calibration activations `X` are captured and the compressed
weights chosen to minimise the difference between the dense and compressed layer outputs:

```
Dense output       Y     = XW
Compressed output  Y_hat = X(M ∘ Q_b(W))
Reconstruction     L_rec = ‖Y − Y_hat‖²_F
Subject to         sparsity(M) = s,  bit_width(Q_b) = b
```

Layers are processed **one at a time, in depth order**, with activations propagated through the
already-compressed prefix so that error accumulation is realistic rather than optimistic.

**The unit of optimisation is local steps per layer.** All fairness accounting is built on that,
not on global optimiser steps.

## Compressible Modules

The same module selection is used for **every arm**. This is not a convenience — an arm that
compressed more layers than another would show a "gain" that was really a coverage difference.

### Included

- **Linear layers inside the decoder blocks only.**
- **Attention projections.**
  - GPT-NeoX / Pythia: `attention.query_key_value` (a *fused* Q/K/V matrix) and `attention.dense`.
  - Qwen2: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`.
- **MLP projections.**
  - GPT-NeoX / Pythia: `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`.
  - Qwen2: `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` (gated MLP).

The fused QKV matrix in Pythia matters for any structured or semi-structured pattern: a head- or
channel-level pattern must be applied consistently across all three slices, or the attention heads
break. Unstructured pruning is unaffected.

### Excluded

- **Input embeddings** (`gpt_neox.embed_in`, `model.embed_tokens`, `wte`, `wpe`).
- **Positional embeddings.** Pythia uses rotary embeddings, which have no learned weight matrix to
  prune; the exclusion is stated anyway so it is not left ambiguous, and it covers learned
  positional tables in any model added later.
- **Output head / `lm_head`** (`embed_out`, `lm_head`), unless explicitly changed later — and any
  such change must be applied to every arm and recorded in the run record.
- **LayerNorm weights and biases**, and **all biases**. A negligible fraction of the parameters,
  and pruning them tends to cost quality for no size benefit.

### Why embeddings and the head are excluded

Embedding parameters are a large fraction of a 160M model and a much smaller fraction of a 1.4B
one. Including them would make the *effective* compression budget vary with model scale, so a
scale trend in joint gain could not be separated from a scale trend in how much of the model was
being compressed (§2.6).

Two consequences must be stated in the write-up:

1. Reported compression ratios are ratios over the **compressible** parameters, and are therefore
   lower than whole-model ratios.
2. The scale x-axis is **targeted non-embedding parameter count**, with total model size shown
   only as a secondary label.

For Qwen2.5 there is an additional reason: its input and output embeddings are **tied**, so
excluding `lm_head` without also excluding `embed_tokens` would compress the input embedding as a
side effect. Both are excluded in
[`qwen2_5_0_5b.yaml`](../configs/models/qwen2_5_0_5b.yaml).

### Enforcement

Module selection resolves through a single function,
[`select_compressible_modules`](../src/scale_aware_compression/models/adapters.py), which every arm
calls. An empty selection must raise rather than warn: a "compressed" model identical to the dense
one would otherwise look like an excellent compression result.

## Pruning Method

- **Saliency: activation-weighted magnitude** (§3.3). For weight `W_ij` and input column `j`:

  ```
  S_ij = |W_ij| · ‖X_j‖₂
  ```

  A weight is unimportant only if it is *both* small and multiplies a low-energy input. Plain
  magnitude ignores the second factor, and the activation norms are already being accumulated for
  the reconstruction solver, so this costs nothing extra.

- **Granularity: global unstructured**, or fixed blockwise, identical across every model size
  (§3.10). Semi-structured 2:4 remains available because it is the pattern with any prospect of
  kernel support, and therefore the only one where a measured latency gain could plausibly come
  from — subject to the installed backend actually providing such a kernel, which must be verified
  rather than assumed.

- **Masks are realised exactly.** The realised sparsity must equal the target exactly, verified on
  the **converted, reloaded** artefact rather than in memory.

- **No gradual ramp, no mask freezing schedule, no regrowth.** Those belonged to the fine-tuning
  design. Under layerwise PTQ the mask for a layer is decided within that layer's local
  optimisation and then frozen when the layer is finished.

See [protocol_freeze.md#d3](protocol_freeze.md#d3--mask-scoring--activation-weighted-magnitude-scored-under-quantised-weights)
for which weights the saliency is computed on, which is the one place the arms deliberately differ.

### Not implemented

Hessian- and gradient-based *pruning criteria* — SparseGPT, Wanda, OBS/OBD, movement pruning — are
**not implemented and not planned for the first milestone**. §3.3 permits a second-order method
"only if it is already stable". Any future addition must be applied to every arm and reported as a
separate variant.

Note the distinction: the *reconstruction solver* may use second-order information (`H = XᵀX`)
without the *pruning criterion* being second-order. The saliency above is first-order.

## Quantisation Method

- **Weight-only.** Activations remain floating point. Activation quantisation is explicitly not
  part of the core design (§3.9); it changes the latency picture substantially and is a separate
  study.
- **Symmetric**, per-channel by default, per-group with a fixed group size for sub-8-bit schemes.
  Group size is held **constant across model sizes**. What matters for validity is that the
  setting is identical across arms, not which setting wins.
- **W8 and W4** are the two screened precisions (§3.9).
- **Identical scheme across arms** — same bit width, symmetry, granularity, group size, and
  excluded modules. Every arm includes the same
  [`quantisation.yaml`](../configs/compression/quantisation.yaml) rather than restating any of it.
- **Identical calibration data.** Calibration indices derive from a fixed `data.calibration_seed`,
  *not* the run seed, so varying the run seed for error bars does not also change the calibration
  set. Calibration is drawn from the training split and must be disjoint from the evaluation set;
  overlap raises rather than warns. A held-out reconstruction subset detects calibration
  overfitting (§4.8).

### Fake versus real quantisation

Two representations exist and the distinction affects every number reported:

| | Storage | Smaller on disk | Faster |
| --- | --- | --- | --- |
| **Fake quantisation** | FP32, values snapped to the grid | no | no |
| **Real quantisation** | target bit width plus scales | yes | depends on backend |

Fake quantisation is what the layerwise solver works with. **Only the real, converted artefact may
be benchmarked or measured for size.** A model that was fake-quantised but never converted produces
correct-looking quality with a meaningless size and latency; `is_converted` and
`storage_efficiency` in the run record exist to catch exactly that, and §4.8 requires confirming
the bit width is real rather than silently dequantised.

### Bit widths and deployment

Settled in [protocol_freeze.md#d1](protocol_freeze.md#d1--cpu-quantisation-backend--pytorch-native-int8-w4-excluded-from-latency):

| Budget axis | W8 | W4 |
| --- | --- | --- |
| Quality | yes | yes |
| Checkpoint size / effective bits | yes | yes |
| **CPU latency** | yes | **no** |

W4 stays in the study for quality and size. It does not appear in a latency table, because a
packed 4-bit CPU linear would measure the dequantisation kernel rather than the compression. RQ4
(does sparsity yield real CPU speedup?) is answered from the pruning-only arm, whose weights stay
FP32 and therefore benchmark natively at every screened sparsity.

PyTorch's mature CPU quantisation path targets **INT8**; there is no equally mature built-in 4-bit
weight-only CPU kernel, which is the whole reason for the split above.

### The INT8 fallback

W4 is the aggressive precision in the current configs and remains the **main technical risk** in
the plan. If screening (§5.3) finds W4 unstable or catastrophically degraded at high sparsity, the
documented fallback applies:

| Budget | Sparsity | Bit width |
| --- | --- | --- |
| Moderate | 50% | INT8 |
| Aggressive (fallback) | 70% | INT8 |

The fallback holds sparsity as the aggressive variable and keeps precision on the well-supported
INT8 backend, so every row in the study shares one runtime and one artefact format — including the
latency table, which would then cover both budgets rather than only the moderate one.

Two consequences must be stated in the write-up if it is used: precision becomes a constant rather
than a second compression axis, and the compression-ratio range narrows from roughly 8×–27× to
roughly 8×–13×. The fallback is written out in
[`main_scale_sweep.yaml`](../configs/experiments/main_scale_sweep.yaml).

Note this is a *screening outcome*, not an open decision. §5.3's selection rule — two budgets that
are technically stable, measurably but not catastrophically degraded, and separated enough to test
the scale question — decides it on evidence from Pythia-160M and 410M, before 1B runs.

## Mask Scoring

Settled: **rank by activation-weighted magnitude, computed on the quantised weights in the joint
arm.** Recorded with full reasoning in
[protocol_freeze.md#d3](protocol_freeze.md#d3--mask-scoring--activation-weighted-magnitude-scored-under-quantised-weights).

```
joint            S_ij = |Q_b(W_ij)| · ‖X_j‖₂        ← scored on quantised weights
sequential P→Q   S_ij = |W_ij|      · ‖X_j‖₂        ← no quantiser exists yet at mask time
sequential Q→P   S_ij = |Q_b(W_ij)| · ‖X_j‖₂        ← quantiser already fitted
```

This is required, not preferred. §3.7 says to "update mask scores using quantized or
fake-quantized weights", and §3.8 lists mask decisions evaluated under quantised weights as what
*qualifies* as joint — with "prune completely, freeze, then call ordinary PTQ" as what does not. A
mask ranked on untouched FP32 weights is chosen in ignorance of the grid, which is that failure
case.

An earlier revision of this document recommended ranking by FP32 shadow-weight magnitude. That
belonged to the superseded quantisation-aware-training design, where shadow weights existed
because an optimiser was training FP32 parameters behind fake-quantisation nodes. Under layerwise
PTQ there is no training loop and no such parameter, so the argument does not carry over.

Rules that apply regardless:

- Document the rule here, use it consistently across every run and arm, and record it in the run
  record.
- **Do not invent a combined alpha–beta scoring function** (e.g. `α·|w| + β·|w − Q(w)|`) unless it
  is actually implemented and separately ablated. An unablated tunable score is a free parameter
  that could manufacture a joint gain.
- Switching rules mid-study invalidates every earlier comparison.

## Sequential Pipeline

**Primary order, P→Q** (§3.5):

```
collect calibration activations
  -> compute saliency on the dense weights
  -> create the target mask
  -> reconstruct the surviving weights
  -> quantise the survivors at the fixed bit width
  -> reconstruct again, for the same local-step budget the joint arm receives
  -> freeze and evaluate
```

Quantisation is fitted to the **post-pruning** model: scales are estimated after the mask exists,
so they describe the sparse weights rather than the dense ones.

Being precise about what that does and does not imply:

> Quantisation is calibrated on the altered post-pruning distribution, including the increased mass
> at zero and the remaining non-zero weights.

It is **not** claimed that pruning narrows the numerical dynamic range. Magnitude pruning removes
the *smallest* weights, so the largest absolute values — which set the observed min/max and
therefore the scale — generally survive. The post-pruning distribution differs from the dense one
mainly in having far more mass at exactly zero, not in having a smaller range. Whether that makes
the model easier or harder to quantise is an empirical question this study does not presuppose.

What the ordering definitively cannot do is the converse: the mask is chosen with no knowledge of
where the quantisation grid points will land. Closing that gap is what the joint arm attempts,
which is why the comparison isolates one design choice.

**Reverse order, Q→P** (§3.6), at one representative budget:

```
quantise -> compute saliency -> mask -> reconstruct -> freeze
```

Where both orders are available, the **best-of** the two defines the sequential baseline per model
and budget, and the winning order is recorded (§6.1). Joint gain is measured against that, not
against P→Q alone.

## Joint Pipeline

The accurate name, and the one to use in the write-up:

> **alternating layerwise co-optimisation of mask and quantiser**

Per §3.7, for each layer:

```
initialise mask M and quantiser Q_b
repeat K joint iterations:
    fake-quantise the surviving weights
    compute reconstruction loss under the current M and Q_b
    recompute saliency UNDER the quantised weights, at target sparsity   <- §3.8 requirement
    update the mask
    re-estimate quantisation scales on the survivors                      <- §3.8 requirement
    locally optimise surviving weights and scales for fixed local steps
freeze M and Q_b
```

What makes it joint is that the mask decision and the quantised representation each influence the
other, within one shared objective, while both constraints are active.

### What qualifies as joint

§3.8, as a checklist the implementation must satisfy:

| Qualifies | Does **not** qualify |
| --- | --- |
| Mask decisions evaluated under quantised or fake-quantised weights | Prune completely, freeze, then call ordinary PTQ |
| Quantisation scales re-estimated after mask changes | Quantise completely, then prune without re-optimising quantisation |
| One common reconstruction objective while both constraints are active | Run two unrelated scripts and label the output "joint" |
| Final mask and quantised weights jointly refined for fixed steps | Compare arms with different calibration data or optimisation budgets |

Phase 6 carries a regression test that **fails** if the joint arm is implemented as
"prune fully, then plain PTQ". That test is the executable form of this table.

## Matched Compression Budget

Sequential and joint runs must match on **all** of:

| Must match | Mechanism |
| --- | --- |
| Target sparsity | Every arm includes the same `pruning.yaml`; asserted by test |
| Quantisation bit width | Every arm includes the same `quantisation.yaml`; asserted by test |
| Included and excluded modules | Single `select_compressible_modules` call path |
| Layer processing order | Shared layerwise driver, depth order |
| Calibration data | Indices from a fixed `data.calibration_seed`, not the run seed |
| Deployment backend | Same `quantisation.backend`, recorded per run |
| Output format | Same conversion code path; `storage_efficiency` cross-checks |

`budget_label` is recorded on every run and carried through `joint_gain_summary`, so a comparison
accidentally made across budgets is visible in the output rather than silent.

A joint arm at 50% sparsity compared against a sequential arm at 60% measures sparsity, not
pipeline design.

## Matched Optimisation Budget

> **Extra optimisation must not be mistaken for a joint-method advantage.**

This is the single easiest way for this study to produce a wrong result. §3.11 states it directly:
if the joint method receives more local steps, more calibration tokens, or model-specific tuning
than the sequential baseline, a higher score cannot be attributed to joint optimisation.

The arms must match on:

- **total local reconstruction steps**, summed across layers — the fairness unit under this method
- **approximately equal objective evaluations**
- **identical calibration tensors** — same examples, same order, same token count, same sequence
  length
- **identical targeted modules and layer processing order**
- **the same seed set** for any direct comparison
- **hyperparameters tuned on the screening model only** — not per scale, unless that is a declared
  ablation

Note the sequential P→Q pipeline has a second reconstruction pass specifically so its local-step
total can equal the joint arm's. Without it the joint arm would receive strictly more optimisation.

Mechanisms:

- Every stage records its local steps and tokens processed.
- `training_cost_overhead` reports the joint/sequential ratio; the headline comparison requires
  1.00.
- `matched_budget()` compares two budgets programmatically.
- A fairness assertion compares calibration tensor hashes and module lists between arms.

One honest caveat: **equal steps is not equal difficulty.** The joint arm is solving a harder
problem in the same budget, so a matched comparison may understate what joint could achieve with a
budget tuned for it — and an unmatched comparison overstates it. The matched comparison is the
defensible one, and the limitation belongs in the write-up.

## Deployment Backend

**CPU latency comparisons are valid only when every arm uses the same runtime and artefact
format.** A latency difference between two backends is a property of the backends.

Requirements:

- One backend for all arms within a results table (`quantisation.backend`, recorded per run).
- One artefact format for all arms; every conversion goes through the same code path.
- One machine per results table, with a pinned thread count and a fixed power profile. Full
  hardware metadata is recorded so a violation is detectable after the fact.
- **Never infer speedup from sparsity alone** (§3.12). Report three separate quantities:
  theoretical non-zero parameter reduction, actual serialised checkpoint size, and measured CPU
  latency.

A null latency result is a valid finding, not a failure (§10.1). Unstructured sparsity in a dense
GEMM kernel should be expected to deliver no speedup: the multiply-accumulates still happen, they
just multiply by zero.

Measurement rules are in [benchmarking_protocol.md](benchmarking_protocol.md); the frozen backend
choice is in [protocol_freeze.md](protocol_freeze.md).

## Optional recovery ablation

The `training/` package — trainer, recovery, callbacks — is **off the critical path**. The core
method does no fine-tuning at all; `recover` is a no-op.

Those files are retained for one optional ablation: a short full-model fine-tune applied *equally*
to the sequential and joint artefacts, to test whether a small recovery budget closes or widens
the gap. It runs only if the schedule allows, after the analysis gate, and it must be reported as a
separate variant with its own budget accounting. It is not part of the minimum viable result set
(§5.7).

## `Compressor` stage semantics

The five-stage ABC survives the change of method, with reinterpreted meanings:

| Stage | Meaning under layerwise PTQ |
| --- | --- |
| `prepare` | select targeted modules, build the calibration set, install activation capture |
| `apply` | **the layerwise loop** — the whole algorithm lives here |
| `recover` | no-op in the core method; retained for the optional ablation above |
| `convert` | pack to real low-bit storage, fold masks, emit the deployable artefact |
| `report_statistics` | unchanged, plus per-layer reconstruction losses |

## Current Status

**All compression algorithms are placeholders.** The interfaces, stage ordering, configuration,
metrics, and CPU benchmarking harness are implemented; the algorithms that would populate them are
not. Placeholder stages raise `NotImplementedError` naming the module to edit, rather than
returning a plausible model — a silent no-op would produce an excellent-looking compression result.

| Component | Status |
| --- | --- |
| Module selection (`select_compressible_modules`) | **placeholder** |
| Activation capture (`H = XᵀX`, `‖X_j‖₂`) | implemented |
| Activation-weighted saliency | implemented |
| Mask construction, per tensor (`build_mask_from_scores`) | implemented |
| Mask construction, model-wide (`build_masks`) | **placeholder** — needs the driver's signature |
| Mask application / folding | **placeholder** |
| Weight quantiser, per tensor (symmetric, per-channel / group, W8 + W4) | implemented |
| Low-bit packing and unpack round-trip | implemented |
| Reconstruction solver (damped ALS) | implemented |
| Layerwise driver | **not started** |
| Pruning-only arm (`Pruner`) | **placeholder** (statistics implemented) |
| Quantisation-only arm (`Quantiser`) | **placeholder** (statistics implemented) |
| Sequential arm P→Q (`SequentialCompressor`) | **placeholder** (stages defined) |
| Sequential arm Q→P | **not started** |
| Joint arm (`JointCompressor`) | **placeholder** (stages defined) |
| Sparsity schedules | implemented — *retained for the optional ablation only* |
| Training loop and callbacks | **placeholder** — *off the critical path* |
| Perplexity / agreement / generation evaluation | implemented |
| Downstream tasks (HellaSwag, PIQA, ARC-Easy) | **not started** |
| CPU benchmark harness | implemented (single-shape; prefill/decode split outstanding) |
| Metrics (sparsity, ratio, retention, joint gain) | implemented |
| Experiment records (JSON + CSV) | implemented |

The open decisions that used to sit here are resolved in
[protocol_freeze.md](protocol_freeze.md). What remains genuinely unsettled is listed there under
*Still open*, and none of it can be resolved from the code.

## Related documents

- [research_plan.pdf](research_plan.pdf) — the authoritative source
- [protocol_freeze.md](protocol_freeze.md) — the frozen decisions and the environment record
- [implementation_plan.md](implementation_plan.md) — build phases and exit tests
- [research_question.md](research_question.md) — the questions and the definition of joint gain
- [methodology.md](methodology.md) — variables, controls, fair-comparison requirements
- [experiment_protocol.md](experiment_protocol.md) — the run tables
- [benchmarking_protocol.md](benchmarking_protocol.md) — CPU measurement rules
- [validity_threats.md](validity_threats.md) — what could still make these results wrong
- [reproducibility.md](reproducibility.md) — seeds, pins, record contents, promotion checklist
