# Method Definition

This document fixes exactly what "sequential" and "joint" mean in this study. It exists so that the
two arms cannot drift apart during implementation, and so that a reader can tell precisely what was
compared.

Where a choice is still open, it is marked **DECISION REQUIRED** and listed again in
[Current Status](#current-status). Those must be settled before the main experiments start, not after
seeing results.

## Scope

This study compares **one specific sequential implementation** against **one specific joint
implementation**. It does not claim to represent pruning or quantisation in general.

Concretely, the comparison is:

- **Sequential:** unstructured magnitude pruning → recovery fine-tuning → post-training weight-only
  quantisation → conversion.
- **Joint:** fake-quantisation preparation → gradual unstructured magnitude pruning during
  quantisation-aware fine-tuning → mask freezing → joint recovery → conversion.

What follows from that scope:

- A result here is evidence about *these two pipelines*, not about "joint compression" as a family.
  A different joint method — one using a Hessian-based criterion, learned masks, or a differentiable
  sparsity budget — could behave differently at any scale.
- The pruning and quantisation methods are deliberately **standard baselines**, not
  state-of-the-art. That is the point: a stronger base method would raise both arms, and the quantity
  of interest is the *difference* between the pipelines. A study built on a bespoke pruning criterion
  could not separate "joint helps" from "our criterion helps".
- No claim is made that either arm is the best available way to compress these models.
- Nothing in the design presumes the joint arm wins. A null or negative joint gain is a valid,
  reportable outcome, and given the seed variance involved it is a likely one at the moderate budget.

## Compressible Modules

The same module selection is used for **every arm**. This is not a convenience — a joint arm that
compressed more layers than the sequential arm would show a "joint gain" that was really a coverage
difference.

### Included

- **Linear (fully connected) layers inside the decoder blocks only.**
- **Attention projections: included.**
  - GPT-NeoX / Pythia: `attention.query_key_value` (a *fused* Q/K/V matrix) and `attention.dense`.
  - Qwen2: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`.
- **MLP projections: included.**
  - GPT-NeoX / Pythia: `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h`.
  - Qwen2: `mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj` (gated MLP).

The fused QKV matrix in Pythia matters for any structured or semi-structured pattern: a head- or
channel-level pattern must be applied consistently across all three slices, or the attention heads
break. Unstructured pruning is unaffected.

### Excluded

- **Input embeddings** (`gpt_neox.embed_in`, `model.embed_tokens`, `wte`, `wpe`).
- **Positional embeddings.** Pythia uses rotary embeddings, which have no learned weight matrix to
  prune; the exclusion is stated anyway so it is not left ambiguous, and it covers learned positional
  tables in any model added later.
- **Output head / `lm_head`** (`embed_out`, `lm_head`), **unless explicitly changed later** — and any
  such change must be applied to both arms and recorded in the run record.
- **LayerNorm weights and biases**, and **all biases**. These are a negligible fraction of the
  parameters and pruning them tends to cost quality for no size benefit.

### Why embeddings and the head are excluded

Embedding parameters are a large fraction of a 160M model and a much smaller fraction of a 1.4B one.
Including them would make the *effective* compression budget vary with model scale, so a scale trend
in joint gain could not be separated from a scale trend in how much of the model was being
compressed. Excluding them keeps the budget comparable across the sweep.

The consequence, which must be stated in the write-up: reported compression ratios are ratios over
the **compressible** parameters, and are therefore lower than whole-model ratios.

For Qwen2.5 there is an additional reason. Its input and output embeddings are **tied**, so excluding
`lm_head` without also excluding `embed_tokens` would compress the input embedding as a side effect.
Both are excluded in [`qwen2_5_0_5b.yaml`](../configs/models/qwen2_5_0_5b.yaml).

### Enforcement

Module selection resolves through a single function,
[`select_compressible_modules`](../src/scale_aware_compression/models/adapters.py), which both arms
call. An empty selection must raise rather than warn: a "pruned" model identical to the dense one
would otherwise look like an excellent compression result.

## Pruning Method

The initial planned method, for both arms:

- **Unstructured magnitude pruning.** Weights are ranked by absolute value and the smallest are set
  to zero. Ranking is **per layer** by default (`global_ranking: false`); global ranking is available
  but changes what a per-layer latency measurement means, so it is not the default.
- **Gradual sparsity schedule for joint training.** Sparsity ramps from `initial_sparsity` to the
  target over `[schedule_start_step, schedule_end_step]` using the cubic ramp of Zhu & Gupta:
  `s(t) = s_f + (s_i - s_f)(1 - t/n)^3`. Implemented in
  [`schedules.py`](../src/scale_aware_compression/compression/schedules.py).
- **Masks periodically updated before the freeze point.** Every `mask_update_interval` steps, masks
  are recomputed at the current scheduled sparsity. Re-ranking every step is expensive and
  near-redundant between adjacent steps.
- **Masks frozen during the final recovery phase.** After
  `mask_freeze_step = total_steps × freeze_masks_after_ratio` (0.8 by default), masks stop changing,
  so the final phase is pure recovery at the target sparsity. Without this, the last mask update
  lands with too few steps left to recover from.
- **Masked weights re-zeroed after optimiser updates.** Momentum and weight decay will move a masked
  weight off zero even when the mask itself has not changed, so masks are re-applied after *every*
  optimiser step, not only on update steps.
- **No regrowth after mask freezing.** Once frozen, a pruned position stays pruned for the rest of
  training. There is no RigL-style regrow phase.

The **same schedule parameters** are used by both arms. If the joint arm reached its target sparsity
at a different step, it would also get a different amount of recovery at full sparsity, and part of
any measured gain would be a schedule artefact. This is asserted by a test over the shipped configs.

### Not implemented

Hessian-based and gradient-based criteria — SparseGPT, Wanda, OBS/OBD, movement pruning — are **not
implemented and not planned for the first milestone**. They appear in
[`paper_outline.md`](paper_outline.md) as related work only. Any future addition must be applied to
both arms and reported as a separate variant.

## Quantisation Method

- **Weight-only quantisation initially.** Weights are quantised; activations are not.
- **Activations remain floating point** unless a later experiment explicitly enables activation
  quantisation (`compression.quantisation.quantise_activations`, default `false`). Activation
  quantisation changes the latency picture substantially and is a separate study; holding it off keeps
  the variable fixed.
- **Per-channel quantisation where supported** (`granularity: per_channel`), falling back to
  per-tensor only where a backend cannot express per-channel scales. Per-group is available for
  sub-8-bit schemes via `group_size`. What matters for validity is that the setting is *identical
  across arms*, not which setting wins.
- **Symmetric scheme** by default. Note that with an asymmetric scheme zero is not necessarily an
  exact grid point, which interacts with pruning: quantising after folding masks can map a pruned
  zero onto a non-zero value. Measured sparsity is verified after conversion either way.
- **Identical quantisation scheme for sequential and joint arms** — same bit width, scheme,
  granularity, group size, observer, and excluded modules. Both arms include the same
  [`quantisation.yaml`](../configs/compression/quantisation.yaml) rather than restating any of it.
- **Identical calibration data.** Calibration indices derive from a fixed `data.calibration_seed`,
  *not* from the run seed, so varying the run seed for error bars does not also change the calibration
  set. Calibration is drawn from the training split and must be disjoint from the evaluation set;
  overlap raises rather than warns.
- **Identical output artefact format and deployment backend.** See
  [Deployment Backend](#deployment-backend).

### Fake versus real quantisation

Two representations exist and the distinction affects every number reported:

| | Storage | Differentiable | Smaller on disk | Faster |
| --- | --- | --- | --- | --- |
| **Fake quantisation** | FP32 | yes (with STE) | no | no |
| **Real quantisation** | target bit width | no | yes | depends on backend |

Fake quantisation is what training uses. **Only the real, converted artefact may be benchmarked or
measured for size.** A model that was fake-quantised but never converted produces correct-looking
quality with a meaningless size and latency; `is_converted` and `storage_efficiency` in the run record
exist to catch exactly that.

### Bit widths and the 4-bit risk

- **INT8** is the moderate budget and the width with the strongest native PyTorch CPU support.
- **4-bit** is the aggressive budget in the current configs, and the main technical risk in the plan.
  PyTorch's native CPU quantisation path targets INT8; there is no equally mature built-in 4-bit
  weight-only CPU kernel, so 4-bit deployment generally requires a separate backend — a packed-weight
  custom linear module, or an external runtime.

**DECISION REQUIRED.** Before the main experiments:

1. Confirm a single 4-bit CPU path exists that both arms can use.
2. Confirm both arms emit the same artefact format and run on the same backend.

If either fails, use the documented fallback:

| Budget | Sparsity | Bit width |
| --- | --- | --- |
| Moderate | 50% | INT8 |
| Aggressive (fallback) | 70% | INT8 |

The fallback holds sparsity as the aggressive variable and keeps precision on the well-supported
path, so every row in the study shares one runtime and one artefact format. Two consequences must be
stated in the write-up if it is used: precision becomes a constant rather than a second compression
axis, and the compression-ratio range narrows from roughly 8×–27× to roughly 8×–13×.

4-bit support stays in the configuration system either way. What is undecided is whether the *main
study* uses it. The fallback is written out in
[`main_scale_sweep.yaml`](../configs/experiments/main_scale_sweep.yaml).

## Sequential Pipeline

```
dense model
  -> pruning
  -> recovery
  -> quantisation preparation
  -> calibration
  -> quantisation
  -> conversion
```

Quantisation is **fitted to the post-pruning model**: observers are inserted and calibrated after
pruning and recovery are complete, so the quantisation parameters describe the sparse model rather
than the dense one.

Being precise about what that does and does not imply:

> Quantisation is calibrated on the altered post-pruning distribution, including the increased mass at
> zero and the remaining non-zero weights.

It is **not** claimed that pruning narrows the numerical dynamic range. Magnitude pruning removes the
*smallest-magnitude* weights, so the largest absolute values — which are what set the observed min/max
and therefore the scale — generally survive. The post-pruning distribution differs from the dense one
mainly in having far more mass at exactly zero, not in having a smaller range. Whether that makes the
model easier or harder to quantise is an empirical question this study does not presuppose.

What the ordering definitively cannot do is the converse: the pruning decision is made with no
knowledge of where the quantisation grid points will land. Closing that gap is what the joint arm
attempts, which is why the comparison isolates one design choice.

Implemented in [`sequential.py`](../src/scale_aware_compression/compression/sequential.py) by
composing `Pruner` and `Quantiser` rather than reimplementing either, so this arm cannot drift from
the single-method arms it is the composition of.

## Joint Pipeline

```
dense model
  -> fake-quantisation preparation
  -> gradual magnitude pruning during optimisation
  -> mask freezing
  -> joint recovery
  -> conversion
```

The first implementation is:

> **joint magnitude pruning with quantisation-aware fine-tuning**

That is the accurate name and it should be used in the write-up. It is **not** a universal joint
compression algorithm, and no claim is made that it is the best way to combine the two objectives. It
is one concrete, reproducible instantiation of "optimise for both at once", chosen because it is
simple enough to implement correctly and to match against the sequential arm step for step.

What makes it joint: fake quantisation is installed and active *before* any weight is pruned, so mask
selection and weight updates both happen in the presence of the quantisation grid, and one
optimisation run adapts to both perturbations rather than recovering from pruning and then absorbing
an unrecovered quantisation error.

Stage detail:

1. **Fake-quantisation preparation** — observers attached and calibrated, fake-quantisation nodes
   inserted on the weights of the selected modules, straight-through estimator enabled. Without an STE
   the rounding has zero gradient almost everywhere and quantisation-aware training cannot adapt the
   weights at all, which would reduce this arm to extra fine-tuning under a frozen perturbation.
2. **Gradual magnitude pruning during optimisation** — masks ramped on the shared cubic schedule,
   recomputed every `mask_update_interval` steps, re-applied after every optimiser step. Optionally
   preceded by `quantisation_warmup_steps` dense steps, since the combined perturbation is largest at
   the start.
3. **Mask freezing** — at `freeze_masks_after_ratio` of training. No regrowth afterwards.
4. **Joint recovery** — the remaining steps train at the frozen target sparsity with fake
   quantisation still active.
5. **Conversion** — masks folded into the weights, fake quantisation converted to real low-precision
   storage through the *same* code path the sequential arm uses.

Implemented in [`joint.py`](../src/scale_aware_compression/compression/joint.py).

## Mask Scoring

The scoring rule is a **choice**, not a property of the method, and the two reasonable options differ
only for weights near a grid boundary. Both are defensible:

| Option | Rule |
| --- | --- |
| **A** | Rank by absolute **fake-quantised** weight magnitude. |
| **B** | Rank by absolute **FP32 shadow-weight** magnitude, with fake quantisation active throughout optimisation. |

**Planned default: Option B** — rank by absolute FP32 shadow-weight magnitude while fake quantisation
remains active during optimisation.

Reasons, in order of weight:

1. **Stability across mask updates.** A weight sitting near a grid boundary can have its quantised
   value jump between grid points from step to step, making an Option A ranking noisy in exactly the
   region where the pruning decision is marginal. Shadow weights move smoothly.
2. **The scoring function stays identical to the sequential arm's.** Both arms then rank by absolute
   weight magnitude, so the arms differ in the *pipeline* — when quantisation is introduced — and not
   in the criterion. That is a cleaner isolation of the variable under study.
3. **Simplicity and reproducibility.** Shadow weights are the parameters themselves; no dependence on
   observer state at the moment of ranking, so a mask rebuild is a pure function of the weights.

Note that Option B still gives a genuinely *quantisation-aware* mask: the shadow weights being ranked
have been shaped by many steps of training with fake quantisation active. The awareness enters through
the optimisation, not through the ranking function.

Rules that apply whichever option is chosen:

- Document it precisely here, use it consistently across every run and every arm, and record it in the
  run record.
- **Do not invent a combined alpha–beta scoring function** (e.g. `α·|w| + β·|w − quantise(w)|`) unless
  it is actually implemented and separately ablated. An unablated tunable score is a free parameter
  that could manufacture a joint gain.
- Switching rules mid-study invalidates every earlier comparison.

**DECISION REQUIRED:** confirm Option B (or deliberately choose A) before implementing
`JointCompressor.apply`. The default above is a recommendation with stated reasons, not a settled
fact.

## Matched Compression Budget

Sequential and joint runs must match on **all** of:

| Must match | Mechanism |
| --- | --- |
| Target sparsity | Both arms include the same `pruning.yaml`; asserted by test |
| Quantisation bit width | Both arms include the same `quantisation.yaml`; asserted by test |
| Included modules | Single `select_compressible_modules` call path |
| Excluded modules | Same `exclude_patterns`, from the shared configs |
| Calibration data | Indices from fixed `data.calibration_seed`, not the run seed |
| Deployment backend | Same `quantisation.backend`, recorded per run |
| Output format | Same conversion code path; `storage_efficiency` cross-checks |

`budget_label` is recorded on every run and carried through `joint_gain_summary`, so a comparison
accidentally made across budgets is visible in the output rather than silent.

A joint arm at 50% sparsity compared against a sequential arm at 60% measures sparsity, not pipeline
design.

## Matched Optimisation Budget

Sequential and joint runs must match as closely as possible on:

- **optimiser steps**
- **training tokens**
- **recovery data** (same corpus, same split, same order)
- **effective batch size** (`batch_size × gradient_accumulation_steps`)
- **learning-rate schedule** (same shape: linear warmup then cosine decay, same warmup ratio)

> **Extra training compute must not be mistaken for a joint-method advantage.**

This is the single easiest way for this study to produce a wrong result. The joint arm naturally wants
a longer run — it is doing more in one pass — and if it gets one, the measured "joint gain" is partly
or wholly the extra training.

Mechanisms:

- `compression.joint.joint_max_steps` is set equal to the sequential arm's
  `compression.recovery.max_steps` in the shipped configs, and a test asserts it.
- Every stage records its `optimiser_steps` and `tokens_processed`.
- `match_sequential_budget` is stored in the run record.
- `training_cost_overhead` reports the joint/sequential ratio; the headline comparison requires 1.00.
- `matched_budget()` compares two budgets programmatically.

One honest caveat: **equal optimiser steps is not equal optimisation difficulty.** The joint arm is
solving a harder problem in the same number of steps, so a matched-budget comparison may understate
what the joint method could achieve with a budget tuned for it — and an unmatched comparison overstates
it. The matched comparison is the defensible one, and the limitation belongs in the write-up.

## Deployment Backend

**CPU latency comparisons are valid only when both methods use the same runtime and artefact format.**

A latency difference between two backends is a property of the backends. If the joint artefact runs on
a packed-weight custom module and the sequential artefact runs on PyTorch's native INT8 path, the
measured difference says nothing about the compression methods.

Requirements:

- One backend for all arms within a results table (`quantisation.backend`, recorded per run).
- One artefact format for all arms; both conversions go through the same code path.
- One machine per results table, with a pinned thread count. Full hardware metadata is recorded so a
  violation is detectable after the fact.
- If moderate and aggressive budgets end up on *different* runtimes, their latency numbers are not
  comparable with each other. Either use the INT8 fallback so both share a runtime, or report the two
  budgets as separate, non-comparable measurement sets and say so explicitly.

**DECISION REQUIRED:** the final CPU quantisation backend. Candidates: PyTorch native `x86` /
`fbgemm` (INT8 only), or an external runtime with 4-bit CPU support. This choice constrains the bit
widths available to the main study, so it must be made first.

See [`benchmarking_protocol.md`](benchmarking_protocol.md) for the measurement rules.

## Current Status

**All compression algorithms are placeholders.** The interfaces, stage ordering, configuration,
metrics, and CPU benchmarking harness are implemented; the algorithms that would populate them are
not. Placeholder stages raise `NotImplementedError` naming the module to edit, rather than returning a
plausible model.

| Component | Status |
| --- | --- |
| Module selection (`select_compressible_modules`) | **placeholder** |
| Mask construction (`build_masks`) | **placeholder** |
| Mask application / hooks / folding | **placeholder** |
| Sparsity schedules | implemented |
| Pruning arm (`Pruner`) | **placeholder** (statistics implemented) |
| Quantisation observers and calibration | **placeholder** |
| Fake quantisation and STE | **placeholder** |
| Real conversion (INT8) | **placeholder** |
| Real conversion (4-bit) | **placeholder**, backend undecided |
| Quantisation arm (`Quantiser`) | **placeholder** (statistics implemented) |
| Sequential arm (`SequentialCompressor`) | **placeholder** (stages defined, statistics implemented) |
| Joint arm (`JointCompressor`) | **placeholder** (stages defined, statistics implemented) |
| Training loop and callbacks | **placeholder** |
| Perplexity / agreement / generation evaluation | **placeholder** (pure helpers implemented) |
| CPU benchmark harness | implemented |
| Metrics (sparsity, ratio, retention, joint gain) | implemented |
| Experiment records (JSON + CSV) | implemented |

### Open decisions requiring a human choice

1. **CPU quantisation backend.** Constrains every downstream bit-width choice. Decide first.
2. **Mask scoring rule.** Option B (FP32 shadow-weight magnitude) is recommended above with reasons;
   confirm or override.
3. **Whether 4-bit stays in the main study** or the INT8 fallback is adopted. Follows from (1).

None of these can be resolved from the code, and none should be resolved after seeing results.

## Related documents

- [research_question.md](research_question.md) — the questions and the definition of joint gain
- [methodology.md](methodology.md) — variables, controls, fair-comparison requirements
- [experiment_protocol.md](experiment_protocol.md) — the run tables
- [benchmarking_protocol.md](benchmarking_protocol.md) — CPU measurement rules
- [validity_threats.md](validity_threats.md) — what could still make these results wrong
- [reproducibility.md](reproducibility.md) — seeds, pins, record contents, promotion checklist
