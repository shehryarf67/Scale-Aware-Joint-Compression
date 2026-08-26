# Post-hoc exploratory ablation: end-to-end recovery

> **This is a post-hoc exploratory ablation motivated by the observed local-to-global gap. It does
> not alter or replace the frozen confirmatory study.** [F-37](findings_log.md#f-37) and
> [F-41](findings_log.md#f-41) remain final. Validation split only; the test split is never used
> here. Nothing in this ablation can satisfy, resume, or contaminate the confirmatory sweep, its
> manifest, or its audit.

## Rationale

[F-38](findings_log.md#f-38) found the **local** objective improving where the **global** one does
not. At 1B the mask divergence (3.41%), layer gain (1.49%) and layer-objective advantage (+2.32%)
are all at or above their 160M values, while the end-to-end joint gain collapses from +1.01 pp to
+0.13 pp. At 160M the headline gain is +1.012 pp but only 7 of 8 replicates are positive.

That pattern is consistent with the joint solution containing structure the **layerwise
reconstruction objective cannot translate into language-model quality**. This ablation tests that
directly: give both arms the same short *global* recovery phase and see whether the gap moves.

## What is compared

| | |
| --- | --- |
| Model | Pythia-160M, revision `50f5173d` |
| Budget | **30% unstructured pruning + W4** weight-only, symmetric per-output-channel — the frozen headline budget, unchanged |
| Arms | **sequential P→Q** (the frozen 160M/W4 order) and **joint**, each followed by identical recovery |
| Split | **validation** |
| Replicates | **3 paired** calibration draws, reusing `CALIBRATION_REPLICATE_SEEDS` |

**This is not a way to make joint stronger.** Recovery compute and every optimisation setting are
identical between arms; only the compressed initialisation differs. If sequential catches up, that
is the finding.

## What is frozen and what is trainable

**Frozen — the pruning mask.** Applied inside every forward pass as `weight * mask`, so a pruned
position contributes nothing and receives exactly zero gradient (`d(w*m)/dw = m = 0`). No regrowth,
no reselection, no recomputation, no sparsity change. `assert_masks_still_hold` fails the run if
sparsity moves by more than `1e-9`, or if any pruned position is non-zero in the effective weight.

**Frozen — the precision.** W4 fake quantisation stays live during recovery, using the repository's
own `fake_quantise` with the same symmetric per-output-channel semantics. A straight-through
estimator carries gradients to the shadow weight; without it `round` would deliver no gradient and
recovery would run while changing nothing. `assert_fake_quantisation_ran` fails the run if any
wrapped module never took a fake-quantised forward pass, so this cannot silently become FP32
fine-tuning.

**Trainable — the shadow weight.** Full precision, off-grid between steps, re-snapped every forward.
Biases train normally; the compression pipeline targets weights only, and so does this.

**Two ordering decisions, recorded because they are choices.** `fake_quantise(weight * mask)` masks
first, so scales are fitted to what survives rather than to values about to be discarded. Scales are
**refitted every forward** rather than frozen, because the final artefact's scales are fitted from
the final weights — freezing stale ones would optimise against a grid the evaluated model does not
use.

## Recovery data is disjoint from calibration

The arms are **fitted** on the calibration sequences, so recovering on those same tokens would
partly be re-fitting on seen data: the absolute improvements would be optimistic and the comparison
would measure memorisation as much as recovery. It would still be *fair* -- both arms see identical
data -- but it would answer a weaker question.

The recovery slice is therefore drawn from the same `train` split with **every calibration index
excluded**, so overlap is zero by construction rather than merely unlikely. A fixed generator
(`recovery.seed + replicate`) makes the batches and their order byte-identical across the two arms
and across re-runs. The run logs the disjointness explicitly:

```
Recovery slice: N sequence(s) from train, DISJOINT from the 128 calibration sequences (overlap 0 by construction)
```

## Recovery objective

Standard causal-language-model cross-entropy over the **whole decoder**. Deliberately *not* the
layerwise reconstruction objective: the question is precisely whether a global objective can exploit
structure the local one leaves behind.

## Recovery compute budget

```
200 optimiser steps x 2 microbatch x 4 gradient accumulation x 512 tokens = 819,200 tokens
```

AdamW, lr 5e-5, weight decay 0.0, linear warmup 5% then cosine decay, gradient clipping 1.0, fp32
(no mixed precision), no gradient checkpointing, seed 1234.

**Why this size.** 819k tokens is ~0.0003 of Pythia-160M's 300B pretraining tokens and about 1.6x
the evaluation window — enough for a global objective to move weights measurably, far too little to
be a retraining or a QAT study. Mixed precision is off because bf16/fp16 rounding on top of a 4-bit
grid is a second error source that would be confounded with the first. `max_steps` is explicit
rather than derived from epochs so the two arms cannot diverge if their loaders ever differ in
length.

## Metrics recorded

Per arm and replicate, **before** recovery: perplexity, total NLL, retention, realised mask
sparsity, bit width, and the layerwise diagnostics. **After**: the same, plus recovery steps,
tokens, duration, optimiser, learning rate, scheduler, seed, the resolved budget, mask sparsity, and
the fake-quantised forward-call count.

Derived: `improvement_pp` per arm, and the joint gain **before** and **after** recovery, where
`joint_gain_pp = joint_retention - sequential_retention` on the same replicate.

## Interpretation rules, fixed in advance

| Observation | Reading |
| --- | --- |
| Gap grows substantially | The joint mask/grid coordination held usable structure the layerwise objective could not exploit |
| Both improve, gap similar | The ~1 pp joint benefit is probably intrinsic to this method at this budget |
| Sequential catches up, gap closes | Joint initialisation is not a durable advantage once both arms receive equal global recovery |

**No outcome is encoded anywhere in the implementation.** Three paired draws reach at best p = 0.25
on a sign test, so this ablation can show an effect size and a direction and **nothing statistical**.
It cannot meet §6.3 and must never be reported as if it could.

## Reproduction

```bash
python scripts/run_recovery_ablation.py \
  --config configs/experiments/recovery_ablation_160m_w4.yaml --smoke   # validate the path
python scripts/run_recovery_ablation.py \
  --config configs/experiments/recovery_ablation_160m_w4.yaml           # the real 3-replicate run
```

Records land in `outputs/recovery_ablation/`, tagged `exploratory`, `post-hoc`,
`recovery-ablation`, `not-confirmatory`, with `confirmatory: false` in every record.
