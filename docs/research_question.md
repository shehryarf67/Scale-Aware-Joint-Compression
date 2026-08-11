# Research question

> # ⚠️ SUPERSEDED IN PART — read [F-37](findings_log.md#f-37) first
>
> **Updated 2026-08-11.** This document was written **before** the confirmatory run. A1 step 10 has
> since been executed once on the test split and the study has its answer:
> **[F-37](findings_log.md#f-37)** (the result), **[F-38](findings_log.md#f-38)** (mechanism
> diagnostics), **[limitations.md](limitations.md)** (what the result may not claim).
>
> **Specifically stale below, and superseded rather than corrected in place** — this file records
> what was *planned*, which is worth keeping legible:
>
> - **The budget grid.** 50% and 70% sparsity appear here as planned budgets. They were screened and rejected; every reported number is **30%** sparsity at W8 or W4.
> - **The scale axis is confounded with depth.** pythia-1b has **16 blocks against pythia-410m's 24** ([F-38](findings_log.md#f-38)), so "scale" is not a clean single factor across the sweep.
> - **The seed-spread clause is withdrawn** as vacuous; see [protocol_freeze.md](protocol_freeze.md#the-amended-practical-importance-rule).
> - **2:4 / structured sparsity was never run** end to end.
> - **The primary question now has an answer:** the advantage did **not** increase with scale, the observed direction was opposite, and **the decline is not statistically established**.
>
> The authoritative statement of what the paper may and may not claim is **§6 of
> [findings_log.md](findings_log.md)**. Where this file and that section disagree, that section wins.

## Primary question

> **How does model scale influence the effectiveness of joint versus sequential pruning and
> quantisation in decoder-only language models?**

Pruning and quantisation are usually applied in sequence: prune the model, fine-tune to recover
the lost quality, then quantise the result. The alternative is to optimise for both objectives at
once, so that the pruning criterion is aware of the quantisation grid and a single optimisation
run compensates for both perturbations together.

Joint optimisation is more expensive and more fragile to implement. The published evidence for it
is almost always reported at a single model size, which leaves the practical question unanswered:
if a research group has a fixed compute budget and a model of a particular size, is the joint
pipeline worth building?

This study measures the difference between the two pipelines at several model sizes, under
identical conditions, and reports whether that difference grows, shrinks, or stays flat as models
get larger.

## The quantity being measured

**Joint gain** is the quality of the joint pipeline minus the quality of the sequential pipeline at
a matched compression budget:

```
joint_gain = joint_quality_score - sequential_quality_score
```

Positive means joint won. Where the metric is lower-is-better (perplexity, or quality lost relative
to dense), the subtraction is reversed so that positive still means joint won:

```
joint_gain = sequential_quality_loss - joint_quality_loss
```

Both conventions are implemented in
[joint_gain.py](../src/scale_aware_compression/metrics/joint_gain.py).

The score used is **perplexity retention relative to each model's own dense FP32 baseline**, not
raw perplexity. Absolute perplexity falls as models get larger, so a gain measured in raw
perplexity points would trend with scale for reasons having nothing to do with compression.

A comparison is only valid when the two arms share:

- the same model and the same dense baseline
- the same compression budget (sparsity target *and* bit width)
- the same evaluation data, tokenisation window, and calibration set
- the same seed
- the same optimisation budget, in optimiser steps

The last is the easiest to get wrong and the most damaging. The joint arm naturally wants a longer
training run, and if it gets one, the measured gain is confounded with extra training. See
`compression.joint.match_sequential_budget` in
[joint.yaml](../configs/compression/joint.yaml).

## Secondary questions

### 1. Does joint gain increase as model size increases?

The primary question, stated as a testable trend. With three or four Pythia sizes the sweep can
distinguish a monotone trend from a flat one, but it cannot fit a scaling law — three points do not
support extrapolation, and the study does not claim any.

**How it is answered:** joint gain per (model, budget), averaged over seeds, plotted against
parameter count on a log axis. The seed-to-seed spread is the error bar. A gain smaller than that
spread is not a finding.

**Possible outcomes, all reportable:**

- gain grows with scale — joint optimisation matters more for larger models
- gain is flat — the pipeline choice is scale-independent, so pick on engineering cost
- gain shrinks — larger models have enough redundancy that the ordering stops mattering
- gain is within noise at every scale — the honest answer is that this experiment cannot
  distinguish the two pipelines, which is itself worth publishing

### 2. At which compression budgets does joint optimisation become useful?

At mild compression a well-tuned sequential pipeline loses very little quality, leaving little room
for a joint pipeline to improve on. At aggressive compression both degrade, and the question is
whether joint degrades more slowly.

**How it is answered:** two budgets per model — moderate (50% sparsity, 8-bit) and aggressive
(70% sparsity, 4-bit). If joint gain is near zero at moderate and clearly positive at aggressive,
the answer is a budget threshold rather than a scale effect, and that distinction changes the
practical recommendation entirely.

### 3. Does the observed trend transfer from Pythia to Qwen?

The Pythia sweep controls model scale properly, because the suite holds data order, tokeniser, and
recipe fixed across sizes. That control is also its limitation: a trend found within Pythia might be
a property of Pythia.

**How it is answered:** the same arms, budgets, and seeds on Qwen2.5-0.5B, whose size falls between
`pythia-410m` and `pythia-1b`. The measured joint gain is compared against the Pythia trend
interpolated at that parameter count. Only the *sign* and rough magnitude are expected to agree:
Qwen has a different tokeniser and corpus, so its absolute perplexity is on a different scale and
is never compared with Pythia's. See
[validation.py](../src/scale_aware_compression/experiments/validation.py).

### 4. Does theoretical sparsity produce real CPU latency improvements?

A model reported at 70% sparsity and 4-bit weights implies roughly a 13x size reduction and, if the
sparsity were fully exploited, a 3.3x speedup. On a dense GEMM kernel it should be expected to
deliver none of that speedup: the multiply-accumulates still happen, they just multiply by zero.
Realising a gain from sparsity requires a kernel that exploits the pattern, which is a property of
the runtime rather than of the compression method.

**How it is answered:** every arm is benchmarked on CPU with a pinned thread count, and the measured
latency is reported next to the theoretical bound `1 / (1 - sparsity)`. The ratio between them is
`sparsity_realisation` in
[efficiency.py](../src/scale_aware_compression/metrics/efficiency.py). Running both unstructured and
2:4 semi-structured sparsity — the latter being the pattern most likely to admit such a kernel —
separates the effect of the pattern from that of the runtime. Whether the installed CPU backend
provides a 2:4 kernel at all must be verified and recorded, not assumed: CPU support for
semi-structured sparsity is considerably less established than the GPU equivalent.

This question is why the deployment measurements are CPU-only. A GPU latency number would not answer
it, and neither would a theoretical FLOP count.

### 5. How much additional training cost does joint optimisation require?

A quality gain that costs three times the training compute is a different recommendation from the
same gain at equal cost.

**How it is answered:** every stage records its optimiser steps, tokens processed, and wall-clock
time. `training_cost_overhead` reports the ratio of joint to sequential cost. The headline
comparison is run at a *matched* budget, so the overhead figure covers the additional
implementation and engineering cost separately from any accuracy-per-step advantage.

## What this study does not claim

- **No scaling law.** Three or four points establish a direction, not a functional form.
- **No claim beyond ~1.4B parameters.** Nothing here supports extrapolation to 7B or beyond.
- **No claim about GPU deployment.** The latency findings are CPU findings.
- **No claim about downstream task accuracy.** Perplexity and prediction agreement are the quality
  metrics. A perplexity-neutral compression can still change downstream behaviour.
- **No claim of state-of-the-art compression.** The pruning and quantisation methods are standard
  baselines, chosen so that the *comparison between pipelines* is clean. A better pruning criterion
  would raise both arms and might change the gap either way.

## Related documents

- [method_definition.md](method_definition.md) — exactly what the two arms are
- [methodology.md](methodology.md) — variables, controls, and fair-comparison requirements
- [experiment_protocol.md](experiment_protocol.md) — the run tables
- [benchmarking_protocol.md](benchmarking_protocol.md) — CPU measurement rules
- [validity_threats.md](validity_threats.md) — what could still make these results wrong
- [reproducibility.md](reproducibility.md) — seeds, pins, and record contents
- [paper_outline.md](paper_outline.md) — how the results become a write-up
