# Methodology

## Design

A controlled scale sweep within one model family, plus an optional external validation run in a
second family.

```
              MAIN Pythia scale sweep (controlled)          EXTENDED (optional)
    160M ────────── 410M ────────── 1B          ┊ ┈┈┈┈┈┈┈┈ 1.4B
      │               │              │          ┊            ┊
      └── 5 arms x 2 budgets x 3 seeds ─────────┘   same, but only if the
                                                     settings are identical

                    External validation (not a sweep point)
                          Qwen2.5-0.5B
                     same arms, budgets, seeds
```

### Why the Pythia suite

Model scale is only a usable independent variable if nothing else changes with it. The Pythia suite
is trained with the same data in the same order with the same recipe at every size, which is what
makes a like-for-like comparison across sizes possible. Comparing, say, GPT-2 to Llama-2-7B would
vary scale, data, tokeniser, architecture, and training budget simultaneously, and no result could
be attributed to scale.

**Main sweep: 160M, 410M, 1B.** Three points spanning roughly one order of magnitude — enough to
establish the direction of a trend, not enough to fit a scaling law.

**Extended sweep: adds 1.4B, optional and hardware-dependent.** It is separated out because joint
training at that size may not fit in available memory, and the workarounds — bf16, gradient
checkpointing, a smaller effective batch size — would each make the largest point differ from the
others in more than scale. Since it sits at the end of the trend line it has outsized influence on the
slope, which is the headline result. So: it counts as a fourth scale point only if it runs under
settings identical to the main sweep, and otherwise is reported separately and excluded from the trend.
A three-point trend with an honest footnote beats a four-point trend with a hidden confound.

One caveat about the suite itself: scale in Pythia is not a perfectly isolated scalar. Depth and width
both grow, head counts change, and the tokens-per-parameter ratio falls across the suite. See
[validity_threats.md](validity_threats.md#scale-as-an-independent-variable).

### Why Qwen2.5-0.5B for validation

The suite's internal consistency is also its limitation: a trend found within Pythia might be a
property of Pythia's data or recipe rather than of transformer compression. Qwen2.5-0.5B is a
different family, tokeniser, and corpus, and its size falls between `pythia-410m` and `pythia-1b`,
so the Pythia trend can be interpolated at its parameter count and compared.

It is **not** a sweep point. Its absolute perplexity is on a different scale from any Pythia number
because of the different tokeniser and vocabulary, so the two are never averaged or plotted on the
same quality axis. Only the sign and rough magnitude of the joint gain — both defined relative to
the model's own dense baseline — are expected to transfer.

## Variables

### Independent variable

**Model scale**, operationalised as parameter count: 162M, 405M, 1.01B, (1.41B).

### Dependent variables

| Variable            | Measurement                                                  | Device | Direction |
| ------------------- | ------------------------------------------------------------ | ------ | --------- |
| Quality retention   | perplexity relative to the model's own dense FP32 baseline   | CPU    | higher better |
| Prediction agreement| top-1 / top-5 match with the dense model, mean KL divergence | CPU    | higher better |
| Model size          | on-disk size of the weight files only                        | n/a    | lower better |
| Latency             | median and p95 per forward pass                              | CPU    | lower better |
| Throughput          | tokens per second, derived from median latency               | CPU    | higher better |
| Peak memory         | peak resident set size of the serving process                | CPU    | lower better |
| Training cost       | optimiser steps, tokens processed, wall-clock seconds        | GPU ok | lower better |

**Joint gain** is the derived quantity of interest: joint minus sequential quality retention at a
matched budget. See [research_question.md](research_question.md).

### Controlled variables

Held identical across every arm at a given model scale:

| Held fixed                  | Value / mechanism                                                       |
| --------------------------- | ----------------------------------------------------------------------- |
| Evaluation corpus           | WikiText-2 raw, validation split                                        |
| Tokenisation window         | 512 tokens, non-overlapping, final partial block dropped                 |
| Evaluation sample count     | fixed, and the token stream is fingerprinted per run                     |
| Calibration set             | same sequences in the same order; indices derive from a *fixed* calibration seed, not the run seed |
| Calibration/evaluation split| calibration is drawn from train; overlap raises an error, not a warning  |
| Sparsity target             | identical between the pruning, sequential, and joint arms                |
| Bit width                   | identical between the quantisation, sequential, and joint arms           |
| Excluded modules            | embeddings and the output head, in every arm                             |
| Optimisation budget         | equal optimiser steps for the sequential and joint arms                  |
| LR schedule shape           | linear warmup then cosine decay, identical in both arms                  |
| Seed set                    | {1234, 2345, 3456} for every cell                                        |
| Benchmark machine           | one machine per results table                                            |
| Benchmark thread count      | pinned, default 4                                                        |
| Benchmark batch / sequence  | 1 x 128                                                                  |
| Warmup / measured runs      | 5 / 30                                                                   |

### Why embeddings and the output head are excluded

Embedding parameters are a large fraction of a 160M model and a much smaller fraction of a 1.4B one.
Including them in the compressible set would make the effective compression budget vary with scale,
so a scale trend in joint gain could not be separated from a scale trend in what fraction of the
model was actually being compressed. Excluding them keeps the budget comparable across sizes.

This matters more for Qwen2.5, whose input and output embeddings are **tied**: excluding `lm_head`
without also excluding `embed_tokens` would compress the input embedding as a side effect. Both are
excluded in [qwen2_5_0_5b.yaml](../configs/models/qwen2_5_0_5b.yaml).

## Compression pipelines

### 1. Dense FP32 baseline

```
load -> evaluate (CPU) -> benchmark (CPU)
```

The reference point for every other arm at that scale. FP32 specifically, so that "quantisation to
8-bit" is a 4x precision reduction rather than 2x, and so the baseline is unambiguous.

### 2. Pruning only

```
dense -> prepare -> prune -> recovery fine-tune -> convert -> evaluate -> benchmark
```

Isolates the effect of sparsity. Magnitude criterion, cubic gradual schedule, per-layer ranking.

### 3. Quantisation only

```
dense -> prepare (observers) -> calibrate -> quantise -> convert -> evaluate -> benchmark
```

Isolates the effect of reduced precision. Post-training, no recovery, which is how weight-only
quantisation is normally deployed. Symmetric, per-channel, weight-only.

### 4. Sequential: pruning then quantisation

```
dense -> prune -> recovery fine-tune -> quantise -> convert -> evaluate -> benchmark
```

The baseline pipeline and standard practice. Quantisation is calibrated on the altered post-pruning
distribution, including the increased mass at zero and the remaining non-zero weights, so the grid is
fitted to the pruned model. Whether that distribution is *easier* to quantise is left as an empirical
question: magnitude pruning removes the smallest-magnitude weights, so the surviving values are not
necessarily confined to a narrower range and the observed min/max may be unchanged. What the ordering
definitely cannot do is the converse — the pruning decision is made with no knowledge of the
quantisation grid.

Implemented by composing `Pruner` and `Quantiser` rather than reimplementing either, so the
sequential arm cannot drift from the single-method arms it is supposed to be the composition of.

### 5. Joint: pruning-aware quantisation

```
dense
  -> fake-quantisation preparation      (observers calibrated, STE nodes inserted)
  -> gradual pruning during optimisation (masks ramped while training continues)
  -> recovery / joint fine-tuning        (one run compensating for both perturbations)
  -> final conversion                    (masks folded in, real low-precision storage)
  -> evaluate -> benchmark
```

The arm under test, and specifically **joint magnitude pruning with quantisation-aware fine-tuning** —
one implementation of a joint pipeline, not a general joint compression algorithm. Fake quantisation
goes in **before** any weight is pruned, so mask selection happens while quantisation-aware training
is active rather than on weights that have never been rounded, and a single optimisation run adapts
to both perturbations.

The mask scoring rule is a documented choice rather than a property of the method; see
[method_definition.md](method_definition.md). Nothing in this design presumes the joint arm wins: a
null or negative joint gain is a valid outcome.

## Fair-comparison requirements

Each of these has a specific mechanism, because each is a way the comparison could silently break.

### 1. Matched optimisation budget

The joint arm gets one training run covering both perturbations; the sequential arm gets one covering
only pruning. If the joint run is longer, the measured gain includes extra training.

*Mechanism:* `joint_max_steps` equals the sequential arm's `recovery.max_steps` in the shipped
configs; every stage records its optimiser steps; `match_sequential_budget` is stored in the run
record; `matched_budget()` compares two budgets programmatically.

### 2. Matched compression budget

A joint arm at 50% sparsity compared against a sequential arm at 60% is measuring sparsity, not
pipeline design.

*Mechanism:* both arms include the same `pruning.yaml` and `quantisation.yaml`; `budget_label` is
recorded and `joint_gain_summary()` carries it, so a mismatched pair is visible in the output.

### 3. Identical data

*Mechanism:* the evaluation token stream is fingerprinted and the fingerprint is stored;
`compute_retention()` refuses to compute retention across differing fingerprints. Calibration
indices derive from a fixed calibration seed, so varying the run seed for error bars does not also
change the calibration set.

### 4. Identical artefact format

If the joint arm's artefact is serialised differently from the sequential arm's, the size and
latency comparison measures the serialisation.

*Mechanism:* both conversions go through the same code path; `storage_efficiency` compares the
measured checkpoint against the size its budget implies and warns when they diverge.

### 5. Measured, not assumed, compression

A pipeline that configures 50% sparsity but leaves masks unapplied produces an excellent-looking
result.

*Mechanism:* `measure_sparsity()` counts actual zeros in the converted model and every arm's
`report_statistics()` puts the measured value next to the target.

### 6. Same benchmark conditions

*Mechanism:* thread count pinned and verified against `torch.get_num_threads()`, with a hard failure
on mismatch by default; full hardware metadata in every record; benchmark device constrained to CPU
by the config loader.

### 7. Seed repeats before conclusions

*Mechanism:* three seeds per cell; the spread is reported alongside the mean. A gain smaller than
the spread is reported as inconclusive.

## Threats to validity

Summary only. The full analysis — construct, internal, external, statistical, and backend validity —
is in [validity_threats.md](validity_threats.md).

| Threat | Mitigation | Residual risk |
| --- | --- | --- |
| Joint arm gets more effective training | matched step budgets, recorded per stage | equal steps is not identical optimisation difficulty |
| Small evaluation set makes gains noisy | 512 sequences, three seeds, spread reported | perplexity differences of <1% remain hard to resolve |
| Unstructured sparsity gives no CPU speedup | measured latency reported against the theoretical bound | a null latency result is a finding about the runtime as much as the method |
| 4-bit may need a different backend from INT8 | one backend per table; INT8 fallback documented | **decision still open**; cross-budget latency comparison invalid if unresolved |
| Only one corpus | fingerprinted and fixed; a second corpus is future work | quality findings are WikiText-2 findings |
| Three scale points (four with the extended sweep) | trend direction only, no scaling law claimed | a non-monotone trend could be missed between points |
| Largest model may need different training settings | 1.4B separated into the extended sweep, excluded from the trend unless settings match | discipline is documented, not enforced by code |
| Qwen differs in more than family | only sign and magnitude of gain compared | a transfer failure has several possible causes |
| Pruning/quantisation are standard baselines | keeps the pipeline comparison clean | a better base method could change the gap |
| Embeddings excluded | keeps the budget comparable across scale | reported ratios are lower than whole-model ratios |
| Mask scoring rule could be tuned to favour joint | rule fixed in advance, no unablated combined score | **choice still open**; must be settled before implementation |

## Related documents

- [research_question.md](research_question.md) — questions and definitions
- [method_definition.md](method_definition.md) — the exact methods, module selection, and mask scoring
- [experiment_protocol.md](experiment_protocol.md) — the run tables
- [benchmarking_protocol.md](benchmarking_protocol.md) — CPU measurement rules
- [validity_threats.md](validity_threats.md) — threats to validity, in full
- [reproducibility.md](reproducibility.md) — seeds, pins, and record contents
