# Paper outline

Working title:

> **Does model scale change whether joint pruning-aware quantisation beats sequential compression?**

Target: a short empirical paper or workshop submission, 8 pages plus appendix. The contribution is a
controlled measurement, not a new compression method — so the framing throughout is "here is what
happens when you hold everything else fixed", and the honesty of the controls carries as much weight
as the numbers.

---

## Abstract (~150 words)

- Pruning and quantisation are normally applied sequentially; joint optimisation is proposed as
  better but is almost always evaluated at a single model size.
- We measure joint gain — joint minus sequential quality retention at a matched compression budget —
  across the Pythia suite from 160M to 1.4B parameters, at two budgets, over three seeds, with
  matched optimisation budgets.
- All deployment measurements are CPU-based, so we also report whether the theoretical sparsity and
  bit-width reductions translate into measured latency.
- We validate the trend on Qwen2.5-0.5B, a different family.
- One-sentence headline result. **Write this last, and write whatever the data says** — including
  "the difference is within seed noise at every scale we tested", which is a publishable answer to a
  question nobody has checked.

---

## 1. Introduction

1. Compressed small language models are deployed on CPU, where compression is not an optimisation
   but a precondition.
2. Pruning and quantisation are usually sequential. Joint optimisation is proposed as better.
3. The gap: published joint-versus-sequential results are reported at one model size. The
   practitioner's question — "at *my* model size, is the extra pipeline complexity worth it?" —
   is unanswered.
4. A second gap: compression papers report theoretical sparsity and bit-width reductions more often
   than measured latency on the hardware the model will run on.
5. Contributions:
   - a controlled measurement of joint gain across one order of magnitude of model scale, with the
     data, tokeniser, and recipe held fixed by construction
   - matched optimisation budgets, so the comparison is about method rather than training compute
   - CPU deployment measurements throughout, with the realised fraction of the theoretical speedup
     reported explicitly
   - an external validation run in a second model family
   - the code and configurations, with the fairness constraints enforced rather than documented

---

## 2. Background and related work

- **Pruning.** Magnitude pruning; gradual schedules (Zhu & Gupta); semi-structured N:M sparsity and
  the state of kernel support for it, GPU versus CPU; Wanda and SparseGPT as stronger criteria that
  this study deliberately does not use (see Limitations).
- **Quantisation.** Post-training quantisation; symmetric versus asymmetric, per-tensor versus
  per-channel; quantisation-aware training and the straight-through estimator; weight-only 4-bit
  methods.
- **Combining the two.** Sequential pipelines in practice; the small literature on joint
  sparse-and-quantised optimisation; what is claimed and at what scale.
- **Scale as a variable.** Scaling laws; the Pythia suite as a controlled ladder; work showing that
  compression sensitivity varies with model size.
- **Positioning.** We introduce no new compression method. We measure an existing design choice under
  controls that the existing literature does not apply.

---

## 3. Method

### 3.1 The five arms

Dense FP32; pruning only; quantisation only; sequential; joint. Stage diagrams for the last two, with
the ordering difference made explicit: sequential fits the quantisation grid to already-pruned
weights but cannot let the pruning decision see the grid; joint inserts fake quantisation first so it
can.

### 3.2 Joint gain

The definition, both sign conventions, and why the comparison is run on retention against each
model's own dense baseline rather than on raw perplexity.

### 3.3 Controls

The fair-comparison requirements table from [methodology.md](methodology.md), and the mechanism
enforcing each. This section is doing real work: the credibility of the result rests on it. Include
the ones that are easy to get wrong —

- matched optimiser steps between arms, recorded per stage
- identical calibration set, derived from a fixed calibration seed rather than the run seed
- embeddings and output head excluded, so the effective budget does not vary with scale
- measured sparsity reported next to target sparsity
- identical artefact serialisation between arms

### 3.4 Experimental grid

**3 Pythia sizes** (160M, 410M, 1B) × 5 arms × 2 budgets × 3 seeds. WikiText-2, 512-token
non-overlapping windows.

State plainly that the 1.4B point is optional and reported separately unless it was run under settings
identical to the main sweep — and if it was excluded, say why in this section rather than burying it in
Limitations. Full definitions of the two arms are in `docs/method_definition.md`; cite it, do not
paraphrase it, so the paper and the code cannot disagree about what was run.

---

## 4. Deployment measurement protocol

A short section, because it is a contribution rather than housekeeping.

- Everything reported is measured on one CPU: pinned threads, fixed batch and sequence length,
  warm-up, 30 runs, median and p95.
- Why CPU: it is the deployment target for this size class, and it is the only setting in which the
  "does sparsity become speed?" question has an answer.
- Weight-file-only size measurement, and `storage_efficiency` as the check that a conversion actually
  happened.
- Peak process RSS rather than an allocator statistic, and why.

---

## 5. Results

### 5.1 Baselines and single-method arms

Tables 1–3 from [experiment_protocol.md](experiment_protocol.md). Establishes that the pruning and
quantisation implementations behave sensibly before any joint claim is made.

### 5.2 Joint gain versus scale — the main result

**Figure 1:** joint gain against parameter count, log x axis, one line per budget, error bars from
the seed spread, horizontal line at zero. The Qwen point marked distinctly and excluded from any fit.

Table 6. State plainly whether the gain exceeds the seed spread at each scale.

### 5.3 Budget dependence

Whether joint gain is a scale effect, a budget effect, or both. If it appears only at the aggressive
budget, that changes the practical recommendation entirely and should be said in those terms.

### 5.4 Does sparsity become speed?

**Figure 2:** measured latency against sparsity, normalised to each model's dense baseline, with the
`1/(1-sparsity)` bound overlaid. Unstructured versus 2:4.

If unstructured sparsity yields no measured speedup — the expected outcome — report it as a finding
about the deployment path, not a limitation of the study.

### 5.5 Quality–size trade-off

**Figure 3:** retention against checkpoint size, one panel per model, series per arm. Whether the
joint arm sits above and to the left of the sequential one.

### 5.6 Training cost

**Figure 4:** joint gain against training-cost overhead. Headline comparisons are at 1.00x by
construction; the section reports the additional implementation and engineering cost separately.

### 5.7 External validation

Table 7. Whether the sign and rough magnitude transfer to Qwen2.5-0.5B. If they do not, enumerate
the candidate causes rather than choosing one.

---

## 6. Discussion

- What the trend means for a practitioner choosing a pipeline at a given model size.
- Where the joint pipeline's extra complexity is and is not justified.
- The gap between theoretical and realised compression benefit, and what would close it
  (semi-structured patterns, better kernels).
- Why a null or inconclusive result on joint gain is still useful: it bounds how much a practitioner
  should expect to gain from the extra engineering.

---

## 7. Limitations

State these plainly and early enough that a reader does not have to find them. Draw from
[validity_threats.md](validity_threats.md), which is the fuller treatment; this section is the subset a
reader needs in the paper itself.

- Three scale points (four if the extended sweep ran under comparable settings) establish a direction,
  not a scaling law. No extrapolation beyond ~1.4B.
- Scale in Pythia is not a perfectly isolated variable: depth, width, head counts, and the
  tokens-per-parameter ratio all change along the suite.
- One specific sequential implementation versus one specific joint implementation. Not a claim about
  pruning or quantisation in general.
- One evaluation corpus. Quality findings are WikiText-2 findings.
- Perplexity and agreement, not downstream task accuracy.
- CPU latency findings are specific to the framework, backend, thread count, and machine used. Sparsity
  yields no speedup without a kernel that exploits it, so a null latency result is a finding about the
  runtime as much as about the method.
- If the moderate and aggressive budgets ran on different backends, their latency and size numbers are
  not comparable with each other — say so rather than plotting them as one curve.
- Standard pruning and quantisation baselines; a stronger base method could change the gap in either
  direction.
- Matched optimiser steps is not the same as matched optimisation difficulty.
- Three seeds give a weak variance estimate. Report gains smaller than the seed spread as
  inconclusive, not as small positive effects.
- The pre-registered analysis is the trend across scale, not the maximum over cells; any cell-level
  claim is exploratory.
- One external validation model, at one scale.

---

## 8. Conclusion

Two or three sentences: the question, the measured answer, and the recommendation that follows.

---

## Appendices

- **A. Full results tables** — every arm, model, budget, and seed, unaveraged.
- **B. Configurations** — the shipped YAML, with the include structure explained.
- **C. Hardware and software** — the benchmark machine, backend, and frozen environment.
- **D. Reproduction instructions** — from [reproducibility.md](reproducibility.md).
- **E. Generation samples** — completions at each budget, showing where degeneracy sets in.
- **F. Schedule details** — sparsity ramp, mask update cadence, freeze point, LR schedule.
- **G. Method definition** — the module selection, mask scoring rule, and matched-budget requirements,
  from [method_definition.md](method_definition.md).

---

## Figure inventory

| # | Figure | Source | Answers |
| - | ------ | ------ | ------- |
| 1 | Joint gain vs scale | `plot_joint_gain_vs_scale` | primary question, secondary 1 |
| 2 | Latency vs sparsity, with bound | `plot_latency_vs_sparsity` | secondary 4 |
| 3 | Quality vs checkpoint size | `plot_quality_vs_size` | secondary 2 |
| 4 | Joint gain vs training cost | `plot_training_cost` | secondary 5 |
| — | Qwen point on Figure 1 | validation records | secondary 3 |

## Writing order

1. Method and protocol sections — they are fixed by the code and can be written before results exist.
2. Limitations — writing these early stops the results section from overclaiming.
3. Results, from the tables and figures, with no interpretation yet.
4. Discussion, then Introduction, then Abstract. The abstract is written last because the headline
   is whatever the data turned out to say.
