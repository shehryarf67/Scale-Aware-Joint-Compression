# Threats to Validity

An honest account of what could make the results of this study wrong, misleading, or narrower than
they appear. Written before any result exists, so that the limitations are not chosen retrospectively
to fit whatever came out.

Each section states the threat, what the design does about it, and what risk remains. **Residual risk
that cannot be mitigated belongs in the paper's Limitations section**, not in a footnote.

## Construct Validity

*Does "joint gain" actually measure the benefit being claimed?*

Joint gain is defined as the quality of the joint pipeline minus the quality of the sequential
pipeline at a matched compression budget. The claim it is meant to support is "joint optimisation is
worth building". Several gaps between the measure and the claim:

- **Quality is operationalised as perplexity retention.** A pipeline that preserves perplexity may
  still change downstream behaviour. Perplexity is an average over tokens and can hide systematic
  changes on rare-but-important tokens. Prediction agreement against the dense model is measured
  alongside, precisely because it is sensitive to behavioural change that perplexity averages away —
  but neither is task accuracy.
- **A difference in quality is not the same as a difference in value.** A 0.5% perplexity retention
  gain is arithmetically real and practically irrelevant if the joint pipeline takes three times as
  long to implement and debug. The study reports training cost next to the gain, but *implementation*
  cost — engineer-hours, fragility, maintenance — is not quantified anywhere, and it is plausibly the
  dominant term in a practitioner's decision.
- **The measure is a difference of two placements on a quality–size trade-off curve, evaluated at one
  point on that curve.** Two pipelines could cross: joint better at high sparsity, sequential better at
  low. Two budgets are run partly to detect this, but two points do not characterise a curve.
- **"Matched budget" is a construct too.** Equal optimiser steps is a defensible operationalisation of
  "same training cost", but it is not the only one; equal wall-clock time or equal GPU-seconds would
  give a different answer, since the joint arm's steps are more expensive. The choice is documented in
  [method_definition.md](method_definition.md) and the alternatives are not run.
- **Retention is measured against each model's own dense baseline**, which is the right normalisation
  for comparing across scales, but it means joint gain inherits any noise in the dense measurement.

**Residual risk:** joint gain is a reasonable proxy for the claim, not a direct measure of it. The
write-up must not slide from "joint gain was positive" to "joint compression is worth adopting".

## Internal Validity

*Could something other than the compression pipeline explain a measured difference?*

This is where the study is most likely to go wrong, because most of these failures produce
plausible-looking numbers rather than errors.

### Unmatched optimiser steps

The joint arm naturally wants a longer training run. If it gets one, the measured gain is partly or
wholly the extra training.

*Mitigation:* `joint_max_steps` equals the sequential arm's `recovery.max_steps` in the shipped
configs, asserted by a test; every stage records `optimiser_steps` and `tokens_processed`;
`training_cost_overhead` must read 1.00 for a headline comparison; `match_sequential_budget` is stored
in every record.

*Residual risk:* equal steps is not equal optimisation difficulty. The joint arm solves a harder
problem in the same budget, so the matched comparison may *understate* the joint method's potential.
The asymmetry is unavoidable — an unmatched comparison overstates it — and the matched one is the
defensible choice.

### Different calibration sets

Calibration determines the quantisation scales and therefore the quality. Different calibration draws
between arms would appear as a compression-method effect.

*Mitigation:* calibration indices derive from a fixed `data.calibration_seed`, **not** the run seed, so
varying the run seed for error bars does not change the calibration set. The set is cached to disk so
every arm provably reads the same bytes. Calibration comes from the training split; overlap with the
evaluation split raises rather than warns.

*Residual risk:* if the calibration set is small (128 sequences by default) the scales it produces are
noisy in absolute terms — but identically noisy for both arms, which is what matters here.

### Different layer coverage

An arm that compresses more modules than the other shows a coverage difference, not a method
difference.

*Mitigation:* one `select_compressible_modules` code path; shared `exclude_patterns` from the shared
configs; an empty selection raises.

*Residual risk:* the joint arm's implementation could diverge during development — e.g. by resolving
modules separately for its pruning and quantisation paths. Only code review catches this; there is no
automatic assertion that the two arms saw the same module list, and adding one would be worthwhile.

### Different output formats

If the two arms serialise differently, the size and latency comparison measures serialisation.

*Mitigation:* both conversions go through the same code path; `storage_efficiency` compares measured
size against what the budget implies and warns when they diverge; `is_converted` distinguishes a real
artefact from a fake-quantised FP32 one.

### Different inference runtimes

A latency difference between backends is a property of the backends.

*Mitigation:* `quantisation.backend` is recorded per run; one backend per results table is required.

*Residual risk:* **substantial and currently unresolved.** If 4-bit needs a different backend from
INT8, the moderate and aggressive budgets are not comparable with each other even when each is
internally consistent. See [Backend Limits](#backend-limits).

### Different precision or memory-saving strategies at larger scales

The most insidious threat in a scale study: the largest model is exactly the one most likely to need
bf16 training, gradient checkpointing, a smaller batch size, or an 8-bit optimiser. Any of those makes
the largest point differ from the others in more than scale — and since it sits at the end of the
trend line, it has outsized influence on the trend's slope, which is the headline result.

*Mitigation:* the default sweep is **three models (160M, 410M, 1B)** with 1.4B moved to
[`extended_scale_sweep.yaml`](../configs/experiments/extended_scale_sweep.yaml), which states
explicitly that its result must not be mixed into the main trend if it required different settings.
Per-model configs preserve effective batch size via gradient accumulation rather than shrinking it.

*Residual risk:* the discipline is documented, not enforced by code. Nothing prevents someone running
1.4B in bf16 and plotting it on the main trend. A pre-run check comparing effective batch size,
sequence length, and dtype across a sweep's models would close this and does not yet exist.

### Placeholder implementations

*All compression algorithms are currently unimplemented.* Every threat above is stated about a
pipeline that does not yet exist, so this document will need revisiting once real code makes real
choices.

## External Validity

*How far do the conclusions generalise?*

- **One main model family.** All sweep points are Pythia. A trend within Pythia could be a property of
  Pythia's data, tokeniser, or training recipe rather than of transformer compression. This is the
  price of controlling scale properly: the same suite that makes scale a clean variable also makes the
  family a constant.
- **Optional Qwen validation.** Qwen2.5-0.5B tests transfer to a second family, and its size falls
  between `pythia-410m` and `pythia-1b` so the Pythia trend can be interpolated at its parameter count.
  Limits: it is **one** model at **one** scale, so it can detect a sign disagreement but cannot
  characterise how the trend differs across families. Its absolute perplexity is not comparable with
  Pythia's — different tokeniser and vocabulary — so only retention ratios and the sign and rough
  magnitude of the gain transfer. And a transfer *failure* has several candidate causes (family, data,
  tokeniser, tied embeddings, grouped-query attention) that a single run cannot separate.
- **Limited scale points.** Three, optionally four, spanning roughly one order of magnitude. Enough to
  establish a direction; **not enough to fit a scaling law**, and a non-monotone trend between points
  could be missed entirely. No extrapolation beyond ~1.4B is supportable.
- **Conclusions limited to decoder-only models similar to those tested.** Nothing here speaks to
  encoder-decoder models, mixture-of-experts, or models above a few billion parameters — where the
  ratio of attention to MLP parameters, the degree of over-parameterisation, and the amount of
  redundancy available to exploit all differ.
- **CPU-specific deployment results.** The latency, throughput, and memory findings are properties of
  one CPU, one thread count, one framework, and one backend. They do not transfer to GPU, to mobile
  NPUs, or even to a different CPU microarchitecture with different vector widths and cache sizes.
- **One corpus.** Quality findings are WikiText-2 findings.

## Scale as an Independent Variable

The Pythia suite is used because it holds data order, tokeniser, and training recipe fixed across
sizes, which removes the confounds that would dominate a comparison across model families. That is a
genuine and unusual strength.

**But "scale" is not a perfectly isolated scalar variable.** Increasing parameter count in Pythia
changes several things at once:

- **depth** (number of decoder blocks) and **width** (hidden size) both grow, on a schedule chosen by
  the suite's authors rather than by this study;
- **attention head count and head dimension** change with width;
- the **ratio of embedding parameters to decoder parameters** falls as models grow — which is why
  embeddings are excluded from the compressible set, but the excluded fraction still varies across the
  sweep;
- the **tokens-per-parameter ratio** differs, since all Pythia models see the same token budget;
- the **degree of over-parameterisation relative to the data** therefore differs, and this is plausibly
  the mechanism behind any compression-sensitivity trend.

So a finding of the form "joint gain grows with scale" is really "joint gain grows along the Pythia
suite's particular depth/width/data-ratio trajectory". A study that varied depth at fixed width, or
width at fixed depth, would decompose this — and is out of scope here.

Parameter count is used as the x axis because it is the quantity a practitioner knows about their own
model. That is a pragmatic choice, not a claim that parameter count is the causal variable.

## Sparsity and Hardware Speed

**Unstructured sparsity may not produce latency improvements without sparse kernels.** A dense GEMM
kernel performs the same multiply-accumulates whether or not the operands are zero; the zeros are
multiplied, not skipped. A speedup requires a kernel that exploits the sparsity pattern, which is a
property of the runtime, not of the compression method.

Consequences for this study:

- A near-zero `sparsity_realisation` for unstructured sparsity is the **expected** result and a
  legitimate finding about the deployment path — not a bug and not a failure of the pruning method.
- Semi-structured patterns (2:4, 4:8) are the ones most likely to admit a sparsity-exploiting kernel,
  and a 2:4 variant is run for that reason. But **CPU support for semi-structured sparsity is
  considerably less established than the GPU equivalent**, and the installed backend may provide no
  such kernel at all. Whether it does must be verified and recorded rather than assumed; otherwise the
  2:4 row measures a dense kernel operating on a patterned weight matrix.
- Size reduction and latency reduction are therefore **separate findings** that must not be conflated.
  A 4× smaller checkpoint that runs at the same speed is a real and reportable outcome.
- Reported theoretical bounds (`1/(1 - sparsity)`) are optimistic ceilings assuming a perfect
  zero-skipping kernel at no overhead. They are printed next to measured values to make the gap
  visible, not as predictions.

## Dataset and Evaluation Limits

- **A single corpus.** WikiText-2 validation only. Compression effects are known to be
  distribution-dependent, and a method that preserves quality on encyclopaedic English may degrade on
  code, dialogue, or another language.
- **Perplexity as the primary metric.** It is well-defined identically across every model in the sweep
  and needs no task head, which is exactly why it was chosen — but it is an average over tokens, it is
  not what anyone deploys a model to do, and small perplexity differences do not map linearly onto
  usefulness. Prediction agreement is measured alongside for behavioural change; neither is downstream
  task accuracy.
- **No downstream tasks.** No zero-shot benchmark suite, no instruction-following evaluation, no
  generation quality judgement beyond degeneracy diagnostics. A compression method could be
  perplexity-neutral and still damage in-context learning.
- **Evaluation set size.** 512 sequences by default. Enough for a stable perplexity estimate; the
  binding constraint on resolving small differences is seed variance rather than evaluation noise, but
  both contribute.
- **A fixed evaluation window** (512 tokens, non-overlapping). A sliding window with a smaller stride
  would give lower, more favourable perplexities; the setting is held constant and recorded so runs are
  comparable, but the absolute values are not comparable with papers using a different protocol.
- **Generation diagnostics are qualitative.** Repetition rate and distinct-token ratio catch collapse,
  not subtle quality loss.

## Statistical Limits

- **Number of seeds.** Three per cell. That gives a crude spread estimate, not a confidence interval
  worth the name. Three samples cannot distinguish a heavy-tailed distribution from a well-behaved one,
  and the standard deviation of three values is itself a high-variance quantity.
- **Variance sources.** Seed variance conflates recovery-training stochasticity, batch ordering, and
  (on GPU) non-deterministic kernels. Calibration is deliberately excluded from this by using a fixed
  calibration seed, so the reported spread understates total pipeline variance.
- **Joint gains smaller than seed variation.** The most likely outcome at the moderate budget, where a
  well-tuned sequential pipeline loses little and leaves little room for improvement. Such a result
  **must be reported as inconclusive** — not as a small positive effect, and not as evidence of no
  effect. The distinction matters: "we could not resolve a difference at this sample size" is a
  different statement from "there is no difference".
- **Multiple comparisons.** With models × budgets × arms, the study makes many implicit comparisons.
  At three models × two budgets, examining six joint-gain values and reporting the largest as the
  finding would be a straightforward multiple-comparisons error. The pre-registered analysis is the
  *trend across scale*, not the maximum over cells; any cell-level claim needs correction or explicit
  labelling as exploratory.
- **Trend fitting on three points.** A slope through three points with error bars is barely
  identifiable. Report the direction and the overlap of error bars; do not report a fitted exponent, an
  R², or a p-value on a scaling relationship.
- **Practical versus statistical significance.** These come apart in both directions here. A gain that
  survives the seed spread may still be too small to justify the joint pipeline's engineering cost; and
  a gain too small to resolve at three seeds may still be worth having if it were free. Both quantities
  should be reported side by side, and neither substituted for the other.
- **No pre-registered effect size.** The study does not state in advance what magnitude of joint gain
  would count as practically meaningful. It should, before results are read.

## Backend Limits

**INT8 and 4-bit may require different backend support and therefore may not be directly comparable.**

- PyTorch's native CPU quantisation path targets INT8. There is no equally mature built-in 4-bit
  weight-only CPU kernel, so 4-bit deployment generally requires a separate backend — a packed-weight
  custom linear module, or an external runtime.
- If the moderate budget (INT8) and the aggressive budget (4-bit) run on different runtimes or emit
  different artefact formats, **their latency and size numbers are not comparable with each other**.
  Each may be internally valid while the cross-budget comparison is meaningless.
- The same 4-bit path must be available to **both** arms. A 4-bit joint artefact compared against an
  INT8 sequential artefact is not a joint-gain measurement at all.
- Kernel maturity differs even within INT8: per-channel versus per-tensor, and `x86` versus `fbgemm`
  versus `qnnpack`, have different performance characteristics. The backend is recorded per run for
  this reason.

*Mitigation:* a documented INT8 fallback — moderate at 50% sparsity + INT8, aggressive at 70% sparsity
+ INT8 — keeps every row on one runtime and one artefact format, at the cost of making precision a
constant rather than a second compression axis. See
[method_definition.md](method_definition.md#bit-widths-and-the-4-bit-risk).

*Residual risk:* **this decision is still open** and it constrains the scope of the whole study. It
must be made before the main experiments, not after seeing which choice produces a nicer result.

## The Joint Mechanism May Be Inert at Moderate Precision

**Discovered 2026-07-28 on implementing Phase 6, then investigated on real layers. This is the most
serious threat currently known, because it predicts a weak result for structural reasons rather than
empirical ones.**

§3.8 lists two things that make a method joint: the mask is scored under quantised weights, and the
quantisation scales are re-estimated after the mask changes. Both are weak at the precisions the
study plans to use.

### Measurements on six real Pythia-160M layers

Layers 0, 5 and 11, attention and MLP, against the real WikiText calibration set at 50% sparsity.
An earlier version of this section quoted a synthetic layer; the real numbers replace it.

| Bits | Joint vs sequential mask differs | Channels whose max-abs scale moves on refit |
| --- | --- | --- |
| W8 | **0.46%** | 0.2% |
| W4 | **8.86%** | 0.2% |
| W3 | 15.42% | 0.2% |
| W2 | 45.54% | 0.2% |

**The mask mechanism is live at W4 but effectively inert at W8.** The synthetic layer understated
this — it showed *zero* divergence at W4, where real layers show 8.86%. Real weights have heavier
tails and real activations have outlier channels, both of which let rounding reorder the saliency.
At W8 quantisation error is small enough that the ranking survives intact.

**The scale mechanism is inert everywhere.** A symmetric per-channel scale is `max|W_row| / qmax`,
and saliency pruning removes the smallest entries, so each row's maximum almost always survives.

One correction, and it matters for how strongly this can be stated: this is **not** a proof. Because
the saliency is activation-*weighted*, a row's largest weight can be pruned when it sits on a
low-energy input column — and it does happen, at 1.3% of channels in layer 11's MLP. The claim is
empirical, not algebraic.

### Two proposed fixes, both measured, both rejected

Layer-objective joint gain, mean over the same six layers, matched budget. Positive means the joint
arm reconstructs better than sequential:

| Configuration | W8 | W4 |
| --- | --- | --- |
| max-abs scales, magnitude scoring (**current default**) | −0.49% | **+1.12%** |
| + error-minimising clipping scale search | −1.51% | −0.99% |
| + quantisation-aware keep-benefit scoring | −11.83% | **−16.15%** |

**Error-minimising clipping scale search** does exactly what it promises in isolation: it cuts naive
quantisation error by 12.8% at W4, and it makes scale re-estimation genuinely mask-dependent — 70%
of channels move their grid on refit, against 0.2% for max-abs. But the layer gets *worse* after
reconstruction. The two objectives are different: clipping saturates outliers, and a saturated weight
cannot be repaired by error compensation. Optimising pre-reconstruction error is the wrong target.

**Keep-benefit scoring** `B_ij = ||X_j||²[W_ij² − (W_ij − Q(W_ij))²]` is worse still, and the reason
is analytic. For round-to-nearest symmetric quantisation the score is bounded below by zero, and for
any weight above the step size `(W − Q(W))²` is nearly independent of `W`, leaving
`B ≈ ||X_j||²·W_ij²` minus a near-constant — a monotone transform of activation-weighted magnitude.
So it largely *reproduces* the magnitude ranking, and where it deviates it prefers weights that
happen to sit near a grid point, which is a property of the current scale rather than a statement
about importance.

Both remain implemented behind `compression.reconstruction.scale_search` and
`.keep_benefit_saliency`, defaulting to off. "The obvious quantisation-aware criterion is worse than
magnitude" is a reportable ablation, not a dead end.

### Where this leaves the study

- **W4 is the budget where the mechanism is live** (8.86% mask divergence, +1.12% layer gain). It
  should carry the headline comparison.
- **W8 should be read as a control.** Near-zero joint gain there is the expected result when
  quantisation error is small, not a failure — and reporting it as a scale-independent null would be
  a misreading.
- **Do not move to W2 to chase a larger effect.** The mechanism is far more active there, but
  choosing a precision because it produces a positive result is exactly the selection §6.3 forbids.
- A criterion that respects error compensation would need the inverse-Hessian term rather than a
  diagonal approximation. That is a genuine research direction, not a configuration change.

Every measurement above is pinned by tests in `tests/test_layerwise.py`, so none of it can regress
silently or be rediscovered during writing-up.

## Summary of unresolved risks

| Risk | Status |
| --- | --- |
| **Joint mechanism weak: mask inert at W8 (0.46% divergence), scale re-estimation inert everywhere (0.2%)** | **characterised — W4 carries the comparison, W8 is a control; two candidate fixes measured and rejected** |
| CPU quantisation backend undecided | settled — PyTorch native INT8, engine `onednn` (D1) |
| Whether 4-bit stays in the main study or the INT8 fallback is used | settled — W4 for quality and size, never for latency (D1) |
| Mask scoring rule not finalised | settled — activation-weighted magnitude on quantised weights (D3) |
| No automatic check that both arms saw the same module list | closed — `assert_matched_plans` checks coverage, calibration and local steps |
| No automatic check that a sweep's models share training settings | gap in tooling |
| No pre-registered practically-meaningful effect size | should be set before reading results |
| Three seeds give a weak variance estimate | accepted; report inconclusive results as inconclusive |
| Three scale points cannot support a scaling law | accepted; report direction only |
| Single corpus, perplexity-centric evaluation | accepted; state in Limitations |
| Unstructured sparsity may yield no CPU speedup | expected; report as a deployment-path finding |

## Related documents

- [research_question.md](research_question.md) — the questions and what the study does not claim
- [method_definition.md](method_definition.md) — exactly what the two arms are
- [methodology.md](methodology.md) — variables, controls, fair-comparison mechanisms
- [benchmarking_protocol.md](benchmarking_protocol.md) — CPU measurement rules
- [reproducibility.md](reproducibility.md) — seeds, pins, promotion checklist
