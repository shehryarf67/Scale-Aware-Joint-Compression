# Experiment protocol

Every table below is a **placeholder**: the settings are the planned defaults and the result columns
are empty until runs are executed. Nothing in this repository has produced a number yet.

Fill the result columns from `outputs/metrics/results.csv`, not by hand.

## Notation

- **Sparsity** — target fraction of prunable weights set to zero. Excludes embeddings and the output
  head; see [methodology.md](methodology.md).
- **Bits** — bits per stored weight after conversion. 32 means unquantised.
- **Steps** — optimiser steps, the training-cost unit compared between arms.
- **Retention** — perplexity retention against that model's own dense FP32 baseline, in percent.
  Higher is better; 100 means no loss.
- **Latency** — median wall-clock milliseconds per forward pass, CPU, 4 threads, batch 1,
  sequence 128.
- **Size** — on-disk size of the weight files only, MiB.

## Compression budgets

| Budget       | Sparsity | Bits | Theoretical size reduction | Rationale |
| ------------ | -------- | ---- | -------------------------- | --------- |
| `moderate`   | 0.50     | 8    | ~8x                        | A well-tuned sequential pipeline should lose little here, so any joint gain is small and needs seed repeats to be credible. |
| `aggressive` | 0.70     | 4    | ~27x                       | Quality degrades enough that the arms should separate. Where the "at which budgets does joint help?" question is answered. **Subject to the backend decision below.** |
| `pilot`      | 0.50     | 8    | ~8x                        | Reduced step and sample counts for pipeline validation. **Not a results budget.** |

Theoretical reduction is `32 / (bits * (1 - sparsity))` and assumes a storage format that actually
exploits both. The measured checkpoint size is what gets reported; see `storage_efficiency`.

### Backend decision required before running the aggressive budget

- PyTorch's native CPU quantisation support is strongest for **INT8**.
- **4-bit weight-only CPU deployment may require a separate backend** (a packed-weight custom linear
  module, or an external runtime); there is no equally mature built-in 4-bit CPU kernel.
- **Latency and size results are not comparable if the moderate and aggressive settings use different
  runtimes or artefact formats.** The same applies across arms with more force: a 4-bit joint artefact
  measured against an INT8 sequential artefact is not a joint-gain measurement.
- **Decide the final backend before the main experiments start**, not after seeing results.

Fallback if a single 4-bit CPU path cannot be implemented for both arms:

| Budget                  | Sparsity | Bits | Theoretical size reduction |
| ----------------------- | -------- | ---- | -------------------------- |
| `moderate`              | 0.50     | 8    | ~8x                        |
| `aggressive` (fallback) | 0.70     | 8    | ~13x                       |

The fallback keeps every row on one runtime and one artefact format. It makes precision a constant
rather than a second compression axis, and narrows the compression-ratio range — both of which must be
stated in the write-up if it is used. 4-bit stays in the configuration system regardless; see
[method_definition.md](method_definition.md#bit-widths-and-the-4-bit-risk).

## Sweep scope

The **main** sweep is three models: `pythia-160m`, `pythia-410m`, `pythia-1b`
([`main_scale_sweep.yaml`](../configs/experiments/main_scale_sweep.yaml)).

`pythia-1.4b` is **optional and hardware-dependent**, and lives in
[`extended_scale_sweep.yaml`](../configs/experiments/extended_scale_sweep.yaml). Rows for it appear in
the tables below so the protocol is complete, but they count as a fourth scale point **only if the run
used settings identical to the main sweep** — same precision, sequence length, effective batch size,
optimiser steps, and no memory-saving technique the smaller runs did not also use. If any of those had
to change, report the 1.4B result separately and exclude it from the scale trend; a three-point trend
with an honest footnote beats a four-point trend with a hidden confound.

Run `sajc sweep --config <file> --plan-only` for the exact run count of any sweep. The plan is the
authority, not these tables.

Throughout the tables below, **`pythia-1.4b*`** marks a row that is part of the extended sweep only:
optional, hardware-dependent, and valid as a scale point only under comparable settings.

---

## Table 1 — Dense FP32 baseline

One run per model. Every other row in the study is measured against its own model's baseline.

| Model | Params | Seed | Perplexity | Latency (ms) | p95 (ms) | Throughput (tok/s) | Peak mem (MiB) | Size (MiB) |
| ----- | ------ | ---- | ---------- | ------------ | -------- | ------------------ | -------------- | ---------- |
| pythia-160m  | 162M  | 1234 | — | — | — | — | — | — |
| pythia-410m  | 405M  | 1234 | — | — | — | — | — | — |
| pythia-1b    | 1.01B | 1234 | — | — | — | — | — | — |
| pythia-1.4b*  | 1.41B | 1234 | — | — | — | — | — | — |
| qwen2.5-0.5b | 494M  | 1234 | — | — | — | — | — | — |

Notes:

- FP32, no compression, `eval_mode: true`.
- One seed is sufficient: with no training and greedy evaluation the run is deterministic. Seeding
  is still recorded.
- Qwen's perplexity is on a different scale from Pythia's and is never compared with it.

Command:

```bash
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml
```

---

## Table 2 — Pruning only

Isolates sparsity. Magnitude criterion, cubic gradual schedule, per-layer ranking, recovery enabled.

| Model | Budget | Sparsity | Granularity | Steps | Measured sparsity | Retention (%) | Latency (ms) | Size (MiB) |
| ----- | ------ | -------- | ----------- | ----- | ----------------- | ------------- | ------------ | ---------- |
| pythia-160m | moderate   | 0.50 | unstructured | 500 | — | — | — | — |
| pythia-160m | aggressive | 0.70 | unstructured | 500 | — | — | — | — |
| pythia-410m | moderate   | 0.50 | unstructured | 500 | — | — | — | — |
| pythia-410m | aggressive | 0.70 | unstructured | 500 | — | — | — | — |
| pythia-1b   | moderate   | 0.50 | unstructured | 500 | — | — | — | — |
| pythia-1b   | aggressive | 0.70 | unstructured | 500 | — | — | — | — |
| pythia-1.4b* | moderate   | 0.50 | unstructured | 500 | — | — | — | — |
| pythia-1.4b* | aggressive | 0.70 | unstructured | 500 | — | — | — | — |

**Measured sparsity must match the target.** A gap means masks were not applied, or the optimiser
refilled pruned positions.

**Expect little or no latency improvement from unstructured sparsity.** A dense GEMM kernel performs
the same multiply-accumulates whether or not the operands are zero, so a speedup requires a kernel
that exploits the sparsity pattern. That is a finding to report, not a bug — and it is why the 2:4
variant below exists.

### Table 2b — Semi-structured 2:4 sparsity

Same models, `granularity: "2:4"`, which fixes sparsity at 0.50 by definition. 2:4 is the pattern
most likely to admit a sparsity-exploiting kernel, so this table is what tests whether *any* sparsity
pattern converts into measured latency on the target backend.

**Verify before reading this table:** CPU support for semi-structured sparsity is much less
established than the GPU equivalent, and the installed backend may provide no 2:4 kernel at all. If
it does not, the row measures a dense kernel operating on a 2:4-patterned weight matrix, and the
correct conclusion is about the deployment path rather than about the pruning pattern. Record which
kernel was actually used.

| Model | Sparsity | Granularity | Retention (%) | Latency (ms) | Speedup vs dense | Sparsity realisation |
| ----- | -------- | ----------- | ------------- | ------------ | ---------------- | -------------------- |
| pythia-160m | 0.50 | 2:4 | — | — | — | — |
| pythia-410m | 0.50 | 2:4 | — | — | — | — |
| pythia-1b   | 0.50 | 2:4 | — | — | — | — |

Command:

```bash
python scripts/run_pruning.py --config configs/experiments/pilot.yaml \
  --override compression.pruning.granularity=2:4
```

---

## Table 3 — Quantisation only

Isolates precision. Post-training, weight-only, symmetric, per-channel, 128 calibration samples, no
recovery.

| Model | Budget | Bits | Scheme | Granularity | Calib samples | Retention (%) | Latency (ms) | Size (MiB) | Compression ratio |
| ----- | ------ | ---- | ------ | ----------- | ------------- | ------------- | ------------ | ---------- | ----------------- |
| pythia-160m | moderate   | 8 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-160m | aggressive | 4 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-410m | moderate   | 8 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-410m | aggressive | 4 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-1b   | moderate   | 8 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-1b   | aggressive | 4 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-1.4b* | moderate   | 8 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-1.4b* | aggressive | 4 | symmetric | per_channel | 128 | — | — | — | — |

**Check `is_converted` and `storage_efficiency` before reading these rows.** A model that was
fake-quantised but never converted is numerically quantised and still FP32 on disk, which produces
correct-looking retention with a meaningless size and latency.

---

## Table 4 — Sequential (prune → recover → quantise)

The baseline pipeline. Stages: dense → prune → recovery → quantise → convert.

| Model | Budget | Sparsity | Bits | Recovery steps | Retention (%) | Latency (ms) | Size (MiB) | Compression ratio |
| ----- | ------ | -------- | ---- | -------------- | ------------- | ------------ | ---------- | ----------------- |
| pythia-160m | moderate   | 0.50 | 8 | 500 | — | — | — | — |
| pythia-160m | aggressive | 0.70 | 4 | 500 | — | — | — | — |
| pythia-410m | moderate   | 0.50 | 8 | 500 | — | — | — | — |
| pythia-410m | aggressive | 0.70 | 4 | 500 | — | — | — | — |
| pythia-1b   | moderate   | 0.50 | 8 | 500 | — | — | — | — |
| pythia-1b   | aggressive | 0.70 | 4 | 500 | — | — | — | — |
| pythia-1.4b* | moderate   | 0.50 | 8 | 500 | — | — | — | — |
| pythia-1.4b* | aggressive | 0.70 | 4 | 500 | — | — | — | — |

---

## Table 5 — Joint (pruning-aware quantisation)

The arm under test. Stages: dense → fake-quantisation prep → gradual pruning during optimisation →
joint fine-tune → convert.

| Model | Budget | Sparsity | Bits | Joint steps | Mask updates | Retention (%) | Latency (ms) | Size (MiB) | Cost vs sequential |
| ----- | ------ | -------- | ---- | ----------- | ------------ | ------------- | ------------ | ---------- | ------------------ |
| pythia-160m | moderate   | 0.50 | 8 | 500 | — | — | — | — | 1.00x |
| pythia-160m | aggressive | 0.70 | 4 | 500 | — | — | — | — | 1.00x |
| pythia-410m | moderate   | 0.50 | 8 | 500 | — | — | — | — | 1.00x |
| pythia-410m | aggressive | 0.70 | 4 | 500 | — | — | — | — | 1.00x |
| pythia-1b   | moderate   | 0.50 | 8 | 500 | — | — | — | — | 1.00x |
| pythia-1b   | aggressive | 0.70 | 4 | 500 | — | — | — | — | 1.00x |
| pythia-1.4b* | moderate   | 0.50 | 8 | 500 | — | — | — | — | 1.00x |
| pythia-1.4b* | aggressive | 0.70 | 4 | 500 | — | — | — | — | 1.00x |

The cost column must read 1.00x for a headline comparison. Anything else means the arms were not
budget-matched and the corresponding joint gain is confounded with extra training.

---

## Table 6 — Joint gain versus scale (the result)

> ⚠️ **This table's columns are superseded.** It was written against the run-seed axis, which
> Amendment A1 §5.1 **withdrew**: the pipeline is deterministic post-training reconstruction, so two
> runs at different run seeds are bit-identical ([F-15](findings_log.md#f-15)) and the "seed spread"
> column is exactly **zero** for every cell. A gate against zero passes for any nonzero gain.
>
> **Current form:** matched on model, budget and **calibration replicate**, with the per-replicate
> gains listed individually, their mean and sd, the count positive, ties counted separately, an exact
> sign-test p over the non-tied replicates, and **R reported per cell** together with whether
> significance was reachable at that R. Produced by `metrics.replicates.summarise_replicates`. See
> [protocol_freeze.md](protocol_freeze.md#the-amended-practical-importance-rule).

Derived from Tables 4 and 5, matched on model, budget, and **calibration replicate**.

| Model | Params | Budget | Sequential retention | Joint retention | Joint gain | Per-draw gains | R | Sign-test p | Reachable? |
| ----- | ------ | ------ | -------------------- | --------------- | ---------- | -------------- | - | ----------- | ---------- |
| pythia-160m  | 162M  | moderate   | — | — | — | — | — |
| pythia-410m  | 405M  | moderate   | — | — | — | — | — |
| pythia-1b    | 1.01B | moderate   | — | — | — | — | — |
| pythia-1.4b*  | 1.41B | moderate   | — | — | — | — | — |
| pythia-160m  | 162M  | aggressive | — | — | — | — | — |
| pythia-410m  | 405M  | aggressive | — | — | — | — | — |
| pythia-1b    | 1.01B | aggressive | — | — | — | — | — |
| pythia-1.4b*  | 1.41B | aggressive | — | — | — | — | — |

**A gain whose sign is not consistent across replicates is inconclusive and must be reported as
such.** The superseded form of this sentence compared the gain to the *seed* spread, which is zero —
so it excluded nothing. Sign consistency across paired calibration draws is the measurable
replacement, and F-26 is the case it would have caught: three draws spanning −0.50 to +0.98 pp,
reported as a point estimate of +0.68.

---

## Table 7 — External validation (Qwen2.5-0.5B)

Same arms, budgets, and seeds. Compared against the Pythia trend interpolated at 494M parameters,
which falls between `pythia-410m` and `pythia-1b`.

| Model | Params | Budget | Sequential retention | Joint retention | Joint gain | Expected from Pythia trend | Sign agrees? | Transfers? |
| ----- | ------ | ------ | -------------------- | --------------- | ---------- | -------------------------- | ------------ | ---------- |
| qwen2.5-0.5b | 494M | moderate   | — | — | — | — | — | — |
| qwen2.5-0.5b | 494M | aggressive | — | — | — | — | — | — |

Only the sign and rough magnitude are expected to agree. Absolute perplexity is not comparable
across tokenisers.

---

## Execution order

Order is not arbitrary. Every non-dense arm needs its model's dense perplexity as the retention
reference, so a model's dense run must complete first.

```
0. pilot.yaml                       -> pipeline validation only; produces no results

main sweep, for each model, smallest first:
    1. dense baseline               -> establishes the retention reference
    2. pruning only                 -> isolates sparsity
    3. quantisation only            -> isolates precision
    4. sequential                   -> the baseline pipeline
    5. joint                        -> the arm under test
    (repeat 2-5 per budget, per seed)
then:
    6. Qwen validation
    7. extended sweep (1.4B)        -> OPTIONAL, only if the main sweep succeeded and the
                                       settings can be held identical
    8. aggregate, figures, tables
```

Run the pilot config end to end before starting the sweep. A pipeline bug found at pythia-1b costs
hours; the same bug found at pythia-160m costs minutes. The pilot is a pipeline-validation run: its
numbers are not results and must never appear in the write-up.

```bash
# validate configuration without running anything
python scripts/run_sequential.py --config configs/experiments/pilot.yaml --dry-run

# pipeline validation: minutes, not hours
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml
python scripts/run_sequential.py     --config configs/experiments/pilot.yaml
python scripts/run_joint.py          --config configs/experiments/pilot.yaml

# see the sweep expansion before committing compute
sajc sweep --config configs/experiments/main_scale_sweep.yaml --plan-only
```

## Pre-run checklist

- [ ] **CPU quantisation backend chosen and recorded** (see the backend decision above)
- [ ] **4-bit-versus-INT8-fallback decision made**, before any main run
- [ ] **Mask scoring rule fixed** in [method_definition.md](method_definition.md#mask-scoring)
- [ ] Model revisions pinned to commit SHAs in the model configs (pilot runs may leave them unpinned)
- [ ] Pilot config completes end to end
- [ ] Dense baselines exist for every model in the sweep
- [ ] `budget_overrides` identical between the sweep and the validation config
- [ ] Sequential `recovery.max_steps` equals joint `joint_max_steps`
- [ ] Both arms resolve the same compressible-module list
- [ ] Benchmark machine idle; thread count pinned
- [ ] Git tree clean, so recorded commits are meaningful
- [ ] Disk space for the checkpoints the sweep will write

## Post-run checklist

- [ ] Every record's measured sparsity matches its target
- [ ] Every quantised record has `is_converted: true` and a plausible `storage_efficiency`
- [ ] Every record's evaluation `dataset_fingerprint` matches its model's dense run
- [ ] Every joint record's `training_cost_overhead` is 1.00
- [ ] `find_comparison_pairs()` reports a complete pair at every scale and budget
- [ ] All benchmark records share one `hardware_cpu_model` and one thread count
- [ ] All records in a table share one `quantisation.backend` and one artefact format
- [ ] Latency coefficient of variation under 15% on every benchmark
- [ ] Generation diagnostics show no degenerate outputs at the aggressive budget
- [ ] Any 1.4B result was run under settings identical to the main sweep, or is reported separately
      and excluded from the scale trend
- [ ] Every joint gain lists its **per-replicate values**, not only a mean — a mean that hides a sign
      flip is what forced the F-25 → F-26 retraction
- [ ] Every joint gain reports **R**, its exact sign-test p over the **non-tied** replicates, and
      whether significance was reachable at that R (it is not at R=5, whatever the effect size)
- [ ] Any gain whose sign is inconsistent across replicates is reported as inconclusive rather than as
      a small positive effect
- [ ] No gain is gated on the **seed spread** — that clause is withdrawn and was vacuous (F-15)

Promoting anything from `outputs/` to `results/` has its own checklist — see
[reproducibility.md](reproducibility.md#promotion-checklist).
