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
| `aggressive` | 0.70     | 4    | ~27x                       | Quality degrades enough that the arms should separate. Where the "at which budgets does joint help?" question is answered. |
| `pilot`      | 0.50     | 8    | ~8x                        | Reduced step and sample counts for pipeline validation. **Not a results budget.** |

Theoretical reduction is `32 / (bits * (1 - sparsity))` and assumes a storage format that actually
exploits both. The measured checkpoint size is what gets reported; see `storage_efficiency`.

---

## Table 1 — Dense FP32 baseline

One run per model. Every other row in the study is measured against its own model's baseline.

| Model | Params | Seed | Perplexity | Latency (ms) | p95 (ms) | Throughput (tok/s) | Peak mem (MiB) | Size (MiB) |
| ----- | ------ | ---- | ---------- | ------------ | -------- | ------------------ | -------------- | ---------- |
| pythia-160m  | 162M  | 1234 | — | — | — | — | — | — |
| pythia-410m  | 405M  | 1234 | — | — | — | — | — | — |
| pythia-1b    | 1.01B | 1234 | — | — | — | — | — | — |
| pythia-1.4b  | 1.41B | 1234 | — | — | — | — | — | — |
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
| pythia-1.4b | moderate   | 0.50 | unstructured | 500 | — | — | — | — |
| pythia-1.4b | aggressive | 0.70 | unstructured | 500 | — | — | — | — |

**Measured sparsity must match the target.** A gap means masks were not applied, or the optimiser
refilled pruned positions.

**Expect little or no latency improvement from unstructured sparsity.** Dense CPU GEMM kernels do
not skip scattered zeros. That is a finding to report, not a bug — and it is why the 2:4 variant
below exists.

### Table 2b — Semi-structured 2:4 sparsity

Same models, `granularity: "2:4"`, which fixes sparsity at 0.50 by definition. 2:4 is the pattern
with actual CPU kernel support, so this table is what answers whether *any* sparsity pattern
converts into measured latency.

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
| pythia-1.4b | moderate   | 8 | symmetric | per_channel | 128 | — | — | — | — |
| pythia-1.4b | aggressive | 4 | symmetric | per_channel | 128 | — | — | — | — |

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
| pythia-1.4b | moderate   | 0.50 | 8 | 500 | — | — | — | — |
| pythia-1.4b | aggressive | 0.70 | 4 | 500 | — | — | — | — |

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
| pythia-1.4b | moderate   | 0.50 | 8 | 500 | — | — | — | — | 1.00x |
| pythia-1.4b | aggressive | 0.70 | 4 | 500 | — | — | — | — | 1.00x |

The cost column must read 1.00x for a headline comparison. Anything else means the arms were not
budget-matched and the corresponding joint gain is confounded with extra training.

---

## Table 6 — Joint gain versus scale (the result)

Derived from Tables 4 and 5, matched on model, budget, and seed. Mean over three seeds, with the
seed-to-seed spread.

| Model | Params | Budget | Sequential retention | Joint retention | Joint gain | Seed spread | Gain > spread? |
| ----- | ------ | ------ | -------------------- | --------------- | ---------- | ----------- | -------------- |
| pythia-160m  | 162M  | moderate   | — | — | — | — | — |
| pythia-410m  | 405M  | moderate   | — | — | — | — | — |
| pythia-1b    | 1.01B | moderate   | — | — | — | — | — |
| pythia-1.4b  | 1.41B | moderate   | — | — | — | — | — |
| pythia-160m  | 162M  | aggressive | — | — | — | — | — |
| pythia-410m  | 405M  | aggressive | — | — | — | — | — |
| pythia-1b    | 1.01B | aggressive | — | — | — | — | — |
| pythia-1.4b  | 1.41B | aggressive | — | — | — | — | — |

**A gain smaller than the seed spread is inconclusive and must be reported as such.**

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
for each model, smallest first:
    1. dense baseline           -> establishes the retention reference
    2. pruning only             -> isolates sparsity
    3. quantisation only        -> isolates precision
    4. sequential               -> the baseline pipeline
    5. joint                    -> the arm under test
    (repeat 2-5 per budget, per seed)
then:
    6. Qwen validation
    7. aggregate, figures, tables
```

Run the pilot config end to end before starting the sweep. A pipeline bug found at pythia-1b costs
hours; the same bug found at pythia-160m costs minutes.

```bash
# validate configuration without running anything
python scripts/run_sequential.py --config configs/experiments/pilot.yaml --dry-run

# see the full sweep expansion before committing compute
sajc sweep --config configs/experiments/main_scale_sweep.yaml --plan-only
```

## Pre-run checklist

- [ ] Model revisions pinned in the model configs
- [ ] Pilot config completes end to end
- [ ] Dense baselines exist for every model in the sweep
- [ ] `budget_overrides` identical between the sweep and the validation config
- [ ] Sequential `recovery.max_steps` equals joint `joint_max_steps`
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
- [ ] Latency coefficient of variation under 15% on every benchmark
- [ ] Generation diagnostics show no degenerate outputs at the aggressive budget
