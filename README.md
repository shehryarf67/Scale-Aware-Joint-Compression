# Scale-Aware Joint Compression

**How does model scale influence the effectiveness of joint versus sequential pruning and
quantisation in decoder-only language models?**

---

## Research question

Compressing a language model usually means two things: removing weights (**pruning**) and
reducing the numerical precision of the weights that remain (**quantisation**). In practice these
are applied one after the other — prune, then quantise what remains. An alternative is to decide both
together, so the pruning decision is aware of the quantisation grid and vice versa (**joint
compression**).

Joint optimisation costs more to run. The question this repository investigates is whether that extra
cost pays off, and specifically whether it pays off *more as models get bigger*:

> How does model scale influence the effectiveness of joint versus sequential pruning and
> quantisation in decoder-only language models?

Secondary questions and the full framing live in [docs/research_question.md](docs/research_question.md).

## Motivation

- Nearly all published pruning + quantisation results are reported at a **single model size**, so
  it is unclear whether an advantage measured on a small model still exists at a larger one.
- Compression papers frequently report *theoretical* sparsity and bit-width reductions rather
  than **measured CPU latency**, which is what actually matters for commodity deployment.
- Joint optimisation costs additional training compute. That cost needs to be reported next to the
  quality benefit so the trade-off is legible.
- The Pythia suite is trained with an identical data order and recipe across sizes, which makes it
  one of the few model families where "model scale" can be treated as a genuinely controlled
  independent variable.

## Models

The main scale sweep uses the EleutherAI Pythia suite, which holds data, tokeniser, and training
recipe fixed across sizes:

| Short name     | Hugging Face ID           | Role                                          |
| -------------- | ------------------------- | --------------------------------------------- |
| `pythia-160m`  | `EleutherAI/pythia-160m`  | **main** scale sweep                          |
| `pythia-410m`  | `EleutherAI/pythia-410m`  | **main** scale sweep                          |
| `pythia-1b`    | `EleutherAI/pythia-1b`    | **main** scale sweep                          |
| `pythia-1.4b`  | `EleutherAI/pythia-1.4b`  | *extended* sweep — optional, hardware-dependent |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B`       | optional external validation                  |

The **main sweep is three models** (`main_scale_sweep.yaml`). `pythia-1.4b` lives in
`extended_scale_sweep.yaml` and is run only after the main sweep succeeds — and it counts as a fourth
scale point only if it can be run with settings identical to the main sweep. A 1.4B run that needed
bf16, a smaller effective batch size, or gradient checkpointing differs from the 1B run in more than
scale, so mixing it into the trend would attribute a training-settings difference to model scale. In
that case it is reported separately and excluded from the trend.

Qwen2.5-0.5B is deliberately *not* part of the scale sweep. It is a different family with a
different tokeniser and training recipe, and is used only to check whether a trend observed within
Pythia transfers outside it.

Model revisions ship unpinned. **Pilot runs may leave them unpinned; every result cited in the paper
must use a pinned commit SHA** — see [reproducibility.md](docs/reproducibility.md#model-revisions).

## Compression methods compared

All five arms are **layerwise post-training reconstruction** (plan §3.1) through one shared solver —
not full-model fine-tuning. For each linear layer they minimise `‖X·Wᵀ − X·(M∘Q_b(W))ᵀ‖²_F` using only
the Gram matrix `H = XᵀX` from a fixed calibration set. The arms differ in *call order*, nothing else.

| ID              | Method                            | Pipeline                                                                    |
| --------------- | --------------------------------- | --------------------------------------------------------------------------- |
| `dense`         | Dense FP32 baseline               | reference point for every other row                                         |
| `pruning`       | Pruning only                      | score → mask → reconstruct (stays FP32; the only arm in a latency table)     |
| `quantisation`  | Quantisation only                 | fit scales → quantise → reconstruct                                         |
| `sequential`    | Sequential P→Q                    | mask → reconstruct → fit scales on that result → quantise → reconstruct     |
| `sequential_qp` | Reverse sequential Q→P            | fit scales on dense → quantise → mask → reconstruct, **reusing** those scales |
| `joint`         | Joint                             | mask scored on the *quantised* weights, refit each iteration, best kept      |

The joint arm's defining property is **decision D3**: it scores saliency as
`S_ij = |Q_b(W_ij)| · ‖X_j‖₂`, so the mask is chosen against the grid the weights will actually live
on. An outer loop alternates mask and scales, keeping a proposal only when it improves the objective.

The central quantity of interest is **joint gain**: the quality of the joint pipeline minus the
quality of the sequential pipeline at a matched compression budget. See
[joint_gain.py](src/scale_aware_compression/metrics/joint_gain.py).

The study compares **one specific sequential implementation against one specific joint
implementation** and does not claim to represent pruning or quantisation in general. The joint arm is
*quantisation-aware mask selection inside layerwise reconstruction*, not a universal joint compression
algorithm. Nothing in the design presumes it wins: a null or negative joint gain is a valid outcome,
and the sequential arm has in fact been ahead at several measured budgets.
Exact definitions — compressible modules, pruning and quantisation methods, mask scoring, and the
matched-budget requirements — are in [method_definition.md](docs/method_definition.md), and what could
still make the results wrong is in [validity_threats.md](docs/validity_threats.md).

### The frozen budgets

**Moderate is 30% sparsity at INT8; aggressive is 30% sparsity at 4-bit.** Frozen 2026-07-29 from the
Phase 7 screening grid — see [protocol_freeze.md](docs/protocol_freeze.md#the-frozen-compression-budgets).

The superseded pair was 50% + INT8 and 70% + 4-bit. Screening measured **both as catastrophic** on
Pythia-160M — 22.9% and 0.8% retention — and since the smallest model sets the ceiling for all three,
neither was usable anywhere.

Two consequences worth reading before interpreting any table:

- **The frozen pair varies precision, not sparsity.** Both budgets prune 30%. So the sparsity-versus-
  latency curve research question 4 asks for does **not** come from these budgets; it comes from
  benchmark-only runs of the pruning-only arm at several sparsities, which are cheap because that arm
  stays FP32.
- **4-bit was chosen because it is the only regime where the joint mechanism is measurably live** —
  8.86% mask divergence at W4 against 0.46% at W8. Two 8-bit budgets could not detect the effect this
  study exists to measure, and would produce a confident null that was an artefact of the design.
  INT8 is therefore the **control**, where a near-zero joint gain is the expected result.

### The 4-bit backend risk

The aggressive budget carries a real constraint, resolved as **decision D1**:

- PyTorch's native CPU quantisation support is **strongest for INT8**.
- **4-bit weight-only CPU deployment may require a separate backend** — a packed-weight custom linear
  module, or an external runtime. There is no equally mature built-in 4-bit CPU kernel.
- **Latency and size results are not comparable if the moderate and aggressive settings use different
  runtimes or artefact formats.** The same applies across arms with more force: a 4-bit joint artefact
  measured against an INT8 sequential artefact is not a joint-gain measurement at all.
**Resolved as D1:** PyTorch native CPU **INT8**, engine **`onednn`**, is the sole latency backend. W4
contributes quality and checkpoint size only and **never appears in a latency table**. Research
question 4 survives because the sparsity→latency curve comes from the pruning-only arm, whose weights
stay FP32.

One correction found by probing rather than reading documentation: every PyTorch tutorial names the
engine `x86`, but on the pinned torch 2.13.0+cu126 `supported_engines` is **`['onednn']` only**. The
shipped configs said `x86`, so conversion would have failed *after* the compression compute was spent.
A `requires_torch` test now asserts the shipped backend against the installed torch, so an upgrade that
renames engines fails a test rather than a run.

## Experimental workflow

```
0.  pilot pipeline validation           configs/experiments/pilot.yaml   (produces no results)
1.  prepare data and calibration sets       scripts/prepare_data.py
2.  dense FP32 baseline per model           scripts/run_dense_baseline.py
3.  pruning only                            scripts/run_pruning.py
4.  quantisation only                       scripts/run_quantisation.py
5.  sequential pruning -> quantisation      scripts/run_sequential.py
6.  joint pruning-aware quantisation        scripts/run_joint.py
7.  quality evaluation (all variants)       scripts/run_evaluation.py
8.  CPU deployment benchmark (all variants) scripts/run_cpu_benchmark.py
9.  main scale sweep (3 models)             scripts/run_scale_sweep.py
10. extended sweep (+1.4B, optional)        scripts/run_scale_sweep.py
11. figures and tables                      scripts/generate_plots.py
```

Step 0 is not optional in practice. `pilot.yaml` is a **pipeline-validation run**: one model, one
seed, one budget, tiny evaluation and calibration sets, ~60 optimiser steps. It exists to prove the
pipeline executes and writes a well-formed record. **Its numbers are not results** and must never
appear in the write-up — a pipeline bug found at 160M costs minutes; the same bug found at 1B costs
hours.

Every run writes one structured record (JSON, plus a row appended to CSV) under `outputs/`, keyed
by experiment ID. See [docs/experiment_protocol.md](docs/experiment_protocol.md).

## CPU-only evaluation policy

**All final deployment measurements — latency, throughput, peak memory, checkpoint size — are
produced on CPU.** GPUs are used only for work that does not appear in a reported deployment
number:

| Allowed on GPU                     | Must be CPU                       |
| ---------------------------------- | --------------------------------- |
| activation capture, Gram matrices  | latency (mean / median / p95)     |
| layerwise reconstruction solves    | throughput (tokens/s)             |
| correctness anchors and diagnostics| peak process memory               |
| exploratory quality evaluation     | final reported quality evaluation |

Benchmarks pin the PyTorch CPU thread count, fix batch size and sequence length, run warm-up
iterations before measurement, and report median and p95 rather than a single timing. Every
benchmark record carries its hardware metadata so incomparable runs can be detected after the
fact. The full rules are in [docs/benchmarking_protocol.md](docs/benchmarking_protocol.md).

## Setup

Requires **Python 3.11 or newer**.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

make install-dev                 # or: pip install -e . -r requirements-dev.txt
```

`torch` is intentionally not pinned to a CUDA build. Install the wheel that matches your platform
first if you need GPU support, then install this package:

```bash
# example: CPU-only wheel
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

Nothing in this package downloads a model at import time. Model weights are fetched only when a
loader is called explicitly, or ahead of time via `scripts/download_models.py`.

## Example commands

```bash
# environment and registry sanity check (no downloads, no compute)
sajc info

# validate a configuration without running anything
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml --dry-run

# fetch the weights for the sweep into the local Hugging Face cache
python scripts/download_models.py --models pythia-160m pythia-410m pythia-1b

# build the evaluation and calibration splits
python scripts/prepare_data.py --config configs/experiments/pilot.yaml

# single-model pipelines
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml
python scripts/run_pruning.py        --config configs/experiments/pilot.yaml
python scripts/run_quantisation.py   --config configs/experiments/pilot.yaml
python scripts/run_sequential.py     --config configs/experiments/pilot.yaml
python scripts/run_joint.py          --config configs/experiments/pilot.yaml

# evaluation and CPU benchmarking
python scripts/run_evaluation.py     --config configs/experiments/pilot.yaml
python scripts/run_cpu_benchmark.py  --config configs/experiments/pilot.yaml --threads 4

# see the sweep expansion before committing compute
python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml --plan-only

# the main scale sweep (3 models), then figures
python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml
python scripts/generate_plots.py  --results outputs/metrics --output outputs/figures

# optional external validation
python scripts/run_scale_sweep.py --config configs/experiments/qwen_validation.yaml

# optional extended sweep: adds pythia-1.4b. Run only after the main sweep succeeds, and only if
# the settings can be held identical to it -- see the header of the config.
python scripts/run_scale_sweep.py --config configs/experiments/extended_scale_sweep.yaml
```

The same operations are available through the `sajc` console script, e.g.
`sajc sequential --config configs/experiments/pilot.yaml`. Add `--help` to any command.

## Repository structure

```
.
├── .github/workflows/ CI: ruff check, ruff format --check, fast tests (no downloads)
├── configs/           YAML configs: models, compression, evaluation, experiments
├── data/              raw / processed / calibration data (contents git-ignored)
├── docs/              research question, method definition, protocols, validity threats
├── notebooks/         exploratory analysis, kept output-free
├── scripts/           argparse entry points, one per pipeline stage
├── src/scale_aware_compression/
│   ├── benchmarking/  CPU latency, throughput, memory, checkpoint size
│   ├── compression/   base interface + pruning / quantisation / sequential / joint
│   ├── data/          dataset loading, preprocessing, calibration sampling
│   ├── evaluation/    perplexity, generation, dense-vs-compressed agreement
│   ├── experiments/   run records, tracker, scale sweep, external validation
│   ├── metrics/       sparsity, compression ratio, retention, joint gain
│   ├── models/        registry, safe loader, architecture adapters
│   ├── anchors/       independent references that check our own (Wanda, exact optimum)
│   ├── training/      superseded fine-tuning scaffold; importable, unregistered
│   └── visualisation/ plots and tables
├── tests/             lightweight tests; no downloads, no training
├── outputs/           RAW, UNVERIFIED run artefacts -- safe to delete, never cited
└── results/           VERIFIED, CURATED, FROZEN artefacts -- what the paper cites
```

### `outputs/` versus `results/`

The two directories carry different guarantees, and the distinction is what stops an unverified number
reaching the paper.

| | `outputs/` | `results/` |
| --- | --- | --- |
| **Contents** | all raw run artefacts, logs, temporary checkpoints, unverified metrics, benchmark records | verified, curated, frozen artefacts used in the paper |
| **Written by** | scripts, automatically | a human, deliberately |
| **Reviewed** | no | yes — via the promotion checklist |
| **Mutable** | yes; regenerated freely | no; frozen once promoted |
| **Safe to delete** | yes, entirely | no |
| **Cited in the write-up** | **never** | yes |

`outputs/` will contain runs that should never be reported: crashed sweeps, wrong thread counts, noisy
benchmarks, unmatched budgets, and a re-run sitting next to the buggy run it replaced. Nothing is
filtered on the way in.

Promotion is a one-way door — `outputs/` → checklist → `results/` — and an artefact only passes with
**all** of the following true:

- [ ] successful run completion (no crash, no interruption, no partial checkpoint)
- [ ] resolved configuration saved in the record
- [ ] git commit recorded, with no `-dirty` suffix
- [ ] hardware metadata recorded
- [ ] matched sequential and joint budgets (`training_cost_overhead` = 1.00)
- [ ] no benchmark anomaly (CV < 15%, thread count honoured, ≥5 warm-up and ≥30 measured runs)
- [ ] final quality metrics verified (measured sparsity matches target, `is_converted` true,
      `dataset_fingerprint` matches the dense baseline, evaluated on CPU)
- [ ] consistent backend and output format across the promoted set

Full detail, including the extra items for a promoted *set*, is in
[reproducibility.md](docs/reproducibility.md#promotion-checklist).

## Reproducibility notes

- One seed per run, recorded in the run record; the sweep repeats each cell over several seeds.
- Every record stores the git commit, hardware metadata, and resolved library versions.
- Configurations are files, not command-line flags. Overrides are possible
  (`--override key.path=value`) but are serialised into the record, so a run is always
  reconstructible.
- Model revisions stay configurable. Pilot runs may use an unpinned revision; **every result cited in
  the paper must use a pinned commit SHA**, since a Hub repository can be updated in place.
- Benchmarks from different machines are never averaged together; thread count, batch size, and
  sequence length are fixed and recorded.
- Nothing is promoted from `outputs/` to `results/` without passing the checklist above.

Details in [docs/reproducibility.md](docs/reproducibility.md).

## Current project status

**The study is complete.** All experiments have run; what remains is writing.

**The headline result is negative on practical importance.** The confirmatory grid
([F-37](docs/findings_log.md#f-37)) ran once on the held-out test split — 171 cells, 42 pairs, 0
failures — and **no cell meets the pre-registered practical-importance bar** of ≥1.0 pp with a
consistent sign. The one statistically significant cell, pythia-410m at 30% + W4, survives
multiple-comparison correction at **Holm-adjusted p = 0.0469** while landing **0.065 pp short** of
the bar: real, small, and fragile.

| Scale | 30% + W4 | 30% + W8 |
| --- | --- | --- |
| pythia-160m | +1.0120 pp (7/8) | +0.0381 pp |
| pythia-410m | +0.9348 pp (8/8, Holm *p* = 0.0469) | +0.0289 pp |
| pythia-1b | +0.1316 pp (4/5) | −0.1794 pp (0/5) |
| **qwen2.5-0.5b** (external) | **+0.4213 pp (7/8)** | **−0.0321 pp** |

**The motivating hypothesis is not supported:** the joint advantage did not increase with scale.
The observed direction is the opposite, but the cross-scale decline is **not statistically
established** and the 410M→1B step is confounded with depth
([F-38](docs/findings_log.md#f-38): pythia-1b has 16 blocks against pythia-410m's 24).

Qwen2.5-0.5B is **external validity, not a fourth scale point** ([F-41](docs/findings_log.md#f-41)).

| Area | Status |
| --- | --- |
| Repository layout, packaging, tooling | done |
| CI (lint, format, fast tests) | done |
| Configuration system and validation | done |
| Model registry and safe loader | done |
| Data loading, chunking, calibration | done |
| Perplexity, agreement, generation | done |
| CPU benchmarking harness and statistics | done |
| Experiment records (JSON + CSV) | done |
| Pruning, quantisation, sequential, joint | **done** — all five arms through one layerwise driver |
| Layerwise reconstruction + activations | **done**, verified against three independent anchors |
| Confirmatory grid (A1 step 10) | **done** — 171 cells, run once |
| External validation (Qwen2.5-0.5B) | **done** — 65 cells |
| Figures and tables | **done** — committed under `results/` |
| Paper | not written |

**Read these before quoting any number:**

- [docs/findings_log.md](docs/findings_log.md) — every measurement with its conditions, and **§6,
  which states exactly what the paper may and may not claim**
- [docs/limitations.md](docs/limitations.md) — 16 items, written *before* any interpretation
- [docs/STATUS.md](docs/STATUS.md) — current state and next steps

Unimplemented paths still raise `NotImplementedError` naming the module to edit, rather than
silently returning plausible-looking numbers.

### Open decisions requiring a human choice

Three methodological questions cannot be resolved from the code, and all three should be settled
**before** the first main experiment — not after seeing which choice produces a nicer result:

1. **The final CPU quantisation backend.** Constrains every downstream bit-width choice, so decide it
   first. Candidates: PyTorch native `x86` / `fbgemm` (INT8 only), or an external runtime with 4-bit
   CPU support.
2. **The exact joint mask-scoring rule.** Rank by absolute fake-quantised weight magnitude, or by
   absolute FP32 shadow-weight magnitude with fake quantisation active throughout?
   [method_definition.md](docs/method_definition.md#mask-scoring) recommends the latter, with reasons;
   confirm or override it.
3. **Whether 4-bit stays in the main study** or the INT8 fallback is adopted. Follows from (1).

## Documentation

| Document | Contents |
| --- | --- |
| [research_question.md](docs/research_question.md) | primary and secondary questions; what the study does *not* claim |
| [method_definition.md](docs/method_definition.md) | exactly what the two arms are: modules, methods, mask scoring, matched budgets |
| [methodology.md](docs/methodology.md) | variables, controls, fair-comparison mechanisms |
| [experiment_protocol.md](docs/experiment_protocol.md) | the run tables, execution order, pre/post-run checklists |
| [benchmarking_protocol.md](docs/benchmarking_protocol.md) | CPU measurement rules and backend constraints |
| [validity_threats.md](docs/validity_threats.md) | what could still make these results wrong |
| [reproducibility.md](docs/reproducibility.md) | seeds, revision pinning, record contents, promotion checklist |
| [paper_outline.md](docs/paper_outline.md) | how the results become a write-up |

## Disclaimer

This project is under **active research development**. Interfaces, configuration keys, and the
result schema are expected to change without notice, and no result produced by the current state
of this repository should be treated as a finding.

## Licence

MIT — see [LICENSE](LICENSE).
