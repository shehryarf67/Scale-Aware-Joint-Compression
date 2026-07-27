# Scale-Aware Joint Compression

**How does model scale influence the effectiveness of joint versus sequential pruning and
quantisation in decoder-only language models?**

---

## Research question

Compressing a language model usually means two things: removing weights (**pruning**) and
reducing the numerical precision of the weights that remain (**quantisation**). In practice these
are applied one after the other — prune, recover, then quantise. An alternative is to optimise for
both objectives at the same time, so that the pruning decision is aware of the quantisation grid
and vice versa (**joint pruning-aware quantisation**).

Joint optimisation is more expensive to train. The question this repository investigates is
whether that extra cost pays off, and specifically whether it pays off *more as models get
bigger*:

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

| ID             | Method                            | Pipeline                                                                                         |
| -------------- | --------------------------------- | ------------------------------------------------------------------------------------------------ |
| `dense`        | Dense FP32 baseline               | reference point for every other row                                                              |
| `pruning`      | Pruning only                      | dense → prune → recover                                                                          |
| `quantisation` | Quantisation only                 | dense → calibrate → quantise → convert                                                           |
| `sequential`   | Sequential pruning → quantisation | dense → prune → recover → quantise → convert                                                     |
| `joint`        | Joint pruning-aware quantisation  | dense → fake-quantisation prep → gradual pruning during optimisation → joint fine-tune → convert |

The central quantity of interest is **joint gain**: the quality of the joint pipeline minus the
quality of the sequential pipeline at a matched compression budget. See
[joint_gain.py](src/scale_aware_compression/metrics/joint_gain.py).

The study compares **one specific sequential implementation against one specific joint
implementation** and does not claim to represent pruning or quantisation in general. The joint arm is
*joint magnitude pruning with quantisation-aware fine-tuning*, not a universal joint compression
algorithm. Nothing in the design presumes it wins: a null or negative joint gain is a valid outcome.
Exact definitions — compressible modules, pruning and quantisation methods, mask scoring, and the
matched-budget requirements — are in [method_definition.md](docs/method_definition.md), and what could
still make the results wrong is in [validity_threats.md](docs/validity_threats.md).

### Bit widths and the 4-bit backend risk

The moderate budget is 50% sparsity at **INT8**; the aggressive budget is 70% sparsity at **4-bit**.
The second carries a real risk that must be resolved before the main experiments:

- PyTorch's native CPU quantisation support is **strongest for INT8**.
- **4-bit weight-only CPU deployment may require a separate backend** — a packed-weight custom linear
  module, or an external runtime. There is no equally mature built-in 4-bit CPU kernel.
- **Latency and size results are not comparable if the moderate and aggressive settings use different
  runtimes or artefact formats.** The same applies across arms with more force: a 4-bit joint artefact
  measured against an INT8 sequential artefact is not a joint-gain measurement at all.
- **The final backend decision must be made before the main experiments**, not after seeing results.

Documented fallback if a single 4-bit CPU path cannot be implemented for both arms:

| Budget                  | Sparsity | Bit width |
| ----------------------- | -------- | --------- |
| Moderate                | 50%      | INT8      |
| Aggressive (fallback)   | 70%      | INT8      |

This keeps every row on one runtime and one artefact format, at the cost of making precision a
constant rather than a second compression axis. 4-bit support stays in the configuration system
either way; what the fallback changes is whether the *main study* uses it.

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

| Allowed on GPU                   | Must be CPU                       |
| -------------------------------- | --------------------------------- |
| fine-tuning, pruning recovery    | latency (mean / median / p95)     |
| quantisation calibration         | throughput (tokens/s)             |
| joint compression training       | peak process memory               |
| exploratory quality evaluation   | final reported quality evaluation |

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
│   ├── training/      trainer, pruning recovery, callbacks
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

This repository is a **scaffold**. The structure, configuration system, model registry, metrics,
and CPU benchmarking harness are in place. The compression algorithms are not.

| Area                                      | Status                       |
| ----------------------------------------- | ---------------------------- |
| Repository layout, packaging, tooling     | done                         |
| CI (lint, format, fast tests)             | done                         |
| Configuration system and validation       | done                         |
| Model registry and safe loader            | done                         |
| Metrics utilities (sparsity, ratio, gain) | done                         |
| CPU benchmarking harness and statistics   | done                         |
| Experiment records (JSON + CSV)           | done                         |
| Method definition and validity analysis   | done                         |
| **Data loading, chunking, calibration**   | **done**                     |
| **Perplexity, agreement, generation**     | **done**                     |
| **CPU benchmark workload (forward/decode)** | **done**                   |
| **Dense baseline, end to end**            | **done**                     |
| Pruning                                   | placeholder                  |
| Quantisation                              | placeholder                  |
| Sequential pipeline                       | placeholder (stages defined) |
| Joint pruning-aware quantisation          | placeholder (stages defined) |
| Layerwise reconstruction + activations    | not started                  |
| Figures and tables                        | placeholder                  |

Placeholder modules raise `NotImplementedError` with a pointer to what needs writing, rather than
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
