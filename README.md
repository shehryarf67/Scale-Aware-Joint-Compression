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

| Short name     | Hugging Face ID           | Role                                        |
| -------------- | ------------------------- | ------------------------------------------- |
| `pythia-160m`  | `EleutherAI/pythia-160m`  | scale sweep                                 |
| `pythia-410m`  | `EleutherAI/pythia-410m`  | scale sweep                                 |
| `pythia-1b`    | `EleutherAI/pythia-1b`    | scale sweep                                 |
| `pythia-1.4b`  | `EleutherAI/pythia-1.4b`  | scale sweep (optional, if hardware permits) |
| `qwen2.5-0.5b` | `Qwen/Qwen2.5-0.5B`       | optional external validation                |

Qwen2.5-0.5B is deliberately *not* part of the scale sweep. It is a different family with a
different tokeniser and training recipe, and is used only to check whether a trend observed within
Pythia transfers outside it.

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

## Experimental workflow

```
1.  prepare data and calibration sets       scripts/prepare_data.py
2.  dense FP32 baseline per model           scripts/run_dense_baseline.py
3.  pruning only                            scripts/run_pruning.py
4.  quantisation only                       scripts/run_quantisation.py
5.  sequential pruning -> quantisation      scripts/run_sequential.py
6.  joint pruning-aware quantisation        scripts/run_joint.py
7.  quality evaluation (all variants)       scripts/run_evaluation.py
8.  CPU deployment benchmark (all variants) scripts/run_cpu_benchmark.py
9.  aggregate the scale sweep               scripts/run_scale_sweep.py
10. figures and tables                      scripts/generate_plots.py
```

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

# the full scale sweep, then figures
python scripts/run_scale_sweep.py --config configs/experiments/main_scale_sweep.yaml
python scripts/generate_plots.py  --results outputs/metrics --output outputs/figures

# optional external validation
python scripts/run_scale_sweep.py --config configs/experiments/qwen_validation.yaml
```

The same operations are available through the `sajc` console script, e.g.
`sajc sequential --config configs/experiments/pilot.yaml`. Add `--help` to any command.

## Repository structure

```
.
├── configs/           YAML configs: models, compression, evaluation, experiments
├── data/              raw / processed / calibration data (contents git-ignored)
├── docs/              research question, methodology, protocols, paper outline
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
├── outputs/           run artefacts: checkpoints, logs, metrics, benchmarks, figures
└── results/           curated results promoted out of outputs/ for the write-up
```

## Reproducibility notes

- One seed per run, recorded in the run record; the sweep repeats each cell over several seeds.
- Every record stores the git commit, hardware metadata, and resolved library versions.
- Configurations are files, not command-line flags. Overrides are possible
  (`--override key.path=value`) but are serialised into the record, so a run is always
  reconstructible.
- Model revisions are pinnable per model config, so a silently updated Hub checkpoint cannot
  change results.
- Benchmarks from different machines are never averaged together; thread count, batch size, and
  sequence length are fixed and recorded.

Details in [docs/reproducibility.md](docs/reproducibility.md).

## Current project status

This repository is a **scaffold**. The structure, configuration system, model registry, metrics,
and CPU benchmarking harness are in place. The compression algorithms are not.

| Area                                      | Status                       |
| ----------------------------------------- | ---------------------------- |
| Repository layout, packaging, tooling     | done                         |
| Configuration system and validation       | done                         |
| Model registry and safe loader            | done                         |
| Metrics utilities (sparsity, ratio, gain) | done                         |
| CPU benchmarking harness and statistics   | done                         |
| Experiment records (JSON + CSV)           | done                         |
| Data loading and calibration sampling     | placeholder                  |
| Pruning                                   | placeholder                  |
| Quantisation                              | placeholder                  |
| Sequential pipeline                       | placeholder (stages defined) |
| Joint pruning-aware quantisation          | placeholder (stages defined) |
| Training / recovery loop                  | placeholder                  |
| Quality evaluation                        | placeholder                  |
| Figures and tables                        | placeholder                  |

Placeholder modules raise `NotImplementedError` with a pointer to what needs writing, rather than
silently returning plausible-looking numbers.

## Disclaimer

This project is under **active research development**. Interfaces, configuration keys, and the
result schema are expected to change without notice, and no result produced by the current state
of this repository should be treated as a finding.

## Licence

MIT — see [LICENSE](LICENSE).
