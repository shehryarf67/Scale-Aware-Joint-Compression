# Reproducibility

The goal is that any row in the results table can be traced back to the exact code, configuration,
data, and machine that produced it, and re-run.

## What every run record contains

One JSON file per run under `outputs/metrics/<experiment_id>.json`, plus a flat row appended to
`outputs/metrics/results.csv`.

| Group        | Fields |
| ------------ | ------ |
| Identity     | `experiment_id`, `timestamp` (UTC, ISO-8601), `git_commit`, `schema_version` |
| Model        | `model_name`, `model_size_label`, `parameter_count`, revision, dtype, device |
| Compression  | `compression_method`, `budget_label`, `sparsity`, `quantisation_bits`, per-stage log with optimiser steps and durations, measured sparsity |
| Seed         | `seed`, plus what was actually seeded (`python`, `numpy`, `torch`, `torch_cuda`) |
| Quality      | perplexity, total NLL, token count, retention, top-1/top-5 agreement, mean KL, generation diagnostics, evaluation `dataset_fingerprint` |
| Deployment   | latency mean/median/std/p95/p99/min/max, per-run latencies, throughput, peak memory, checkpoint size, thread report |
| Environment  | full hardware metadata and resolved versions of torch, transformers, datasets, numpy, and the rest |
| Config       | the entire resolved configuration, after includes and overrides |

Defined by `ExperimentRecord` in
[runner.py](../src/scale_aware_compression/experiments/runner.py). The CSV column order is
`RESULT_CSV_COLUMNS` in [constants.py](../src/scale_aware_compression/constants.py) and is versioned
by `RESULT_SCHEMA_VERSION`; appending a row whose schema differs from an existing CSV's header raises
rather than misaligning columns silently.

## Seeding

`set_global_seed(seed, deterministic=True)` seeds `random`, NumPy, and PyTorch (CPU and CUDA), sets
`PYTHONHASHSEED`, requests deterministic algorithms, and disables cuDNN benchmarking. It returns a
record of what it managed to seed, which goes into the run record — so a run made without NumPy
installed is distinguishable from one made with it.

Seeds used: **1234, 2345, 3456**, one run per seed per cell. The spread across them is the error bar.

Determinism costs some throughput. That trade is made deliberately: an irreproducible result is worth
less than a slow one.

### The calibration seed is separate

Calibration indices derive from `data.calibration_seed`, not from `runtime.seed`. Varying the run seed
to get error bars must not also change which sequences the quantiser calibrates on, or the seed spread
would conflate two sources of variation.

## Pinning

### Model revisions

Every model config has a `revision` field. It ships as `null`, which resolves to `main` — **pin it
before collecting results.** A Hub checkpoint can be updated in place, and an unpinned revision means
a re-run may silently load different weights.

```yaml
model:
  name: pythia-410m
  revision: 8b6c9d6c31d2b7a0f7d1e0a1b2c3d4e5f6a7b8c9   # a commit SHA, not a branch
```

### Dependencies

`requirements.txt` uses lower bounds, not exact pins, because `torch` must be chosen per platform.
For a reproduction run, freeze the environment:

```bash
pip freeze > results/summaries/environment-<date>.txt
```

Every record also stores the resolved versions of the packages that can change results, so a record
carries its own environment even without the freeze file.

### Git commit

Captured per run via `git rev-parse HEAD`. A dirty working tree appends `-dirty`, which means the
commit does not fully describe the code that ran. **Commit before a results run.**

## Configuration, not flags

A run is described by a YAML file. Command-line overrides exist (`--override runtime.seed=7`) and are
serialised into the record's `config` block, so an overridden run is still fully described.

Configs compose through `include`, resolved relative to the including file, merged in order, then
overridden by the including document's own keys. The sequential and joint arms both include the same
`pruning.yaml` and `quantisation.yaml`, so their shared settings cannot drift apart.

## Reproducing a result

```bash
# 1. check out the recorded commit
git checkout <git_commit from the record>

# 2. recreate the environment
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-dev.txt
# then match the torch version from the record's software block

# 3. extract the resolved config from the record
python -c "
import json, pathlib, yaml
record = json.loads(pathlib.Path('outputs/metrics/<experiment_id>.json').read_text())
pathlib.Path('repro.yaml').write_text(yaml.safe_dump(record['config'], sort_keys=False))
"

# 4. re-run
python scripts/run_sequential.py --config repro.yaml

# 5. compare the new record against the old
```

Expect exact agreement on quality metrics and approximate agreement on latency. Latency will differ
across machines; that is why the hardware block is recorded.

## What is not reproducible, and why

| Not reproducible | Reason | Mitigation |
| --- | --- | --- |
| Absolute latency across machines | different CPUs, cache sizes, BLAS builds | hardware metadata recorded; never average across machines |
| Latency to the last millisecond on one machine | thermal state, background load | 30 runs, median and p95, CV warning |
| Peak memory to the megabyte | allocator behaviour, OS paging | reported to one decimal, compared as ratios |
| Bit-exact GPU training | non-deterministic kernels remain even under `use_deterministic_algorithms(warn_only=True)` | recovery/joint results reported over three seeds |
| Hub downloads over time | upstream repositories change | pin revisions |

## Artefacts and what is committed

| Path | Committed? | Contents |
| --- | --- | --- |
| `configs/` | yes | every configuration |
| `docs/` | yes | protocols and write-up material |
| `src/`, `tests/`, `scripts/` | yes | all code |
| `notebooks/` | yes, outputs stripped | analysis; `nbstripout` runs as a pre-commit hook |
| `data/*` | no | corpora and calibration caches, rebuildable |
| `outputs/checkpoints/` | no | model weights, large |
| `outputs/logs/`, `outputs/metrics/`, `outputs/benchmarks/` | no | per-run artefacts |
| `outputs/figures/`, `outputs/tables/` | no | regenerated from records |
| `results/summaries/*.md` | yes | curated result summaries promoted for the write-up |

Directory structure is tracked through `.gitkeep` files; the contents are ignored. Model weights and
large tabular exports are excluded by extension as a second line of defence, so an accidental
`git add` of a checkpoint fails rather than committing 4 GB.

## Verifying a results set

- [ ] Every record has a non-null `git_commit` with no `-dirty` suffix
- [ ] Every record's `schema_version` matches
- [ ] Every model config used a pinned `revision`
- [ ] All records to be compared share one evaluation `dataset_fingerprint`
- [ ] All benchmark records share one `hardware_cpu_model` and thread count
- [ ] Three seeds present for every cell
- [ ] `software.torch` identical across records in one table
- [ ] Measured sparsity matches the target on every pruned record
- [ ] `is_converted` true on every quantised record
- [ ] Frozen environment file saved alongside the results
