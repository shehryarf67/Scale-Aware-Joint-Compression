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

Every model config has a `revision` field, and it stays configurable rather than hard-coded. It ships
as `null`, which resolves to whatever `main` points at today.

**The policy:**

| Stage | Revision |
| --- | --- |
| Pilot / pipeline-validation runs | unpinned (`null`) is acceptable |
| Exploratory runs while iterating | unpinned is acceptable |
| **Main sweep, validation, extended sweep** | **must be pinned to a commit SHA** |
| **Any result cited in the paper** | **must be pinned to a commit SHA** |

A Hugging Face repository can be updated in place. An unpinned revision means a re-run months later
may silently load different weights, and the result would be irreproducible with no error to indicate
why.

**How to pin.** Look up the current commit for each model and paste it in — do not copy a SHA from
this document or from any other project, and do not guess one. An invented SHA fails at load time,
which is the good case; a SHA belonging to a different revision than you think is the bad case.

```bash
# print the current main-branch commit for each model in the sweep
python - <<'PY'
from huggingface_hub import HfApi

from scale_aware_compression.models.registry import list_models, resolve_model_id

api = HfApi()
for name in list_models():
    info = api.model_info(resolve_model_id(name))
    print(f"{name:14s} {resolve_model_id(name):26s} {info.sha}")
PY
```

Then set it in the model config:

```yaml
model:
  name: pythia-410m
  # Replace with the real 40-character commit SHA from the command above.
  # A branch name ('main') or tag is NOT a pin -- both can move.
  revision: <PASTE-COMMIT-SHA-HERE>
```

Every run record stores the resolved `revision`, so a record made with an unpinned revision is
distinguishable after the fact from one made with a pin — but only the pin makes the run repeatable.

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

## `outputs/` versus `results/`

Two directories, two different guarantees. The distinction is what stops an unverified number reaching
the paper.

### `outputs/` — raw, unverified, disposable

Everything a run produces, the moment it produces it. Nothing here has been checked.

| Path | Contents |
| --- | --- |
| `outputs/checkpoints/` | temporary model checkpoints, including failed and partial runs |
| `outputs/logs/` | per-run log files |
| `outputs/metrics/` | **unverified** JSON run records and the appended `results.csv` |
| `outputs/benchmarks/` | benchmark records, including runs with anomalous timing |
| `outputs/figures/` | figures regenerated on every analysis pass |
| `outputs/tables/` | tables regenerated on every analysis pass |

Properties:

- **Written automatically** by scripts, with no human review.
- **Safe to delete entirely.** Everything is either regenerable from a config or was never verified.
- **May contain contradictory records** — a re-run under a fixed bug sits next to the buggy one, and
  only the experiment ID and timestamp distinguish them.
- **May contain runs that should never be reported**: crashed mid-sweep, wrong thread count, noisy
  benchmark, unmatched budgets.
- **Never cite a path under `outputs/` in the write-up.**

### `results/` — verified, curated, frozen

Only artefacts that have passed the promotion checklist below, and that the paper actually cites.

| Path | Contents |
| --- | --- |
| `results/raw/` | frozen copies of the verified run records a result rests on |
| `results/processed/` | the aggregated tables the figures were built from |
| `results/summaries/` | curated Markdown summaries, committed to git |

Properties:

- **Written deliberately**, by a human promoting a specific artefact after checking it.
- **Frozen.** Once promoted, a file is not edited. A correction means promoting a new, separately
  identified artefact and recording why the old one was superseded — not overwriting it.
- **One backend, one machine, one schema per promoted set.**
- **Every promoted artefact traces to a git commit and a resolved configuration**, so it can be
  regenerated from scratch.
- The write-up cites `results/`, never `outputs/`.

Promotion is a one-way door: `outputs/` → checklist → `results/`. Nothing moves the other way.

## Promotion checklist

Every item must hold before an artefact moves from `outputs/` to `results/`. A single unchecked box
means it stays in `outputs/`.

- [ ] **Successful run completion.** The run finished; it did not crash, time out, or get interrupted.
      No partial checkpoint, no truncated record.
- [ ] **Resolved configuration saved.** The fully merged config — after includes and overrides — is
      stored in the record's `config` block, so the run is reconstructible without the original
      command line.
- [ ] **Git commit recorded.** Non-null `git_commit` with **no `-dirty` suffix**. A dirty tree means
      the commit does not describe the code that ran.
- [ ] **Hardware metadata recorded.** CPU model, core counts, memory, and thread environment present
      in the record. Absent metadata makes a latency number uninterpretable.
- [ ] **Matched sequential and joint budgets.** `training_cost_overhead` is 1.00, and
      `match_sequential_budget` is true. An unmatched pair cannot support a joint-gain claim.
- [ ] **No benchmark anomaly.** Latency coefficient of variation under 15%; `warmup_runs` ≥ 5 and
      `measured_runs` ≥ 30; `thread_report.torch_num_threads` equals the requested count; no bimodal
      latency distribution suggesting thermal throttling.
- [ ] **Final quality metrics verified.** Measured sparsity matches its target; `is_converted` true
      for every quantised artefact; `storage_efficiency` plausible; evaluation `dataset_fingerprint`
      matches the model's dense baseline; the run was evaluated on CPU.
- [ ] **Consistent backend and output format.** Every artefact in the promoted set shares one
      `quantisation.backend`, one artefact format, and one `software.torch` version.

Additional items for a promoted *set* rather than a single run:

- [ ] All seeds for the cell are present, and the spread across them is recorded alongside the mean.
- [ ] Any joint gain smaller than the seed spread is labelled inconclusive, not reported as a small
      positive effect.
- [ ] Any 1.4B result was produced under settings identical to the main sweep, or is excluded from the
      scale trend and labelled as such.
- [ ] A frozen environment file (`pip freeze`) is saved alongside.

## Artefacts and what is committed

| Path | Committed? | Contents |
| --- | --- | --- |
| `configs/` | yes | every configuration |
| `docs/` | yes | protocols and write-up material |
| `src/`, `tests/`, `scripts/` | yes | all code |
| `notebooks/` | yes, outputs stripped | analysis; `nbstripout` runs as a pre-commit hook |
| `data/*` | no | corpora and calibration caches, rebuildable |
| `outputs/checkpoints/` | no | model weights, large |
| `outputs/logs/`, `outputs/metrics/`, `outputs/benchmarks/` | no | unverified per-run artefacts |
| `outputs/figures/`, `outputs/tables/` | no | regenerated from records |
| `results/raw/`, `results/processed/` | no | large; frozen locally and archived separately |
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
