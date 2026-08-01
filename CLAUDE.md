# CLAUDE.md

Guidance for Claude Code working in this repository. Read
[docs/STATUS.md](docs/STATUS.md) **first, every session** — it holds the current phase, what is
in progress, and the decisions that are still open.

Picking the project up cold, or on a machine that has not worked on it before? Read
[docs/partner_handoff.md](docs/partner_handoff.md) as well — the research question, every finding
with its trust level, and the next tasks with their exact acceptance criteria.

@docs/STATUS.md

## What this project is

A controlled study of whether **joint** pruning-and-quantisation beats **sequential**
pruning-then-quantisation, and whether the gap changes with model scale (Pythia 160M → 410M → 1B).

The governing documents, in order of authority:

1. `docs/research_plan.pdf` — **the authoritative source**. The full 28-page execution plan:
   research questions, method family, experimental matrix, seed policy, schedule, permissible
   claims. Read it when a detail is not in the markdown docs, and prefer it when they disagree.
2. `docs/method_definition.md` — exactly what the two arms are. The engineering spec.
3. `docs/implementation_plan.md` — the ten build phases with exit tests, and the testing plan.
4. `docs/findings_log.md` — **every measurement taken, with its conditions**. Append-only; the
   paper is written from it. Add to it whenever a number is produced, and never silently delete a
   superseded one.
5. `docs/experiment_protocol.md` — the run tables and checklists.
6. `docs/benchmarking_protocol.md` — CPU measurement rules.
7. `docs/validity_threats.md` — what could make the results wrong.

If code and docs disagree, that is a bug in one of them. Say so rather than picking silently.

**The method is layerwise post-training reconstruction** (plan §3.1), not full-model
quantisation-aware fine-tuning. The unit of optimisation is *local steps per layer*. Parts of the
config still carry the older fine-tuning vocabulary; see the reconciliation table in
`docs/implementation_plan.md`.

## Machine policy

Superseded 2026-08-01. The old rule was "the HP Omen is the only machine that runs code". That was
operational shorthand from when one person had one GPU box, and it is **stricter than the protocol
requires** — `benchmarking_protocol.md` and `methodology.md` both say *one machine per results
table*, not one machine per project. Three tiers, by what is actually host-bound:

| Tier | Work | Where |
| --- | --- | --- |
| 1 | Tests, lint, config validation, docs, analysis of existing records | **anywhere** |
| 2 | Compression, activation capture, quality evaluation | **any CUDA machine** |
| 3 | Deployment measurements — latency, throughput, peak memory, checkpoint size | **the designated benchmark host, currently the Omen** |

Tier 2 is portable because compression and perplexity differ across machines only by
floating-point reduction order (~1e-5 relative, against the ~1e-2 effects being measured). Tier 3
is not portable under any correction: a latency is a property of the machine.

Two invariants hold regardless of tier:

- **A comparison never spans machines.** Both arms of a cell, at the same replicate, run on one
  host. The machine is one of §3.11's matched conditions. `ExperimentTracker.exists_valid`
  enforces this — a record from another host is re-run rather than reused (B-33).
- **Never mix machines in one results table.** `scripts/generate_plots.py` refuses to plot when
  *deployment-bearing* records span hosts, and every record carries `hardware` so it is checkable
  after the fact.

`outputs/`, `results/` and `data/` are git-ignored, so they exist only on machines that have run
something. Never assume a run record is present just because the repo is checked out.

**Reproducibility is a separate rule and it did not loosen.** Run from a clean tree at a committed
SHA. The one unusable set of numbers this project has produced came from a working tree 22 commits
behind `main` recorded as `aec5099-dirty` — the fault was the dirty tree, not the hardware.

## Commands

```bash
# environment (`python` on PATH may be a Microsoft Store stub on Windows -- check first)
uv venv --python 3.11
.venv\Scripts\activate          # Windows;  source .venv/bin/activate  on Linux/macOS
# CUDA build; see pytorch.org for the current index-url
uv pip install -e . -r requirements-dev.txt

pytest                                  # full suite, offline, seconds
pytest -m "not requires_torch"          # the subset that runs without torch
ruff check . && ruff format --check .   # must both pass before committing

sajc info                               # registry + hardware, no downloads
python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml --dry-run
```

`--dry-run` validates a config and prints the plan without loading a model. Use it first.

## Hard rules

These are not style preferences. Each one, if broken, silently invalidates results.

- **Deployment measurements are CPU-only, on the designated benchmark host.** Latency,
  throughput, peak memory, checkpoint size. GPU is allowed for compression and activation
  capture, never for measurement. The config loader rejects a non-CPU benchmark device; do not
  work around it.
- **Sequential and joint arms must match** on target sparsity, bit width, calibration data,
  module coverage, optimisation budget, artefact format, and backend. Tests assert this over
  the shipped configs — if a change breaks one of those tests, the change is wrong.
- **Never fabricate or estimate a result.** Unimplemented paths raise `NotImplementedError`
  naming the module to edit. Keep it that way; a plausible-looking number is worse than an error.
- **Never commit anything under `outputs/`, `results/`, or `data/`.** Structure is tracked via
  `.gitkeep`; contents are not.
- **Report measured against target**, always — measured sparsity next to target sparsity,
  `is_converted` on quantised artefacts, `storage_efficiency` on checkpoints.

## Gotchas discovered the hard way

- **Do not `sys.modules.pop("torch")` in tests.** Re-importing torch raises
  `Only a single TORCH_LIBRARY can be used`. Use the `imported_after` /
  `environment_after_import` fixtures in `tests/conftest.py`, which check in a subprocess.
- **Heavy imports are lazy on purpose.** `import scale_aware_compression` must not pull in torch
  or transformers; tests enforce it. Import them inside functions.
- **Patch where a symbol is defined, not where it is used.** Several functions import their
  dependencies locally, so patching the importing module misses.
- **On Windows, do not round-trip UTF-8 through PowerShell `Get-Content`/`Set-Content`** — it
  reads as ANSI and mangles em-dashes. Use the Read/Edit/Write tools.
- **`.gitattributes` pins LF.** Without it, `core.autocrlf` breaks `ruff format --check`.

## Code conventions

- Python 3.11+, `src/` layout, full type hints, Google-style docstrings on every public symbol.
- Ruff for lint and format, 100 columns. `ruff check .` and `ruff format --check .` must pass.
- **British spelling in domain terms**: quantisation, tokenise, normalise, optimisation. The
  package name and public API already use it; stay consistent.
- Comments explain *why*, not *what*. Prefer no comment to a restatement of the code.
- No new abstractions without a second caller.

## Testing

- Every test is offline: no downloads, no training, no real benchmarks.
- The key fixture is `tiny_causal_lm` — a randomly initialised 2-layer `GPTNeoXForCausalLM` built
  from a config object. Same architecture class as Pythia, no download, runs in milliseconds.
  Use it for anything touching a real model.
- `fake_tokenizer` is a byte-level stand-in with the slice of the Transformers API we use.
- Markers: `requires_torch` (runs in CI), `requires_model` and `slow` (excluded from CI).
- A test asserting a *fairness invariant* is more valuable than one asserting a return type.

## Workflow

At the end of a session that changed anything meaningful, **update `docs/STATUS.md`** — current
phase, what moved, what is next, any new open decision — and commit. That file is the handoff.

If the session **produced a number** — a perplexity, a timing, an effect size, a rejected
ablation — append it to **`docs/findings_log.md`** with the conditions that produced it: model
revision, budget, calibration fingerprint, evaluation window, machine. A number without its
conditions cannot go in the paper, and a number that only exists in a commit message will be
re-derived later under different conditions.
