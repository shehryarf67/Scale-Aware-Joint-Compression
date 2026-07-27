# CLAUDE.md

Guidance for Claude Code working in this repository. Read
[docs/STATUS.md](docs/STATUS.md) **first, every session** — it holds the current phase, what is
in progress, and the decisions that are still open.

@docs/STATUS.md

## What this project is

A controlled study of whether **joint** pruning-and-quantisation beats **sequential**
pruning-then-quantisation, and whether the gap changes with model scale (Pythia 160M → 410M → 1B).

The governing documents, in order of authority:

1. `docs/method_definition.md` — exactly what the two arms are. **The spec.**
2. `docs/experiment_protocol.md` — the run tables and checklists.
3. `docs/benchmarking_protocol.md` — CPU measurement rules.
4. `docs/validity_threats.md` — what could make the results wrong.

If code and docs disagree, that is a bug in one of them. Say so rather than picking silently.

## Machine policy

**The HP Omen is the only machine that runs code.** It has the NVIDIA GPU and it is the machine
every benchmark number must come from. The other laptop is for reading and writing only.

Consequences:

- `outputs/` and `results/` are git-ignored, so they exist **only on the Omen**. Never assume a
  run record is present just because the repo is checked out.
- Never mix benchmark numbers from two machines in one table. The protocol forbids it and every
  record carries `hardware.cpu_model` so it is checkable.

## Commands

```bash
# environment (Windows; `python` on PATH may be a Microsoft Store stub -- check first)
uv venv --python 3.11 && .venv\Scripts\activate
# CUDA build on the Omen; see pytorch.org for the current index-url
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

- **Deployment measurements are CPU-only.** Latency, throughput, peak memory, checkpoint size.
  GPU is allowed for compression and activation capture, never for measurement. The config
  loader rejects a non-CPU benchmark device; do not work around it.
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
