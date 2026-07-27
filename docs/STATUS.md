# Project status

**Last updated:** 2026-07-27 · after implementing Steps 1–3 (data, evaluation, dense baseline)

> Read this first. It is the handoff between sessions and between machines. If it looks stale,
> check `git log` — the truth is the commit history, this file is a summary of it.
>
> **This file = where we are now.** For the durable roadmap (all ten phases, exit tests, the
> testing plan) see [implementation_plan.md](implementation_plan.md). For the authoritative
> source, [research_plan.pdf](research_plan.pdf).

---

## Where we are

The **infrastructure is done and the dense baseline runs end to end.** No compression algorithm
is implemented yet.

| | State |
| --- | --- |
| Tests | **494 passing**, offline, ~11 s |
| Lint / format | `ruff check .` and `ruff format --check .` both clean |
| CI | `.github/workflows/ci.yml` — lint, format, tests on push/PR to `main` |
| Runnable today | dense baseline, end to end, producing a full run record |
| Not yet runnable | pruning, quantisation, sequential, joint |

### What works

- **Config system** — YAML with `include:` composition, typed dataclasses, validation at load.
- **Model registry** — 5 models, offline lookup, safe loader that validates before downloading.
- **Data pipeline** — tokenise → chunk → cache → fingerprint; calibration draw from a fixed seed
  with a held-out subset for the overfitting check.
- **Evaluation** — perplexity, dense-vs-compressed agreement, generation diagnostics.
- **CPU benchmark** — pinned threads, warm-up, repeated runs, median/p95/IQR-ready statistics.
- **Run records** — JSON + CSV, with git commit, hardware, software versions, and `status`.
- **`ExperimentRunner.run`** — complete for the dense arm.

### Sanity check that the evaluation is correct

An untrained model with a 259-token vocabulary scored **perplexity 257.18**. A near-uniform model
should score near |V|, so this confirms the loss is computed over the right axis. A shift or shape
bug lands nowhere near it. Dense retention computes as exactly 100% (it is its own reference).

---

## ⚠️ The most important thing to know

**The research plan specifies a different method family than the scaffold's config assumes.**

`Research Plan §3.1` requires **layerwise post-training reconstruction**: for each linear layer,
capture calibration activations `X` and choose the compressed weights to minimise
`‖XW − X(M ∘ Q(W))‖²_F`. It explicitly avoids full-model fine-tuning, because that will not fit
at 1B on a laptop.

The scaffold was built assuming **full-model quantisation-aware fine-tuning** — a `Trainer`,
global optimiser steps, gradual sparsity ramps across training, mask-freeze ratios.

**These are not variations of each other.** The unit of optimisation differs: *local steps per
layer*, not *global optimiser steps*.

What this means in practice:

- The infrastructure survives — config, registry, records, benchmark, metrics, tests.
- `training/` (trainer, recovery, callbacks) **drops off the critical path**. Keep the files;
  they become the optional fine-tune ablation.
- The `Compressor` ABC still fits, with reinterpreted stages: `prepare` = select modules +
  install activation capture, `apply` = **the layerwise loop**, `recover` = no-op in the core
  method, `convert` = pack to real low-bit storage.
- Config needs a `compression.reconstruction` section (`local_steps`, `damping`, `block_size`)
  and `local_steps` becomes the fairness unit in place of `max_steps`.

The full reconciliation list (A1–A9), with sizes and the reinterpreted `Compressor` stage
semantics, is in [implementation_plan.md](implementation_plan.md#reconciliation-plan-versus-scaffold).
A summary is repeated below for convenience.

---

## Next: Step 4 — single-layer compression primitives

**Do not start by compressing a model.** Work on one `nn.Linear` with captured activations, in a
notebook, until each piece is provably right. This mirrors the plan's own 48-hour checklist
(§10.2) and is where the method is actually validated.

| Primitive | What it does |
| --- | --- |
| Activation capture | hook a layer, accumulate `H = XᵀX` and per-column `‖X_j‖₂` |
| Saliency | activation-weighted magnitude: `S_ij = \|W_ij\| · ‖X_j‖₂` |
| Mask | global-unstructured or blockwise at a target sparsity |
| Quantiser | symmetric weight-only, per-channel / groupwise, W8 and W4 |
| Packing | int4/int8 storage + scales, bit-exact unpack round-trip |
| Reconstruct | error-compensated column sweep with ridge damping `(H + λI)` |

**Exit test:** on a synthetic layer — reconstruction strictly reduces `‖Y − Ŷ‖²_F` versus naive
rounding; realised sparsity is exact; quantised weights take ≤ 2^b distinct values per group;
pack → unpack is lossless.

Then Phase 6 (the five arms through one shared solver), Phase 7 (budget screening), Phase 8
(sweep). Full detail and exit tests for every phase:
[implementation_plan.md](implementation_plan.md#phases).

---

## 🔴 Open decisions — settle before implementing Step 4

None of these can be resolved from the code, and none should be resolved after seeing results.

### 1. CPU quantisation backend — decide first, it constrains everything

PyTorch's native CPU quantisation targets **INT8**. There is no equally mature built-in 4-bit
weight-only CPU kernel, so W4 deployment generally needs a separate backend.

Options: (a) report W4 quality and size but benchmark latency only at W8, stating the limitation;
(b) find one 4-bit CPU runtime usable by **both** arms; (c) INT8-only fallback throughout.

**Recommendation: (a).** Layerwise PTQ makes the *compression* side of W4 straightforward; only
deployment is the problem. This keeps W4 in the study for quality claims while avoiding an
incomparable latency table.

### 2. Reconstruction solver depth

Full GPTQ/SparseGPT-style error-compensated Hessian sweep (stronger, ~2–3 days, what reviewers
will expect given refs [4][5][6]) versus simpler alternating least squares (faster, weaker).

**Recommendation:** build ALS first as a working fallback, upgrade to the Hessian sweep if the
schedule allows. This is the largest single implementation risk in the 2–3 week plan.

### 3. Mask scoring rule

Rank by fake-quantised magnitude, or by FP32 shadow-weight magnitude with fake quantisation
active? `docs/method_definition.md#mask-scoring` recommends the latter, with reasons. Confirm or
override. **Do not** invent a combined α·β score unless it is implemented and separately ablated.

### Also unresolved

- **Pythia variant** — standard or deduped. Must be one, consistently (plan §2.7). Registry
  currently points at standard.
- **lm-eval-harness** as a dependency, or three tasks implemented in-repo? Plan cites the harness.
- **Practical-importance threshold** (§6.3) — must be predefined *before* results are viewed.
- **Model revisions** — still unpinned (`revision: null`). Fine for the pilot; **must** be pinned
  commit SHAs before any run whose numbers reach the paper.

---

## Deferred reconciliation (plan vs scaffold)

Known gaps, not yet implemented. Roughly in priority order.

| # | Gap | Size |
| --- | --- | --- |
| A1 | Layerwise reconstruction: `activations.py`, `reconstruct.py`, `layerwise.py` | large |
| A2 | Config: `reconstruction` section; `local_steps` replaces `max_steps` as the fairness unit | medium |
| A3 | Reverse sequential **Q→P**, and joint gain vs **best-of** {P→Q, Q→P} (plan §3.6, §6.1) | medium |
| A4 | Downstream tasks — HellaSwag, PIQA, ARC-Easy are **required** (§4.3) | medium |
| A5 | Prefill vs decode timed **separately**, at 128 and 512 prompt lengths; IQR; model-order rotation (§4.7) | small–medium |
| A6 | **Targeted non-embedding parameter count** as the scale x-axis (§2.6) — currently uses total | small |
| A7 | Seed policy: 1 for screening/first pass, 3 confirmatory only (§5.5). Current sweep runs 3 everywhere → ~40% wasted compute | small |
| A8 | Budget **screening** stage S1–S4 on 160M, freeze 2 budgets before 1B (§5.3) | small |
| A9 | Record fields: per-layer reconstruction loss, compression time, peak GPU memory, effective bits, tokenizer revision | small |

---

## Environment notes

- **Omen is the only machine that runs code.** `outputs/`, `results/`, and `data/` are
  git-ignored and exist only there.
- On the Omen, install the **CUDA** torch build (compression); benchmarks still force CPU.
- The other laptop has no NVIDIA GPU, no Python by default (`python` is a Store stub), and its
  `datasets` import fails on a pandas DLL blocked by an Application Control policy. Every data
  test stubs the corpus, so this does not affect the suite — but it does mean **the real
  WikiText load path has never been executed**. Treat the first `prepare_data.py` run on the
  Omen as its first real test.

---

## Immediate checklist for the Omen

- [ ] `git clone` (after this work is pushed)
- [ ] Python 3.11 + venv; CUDA torch from pytorch.org; `pip install -e . -r requirements-dev.txt`
- [ ] `pytest` → expect 494 passing
- [ ] `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
      — record the GPU and VRAM; the plan's §5.2 go/no-go for 1.4B needs it
- [ ] `python scripts/download_models.py --models pythia-160m`
- [ ] `python scripts/prepare_data.py --config configs/experiments/pilot.yaml` — first real
      exercise of the WikiText path
- [ ] `python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml` — first
      real record, on a real model
- [ ] Settle the three open decisions above
- [ ] Start Step 4
