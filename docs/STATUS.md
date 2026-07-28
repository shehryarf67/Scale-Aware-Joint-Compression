# Project status

**Last updated:** 2026-07-28 · first session on the HP Omen · **Phases 0, 5 and 6 complete**;
all five arms run end to end on a real model

> Read this first. It is the handoff between sessions and between machines. If it looks stale,
> check `git log` — the truth is the commit history, this file is a summary of it.
>
> **This file = where we are now.** For the durable roadmap (all ten phases, exit tests, the
> testing plan) see [implementation_plan.md](implementation_plan.md). For the frozen decisions and
> the environment record, [protocol_freeze.md](protocol_freeze.md). For the authoritative source,
> [research_plan.pdf](research_plan.pdf).

---

## ✅ Resolved: the Smart App Control blocker

Smart App Control was enforcing user-mode code integrity and blocking the unsigned native
extensions the stack depends on — `shm.dll` and `torch_cuda.dll` (torch), three pandas `.pyd`
files, and `_regex` (a transformers dependency). It appeared *progressively*: the suite passed 502
tests right after install, then torch stopped importing about thirty minutes later as the policy
caught up with the newly written binaries.

**Turned off on 2026-07-28 and the machine rebooted.** Verified afterwards:
`VerifiedAndReputablePolicyState = 0`, user-mode code integrity enforcement `0`, and `torch`,
`pandas`, `regex` and `datasets` all import. The suite now runs in ~32 s rather than ~108 s,
because torch is no longer retrying blocked loads.

This is **irreversible** — Smart App Control cannot be re-enabled without reinstalling Windows.
Consequence worth remembering: this machine no longer screens unsigned executables, so be
deliberate about what gets installed on it.

If the environment ever appears to break again, re-check that state value first — a green test run
is not by itself evidence the environment is stable.

---

## Where we are

Infrastructure, the Phase 0 decisions, the compression primitives and the layerwise driver are all
done. **Every arm runs from a config to a run record on real Pythia-160M.**

| | State |
| --- | --- |
| Tests | **732 passing** in ~36 s, offline |
| Lint / format | `ruff check .` and `ruff format --check .` both clean |
| CI | `.github/workflows/ci.yml` — lint, format, tests on push/PR to `main` |
| Environment | verified end to end: torch 2.13.0+cu126, CUDA available, sm_89 |
| Runnable today | **all five arms** plus dense, config to run record, on a real model |
| Not yet done | budget screening (Phase 7), downstream tasks (A4), prefill/decode split (A5) |

### What works

- **Config system** — YAML with `include:` composition, typed dataclasses, validation at load.
- **Model registry** — 5 models, offline lookup, safe loader that validates before downloading.
- **Data pipeline** — tokenise → chunk → cache → fingerprint; calibration draw from a fixed seed
  with a held-out subset for the overfitting check.
- **Evaluation** — perplexity, dense-vs-compressed agreement, generation diagnostics.
- **CPU benchmark** — pinned threads, warm-up, repeated runs, median/p95/IQR-ready statistics.
- **Run records** — JSON + CSV, with git commit, hardware, software versions, and `status`.
- **`ExperimentRunner.run`** — complete for every arm, including calibration injection.

### Sanity check that the evaluation is correct

An untrained model with a 259-token vocabulary scored **perplexity 257.18**. A near-uniform model
should score near |V|, so this confirms the loss is computed over the right axis. A shift or shape
bug lands nowhere near it. Dense retention computes as exactly 100% (it is its own reference).

---

## ✅ Settled this session: the method family and the three open decisions

The repo carried **two generations of plan** that contradicted each other. The older markdown
specified full-model quantisation-aware fine-tuning; `research_plan.pdf` §3.1 specifies **layerwise
post-training reconstruction**, which is what actually fits on a 6 GB laptop GPU. The markdown was
still labelled "the spec", so it was quietly overriding the plan on real choices — D3 below was a
direct casualty.

Fixed:

- [method_definition.md](method_definition.md) **rewritten** against §3.1–3.12 — layerwise
  objective, the five arms through one shared solver, `local steps` as the fairness unit, Q→P
  reverse ablation, best-of-sequential. The fine-tuning material survives only as the optional
  recovery ablation.
- [protocol_freeze.md](protocol_freeze.md) **created** — the §2.7 freeze table, the environment
  record, and the three decisions below with the plan sections they follow from.
- Six new doc-guard tests, so the superseded design cannot creep back silently.

| # | Decision | Settled as |
| --- | --- | --- |
| **D1** | CPU quantisation backend | PyTorch native CPU **INT8**, engine **`onednn`**, is the sole latency backend. W4 keeps quality + size, never appears in a latency table. **RQ4 survives** — the sparsity→latency curve comes free from the pruning-only arm, whose weights stay FP32. |
| **D2** | Reconstruction solver depth | **Damped ALS first**, Hessian sweep as a later drop-in behind the same interface. §3.3 makes second-order optional, not expected. `H = XᵀX` accumulated from the start regardless. Memory is *not* the constraint: the worst-case layer Hessian is 256 MiB. |
| **D3** | Mask scoring rule | **Activation-weighted magnitude, scored on the quantised weights** in the joint arm — `S_ij = \|Q_b(W_ij)\| · ‖X_j‖₂`. This **overrides** the old Option B recommendation, which would have failed §3.8's definition of joint. |

Full reasoning for each is in
[protocol_freeze.md](protocol_freeze.md#the-three-decisions-that-were-open).

---

## ✅ Phase 5 — single-layer compression primitives: **done and passing**

Every primitive is implemented at the **tensor level** and validated against a synthetic layer, as
the plan's §10.2 checklist prescribes. 93 new tests.

| Primitive | Where | Status |
| --- | --- | --- |
| Activation capture — streamed `H = XᵀX`, `‖X_j‖₂`, relative damping, forward hook | `compression/activations.py` | done |
| Saliency — `S_ij = \|W_ij\| · ‖X_j‖₂` | `compression/pruning.py` | done |
| Mask — unstructured + 2:4 / 4:8, **exact** realised sparsity | `compression/masks.py` | done |
| Quantiser — symmetric, per-tensor / channel / group, W8 + W4 (+ W2) | `compression/quantisation.py` | done |
| Packing — int2/4/8 storage, bit-exact unpack, effective-bits accounting | `compression/quantisation.py` | done |
| Reconstruct — damped ALS on `(H + λI)`, per **D2** | `compression/reconstruct.py` | done |

**Exit criteria, all asserted:**

- reconstruction strictly reduces `‖Y − Ŷ‖²_F` versus naive rounding — both pruning-only and
  combined with W4
- realised sparsity is exact, including when scores tie (which activation weighting makes routine,
  since a dead input column zeroes a whole column of scores)
- quantised weights take ≤ 2^b distinct values per group, across all three granularities
- pack → unpack is bit-exact, tested at sizes that are *not* multiples of the lane count, because
  padding is where that breaks

Two design points worth knowing, both load-bearing:

- **The solve redistributes pruned mass.** The right-hand side is `H w` over the *full* dense row,
  so survivors absorb what the pruned weights were contributing. That error compensation is most of
  what reconstruction buys, and there is a test that fails if it is dropped.
- **The refinement loop only accepts improvements.** Projecting onto a discrete grid is not
  guaranteed to reduce a quadratic objective, so an unguarded loop can finish worse than it
  started. Naive rounding is iterate zero and nothing replaces it unless it measurably wins — which
  is what makes "reconstruction improved the layer" safe to report.

### Known limitation, deliberately left

`solve_masked_rows` does one dense solve per output channel, roughly `out_features × |S|³`. Correct
and fine for validation and the small end of the sweep, but it **will not scale** to
`in_features = 8192` as written. Phase 6 needs either mask-grouping (rows sharing a keep-set solved
together) or the Hessian column sweep D2 defers. Chosen simple-and-obviously-correct first, and
flagged in the docstring rather than discovered later.

---

## 🔴 Characterised in Phase 6: the joint mechanism is weak, and W4 must carry the comparison

**Read [validity_threats.md](validity_threats.md#the-joint-mechanism-may-be-inert-at-moderate-precision)
before running any screening.** Investigated on **six real Pythia-160M layers** with the real
calibration set, not the synthetic layer the first pass used.

| Bits | Joint vs sequential mask differs | Max-abs scale moves on refit | Layer-objective joint gain |
| --- | --- | --- | --- |
| W8 | **0.46%** | 0.2% | −0.49% |
| W4 | **8.86%** | 0.2% | **+1.12%** |

- **The mask mechanism is live at W4, inert at W8.** My synthetic measurement said *zero* divergence
  at W4 and was wrong — real weights have heavier tails and real activations have outlier channels.
- **The scale mechanism is inert at every width.** Pruning removes the smallest weights so each row's
  maximum survives. Correction to an earlier claim: this is empirical, not provable — activation
  weighting *can* prune a row's largest weight, and does at 1.3% of channels in layer 11's MLP.

**Two candidate fixes were implemented, measured, and rejected**, both because they made the layer
objective worse:

| Configuration | W8 | W4 |
| --- | --- | --- |
| max-abs + magnitude (**default**) | −0.49% | **+1.12%** |
| + clipping scale search | −1.51% | −0.99% |
| + keep-benefit scoring | −11.83% | **−16.15%** |

The clipping search cuts *naive* quantisation error by 12.8% and does make scale re-estimation live
(70% of channels move, vs 0.2%) — but clipping saturates outliers, and a saturated weight cannot be
repaired by error compensation, so the post-reconstruction result degrades. Keep-benefit scoring is
worse and analytically so: it reduces to a monotone transform of activation-weighted magnitude plus a
near-constant, so it mostly reproduces the magnitude ranking and where it deviates it favours weights
that happen to sit near a grid point.

Both are retained as declared ablations behind `compression.reconstruction.scale_search` and
`.keep_benefit_saliency`, defaulting off.

**Consequences for the experimental design:**

- W4 carries the headline comparison; W8 is a **control**, where a near-zero gain is the expected
  result rather than a failure.
- Do **not** move to W2 to chase a larger effect — the mechanism is more active there, but selecting
  a precision because it yields a positive result is what §6.3 forbids.
- A criterion that respects error compensation needs the inverse-Hessian term, not a diagonal
  approximation. Research direction, not a config change.

Every number above is pinned by tests in `tests/test_layerwise.py`.

---

## ✅ Phase 6 — complete, and verified on a real model

All five arms run through one shared driver, from a YAML config to a run record. Verified end to end
on **real Pythia-160M** with the real WikiText calibration set:

| | |
| --- | --- |
| Target modules | 48 · **84,934,656** targeted parameters (the §2.6 scale x-axis) |
| Measured sparsity | **0.5000** against a 0.5 target |
| Effective bits/weight | **8.03** — real 8-bit plus scale overhead |
| Storage efficiency | **0.89** |
| Reconstruction | **+40.9%** mean objective improvement over naive rounding (27%–65%) |
| Dense perplexity | 34.77 |
| Joint 50% + W8 perplexity | 231.96 → retention **15.0%** |

Delivered this session:

- `compression/arms.py` — the five arms as thin declarations over the driver, plus
  `plan_from_config` so every arm derives its budget from one function. `prepare` refuses to run if
  the arm and the config's method disagree, because the budget comes from the config.
- `compression/packed.py` — `PackedLinear` holding int2/4/8 codes plus fp32 scales and **no mask
  buffer** (a byte-per-weight mask at 4 bits would be twice the size of the weights it describes).
  Scheme metadata is an int64 tensor rather than `get_extra_state`, so a packed model saves through
  the same `save_pretrained` path as the dense baseline — which is what makes their checkpoint sizes
  comparable.
- `SEQUENTIAL_QP` registered; all five methods now runnable. The older fine-tuning compressors stay
  importable but unregistered, so one cannot be run by accident.
- `ExperimentRunner` draws the calibration set once and injects it, so every arm at a budget sees
  byte-identical data.

### 🔴 Open quality question — resolve before Phase 7 screening

**Retention is 15%, and that is much worse than it should be.** The pipeline itself checks out:
sparsity exact, bit width real, reconstruction improving every layer by 27–65%. Calibration size is
**not** the cause — 8× more data (16 → 128 sequences) moved perplexity only from 231.96 to 227.08.

Published one-shot 50% pruning results on comparable-scale models degrade far less than this, so
something in the configuration is costing more than it should. The leading suspect is the **mask
comparison group**: the mask is ranked *globally across each tensor*, which permits some output rows
to be pruned far harder than others — possibly near-entirely. Wanda-style **per-output-row** ranking
is known to do substantially better for exactly this reason, and §3.10 permits either. Measure it on
160M before freezing any budget.

Two smaller things noticed in the same runs:

- **Run IDs collide across arms.** `experiment.id` is `pilot` for every arm, so a compressed run
  overwrites the dense record it needs for retention. §5.6's convention
  (`<family>_<size>_<method>_<sparsity>_<bits>_<seed>`) fixes it; currently on the A-list.
- **`.gitignore` only covered named subdirectories.** A run writes to `outputs/<experiment_id>/`,
  which nothing matched — so the first real run left `outputs/pilot/` untracked *and unignored*, one
  `git add -A` from being committed. Closed with catch-all rules, and now enforced by tests that
  check the git index rather than the working tree (the old test asserted `outputs/` was empty, which
  fails on the one machine that is supposed to fill it).

---

## Next: Phase 7 — budget screening

The driver and all five arms are done, tested, and wired into `ExperimentRunner`.

Done this session:

- **Phase 2 adapters** — `select_compressible_modules` (adapter-gated *and* substring-gated, raising
  `EmptySelectionError` rather than returning the dense model), `get_decoder_blocks`,
  `get_weight_tensors`, `get_linear_modules`, `describe_architecture`,
  `count_targeted_parameters` (A6).
- **A2 config** — a `reconstruction` section (`solver`, `local_steps`, `joint_iterations`,
  `damping`, `block_size`, `activation_order`), with `local_steps` as the fairness unit.
- **`CompressionMethod.SEQUENTIAL_QP`** and the `ReconstructionSolver` / `SaliencyRule` enums (A3).
  `SEQUENTIAL` remains P→Q, the primary order, so no existing config changed meaning.
- **`compression/layerwise.py`** — depth-order block iteration, activations captured *through the
  already-compressed prefix*, all five arms as call-order variations on one solver,
  `assert_matched_plans` enforcing §3.11.

Still to do:

1. Wire `Pruner` / `Quantiser` / `SequentialCompressor` / `JointCompressor` stage methods to the
   driver, and register `SEQUENTIAL_QP` in `COMPRESSOR_REGISTRY`.
2. `convert` — real int4/int8 packing into a deployable artefact, so `is_converted` and
   `storage_efficiency` mean something.
3. Decide the scale-rule question above before screening.

Then Phase 7 (budget screening), Phase 8 (sweep). Full detail and exit tests for every phase:
[implementation_plan.md](implementation_plan.md#phases).

---

## 🔴 Still open

Reduced to the items that genuinely cannot be settled yet. Tracked in
[protocol_freeze.md](protocol_freeze.md#still-open).

- **Calibration indices, token count, sequence length** — frozen by config once `prepare_data.py`
  has run for real. The WikiText load path has still never been executed anywhere.
- **The two final budgets** — output of Phase 7 screening on 160M/410M. §5.3 requires them frozen
  before 1B.
- **W4 latency via `torchao`** — deferred to Phase 6. Would lift D1's "no W4 latency row"
  limitation if one 4-bit CPU path can serve both arms. Needs measuring, not assuming.
- **1.4B go/no-go** — §5.2 needs peak VRAM under ~85% of 6.0 GiB, a **5.1 GiB ceiling**. Tight.
  Decide after Phase 5 profiling.

Settled since the last revision, no longer open: the backend (**`onednn`** — see below), the
solver, the mask scoring rule, the Pythia variant (**standard**), lm-eval-harness (**yes, pinned**),
the practical-importance threshold (**≥ 1.0 pp retention, consistent in sign across all three
confirmatory seeds, exceeding the seed spread**), Smart App Control, the power profile (**High
performance**, no downclocking, never sleeps), the benchmark thread count (**4**, inside the P-core
budget), and all five **model revision SHAs**.

### One correction found by probing rather than reading docs

D1 originally froze the latency backend as `x86`, the name every PyTorch tutorial uses. On the
pinned torch 2.13.0+cu126 `supported_engines` is **`['onednn']` only** — `x86`, `fbgemm` and
`qnnpack` all raise "quantized engine is not supported". The shipped configs said `x86`, so
conversion would have failed *after* the compression compute was spent. Corrected everywhere, and a
`requires_torch` test now asserts the shipped backend against the installed torch so a future
upgrade that renames engines fails a test instead of a run.

Also recorded: `torch.ao.quantization` and the `qint8`/`quint8`/`qint32` dtypes are both deprecated
in favour of `torchao`. Neither blocks the study — §2.7 pins the environment for its duration — but
**the torch version must not be upgraded mid-study**.

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

A3 and D3 interact: `CompressionMethod` gains `SEQUENTIAL_QP`, and joint gain must be computed
against best-of-sequential with the winning order recorded (§6.1).

---

## Environment notes

Full record in [protocol_freeze.md](protocol_freeze.md#environment). Summary:

- **Omen is the only machine that runs code.** `outputs/`, `results/`, and `data/` are git-ignored
  and exist only there.
- CPU i7-13620H (10P/16L) · 13.7 GiB RAM · **RTX 4050 Laptop, 6.0 GiB VRAM**, sm_89 · driver 592.82
- Python 3.11.9 · torch **2.13.0+cu126** · transformers 5.14.1 · datasets 5.0.0 · numpy 2.4.6
- Installed from scratch this session: the Omen had **no Python at all** (`python` was the Microsoft
  Store stub, and there was no uv or conda).
- **13.7 GiB system RAM is worth watching.** CPU-only evaluation of Pythia-1.4B in FP32 is ~5.6 GiB
  of weights before activations, and the benchmark is CPU-bound by design.
- The real **WikiText load path has still never been executed** — every data test stubs the corpus.
  Treat the first `prepare_data.py` run as its first real test, and note `datasets` is currently
  blocked independently of torch.

---

## Immediate checklist for the Omen

- [x] `git clone`
- [x] Python 3.11 + venv; CUDA torch (`cu126`); `pip install -e . -r requirements-dev.txt`
- [x] `pytest` → 502 passing (before the SAC blocks landed)
- [x] Record GPU and VRAM for §5.2 — **RTX 4050, 6.0 GiB**
- [x] Settle the three open decisions → [protocol_freeze.md](protocol_freeze.md)
- [x] Resolve the Smart App Control blocker — **off, rebooted, verified**
- [x] Re-run `pytest` → **511 passing in ~32 s**, stably green
- [x] Power profile → **High performance**; thread count pinned at **4**
- [x] Pin model revision SHAs in all five model configs
- [x] Probe the real quantisation backend → **`onednn`**, not `x86`
- [x] **Phase 5 — single-layer compression primitives**, all exit criteria asserted
- [ ] `python scripts/download_models.py --models pythia-160m`
- [ ] `python scripts/prepare_data.py --config configs/experiments/pilot.yaml` — first real exercise
      of the WikiText path
- [ ] `python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml` — first real
      record, on a real model
- [ ] Phase 6 — the five arms through one shared layerwise driver ← **next**
