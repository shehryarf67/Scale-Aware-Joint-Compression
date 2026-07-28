# Project status

**Last updated:** 2026-07-28 · first session on the HP Omen · **Phase 0 complete**, environment
verified and frozen

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

Infrastructure is done, the dense baseline runs end to end, and **the Phase 0 decisions are now
settled**. No compression algorithm is implemented yet.

| | State |
| --- | --- |
| Tests | **511 passing** in ~32 s, offline |
| Lint / format | `ruff check .` and `ruff format --check .` both clean |
| CI | `.github/workflows/ci.yml` — lint, format, tests on push/PR to `main` |
| Environment | verified end to end: torch 2.13.0+cu126, CUDA available, sm_89 |
| Runnable today | dense baseline, end to end, producing a full run record |
| Not yet implemented | pruning, quantisation, sequential, joint |

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

## Next: Phase 5 — single-layer compression primitives

**Blocked on the environment.** Nothing below can be written *and verified*, and unverified
numerical code is exactly what this project must not accumulate.

**Do not start by compressing a model.** Work on one `nn.Linear` with captured activations until
each piece is provably right. This mirrors the plan's own 48-hour checklist (§10.2).

| Primitive | What it does | Target file |
| --- | --- | --- |
| Activation capture | hook a layer, accumulate `H = XᵀX` and per-column `‖X_j‖₂` | `compression/activations.py` (new) |
| Saliency | activation-weighted magnitude, `S_ij = \|W_ij\| · ‖X_j‖₂` | `compression/pruning.py` |
| Mask | global-unstructured or blockwise at exact target sparsity | `compression/masks.py` |
| Quantiser | symmetric weight-only, per-channel / groupwise, W8 and W4 | `compression/quantisation.py` |
| Packing | int4/int8 storage + scales, bit-exact unpack round-trip | `compression/quantisation.py` |
| Reconstruct | damped ALS on `(H + λI)`, per **D2** | `compression/reconstruct.py` (new) |

**Exit test:** on a synthetic layer — reconstruction strictly reduces `‖Y − Ŷ‖²_F` versus naive
rounding; realised sparsity is exact; quantised weights take ≤ 2^b distinct values per group;
pack → unpack is lossless.

Also outstanding from Phase 0: config needs a `reconstruction` section (`local_steps`, `damping`,
`block_size`) with `local_steps` replacing `max_steps` as the fairness unit, and
`CompressionMethod` needs `SEQUENTIAL_PQ` / `SEQUENTIAL_QP` (A2).

Then Phase 6 (the five arms through one shared solver), Phase 7 (budget screening), Phase 8
(sweep). Full detail and exit tests for every phase:
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
- [ ] `python scripts/download_models.py --models pythia-160m`
- [ ] `python scripts/prepare_data.py --config configs/experiments/pilot.yaml` — first real exercise
      of the WikiText path
- [ ] `python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml` — first real
      record, on a real model
- [ ] Phase 5 — single-layer compression primitives ← **next**
