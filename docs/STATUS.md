# Project status

**Last updated:** 2026-07-28 · first session on the HP Omen · Phase 0 decisions settled,
environment blocked

> Read this first. It is the handoff between sessions and between machines. If it looks stale,
> check `git log` — the truth is the commit history, this file is a summary of it.
>
> **This file = where we are now.** For the durable roadmap (all ten phases, exit tests, the
> testing plan) see [implementation_plan.md](implementation_plan.md). For the frozen decisions and
> the environment record, [protocol_freeze.md](protocol_freeze.md). For the authoritative source,
> [research_plan.pdf](research_plan.pdf).

---

## 🛑 Blocker: Smart App Control is breaking the environment

**Nothing numerical runs on the Omen right now.** Windows Smart App Control is enabled
(`VerifiedAndReputablePolicyState = 1`, user-mode code integrity enforced) and is blocking the
unsigned native extension modules the stack depends on.

Confirmed blocked, from `Microsoft-Windows-CodeIntegrity/Operational` event 3077:

```
shm.dll                           torch, core  -- blocks `import torch` outright
torch_cuda.dll                    torch, CUDA
ccalendar.cp311-win_amd64.pyd     pandas
interval.cp311-win_amd64.pyd      pandas
lib.cp311-win_amd64.pyd           pandas
_regex.cp311-win_amd64.pyd        regex, a transformers dependency
```

**The blocks appeared progressively.** The full suite passed **502 tests** immediately after
install; torch stopped importing roughly thirty minutes later as the policy caught up with the
newly written binaries. So a green run is not evidence the environment is stable — re-check before
trusting any result.

What this takes out: torch (so all compression, evaluation, benchmarking), and separately pandas,
which also takes out `datasets` (the WikiText load path) and the CSV/table writers.

### The options

| Option | Effect | Cost |
| --- | --- | --- |
| **A. Turn Smart App Control off** | Unblocks everything natively; cleanest measurement environment for §4.7 | **Irreversible** — SAC cannot be re-enabled without reinstalling Windows. Lowers the machine's security posture. |
| **B. Move to WSL2** | Linux binaries are not subject to SAC; CUDA works via the existing 592.82 driver | Adds a virtualisation layer under the CPU benchmark. Acceptable only if *every* number comes from it, and it must be recorded as the runtime per §2.7. |
| **C. Avoid the blocked packages** | Not viable — torch is the blocker, and it is not optional. | — |

**Recommendation: A**, because §4.7 wants one pinned, unvirtualised runtime and the Omen is already
designated the only machine that produces numbers. B is the fallback if turning SAC off is
unacceptable. Either way, record the choice in
[protocol_freeze.md](protocol_freeze.md#environment).

This is a system-wide, irreversible setting, so it is deliberately **not** actioned here.

---

## Where we are

Infrastructure is done, the dense baseline runs end to end, and **the Phase 0 decisions are now
settled**. No compression algorithm is implemented yet.

| | State |
| --- | --- |
| Tests | **508** (502 + 6 new doc guards); last full green run was before the SAC blocks landed |
| Lint / format | `ruff check .` and `ruff format --check .` both clean |
| CI | `.github/workflows/ci.yml` — lint, format, tests on push/PR to `main` |
| Runnable today | **nothing** — see the blocker above |
| Runnable once unblocked | dense baseline, end to end, producing a full run record |
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
| **D1** | CPU quantisation backend | PyTorch native CPU **INT8** (`x86`) is the sole latency backend. W4 keeps quality + size, never appears in a latency table. **RQ4 survives** — the sparsity→latency curve comes free from the pruning-only arm, whose weights stay FP32. |
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

- **Smart App Control** — the blocker above. Needs a decision.
- **Power profile** — currently **Balanced**; §4.7 requires a fixed performance mode. Change and
  re-record before any benchmark.
- **Benchmark thread count** — pin once the environment runs.
- **Model revision SHAs** — still `revision: null`. Fine for a pilot; **must** be pinned before any
  run whose numbers reach the paper (§2.7).
- **The two final budgets** — output of Phase 7 screening on 160M/410M. §5.3 requires them frozen
  before 1B.
- **1.4B go/no-go** — §5.2 needs peak VRAM under ~85% of 6.0 GiB, a **5.1 GiB ceiling**. Tight.
  Decide after Phase 5 profiling.

Settled since the last revision, no longer open: the backend, the solver, the mask scoring rule, the
Pythia variant (**standard**), lm-eval-harness (**yes, pinned**), and the practical-importance
threshold (**≥ 1.0 pp retention, consistent in sign across all three confirmatory seeds, exceeding
the seed spread**).

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
- [ ] **Resolve the Smart App Control blocker** ← everything below waits on this
- [ ] Re-run `pytest` and confirm it is *stably* green
- [ ] Switch the power profile off Balanced; pin the benchmark thread count
- [ ] `python scripts/download_models.py --models pythia-160m`
- [ ] Pin model revision SHAs in all five model configs
- [ ] `python scripts/prepare_data.py --config configs/experiments/pilot.yaml` — first real exercise
      of the WikiText path
- [ ] `python scripts/run_dense_baseline.py --config configs/experiments/pilot.yaml` — first real
      record, on a real model
- [ ] Start Phase 5
