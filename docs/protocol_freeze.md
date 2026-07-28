# Protocol freeze

**Status:** draft — decisions recorded, environment section blocked (see [Environment](#environment)).
**Opened:** 2026-07-28 · HP Omen · first session on the machine that runs code.

Research plan §2.7 requires every decision below to be fixed *before* the full experiments, and
§10.2 requires the environment recorded alongside them. This file is that record. It is the
answer to "why is it set that way" for anything a later session or a reviewer asks about.

**Rule:** a value in this file changes only by editing this file, in a commit that says why.
Changing one silently invalidates every run recorded before the change. Nothing here may be
revised after results have been seen (§6.3).

---

## The three decisions that were open

These had no answer in the code and could not be settled from it. Each is resolved below with
the plan section it follows from.

### D1 — CPU quantisation backend · **PyTorch native INT8, W4 excluded from latency**

| | |
| --- | --- |
| Latency backend | PyTorch native CPU INT8 (`x86` / fbgemm), one pinned thread count |
| W8 | quality **and** size **and** latency |
| W4 | quality **and** size only — **never** in a latency table |
| Artefact | both arms convert through the same code path; `is_converted` proves it |

PyTorch's mature CPU quantisation path targets INT8. There is no equally mature built-in 4-bit
weight-only CPU kernel, so a W4 artefact needs a packed-weight custom linear that dequantises on
the fly — which is *slower* than FP32 and would measure the kernel, not the compression.

The plan already separates these concerns and does not require a W4 latency number:

- §3.12 requires three quantities reported **separately** — theoretical non-zero reduction,
  serialised checkpoint size, measured CPU latency — and forbids inferring speedup from sparsity.
- §10.1 rates "CPU sparsity gives no speedup" as *high likelihood* and instructs that a null
  latency result be treated as valid.
- §4.7 forbids comparing across backends. Restricting the latency table to W8 satisfies this
  trivially: one runtime, one thread count, one artefact format.

**Research question 4 survives this intact.** RQ4 asks whether theoretical sparsity produces real
CPU latency gains. That is answerable from the **pruning-only arm**, which §5.4 already mandates
at two budgets per model: pruning-only weights stay FP32, so they benchmark natively at every
screened sparsity with no 4-bit kernel involved. The sparsity→latency curve comes free.

W4 still needs real int4 packing — quality and size claims require a genuinely converted
artefact, not a fake-quantised FP32 one. Phase 5 already requires a bit-exact pack/unpack
round-trip.

**Cost, to be stated in the write-up:** no latency row for a combined W4 cell. Precision is a
compression axis for quality and size, not for latency.

### D2 — Reconstruction solver · **damped ALS first, Hessian sweep as a later drop-in**

| | |
| --- | --- |
| Initial solver | alternating least squares with ridge damping `(H + λI)` |
| `H = XᵀX` | accumulated from the start, regardless of solver |
| Upgrade path | error-compensated column sweep, behind the same interface |

The plan does **not** require a second-order method. §3.3 is explicit: "the minimum viable
baseline is activation-weighted magnitude pruning; a second-order method may be added only if it
is already stable." Reference [6] (Optimal BERT Surgeon) is cited as related work, not as a bar
to clear.

Two further reasons ALS is the right first target:

1. **It matches the fairness unit.** §3.11 requires "equal total local optimisation steps and
   approximately equal objective evaluations". That is natural to define, record, and assert for
   an iterative solver. A single deterministic column sweep has no "steps" in that sense, so the
   budget would have to be matched in sweeps and then hand-reconciled against a K-iteration joint
   loop — harder to defend and harder to test.
2. **§3.7 describes joint as alternating optimisation** — "optimise surviving weights/scales for
   fixed local steps". ALS expresses that directly.

Memory is *not* the constraint. Pythia-1B and 1.4B both use hidden 2048 / intermediate 8192, so
the largest layer Hessian (`mlp.dense_4h_to_h`, input dim 8192) is 8192² × 4 B = **256 MiB** in
fp32, one layer at a time. That fits the 6 GB GPU comfortably. The real constraint is
implementation time: implementation_plan.md puts Phases 5–6 at 4–6 days against the 3–5 the
schedule allots.

Accumulate `H` immediately even though ALS barely needs it: `‖X_j‖₂` is required for saliency
(§3.3), damped ALS wants `(H + λI)`, and Phase 2's exit test already checks streamed `H` against
direct `XᵀX`. Building capture once makes the solver upgrade a contained change.

### D3 — Mask scoring · **activation-weighted magnitude, scored under quantised weights**

```
joint            S_ij = |Q_b(W_ij)| · ‖X_j‖₂        ← scored on quantised weights
sequential P→Q   S_ij = |W_ij|      · ‖X_j‖₂        ← no quantiser exists yet at mask time
```

**This overrides the recommendation previously in `method_definition.md`**, which said to rank by
FP32 shadow-weight magnitude. That recommendation belonged to the superseded full-model
quantisation-aware-training design and does not survive the move to layerwise PTQ.

The authoritative plan requires the opposite:

- §3.7: "update mask scores **using quantized or fake-quantized weights**."
- §3.8, *What qualifies as joint*: "Mask decisions are evaluated under quantized or
  fake-quantized weights." The same table lists "prune completely, freeze the result, then call
  ordinary PTQ" under **does not qualify**.

Ranking by weights the quantiser has not touched chooses a mask in ignorance of the grid, which
is precisely §3.8's failure case. Under layerwise PTQ there is also no training loop, so there are
no shadow weights shaped by thousands of QAT steps — the mechanism that made the FP32 option
defensible does not exist here.

The stability objection to scoring on quantised weights (values jittering across a grid boundary
between mask updates) is weak in this setting: there are K joint iterations rather than thousands
of optimiser steps, and §3.7 *deliberately* re-estimates scales after each mask change. That
movement is the coupling under study, not noise.

Note the saliency is **activation-weighted** magnitude per §3.3, not plain magnitude — another
point where the older markdown diverged from the plan.

The asymmetry between the two arms is not unfairness. The sequential arm *cannot* score under
quantisation because it prunes first; that is what makes it sequential, and it is the single
variable the study isolates. Fairness is enforced where §3.11 puts it: identical calibration
tensors, identical module lists and order, equal total local steps.

**Standing prohibition, unchanged:** do not introduce a combined `α·|w| + β·|w − Q(w)|` score
unless it is implemented and separately ablated. An unablated tunable score is a free parameter
that can manufacture a joint gain.

---

## §2.7 freeze table

| Decision | Frozen value | Source |
| --- | --- | --- |
| Pythia variant | **standard**, never deduped, all sizes | §2.7 |
| Model revisions | ⛔ **not yet pinned** — see [Open](#still-open) | §2.7 |
| Target layers | decoder-block linears only, per the adapter table below | §2.6, §3.10 |
| Layer exclusions | embeddings, LM head, LayerNorm, all biases | §3.10 |
| Quantisation | weight-only, symmetric, per-channel; W8 and W4; group size 128 when per-group | §3.9 |
| Activation quantisation | **off** — not part of the core design | §3.9 |
| Pruning granularity | global unstructured, identical across all sizes | §3.10 |
| Saliency | activation-weighted magnitude; under `Q_b(W)` for joint (**D3**) | §3.3, §3.7 |
| Screening sparsities | 30% / 50% / 70% of targeted weights | §3.10 |
| Benchmark runtime | PyTorch native CPU INT8, `x86`; latency at W8 only (**D1**) | §2.7, §4.7 |
| Scale x-axis | targeted non-embedding parameter count | §2.6 |
| Seeds | 1 screening · 1 first pass · 3 confirmatory | §5.5 |
| Run IDs | `<family>_<size>_<method>_<sparsity>_<bits>_<seed>` | §5.6 |

### Targeted modules

| Architecture | Attention | MLP | Excluded |
| --- | --- | --- | --- |
| `GPTNeoXForCausalLM` (Pythia) | `attention.query_key_value` (fused QKV), `attention.dense` | `mlp.dense_h_to_4h`, `mlp.dense_4h_to_h` | `gpt_neox.embed_in`, `embed_out` |
| `Qwen2ForCausalLM` | `self_attn.{q,k,v,o}_proj` | `mlp.{gate,up,down}_proj` | `model.embed_tokens`, `lm_head` (**tied** — excluding one without the other would compress the input embedding as a side effect) |

Resolved through a single `select_compressible_modules` call path so both arms cannot diverge. An
empty selection must raise, not warn: a "compressed" model identical to the dense one would
otherwise read as an excellent result.

### Additional decisions taken here

| Decision | Frozen value | Reasoning |
| --- | --- | --- |
| Downstream tasks | `lm-evaluation-harness`, **pinned version**, task versions recorded | §4.3 requires HellaSwag, PIQA, ARC-Easy and §4.8 requires logging task versions. Reimplementing three tasks in-repo risks silent scoring differences from published numbers, which is a worse failure than one heavy dependency. |
| Practical-importance rule | joint gain counts as practically important only when perplexity retention improves by **≥ 1.0 percentage point**, consistently in sign across **all three** confirmatory seeds, **and** the mean improvement exceeds the seed spread (max − min) at that cell | §6.3 requires this predefined. Stated before any compressed result exists. The seed-spread clause is what stops a gain smaller than noise being reported as a finding. |
| Downstream importance rule | ≥ 1.0 percentage point accuracy gain on at least 2 of the 3 tasks, same sign across seeds | Same reasoning; §4.3 tasks are secondary evidence. |

---

## Environment

Recorded per §10.2. **The HP Omen is the only machine that produces numbers.**

| | |
| --- | --- |
| CPU | 13th Gen Intel Core i7-13620H — 10 physical / 16 logical cores |
| RAM | 13.7 GiB |
| GPU | NVIDIA GeForce RTX 4050 Laptop — **6.0 GiB** VRAM, sm_89 (Ada) |
| NVIDIA driver | 592.82 |
| OS | Windows 11 Home 10.0.26200 |
| Power profile | ⛔ **Balanced** — §4.7 requires a fixed performance mode; must be changed and re-recorded before any benchmark |
| Python | 3.11.9 (`winget` user-scope install) |
| torch | 2.13.0+cu126 |
| transformers | 5.14.1 |
| datasets | 5.0.0 |
| numpy | 2.4.6 |
| Thread count for benchmarks | ⛔ not yet pinned |

### ⛔ Blocker: Smart App Control

**Windows Smart App Control is enabled** (`VerifiedAndReputablePolicyState = 1`, user-mode code
integrity enforced) and is blocking the unsigned native extension modules the stack depends on.
Confirmed blocked, from `Microsoft-Windows-CodeIntegrity/Operational` event 3077:

```
torch_cuda.dll                    (torch, CUDA)
shm.dll                           (torch, core -- blocks `import torch` outright)
ccalendar.cp311-win_amd64.pyd     (pandas)
interval.cp311-win_amd64.pyd      (pandas)
lib.cp311-win_amd64.pyd           (pandas)
_regex.cp311-win_amd64.pyd        (regex, a transformers dependency)
```

The blocks appeared **progressively**: the full suite passed 502 tests immediately after install,
then torch stopped importing roughly thirty minutes later as the policy caught up with the newly
written binaries. So a green test run is not evidence the environment is stable.

Consequences while this stands: no torch, therefore no compression, no evaluation, no benchmark.
pandas is separately blocked, which also takes out `datasets` (the WikiText load path) and the
CSV/table writers.

Resolution requires a decision that is **system-wide and irreversible** — Smart App Control
cannot be re-enabled without reinstalling Windows — so it is deliberately not taken here. See
[STATUS.md](STATUS.md) for the options.

---

## Still open

| Item | Why it is not frozen yet |
| --- | --- |
| Model revision SHAs | Requires querying the Hub. **Must** be pinned before any run whose numbers reach the paper (§2.7). Null revisions are acceptable only for a throwaway pilot. |
| Calibration sample indices, token count, sequence length | Frozen by the config once `prepare_data.py` has run for real. Blocked on the environment. |
| The two final budgets | Output of Phase 7 screening (S1–S4 on 160M). §5.3 requires them frozen **before** 1B. |
| 1.4B go/no-go | §5.2 needs measured peak VRAM against 85% of 6.0 GiB — a 5.1 GiB ceiling, which is tight. Decide after Phase 5 profiling. |
| Benchmark thread count | Pin once the environment runs; record here. |
| Power profile | Change from Balanced to a fixed performance mode, then re-record. |

---

## Related

- [research_plan.pdf](research_plan.pdf) — authoritative source
- [method_definition.md](method_definition.md) — the arms, as specified
- [implementation_plan.md](implementation_plan.md) — build phases and exit tests
- [STATUS.md](STATUS.md) — where the work currently stands
