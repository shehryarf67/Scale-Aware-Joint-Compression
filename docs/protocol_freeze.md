# Protocol freeze

**Status:** draft — decisions recorded, environment section blocked (see [Environment](#environment)).
**Opened:** 2026-07-28 · HP Omen · first session on the machine that runs code.

Research plan §2.7 requires every decision below to be fixed *before* the full experiments, and
§10.2 requires the environment recorded alongside them. This file is that record. It is the
answer to "why is it set that way" for anything a later session or a reviewer asks about.

**Rule:** a value in this file changes only by editing this file, in a commit that says why.
Changing one silently invalidates every run recorded before the change. Nothing here may be
revised after results have been seen (§6.3).

> 📋 **Amended by [Protocol Amendment A1](protocol_amendment_a1.md), 2026-07-30.** A1 withdraws the
> §5.5 run-seed axis (the pipeline is deterministic, so it measured nothing), amends §6.3, splits
> evaluation into validation-for-selection and test-for-confirmation, and enforces the best-of-two
> sequential ordering §3.6 always required. **It also declares every result this project has produced
> so far exploratory**, including the 410M table below. The frozen budgets are *not* reopened.

---

## The three decisions that were open

These had no answer in the code and could not be settled from it. Each is resolved below with
the plan section it follows from.

### D1 — CPU quantisation backend · **PyTorch native INT8, W4 excluded from latency**

| | |
| --- | --- |
| Latency backend | PyTorch native CPU INT8, engine **`onednn`**, one pinned thread count |
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
screened sparsity with no 4-bit kernel involved, so the sparsity→latency curve is *obtainable*
without a 4-bit kernel.

⚠️ **"Comes free" was too strong, and this said so before A5 ran.** It comes free only if the
pruning-only arm is actually benchmarked at *several* sparsities. [F-34](findings_log.md#f-34)
measured one (30%, the frozen budgets' value) at three scales and found no commensurate speedup.
That is one point against dense, not a curve. Reporting a curve needs additional sparsities, and
choosing them *after* seeing the null would need declaring as a follow-up rather than presented as
part of the original design.

W4 still needs real int4 packing — quality and size claims require a genuinely converted
artefact, not a fake-quantised FP32 one. Phase 5 already requires a bit-exact pack/unpack
round-trip.

**Cost, to be stated in the write-up:** no latency row for a combined W4 cell. Precision is a
compression axis for quality and size, not for latency.

#### Backend probe, 2026-07-28 — the engine is `onednn`, not `x86`

Probed directly against the pinned torch rather than taken from documentation:

```
torch.__version__                          2.13.0+cu126
torch.backends.quantized.supported_engines ['onednn']
  engine = 'x86'      -> RuntimeError: quantized engine X86 is not supported
  engine = 'fbgemm'   -> RuntimeError: quantized engine FBGEMM is not supported
  engine = 'qnnpack'  -> RuntimeError: quantized engine QNNPACK is not supported
  engine = 'onednn'   -> ok
```

An end-to-end check confirms the path is functional: `quantize_dynamic` on an `nn.Linear`
produces a module whose stored weight dtype is genuinely `torch.qint8`, and its forward pass
runs. So INT8 CPU deployment works — under the engine name `onednn` only.

The shipped configs said `backend: x86`, which would have failed at conversion time. Corrected in
[`quantisation.yaml`](../configs/compression/quantisation.yaml) and the `QuantisationConfig`
default. A `requires_torch` test now asserts the shipped backend is one the installed torch
actually reports, so a future torch upgrade that renames engines fails a test instead of a run.

**Two deprecation warnings worth recording**, because they set a ceiling on how far the torch pin
can move:

1. `torch.ao.quantization` is deprecated, with `torchao` named as the migration target.
2. The quantised dtypes `qint8` / `quint8` / `qint32` are deprecated
   (pytorch/pytorch#184982).

Neither blocks this study: §2.7 requires the environment pinned for its whole duration, so a
deprecated-but-working path is acceptable and reproducible. It does mean **the torch version must
not be upgraded mid-study**, and that a follow-on project should target `torchao` instead.

Open question deferred to Phase 6, not resolved here: `torchao` advertises weight-only int4 CPU
support. If that proves usable by *both* arms under one artefact format, the "no W4 latency row"
limitation above could be lifted. That claim needs measuring, not assuming, and it is not on the
Phase 5 path — the primitives are backend-independent.

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
| Model revisions | **pinned**, see the SHA table below | §2.7 |
| Target layers | decoder-block linears only, per the adapter table below | §2.6, §3.10 |
| Layer exclusions | embeddings, LM head, LayerNorm, all biases | §3.10 |
| Quantisation | weight-only, symmetric, per-channel; W8 and W4; group size 128 when per-group | §3.9 |
| Activation quantisation | **off** — not part of the core design | §3.9 |
| Pruning granularity | global unstructured, identical across all sizes | §3.10 |
| Saliency | activation-weighted magnitude; under `Q_b(W)` for joint (**D3**) | §3.3, §3.7 |
| Screening sparsities | 30% / 50% / 70% of targeted weights | §3.10 |
| Benchmark runtime | PyTorch native CPU INT8, engine `onednn`; latency at W8 only (**D1**) | §2.7, §4.7 |
| Scale x-axis | targeted non-embedding parameter count | §2.6 |
| Seeds | 1 screening · 1 first pass · 3 confirmatory | §5.5 |
| Run IDs | `<family>_<size>_<method>_<sparsity>_<bits>_<seed>` | §5.6 |

### Pinned model revisions

Resolved from the Hub on **2026-07-28** with
`HfApi().model_info(repo_id, revision="main").sha` and written into the five model configs. A
branch name is not a pin — a Hub repository can be updated in place, so an unpinned run may
silently load different weights months later.

| Config | Repository | Commit SHA |
| --- | --- | --- |
| `pythia_160m.yaml` | `EleutherAI/pythia-160m` | `50f5173d932e8e61f858120bcb800b97af589f46` |
| `pythia_410m.yaml` | `EleutherAI/pythia-410m` | `9879c9b5f8bea9051dcb0e68dff21493d67e9d4f` |
| `pythia_1b.yaml` | `EleutherAI/pythia-1b` | `f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2` |
| `pythia_1_4b.yaml` | `EleutherAI/pythia-1.4b` | `fedc38a16eea3bd36a96b906d78d11d2ce18ed79` |
| `qwen2_5_0_5b.yaml` | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |

Note every Pythia repository above is the **standard** variant. None carries a `-deduped` suffix,
which is what §2.7's "never mix standard and deduplicated variants" requires — and it is checkable
at a glance from this table.

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
| Practical-importance rule | ⚠️ **SUPERSEDED — see [the amended rule](#the-amended-practical-importance-rule) below.** As originally frozen: joint gain counts as practically important only when perplexity retention improves by **≥ 1.0 percentage point**, consistently in sign across **all three** confirmatory seeds, **and** the mean improvement exceeds the seed spread (max − min) at that cell | §6.3 required this predefined, and it was stated before any compressed result existed. Its binding clause then turned out to be **vacuous**: run seeds are inert under this method ([F-15](findings_log.md#f-15)), so the seed spread is exactly zero and any nonzero gain passed. Amendment A1 §5.1 replaced the seed axis with paired calibration draws. |
| Downstream importance rule | ⚠️ **WITHDRAWN, not amended.** As frozen: ≥ 1.0 pp accuracy gain on at least 2 of the 3 tasks, same sign across seeds | Same seed-era defect as above, and no replacement is claimed. The downstream run uses **one** calibration draw, so there is no sign to be consistent across. Downstream results are **descriptive secondary endpoints** and no formal joint-superiority claim is made from them; the policy is declared in `configs/experiments/downstream.yaml`. Invoking this rule in the write-up would be invoking a rule the design cannot satisfy. |

### The amended practical-importance rule

**Current governing form**, per Amendment A1 §5.1 and §6.3. This is the rule the write-up uses; the
row above records what it replaced.

A joint gain counts as practically important when, at a cell:

1. perplexity retention improves by **≥ 1.0 percentage point**; **and**
2. the sign is **consistent across every paired calibration replicate** at that cell -- ties are
   excluded and counted, not treated as negatives (B-40); **and**
3. **R is reported**, together with whether significance was reachable at that R. At R=5 the best
   possible outcome is p = 0.0625, so 1B can carry effect-size evidence and never significance.

**What was lost and what was gained, stated plainly because §6.3 requires it.** The original rule was
stricter *on its face* -- it had a third clause -- and unmeasurable *in fact*, because that clause
evaluated to zero for every cell. Replacing an unmeasurable criterion with a weaker measurable one is
both a correction and **a reduction in pre-registered strength**. The paper must say so rather than
present the amended rule as the one that was pre-registered.

The seed-spread clause has no replacement. Paired-draw variance is reported as a standard deviation,
a per-replicate list and an exact sign test instead of being folded into a pass/fail threshold --
because F-26 showed the paired difference is *noisier* than either arm alone, so a spread-based gate
would have been calibrated on the wrong quantity.

---

## Environment

Recorded per §10.2. **The HP Omen is the designated benchmark host: every *deployment* number —
latency, throughput, peak memory, checkpoint size — comes from this machine, and one results table
never spans two.**

**Amended 2026-08-01.** This previously read "the only machine that produces numbers", which was
stricter than §4.7 and than `benchmarking_protocol.md`'s *one machine per results table*.
Compression, activation capture and quality evaluation may run on **any CUDA machine**: they differ
across hosts only by floating-point reduction order, ~1e-5 relative, against ~1e-2 effects. A
*comparison* still may not span machines, and that is enforced in code rather than by convention —
`ExperimentTracker.exists_valid` re-runs a record produced on another host instead of reusing it
(B-33). The freeze on this table itself is unchanged; it describes the benchmark host.

| | |
| --- | --- |
| CPU | 13th Gen Intel Core i7-13620H — 10 physical / 16 logical cores |
| RAM | 13.7 GiB |
| GPU | NVIDIA GeForce RTX 4050 Laptop — **6.0 GiB** VRAM, sm_89 (Ada) |
| NVIDIA driver | 592.82 |
| OS | Windows 11 Home 10.0.26200 |
| Power profile | **High performance** (`ad8e16f4-0e1d-4811-9f3b-165752347277`), set 2026-07-28 |
| Python | 3.11.9 (`winget` user-scope install) |
| torch | 2.13.0+cu126 |
| transformers | 5.14.1 |
| datasets | 5.0.0 |
| numpy | 2.4.6 |
| Thread count for benchmarks | **4** (`benchmark.num_threads`), in every shipped config |

### Power profile, set 2026-07-28

§4.7 requires "a fixed performance mode". The machine was on **Balanced**, whose processor
throttle floor on AC was 5% against a 100% ceiling — the CPU was free to clock up and down
between repetitions, which is variance injected directly into the quantity being measured.

Windows 11 had hidden the legacy schemes on this laptop, so High performance was restored with
`powercfg -duplicatescheme 8c5e7fda-…` and made active. Verified after the change:

| Setting | AC value | Meaning |
| --- | --- | --- |
| `SUB_PROCESSOR PROCTHROTTLEMIN` | `0x64` = 100% | no downclocking |
| `SUB_PROCESSOR PROCTHROTTLEMAX` | `0x64` = 100% | full frequency available |
| `SUB_SLEEP STANDBYIDLE` | `0` | never sleeps — was 600 s, which would have killed a long sweep |
| `SUB_SLEEP HIBERNATEIDLE` | `0` | never hibernates |

Fully reversible: `powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e` restores Balanced.

This pins *frequency policy*, not temperature. §4.7's other requirements still apply and are not
optional: plug into power, close heavy applications, allow the machine to cool between benchmark
blocks, run warm-up iterations, take 20–30 timed repetitions, report median / IQR / p95, and
rotate model order. Thermal throttling under sustained load is exactly what the IQR and the
cooldown rule exist to expose.

### Thread count, and why 4

Already pinned at **4** across every shipped config; recorded here with the reasoning, which was
missing.

The i7-13620H is **heterogeneous**: 6 performance cores (12 threads with SMT) plus 4 efficiency
cores, 10 physical / 16 logical. That matters for a latency benchmark. A thread count above 6
forces work onto E-cores or onto SMT siblings, and which one the scheduler picks can vary between
repetitions — so the *same* model measured twice can differ for reasons that have nothing to do
with compression.

Four threads sits inside the P-core budget, so all four can be scheduled on physical performance
cores with no E-core or SMT contention. It is also batch-size-1 realistic, which §4.7 names as the
primary deployment setting.

The absolute value matters less than that it never changes: §4.7 forbids comparing models measured
under different thread counts, and `hardware` metadata is recorded per run so a violation is
detectable after the fact. `configs/evaluation/cpu_benchmark.yaml` notes that a single-thread
sweep is available as a *separate, self-consistent* comparison for the sparsity-scaling question —
its numbers must not be mixed into a 4-thread table.

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

## The frozen compression budgets

**Frozen 2026-07-29**, from the Phase 7 screening grid on Pythia-160M. Evidence:
[findings_log.md F-10 and F-13](findings_log.md#f-13), and
`outputs/tables/screening_summary.md`.

| | Sparsity | Bits | 160M retention (seq / joint) | Role |
| --- | --- | --- | --- | --- |
| **Moderate** (screening S1) | **30%** | **W8** | 80.4% / 80.6% | **Control.** The mechanism is near-inert at W8, so a gain near zero is the *expected* outcome. |
| **Aggressive** (screening S5) | **30%** | **W4** | 56.0% / 51.4% | **The headline comparison.** The only regime where the joint mechanism is measurably live. |

§6.3 forbids revisiting this once results exist. Written into
`main_scale_sweep.yaml`, `extended_scale_sweep.yaml` and `qwen_validation.yaml`, and pinned by tests
in `tests/test_config.py`.

### What this replaced, and why

The shipped configs previously used **50% + W8** as moderate and **70% + W4** as aggressive. Screening
measured both as **catastrophic on Pythia-160M** — 22.9% and 0.8% retention. Because the budget is a
controlled variable across scales (§2.5), the *smallest* model sets the ceiling for all three, so
neither was usable anywhere in the study.

### Why this pair, out of three eligible candidates

Screening left three budgets eligible: S1 (30% + W8, 80.4%), S5 (30% + W4, 56.0%) and S6
(40% + W8, 55.1%). No pair satisfies everything:

- **S5 + S6** fails §5.3's separation clause — both sit at ~55% retention, so there is no severity
  contrast to test the budget question against.
- **S1 + S6** would vary *sparsity* and keep both budgets on the benchmarkable INT8 path. Rejected
  because it contains **no 4-bit condition at all**, and [F-05](findings_log.md#f-05) measured the
  joint mechanism as near-inert at W8 (0.46% mask divergence against 8.86% at W4). A sweep with two
  8-bit budgets would be structurally incapable of detecting the effect the study exists to measure,
  and would produce a confident null that was an artefact of the design.
- **S1 + S5** was chosen. It varies *precision*, keeps the one live-mechanism regime, and separates
  80.4% from 56.0% retention.

### The consequence, stated plainly

**Both budgets prune 30%, so sparsity never varies across the frozen pair.** The sparsity-versus-
latency curve that research question 4 asks for therefore does **not** come from these budgets.

It comes instead from **benchmark-only runs of the pruning-only arm** at several sparsities. Those are
cheap: that arm stays FP32, so it benchmarks on the native dense kernel, and a latency measurement
does not need the full quality evaluation. This must actually be scheduled — it is the only route to
RQ4 under this pair, and it is easy to forget because it is not part of any budget cell.

Per decision **D1**, the aggressive budget contributes quality and size only and never appears in a
latency table.

### Confirmed on 410M — ⚠️ exploratory only

> 🔴 **Superseded as confirmatory evidence by [Amendment A1 §4](protocol_amendment_a1.md).** These
> numbers are on the **validation** split (now a declared selection surface), predate the B-22/B-23
> corrections that **inflated joint gain**, and carry no uncertainty estimate. The **eligibility**
> conclusion stands — sequential retention has been stable across every version of the code — but the
> **joint-versus-sequential columns must not be quoted.**

§5.3's pre-1B requirement is satisfied. Pythia-410M, same 493 × 512 window, dense 22.17, matched solver
budgets:

| Budget | Sequential | Joint | Seq ret. | Joint ret. | Verdict |
| --- | --- | --- | --- | --- | --- |
| moderate 30% + W8 | 29.08 | 29.09 | **76.2%** | 76.2% | ELIGIBLE |
| aggressive 30% + W4 | 39.51 | 38.03 | **56.1%** | 58.3% | ELIGIBLE |

The aggressive budget gives near-identical *sequential* retention at both scales — 56.0% at 160M
against 56.1% at 410M — which is the behaviour a properly controlled variable should show.

Full detail, including the joint gain changing sign between scales, in
[findings_log.md F-14](findings_log.md#f-14). **That sign flip is not yet a finding** — see the seed
problem immediately below.

## The frozen sequential order (A1 step 7)

Plan §3.6 and §6.1 require joint gain to be measured against **best-of {P→Q, Q→P}**, with the winning
order recorded. Selected on the **validation** split so the choice never touches the confirmatory
split — validation picks the method, test estimates its performance. Evidence:
[findings_log.md F-24](findings_log.md#f-24).

| Model | Budget | **Frozen order** | Evidence | Status |
| --- | --- | --- | --- | --- |
| pythia-160m | aggressive 30% + W4 | **P→Q** | +4.26 pp, one draw | ✅ **frozen on evidence** |
| pythia-410m | aggressive 30% + W4 | **P→Q** | +6.82 pp, one draw | ✅ **frozen on evidence** |
| pythia-160m | moderate 30% + W8 | **P→Q** | indistinguishable over 5 draws | ✅ **frozen by pre-declared fallback** |
| pythia-410m | moderate 30% + W8 | **P→Q** | +0.04 pp, i.e. indistinguishable | ✅ **frozen by the same fallback** |
| **pythia-1b** | **aggressive 30% + W4** | **P→Q** | **+2.15 pp, 3/3 draws** | ✅ **frozen on evidence** |
| **pythia-1b** | **moderate 30% + W8** | **Q→P** | **+0.10 pp, 3/3 draws** | ✅ **frozen on evidence** |

**All six cells are now frozen.** Evidence for the 1B pair: [findings_log.md F-32](findings_log.md#f-32).

### ⚠️ The W8 order is not the same at every scale, and that is by design

1B freezes **Q→P** at the moderate budget while 160M and 410M freeze **P→Q**. A1 §3 freezes the order
**per (model, budget)** precisely so this is permitted: the two smaller scales had an inconsistent sign
and took the pre-declared *fallback*, whereas 1B had a consistent sign across three draws and so took
the *measured* branch of the same rule. Nothing was decided differently; the same rule met different
evidence.

Read the 1B W8 freeze with its two caveats attached. The margin is **0.10 pp**, and three unanimous
draws reach only p = 0.25 on an exact sign test — consistent in sign, not significant. It is also on
the **control** budget, where 96.3% retention leaves almost no headroom and F-05 predicts the mechanism
is inert. Nothing in the headline depends on it.

### W4 — frozen, decisively, at both scales

**P→Q wins at 4 bits and the margin grows with scale**: +4.26 pp at 160M, +6.82 pp at 410M. On the
additive scale the Q→P penalty is 0.078 nats at 160M and 0.124 at 410M.

Consistent with the mechanism in [F-24](findings_log.md#f-24): Q→P reuses the dense-fitted scales
without refitting, which is what keeps it a *sequential* arm rather than a joint one. That is nearly
free at W8 where quantisation is almost lossless, and punishing at W4 where a coarse grid is badly
matched to the post-pruning distribution — worse at scale, because a larger model has more channels
whose distributions shift.

**The aggressive headline is unaffected by best-of**, because P→Q was already the stronger order. Joint
gain stands at +1.08 pp (160M) and +0.68 pp (410M).

### ✅ W8 — resolved. Frozen at P→Q by the pre-declared fallback.

**The two orders are indistinguishable at 8 bits.** Measured across five paired calibration draws
([findings_log.md F-28](findings_log.md#f-28)):

| | Margin, Q→P − P→Q |
| --- | --- |
| rep0 … rep4 | +0.43, **−0.09**, +0.21, +0.22, +0.14 pp |
| mean | **+0.18 pp**, sd 0.19, SE 0.08 |
| Q→P ahead in | 4 / 5 draws — **sign not consistent**, p = 0.375 |

The rule was fixed in `order_selection_w8_replicates.yaml` *before* the measurement, and its second
branch fires: the sign varies, so **W8 is frozen at P→Q — the §3.6 pre-registered primary order — and
the choice is recorded as arbitrary rather than as a measured preference.**

**This is not a finding that P→Q is better.** Q→P is ahead on the mean. Adopting it would mean selecting
a +0.18 pp winner out of noise and then reporting a joint gain against it — which would flip the sign of
the moderate budget's headline. That is exactly the outcome the rule existed to prevent.

**Consequence:** the moderate budget's joint gain is **+0.07 pp** (against P→Q), a clean null and what
**F-05** predicts for a mechanism inert at 8 bits. The −0.36 pp figure derived in F-24 is withdrawn with
the Q→P freeze it rested on.

**An earlier revision of this file froze Q→P** on the 160M margin alone, calling one draw sufficient
because "both margins exceed what a single draw's noise plausibly explains". That was true of the W4
margin and an **assertion** for W8.

**Why more draws will not help.** W8 quantisation is near-lossless (F-07: 99.8% retention W8-only), so
there is little damage for a draw to modulate — each arm's sd is only 0.13 pp. But the difference between
two orderings is also tiny, because with an almost-lossless quantiser it barely matters which runs first.
Small signal, small noise. Resolving a +0.18 pp margin at sd 0.19 would need R ≈ 12, spent on the
*control* budget to settle a question that does not affect the headline.

## Still open

| Item | Why it is not frozen yet |
| --- | --- |
| Calibration sample indices, token count, sequence length | Frozen by the config once `prepare_data.py` has run for real. The WikiText load path has still never been executed. |
| ~~Seed policy (§5.5) and the practical-importance rule (§6.3)~~ | **AMENDED 2026-07-30** — [Amendment A1 §5.1](protocol_amendment_a1.md). The run-seed axis is withdrawn and replaced by **five paired calibration replicates**; §6.3 is reworded, and the loosening is declared as such. Reported as an effect-size study: five draws cannot clear p < 0.05 even when unanimous (2/2⁵ = 0.0625). Evidence: [F-15](findings_log.md#f-15). |
| ~~Evaluation split (§4.1)~~ | **AMENDED 2026-07-30** — [Amendment A1 §5.2](protocol_amendment_a1.md). Validation for selection, **test for confirmation**. No leakage existed (calibration is from train); the defect was that budgets were selected on validation. |
| ~~Sequential ordering (§3.6, §6.1)~~ | **ENFORCED 2026-07-30** — [Amendment A1 §5.3](protocol_amendment_a1.md). Both orders run; the winner is selected on **validation** and frozen per (model, budget) before any test evaluation. Not a change — the documents always required best-of-two; the grid did not implement it. |
| ~~The two final budgets~~ | **FROZEN 2026-07-29** — see [The frozen compression budgets](#the-frozen-compression-budgets). Confirmation on 410M outstanding before 1B. |
| 1.4B go/no-go | §5.2 needs measured peak VRAM against 85% of 6.0 GiB — a 5.1 GiB ceiling, which is tight. Decide after Phase 5 profiling. |
| W4 latency via `torchao` | Deferred to Phase 6. Would lift D1's "no W4 latency row" limitation if a single 4-bit CPU path serves both arms. Needs measuring. |

Closed since this file was opened: Smart App Control (disabled 2026-07-28, environment verified),
the power profile (High performance), the benchmark thread count (4), the model revision SHAs (all
five pinned), and the quantisation engine name (`onednn`, not `x86`).

---

## Related

- [research_plan.pdf](research_plan.pdf) — authoritative source
- [method_definition.md](method_definition.md) — the arms, as specified
- [implementation_plan.md](implementation_plan.md) — build phases and exit tests
- [STATUS.md](STATUS.md) — where the work currently stands
