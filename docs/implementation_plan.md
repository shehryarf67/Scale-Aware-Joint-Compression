# Implementation plan

How the code gets built, in what order, and how each step is verified.

This is the **durable** roadmap. For *where we currently are*, read
[STATUS.md](STATUS.md) — that file is updated every session and this one changes only when the
plan itself changes.

- **What to build:** [method_definition.md](method_definition.md) is the spec for the two arms.
- **Why:** [research_plan.pdf](research_plan.pdf) is the authoritative source document.
- **What could go wrong:** [validity_threats.md](validity_threats.md).

---

## The method, in one paragraph

Research plan §3.1 selects **layerwise post-training reconstruction**. For each linear layer,
calibration activations `X` are captured and the compressed weights chosen to minimise

```
L_rec = ‖ XW − X(M ∘ Q_b(W)) ‖²_F      subject to   sparsity(M) = s,  bit_width(Q_b) = b
```

Layer by layer, one at a time. This avoids full-model fine-tuning, which will not fit at 1B–1.4B
on a laptop, while still allowing a genuine sequential-versus-joint comparison under identical
calibration data and optimisation budgets.

**The unit of optimisation is local steps per layer, not global optimiser steps.** That is the
single most important consequence, and it is what the fairness accounting must be built on.

---

## Reconciliation: plan versus scaffold

The scaffold was originally built assuming full-model quantisation-aware fine-tuning. The
infrastructure survives; the compression method layer does not. These are the known gaps.

| # | Gap | Size | Status |
| --- | --- | --- | --- |
| A1 | Layerwise reconstruction: `activations.py`, `reconstruct.py`, `layerwise.py` | large | not started |
| A2 | Config `reconstruction` section; `local_steps` replaces `max_steps` as the fairness unit | medium | not started |
| A3 | Reverse sequential **Q→P**, and joint gain vs **best-of** {P→Q, Q→P} (§3.6, §6.1) | medium | not started |
| A4 | Downstream tasks — HellaSwag, PIQA, ARC-Easy are **required** (§4.3) | medium | not started |
| A5 | Prefill vs decode timed **separately**, prompts of 128 and 512, IQR, model-order rotation (§4.7) | small–medium | not started |
| A6 | **Targeted non-embedding parameter count** as the scale x-axis (§2.6) | small | not started |
| A7 | Seed policy: 1 screening / 1 first pass / 3 confirmatory (§5.5) | small | not started |
| A8 | Budget screening stage S1–S4 on 160M, freeze two budgets before 1B (§5.3) | small | not started |
| A9 | Record fields: per-layer reconstruction loss, compression time, peak GPU memory, effective bits, tokenizer revision (§7.2) | small | partial |

`training/` (trainer, recovery, callbacks) drops off the critical path. Keep the files — they
become the optional short-fine-tune ablation.

### The `Compressor` ABC still fits

With reinterpreted stage semantics, so the interface and its tests survive:

| Stage | Meaning under layerwise PTQ |
| --- | --- |
| `prepare` | select target modules, build the calibration set, install activation capture |
| `apply` | **the layerwise loop** — the whole algorithm lives here |
| `recover` | no-op in the core method; retained for the optional fine-tune ablation |
| `convert` | pack to real low-bit storage, fold masks, emit the deployable artefact |
| `report_statistics` | unchanged, plus per-layer reconstruction losses |

---

## Phases

Each phase has an **exit test** — the thing that must pass before moving on. Phases 0–4 are
strictly sequential. From 5 onwards there is some parallelism (noted for the §8.3 two-researcher
split).

### Phase 0 — Protocol freeze and schema reconciliation
*Plan: Days 1–2 · §2.7, Appendix A*

| Deliverable | Notes |
| --- | --- |
| `docs/protocol_freeze.md` | Appendix A template, filled in, dated, version-controlled |
| Config schema | A2: `reconstruction` section, `local_steps`, saliency enum |
| `CompressionMethod` extended | `SEQUENTIAL_PQ`, `SEQUENTIAL_QP` |
| Record schema | A9 fields |
| Pythia variant chosen | standard or deduped — **one, consistently** |
| Revisions pinned | real commit SHAs in all five model configs |
| Seed policy + screening configs | A7, A8 |
| Environment lock file | `pip freeze`, plus CPU/GPU/RAM/VRAM recorded |

**Exit test:** all shipped configs load; the existing suite stays green; `protocol_freeze.md` has
no blank rows.

Mostly editing what exists. It is also the phase most likely to be skipped under time pressure —
don't. Every later phase writes records against this schema.

### Phase 1 — Data, calibration, held-out set ✅ *done*
*Plan: §4.1, §4.2 · `data/loaders.py`, `preprocessing.py`, `calibration.py`*

Tokenise, chunk, cache with a content hash, fingerprint. Calibration indices from a fixed
`calibration_seed`, disjoint from evaluation by construction, with a held-out subset for the
overfitting check (§4.8).

**Exit test:** two invocations produce byte-identical caches; calibration, evaluation, and
held-out sets are provably disjoint. *Passing.*

### Phase 2 — Adapters and activation capture
*Plan: §2.6, §3.1 · `models/adapters.py`, new `compression/activations.py`*

- `select_compressible_modules` — transformer-block linears only; exclude embeddings, LM head,
  norms, biases (§3.10).
- `count_targeted_parameters` — the new scale x-axis (A6).
- Block iteration in depth order, propagating activations **through the already-compressed
  prefix** so error accumulation is realistic.
- Streaming accumulation of `H = XᵀX` and `‖X_j‖₂`.

Memory note: at `d = 2048`, `H` in fp32 is ~16 MB per layer — fine. Activation batches must be
chunked.

**Exit test (torch, tiny model):** streamed `H` matches direct `XᵀX` to fp32 tolerance; targeted
parameter count matches a hand count; GPT-NeoX and Qwen2 adapters resolve equivalent module sets.

### Phase 3 — Evaluation ✅ *perplexity, agreement, generation done; downstream outstanding*
*Plan: §4.2–4.4 · `evaluation/*`*

Perplexity (primary, CPU, fixed stride), then downstream tasks (A4), retention exactly as §4.4.

**Exit test:** dense pythia-160m perplexity lands in a plausible published range, and two repeated
evaluations agree exactly (§3.2). *Perplexity passing; downstream not started.*

### Phase 4 — CPU benchmark and dense baseline ✅ *done (single-shape)*
*Plan: §3.2, §4.5, §4.7 · Stage A deliverable*

Prefill and decode callables timed separately (A5 outstanding), median/IQR/p95, raw repetitions
retained, peak RSS, checkpoint size, effective bits. `ExperimentRunner.run` wired for dense.

**Exit test:** a complete dense record with quality + timings + memory + size + full metadata; a
re-run overwrites its own record rather than accumulating duplicates. *Passing for the current
single-shape workload; the prefill/decode split remains.*

> **This is the first vertical slice.** It exercises config → load → evaluate → benchmark →
> record in one pass. Every bug found here is found once instead of five times.

### Phase 5 — Single-layer compression primitives ✅ *done*
*Plan: §3.3, §3.4, §3.7, §10.2 · `compression/{pruning,quantisation,reconstruct}.py`*

Work entirely on **one `nn.Linear` with captured activations**, in a notebook, before touching a
model. This mirrors the plan's own 48-hour checklist and is where the method is validated.

| Primitive | Content |
| --- | --- |
| Saliency | activation-weighted magnitude, `S_ij = \|W_ij\| · ‖X_j‖₂` |
| Mask | global-unstructured or fixed-blockwise at target sparsity |
| Quantiser | symmetric weight-only, per-channel / groupwise, W8 and W4 |
| Packing | int4/int8 storage plus scales; bit-exact unpack round-trip |
| Reconstruct | error-compensated column sweep with ridge damping `(H + λI)` |

**Exit test:** on a synthetic layer — reconstruction strictly reduces `‖Y − Ŷ‖²_F` versus naive
round-to-nearest; realised sparsity is exact; quantised weights take ≤ 2^b distinct values per
group; pack → unpack is lossless. *All four asserted in `tests/test_compression_primitives.py`.*

One limitation carried forward deliberately: `solve_masked_rows` does one dense solve per output
channel (`out_features × |S|³`), which is correct but will not scale to `in_features = 8192`.
Phase 6 needs mask-grouping or the deferred Hessian column sweep.

### Phase 6 — The five arms via the layerwise driver 🔑 ← **next**
*Plan: §3.3–3.8*

All arms call the **same** solver, differing only in call order. That is what makes §3.8 ("what
qualifies as joint") checkable in code rather than by inspection.

```
pruning-only     mask → reconstruct survivors
quant-only       quantise → reconstruct
sequential P→Q   mask → reconstruct → quantise → reconstruct
sequential Q→P   quantise → mask → reconstruct                    (reverse ablation)
joint            repeat K times:
                   fake-quantise survivors
                   recompute saliency UNDER quantised weights     ← §3.8 requirement
                   update mask at target sparsity
                   re-estimate scales on survivors                ← §3.8 requirement
                   local reconstruct for fixed steps
                 freeze M and Q
```

**Exit tests:**
- Every arm hits its target sparsity and bit width exactly, verified on the **converted,
  reloaded** artefact.
- A machine-checkable fairness assertion: identical calibration tensors, identical module lists,
  equal total local steps.
- A regression test that **fails** if joint is implemented as "prune fully, then plain PTQ".

### Phase 7 — Budget screening
*Plan: §5.3 · 160M then 410M, one seed*

Run S1 (30%+W8), S2 (50%+W8), S3 (50%+W4), S4 (70%+W4). Select two that are stable, measurably
but not catastrophically degraded, and separated enough to test the scale question. **Freeze
before 1B.**

**Exit test:** the frozen budgets are written into `protocol_freeze.md` and the sweep configs,
with the screening evidence recorded.

### Phase 8 — Sweep execution, resume, status
*Plan: §5.4–5.6, §10.3*

Run-ID convention per §5.6 (`pythia_410m_joint_s50_w4_seed2`); seed policy per §5.5; **resume
from a completed layer**, not just a completed run; `status` recorded; output paths that cannot
overwrite prior results.

**Exit test:** kill a 410M run mid-sweep, resume, and verify no duplicate, partial, or silently
lost cells.

### Phase 9 — Analysis, figures, tables
*Plan: §6 · 8 figures, 6 tables*

Joint gain vs **log targeted non-embedding parameter count**, per-seed points plus mean/median
with uncertainty; **best-of-sequential** selection recording which order won; paired comparisons
where seed and calibration match; effect sizes alongside any p-value.

**Exit test:** `python scripts/generate_plots.py` reproduces every figure and table from raw
records on a clean checkout.

### Phase 10 — Integrity and audit tooling
*Plan: §4.8, §10.3, §10.4*

A `scripts/audit_results.py` that mechanically checks: independent checkpoint reload; sparsity
survives serialisation; bit width is real, not silently dequantised; one backend and thread count
per table; no optional model mixed into the Pythia regression; confirmatory seeds complete.

**Exit test:** passes on the real result set, and **fails** on a deliberately corrupted record.

---

## Testing plan

Four tiers, so CI stays fast while the expensive checks stay runnable.

| Tier | Marker | Needs | Runtime | When |
| --- | --- | --- | --- | --- |
| T0 existing suite | — | nothing | ~11 s | every commit |
| T1 pure functions | — | nothing | ~1 s | every commit |
| T2 torch units | `requires_torch` | torch, CPU | ~10 s | every commit |
| T3 tiny-model integration | `slow` | torch, CPU | ~1–3 min | pre-merge / nightly |
| T4 real-model smoke | `requires_model` | pythia-160m | ~10 min | manual, before a sweep |

### The key technique: a tiny randomly initialised model

`tiny_causal_lm` in `tests/conftest.py` builds a two-layer `GPTNeoXForCausalLM` **from a config
object** — no download, no pretrained weights, milliseconds to instantiate. It gives genuine
end-to-end coverage of capture → mask → quantise → reconstruct → convert → save → reload →
benchmark, inside CI, offline.

Without it, every real bug waits for a 160M run to surface. Do the same for `Qwen2ForCausalLM`:
tied embeddings and grouped-query attention are exactly what breaks a module selector.

### Properties worth testing — the ones that fail *silently*

**Compression correctness**
- Realised sparsity equals target, measured on the **converted, reloaded** artefact
- Distinct values per quantisation group ≤ 2^b (catches silent dequantisation, §4.8)
- Pack → unpack is bit-exact
- Reconstruction reduces the layer objective versus naive rounding
- Held-out reconstruction loss tracks calibration loss (catches calibration overfitting)
- Checkpoint size shrinks by roughly the predicted factor

**Fairness invariants** — §3.11 as executable assertions
- Both arms consumed identical calibration tensors (hash equality)
- Both arms touched identical module lists, in identical order
- Equal total local steps and comparable objective-evaluation counts
- A joint run implemented as "prune → freeze → plain PTQ" must **fail**

**Measurement integrity**
- Dense evaluation is bit-identical across two runs
- Prefill and decode are separately recorded and internally consistent
- Median / IQR / p95 verified against a hand-computed sample
- A record spanning two CPU models or thread counts is rejected by the audit

**Analysis correctness**
- Joint gain uses best-of-sequential, and the winning order is recorded
- The scale x-axis is targeted non-embedding parameters, not total
- Retention formulas match §4.4 in both directions
- A gain smaller than the seed spread is reported as inconclusive

---

## Schedule fit and the main risk

Mapping to the plan's 21-day schedule (§8.1):

| Plan days | Phase | Status |
| --- | --- | --- |
| 1–2 protocol freeze | 0 | largely covered by the scaffold |
| 2–3 hardware pilot | 1–4 | Phases 1–4 *are* Stage A |
| 3–5 build 160M pipeline | 5–6 | **the real work** |
| 5–6 budget screening | 7 | |
| 7–13 410M and 1B | 8 | mostly compute |
| 13–15 confirmatory seeds | 8 | |
| 16 analysis gate | 9–10 | |
| 17–21 optional work and writing | — | |

**The schedule risk is concentrated in Phases 5–6.** The plan allots roughly days 3–5 for "build
160M pipeline end to end"; a correct Hessian-based reconstruction solver plus five arms plus
their fairness assertions is realistically 4–6 days. Everything before it is largely done and
everything after is mostly compute, so a slip lands in the 410M window, which has slack.

Two mitigations: implement the simpler alternating-least-squares solver first as a working
fallback and upgrade to the Hessian sweep if time allows; and treat Phase 4 as a hard gate,
because it is cheap and it de-risks everything downstream.

The plan's own schedule-protection rule (§8.2) applies: **Qwen and 1.4B must not consume the time
reserved for confirmatory Pythia seeds, analysis, or writing.** A complete three-scale paper is
stronger than an unfinished five-model one.
