# Partner handoff — read this before writing any code

> # ⛔ THE MEASUREMENT PHASE IS OVER. DO NOT RUN ANY EXPERIMENT.
>
> **Updated 2026-08-10.** A1 step 10 — the confirmatory run — **has been executed**: 171/171 cells,
> 42/42 pairs, 0 failures, on the held-out test split, audit passed. **[F-37](findings_log.md#f-37)**
> is the result and it is final. Step 10 runs **once**; the protocol forbids tuning, re-running or
> re-selecting anything afterwards.
>
> **The headline: no cell meets the pre-registered §6.3 practical-importance bar.** Joint gives
> +1.01 pp at 160M (7/8) and +0.93 pp at 410M (8/8, p = 0.0078 — the only significant cell) at 4
> bits, +0.13 pp at 1B, and **nothing at 8 bits**. The motivating hypothesis — that joint pays off
> *more* at scale — is **not supported**; the observed direction is opposite, but the decline is
> not statistically established.
>
> **Task 5 below is DONE.** Tasks 1–4 are done. What remains is analysis and writing; see
> [STATUS.md](STATUS.md) for the ordered next steps. If you are picking this up cold, read
> [F-37](findings_log.md#f-37) and §6 of the findings log **before** anything else in this file —
> Part 2's trust levels below were written when every number was exploratory, and the confirmatory
> numbers supersede them.
>
> **Neither exploratory point estimate replicated.** 160M fell +1.69 → +1.01; 410M rose +0.39 →
> +0.93. Any conclusion in Part 2 resting on those figures is superseded by F-37.

**Date:** 2026-07-31, **superseded in part 2026-08-10** · **Audience:** the second author, and
whatever LLM assistant they are using

This document exists so you can pick the project up without reverse-engineering it from the commit
history. It has three parts:

1. **[Why we are doing this](#part-1--why-any-of-this-exists)** — the research question and why the
   design is what it is. Read it even if you are impatient; several of the tasks below look wrong
   until you know the reasoning.
2. **[What we know so far](#part-2--what-we-actually-know)** — every finding that matters, and
   crucially **which numbers are trusted and which are not**.
3. **[What to do next](#part-3--what-to-do-next)** — five tasks in dependency order, with exact
   commands and exact acceptance criteria.

> **If you are an LLM reading this:** you are being asked to continue a study that has already
> retracted three published-internally numbers. Every retraction came from the same failure mode —
> a plausible number produced by code nobody had checked against an independent reference. Your
> value here is *scepticism*, not throughput. When this document says "verify", it means run the
> named check and compare against the named value, not reason about whether it should pass.

---

## Part 1 — Why any of this exists

### The research question

> **Does model scale change whether *joint* pruning-and-quantisation beats *sequential*
> pruning-then-quantisation?**

Pythia **160M → 410M → 1B**, one architecture family, one calibration corpus, one solver, matched
budgets. Qwen2.5-0.5B is an optional external-validity point.

The motivating intuition was that joint should pay off **more** as models grow, because larger
models have more redundancy for a joint optimiser to exploit. **The evidence so far says the
opposite.** That is a finding, not a failure, and the write-up reports it either way.

### What the two arms actually are

Both arms are **layerwise post-training reconstruction** (research plan §3.1). Not
quantisation-aware fine-tuning — that was an earlier generation of the plan, and parts of the config
still carry its vocabulary. For each targeted linear layer, using only the Gram matrix
`H = XᵀX` accumulated from a fixed calibration draw, minimise

```
‖ X·Wᵀ  −  X·(M ∘ Q_b(W))ᵀ ‖²_F
```

* **Sequential (P→Q)** — choose the mask `M`, reconstruct, *then* quantise to `b` bits and
  reconstruct again.
* **Sequential (Q→P)** — the reverse order. §3.6 requires joint gain to be measured against
  **best-of {P→Q, Q→P}**, not against P→Q alone.
* **Joint** — score the mask **on the quantised weights**: `S_ij = |Q_b(W_ij)| · ‖X_j‖₂`, so the
  mask and the grid are chosen together. This is decision **D3**, and it is what makes the arm
  genuinely joint rather than sequential with extra steps.

**The unit of fairness is `local_steps` per layer.** Both arms get the same number. If you change
one arm's budget you have destroyed the comparison, and `assert_matched_plans` will refuse to run.

### The two frozen budgets, and why they are what they are

| Budget | Recipe | Role |
| --- | --- | --- |
| **moderate** | 30% sparsity + **W8** | **Control.** The joint mechanism is near-inert at 8 bits, so a gain near zero here is the *expected* result |
| **aggressive** | 30% sparsity + **W4** | **Headline.** 4 bits is the only regime where the mechanism is measurably live |

Both prune 30%; the budgets vary **precision**, not sparsity. That is deliberate:
[F-05](findings_log.md#f-05) measured mask divergence between the arms at **8.86% at W4 against
0.46% at W8**. Two 8-bit budgets could not detect the effect this study exists to measure, and would
produce a confident null that was an artefact of the design.

**These are frozen. Do not reopen them.** §6.3 forbids revisiting the budget once results exist, and
they have now been confirmed three times at 160M and once at 410M.

### Why the protocol looks paranoid

Because it had to become paranoid. See [Part 2](#part-2--what-we-actually-know): **every
implementation fault found in this project so far has flattered the joint arm.** Not one ran the
other way. A design that leans on "the code is probably right" would have shipped a false positive
at least three times.

Hence: independent anchors before results, paired calibration replicates instead of run seeds,
validation for selection and test reserved for confirmation, and decision rules written into the
config **before** the run that will resolve them.

### The documents, in order of authority

1. **`docs/research_plan.pdf`** — the authoritative 28-page execution plan. When markdown and PDF
   disagree, the PDF wins.
2. **[protocol_amendment_a1.md](protocol_amendment_a1.md)** — governs how the *remaining* experiments
   run. Adopted 2026-07-30. **Read this second.** It changes the execution order and declares
   everything produced so far exploratory.
3. **[method_definition.md](method_definition.md)** — exactly what the arms are.
4. **[findings_log.md](findings_log.md)** — **every measurement with its conditions.** Append-only.
   The paper is written from this file.
5. **[STATUS.md](STATUS.md)** — where we are right now. The session-to-session handoff.
6. **[protocol_freeze.md](protocol_freeze.md)**, **[validity_threats.md](validity_threats.md)**,
   **[benchmarking_protocol.md](benchmarking_protocol.md)**, **[experiment_protocol.md](experiment_protocol.md)**.

If code and docs disagree, that is a bug in one of them. **Say so rather than picking silently.**

### Hard rules — breaking any of these silently invalidates results

* **Deployment measurements are CPU-only.** Latency, throughput, peak memory, checkpoint size. GPU
  is fine for compression and activation capture, never for a reported measurement. The config
  loader rejects a non-CPU benchmark device — **do not work around it**.
* **The two arms must match** on target sparsity, bit width, calibration data, module coverage,
  optimisation budget, artefact format and backend. Tests assert this over the shipped configs. If
  your change breaks one of those tests, **your change is wrong**.
* **Never fabricate or estimate a result.** Unimplemented paths raise `NotImplementedError` naming
  the module to edit. Keep it that way. A plausible-looking number is worse than an error.
* **Never commit anything under `outputs/`, `results/` or `data/`.** Structure is tracked via
  `.gitkeep`; contents are not.
* **Report measured against target**, always — measured sparsity beside target sparsity,
  `is_converted` on quantised artefacts, `storage_efficiency` on checkpoints.
* **Deployment measurements come from one designated benchmark host.** Everything else can run on
  any CUDA machine. See below — this is the rule that changed on 2026-08-01.

### The machine rule — what it is now, and what it used to say

**You can run this on Colab, or any other CUDA box.** An earlier version of this document said "the
HP Omen is the only machine that runs code". That was operational shorthand from when one person had
one GPU, and it was **stricter than the protocol requires**: `benchmarking_protocol.md` and
`methodology.md` both say *one machine per results table*, not one machine per project. It has been
corrected.

| Tier | Work | Where you may run it |
| --- | --- | --- |
| 1 | Tests, lint, config validation, docs, analysis of existing records | **anywhere**, no GPU needed |
| 2 | Compression, activation capture, quality evaluation | **any CUDA machine** — Colab included |
| 3 | Deployment: latency, throughput, peak memory, checkpoint size | **the designated benchmark host** (currently the Omen) |

**Why tier 2 is portable and tier 3 is not.** Compression and perplexity depend on the weights and
the data; across machines they move only by floating-point reduction order, about **1e-5 relative**
— the same magnitude as the CPU/GPU drift measured in [F-29](findings_log.md#f-29) and the CPU
thread-configuration sensitivity in [F-23](findings_log.md#f-23), and three orders of magnitude
below the ~1e-2 effects this study measures. A **latency** is not like that. It is a property of the
machine, and no correction makes two hosts comparable.

**Two invariants that hold wherever you run:**

1. **A comparison never spans machines.** Both arms of a cell, at the same replicate, on one host.
   The machine is one of §3.11's matched conditions. This is enforced in code —
   `ExperimentTracker.exists_valid` re-runs a record produced on a different host rather than
   reusing it (**B-33**), so `skip_existing` cannot quietly pull your Colab record into a grid
   running on the Omen.
2. **Never mix machines in one results table.** `scripts/generate_plots.py` **refuses to plot**
   when deployment-bearing records span hosts. It no longer warns when only compression records do,
   because that case is benign and a warning that fires when nothing is wrong stops being read.

**What this means in practice for you:** do tasks 1–3 on whatever GPU you have. Copy the resulting
records over if useful, but expect them to re-run on the other machine — that is the guard working,
not a bug. **Task 4's A5 benchmarks and task 5's confirmatory run must happen on the benchmark
host.**

### Reproducibility is a separate rule, and it did not loosen

A previous Phase 7 attempt ran on Colab from a working tree **22 commits behind `main`**, at a
commit recorded as `aec5099-dirty`. Those numbers could not be used — not because of the hardware,
but because *nothing about them was reproducible*: unknown code, uncommitted changes. The work was
not wasted (its test-split fix was genuinely right and is in `main` now), but the measurements were
unusable.

**So: clean tree, committed SHA, `git pull` first, every time.** The record stores the commit and
appends `-dirty` if the tree is not clean. A `-dirty` record is not a result.

---

## Part 2 — What we actually know

### The headline, stated precisely

**At 160M the joint arm wins. At 410M it does not. The gain shrinks with scale.**

| | 160M | 410M |
| --- | --- | --- |
| Joint gain, mean of 3 paired draws | **+1.69 pp** | +0.39 pp |
| sd | 0.63 | 0.78 |
| Draws positive | **3/3** | 2/3 |
| Distance from zero | 2.68 sd | 0.50 sd |
| Above the pre-registered ≥1.0 pp bar? | **3/3 yes** | no |

Per-draw scale difference (160M − 410M): **+0.40, +2.15, +1.36 pp** — mean +1.30, **3/3 positive**.
[F-27](findings_log.md#f-27).

**What you may claim from this: nothing statistically.** Three unanimous draws reach only p = 0.25
on an exact sign test. This is a *consistent-in-sign effect-size result on the validation split*,
which is exactly what A1 §5.1 prescribes for the exploratory stage — and it is why the confirmatory
stage exists.

### The three anchors — why the code is trusted now

| Anchor | What it proved | Result |
| --- | --- | --- |
| [F-19](findings_log.md#f-19) **Wanda mask agreement** | An independent implementation sharing no code produces **exactly our mask** | 0 differing positions across 48 modules / **84,934,656 weights** |
| [F-20](findings_log.md#f-20) **Exact optimum** | The sweep never beats the provable minimiser `ŵ_S = (H_SS)⁻¹H_{S,:}w`, and never loses to naive masking | **0/96 rows** violate either invariant |
| [F-22](findings_log.md#f-22) **External SparseGPT** | Our absolute retention is credible, not inflated | 77.3% of a 25 pp gap traced to the mask **comparison group**; 5.67 pp residual, inside A1's 10% band |

Plus [F-21](findings_log.md#f-21): arm-dependent solver slack is **real** (efficiency 0.6409 under
the sequential mask, 0.5631 under the joint one) but **never inverted which mask was better** across
96 rows. So the *direction* of a measured joint gain is not a solver artefact — and the slack runs
*against* joint, meaning the mechanism is understated rather than flattered.

**Open limitation, stated in the paper:** slack's effect on *magnitude* cannot be measured this way,
because no exact optimum exists for the quantised problem — a discrete grid makes it an integer
program.

### The joint-gain figure has been wrong three times. Know the history.

| Figure | Status and cause |
| --- | --- |
| **−4.55 pp** | **Retracted.** The joint outer loop had no acceptance test, so it discarded better solutions it had already found |
| **+1.03 pp** | **Retracted.** The arms minimised *different objectives* — sequential targeted its own intermediate, joint targeted dense |
| **+1.08 pp** | Single draw ([F-23](findings_log.md#f-23)). First figure from code that passes three independent anchors |
| **+1.08 pp** | **Unchanged** against best-of-sequential ([F-24](findings_log.md#f-24)) |
| **+1.69 pp** | **Current** ([F-27](findings_log.md#f-27)). Mean of three paired draws, 3/3 positive |

**Every fault so far has flattered joint** — B-14, B-17, B-22, B-23, B-30. If you find a bug and it
turns out to have *hurt* joint, that is unusual enough to double-check.

Two things separate the current figure from the retractions: it survived a rewrite that changed the
objective, added an acceptance guard, fixed activation grouping and altered packing; and it survived
being re-measured against the *stronger* of two sequential baselines. A third was added by F-27: it
**replicates**, and the single-draw +1.08 pp turned out to be the **pessimistic** end of the
distribution, not a lucky reading.

### Two facts about noise that will save you a day

1. **Pairing does not cancel calibration noise.** The paired *difference* is **noisier** than either
   arm alone — 1.47 pp spread against 0.50 and 1.28 pp ([F-26](findings_log.md#f-26)). A draw
   changes *which mask each arm picks*, and the arms respond differently by construction, because the
   mask is what distinguishes them. So **never judge a margin against a single arm's spread**; judge
   it against the paired margin's spread.
2. **Run seeds measure nothing.** The pipeline is deterministic post-training reconstruction, so two
   runs at different run seeds are bit-identical ([F-15](findings_log.md#f-15)). Three seeds gave a
   spread of exactly zero, which made §6.3's "must exceed the seed spread" clause pass for any
   nonzero gain. A1 **withdrew** the seed axis and replaced it with **paired calibration
   replicates**: replicate `r` uses `constants.CALIBRATION_REPLICATE_SEEDS[r]`, and every arm in a
   comparison uses the same `r`.

### The other findings you need in your head

* **Mask comparison group is worth 6.7× perplexity** ([F-07](findings_log.md#f-07)). Ranking
  activation-weighted scores *tensor-wide* deletes whole low-energy input columns; **per-output-row**
  makes each row keep its own top-k. Per-output is the default and the right one. This single setting
  explained 77% of the gap to reference SparseGPT.
* **The scale mechanism is inert at every width** ([F-05](findings_log.md#f-05)). Pruning removes the
  smallest weights so each row's maximum usually survives, and the re-fitted scale barely moves
  (0.2%). Two candidate fixes — clipping scale search and keep-benefit scoring — were implemented,
  measured, and **rejected because they made the objective worse**. They remain as declared
  ablations, defaulting off. **Do not re-propose them.**
* **The sequential order differs by budget** ([F-24](findings_log.md#f-24),
  [F-28](findings_log.md#f-28)). At W4, **P→Q** wins decisively (+4.26 pp at 160M, +6.82 pp at
  410M). At W8 the two orders are **indistinguishable** across five paired draws (mean margin
  +0.18 pp, sd 0.19, 4/5 favour Q→P but the sign varies), so the pre-declared rule froze **P→Q** and
  recorded the choice as *arbitrary rather than measured*.
* **An exploratory cell now costs ~1.3 min, down from ~9.3** ([F-29](findings_log.md#f-29)).
  Block-sequential capture made compression 2.7× faster with a **bit-identical** Gram, and GPU
  evaluation is 22.5× faster with 8.3e-06 drift. **The confirmatory run keeps CPU evaluation and its
  ~38 hours** — that is the rule.

### What is still open

| | |
| --- | --- |
| **Pythia-1B** | Blocked on per-block GPU offload. Peak is **6.31 GiB on a 6.00 GiB card** with the model resident |
| **A4 downstream tasks** | HellaSwag, PIQA, ARC-Easy — **required by §4.3**, not started |
| **A5 prefill/decode split** | Timed separately at 128 and 512 prompt lengths, IQR, model-order rotation — **required by §4.7**, not started |
| **S6 mechanistic control** | 12 runs, A1 §5.4 |
| **Confirmatory run** | A1 steps 9–10. Test split, R=8/8/5, **once**, no tuning after |
| **W4 CPU latency via `torchao`** | Would lift D1's "no W4 latency row" limitation. Needs measuring |

---

## Part 3 — What to do next

Five tasks, in dependency order. **Do them in order** — task 2 needs task 1, and task 5 must be last
by protocol.

### Before you touch anything

```bash
git checkout main && git pull            # do NOT work from an old branch
uv venv --python 3.11
.venv\Scripts\activate                   # Windows;  source .venv/bin/activate  elsewhere
uv pip install -e . -r requirements-dev.txt
pytest                                   # expect 974 passing, ~45 s, offline
ruff check . && ruff format --check .     # both must pass
sajc info                                # registry + hardware, no downloads
```

If `pytest` is not green **before** your change, stop and find out why. You cannot attribute a
failure to your own edit otherwise.

Use `--dry-run` on any script that takes a config, first, every time. It validates the config and
prints the plan without loading a model.

**On Colab or another Linux GPU box**, four things differ and all of them are survivable:

* **Do not `pip install torch`** — the CUDA build is already there. Installing over it can pull a
  CPU wheel and silently drop you to no GPU. `uv pip install -e . -r requirements-dev.txt` is fine;
  check `sajc info` afterwards and confirm CUDA is still available.
* **The torch version will not be 2.13.0+cu126.** §2.7 pins the environment *for the confirmatory
  run*, not for development. Exploratory work on a different torch is fine; it goes in the record,
  which is the point of recording it. **Do not upgrade the benchmark host's torch to match yours.**
* **`torch.backends.quantized.supported_engines` differs by platform.** On the pinned Windows build
  it is `['onednn']` only; Linux usually offers `fbgemm` and `x86` too. A `requires_torch` test
  asserts the shipped config's backend exists in *your* torch, so it will tell you rather than
  failing at conversion after the compute is spent.
* **Runs must survive a disconnect.** `skip_existing: true` makes a sweep resumable — an interrupted
  grid re-runs only cells with no record — but Colab wipes local disk on disconnect, so point
  `runtime.output_dir` at mounted Drive or copy `outputs/metrics/` out as you go. A lost record is a
  re-run, not a corruption.

---

### Task 1 — Per-block GPU offload ✅ **done 2026-08-01, [F-31](findings_log.md#f-31)**

Kept in full below because the *way it failed* is the most useful thing in this document for
anyone about to touch `layerwise.py`.

`compression.reconstruction.offload_blocks`, defaulting off. Verified bit-identical at 160M for
both arms (0 of 148 parameters differ), and the full cell reproduces **65.2614 / 64.0413** exactly.

**Two wrong implementations before the right one, and neither was caught by reasoning:**

1. **Captured block-0 inputs on the host** (B-34). Aborting at block 0 means only the embedding
   runs, and a lookup is a gather — so it should be bit-identical. But GPT-NeoX also computes the
   rotary `cos`/`sin` in that forward, CPU and CUDA trigonometry differ in the last bits, and a
   **mask is a discrete function of saliency**. One flipped near-tie moved `attention.dense` in
   block 0 by 2.25 absolute, cascaded through every later block, and **more than doubled the joint
   gain** (+2.35 pp against +1.08). Only the F-23 reproduction gate caught it.
2. **Moved the whole model to the device for the capture** (B-34b). Numerically correct, fine at
   160M and 410M, and it hit `CUDA error: out of memory` at 1B — the one model the whole change
   exists for.

**The lesson to carry:** in this pipeline a last-bit difference is not a small difference, because
the mask thresholds it. Verify with `torch.equal`, never `allclose`.

<details>
<summary>The task as it was originally written</summary>

### Per-block GPU offload. This is what unblocks Pythia-1B.

**The problem.** `load_model` does `model.to(device)`, so the whole model sits on the GPU during
compression. At 1B that is 3.77 GiB of FP32 weights, and the widest layer's inverse-Cholesky
temporaries (`in_features` 8192) add ~2.5 GiB. Measured peak: **6.31 GiB on a 6.00 GiB card.** It
"completes" only by spilling into host memory — the widest-layer solve took **33.9 s against 4.8 s**
standalone, a 7× slowdown, and the real run also holds calibration activations.

**`block_size` does not help.** Tested at 128 / 64 / 32: peak 6.31 / 6.37 / 6.37 GiB. The Gram
factorisation dominates, not the block loop.

**Why the change is small now.** [F-29](findings_log.md#f-29) refactored capture so the **block loop
owns the forward**. `compress_model_layerwise` in
[layerwise.py](../src/scale_aware_compression/compression/layerwise.py) already iterates blocks and
replays one at a time over cached hidden states. What is missing is only *residency*.

**What to implement.** In `compress_model_layerwise`, around the existing block loop
([layerwise.py:948](../src/scale_aware_compression/compression/layerwise.py#L948)):

1. Keep the model on **CPU**; move the embedding and anything before block 0 to the device only for
   the single `_capture_block_inputs` call, then back.
2. For each block: `block.to(device)` → compress its groups → `_advance_cache` → `block.to("cpu")`.
3. One block resident at a time — ~0.2 GiB of weights plus the Gram and its temporaries.
4. Put it behind a config flag so the existing path stays reachable and the 160M/410M reproductions
   below are a like-for-like comparison. Default it off until the gates pass.

**Two traps that are already documented, and will bite again:**

* **`use_cache` must be off for the duration**, in a `try/finally`. A live key/value cache
  accumulates across repeated single-block replays and changes what the solver is fitted to. This is
  **B-28**, found first in the SparseGPT driver and then again here.
* **Blocks with no targeted modules must still advance the cache.** Skipping one leaves the next
  block replaying inputs from the wrong depth — a *silent wrong answer*, not an error.

**One correction to an earlier claim.** STATUS previously said
`scripts/run_sparsegpt_external_anchor.py` "contains a working per-block-offload driver". It does
not — it does block-sequential **replay** with the whole model resident, which is the part we have
already adopted. It is still the right file to read for the replay pattern and the `_StopForwardError`
catcher, but **the offload itself has to be written.**

**Gates — and read this before comparing against any number in this repo.**

The values below (65.261 / 64.041 at 160M, 37.851 / 37.415 at 410M) were produced **on the Omen**.
They are *not* portable targets. A different GPU runs different cuBLAS kernels, so the Gram differs
in its last bits, and a near-tie in the saliency ranking can flip — [F-19](findings_log.md#f-19)
found 4 positions in 85 million flipping between float32 and float64 norms alone. **Expect small
differences on another host, and do not read them as a broken refactor.**

So use a **host-local baseline** instead. It is portable, and for isolating a refactor it is
*stronger* than an absolute target, because it holds the machine constant:

```bash
git stash                 # or check out the pre-change commit
# run one 160M aggressive cell, sequential + joint -> record the two perplexities
git stash pop
# run the same cell again on the same machine
```

| Gate | How | Required result |
| --- | --- | --- |
| 1 | `pytest` and `ruff check . && ruff format --check .` | 974+ passing, both clean |
| 2 | 160M aggressive cell, before vs after **on your machine** | **bit-identical** |
| 3 | `python scripts/run_reconstruction_anchor.py --config configs/experiments/screening.yaml` | 0 rows below the optimum, 0 worse than naive. Efficiency near **0.6409** — this one *is* fairly portable, but compare the invariants, not the fourth decimal |
| 4 | 410M aggressive cell, before vs after **on your machine** | **bit-identical** |
| 5 | Omen re-check, before the change is relied on | reproduces **65.261 / 64.041** and **37.851 / 37.415** exactly |

Gate 5 is the one that needs the benchmark host, and it is cheap — two cells, a few minutes each at
the post-[F-29](findings_log.md#f-29) cost. Everything before it you can do alone.

If gate 2 or 4 moves *at all*, the change altered the numbers. Then you must either find out why or
**bump `METHOD_VERSION`** in [constants.py](../src/scale_aware_compression/constants.py), which
invalidates ~50 existing records and forces a recompute. F-29 avoided that; try to avoid it again,
but **never** paper over a difference to protect the record count.

Then measure the 1B peak and report it against the **5.1 GiB ceiling** (§5.2's 85% of 6.0 GiB).

</details>

**Two reusable things came out of this**, and they are the right tools for any future change to the
driver:

* `scripts/verify_block_offload.py` — compresses the same model twice, resident and offloaded, and
  compares **every parameter with `torch.equal`**. It is what localised B-34 to a single module in
  a single block, which is what made the cause findable.
* `configs/experiments/verify_offload_160m.yaml` and `verify_offload_1b.yaml` — the equivalence
  gate and the peak-memory measurement. Note the split: equivalence is established at 160M where
  **both** paths run and the right answer is already known, and 1B measures only memory, because
  there the resident comparison is impossible by construction.

---

### Task 2 — 1B budget confirmation and order selection ← **next**

Task 1 is done, so this is unblocked.

1. **Dense baseline** at 1B, then the **moderate** and **aggressive** budgets, sequential and joint,
   on the **validation** split with GPU evaluation (exploratory — allowed and 22× faster).
2. **Confirm both budgets** land inside §5.3's "measurably but non-catastrophically degraded" band,
   as they did at 160M and 410M. Report measured against target.
3. **Order selection:** run **both** P→Q and Q→P at both budgets, and freeze the winner **per
   (model, budget)** before anything confirmatory. Use `configs/experiments/order_selection.yaml` as
   the template. **Write the decision rule into the config before you run it** — that is what made
   F-28 costless when my expectation turned out to be wrong twice in opposite directions.
4. Expect **P→Q** to win at W4 (it did by 4–7 pp at both smaller scales) and the W8 orders to be
   indistinguishable. If W4 comes out differently at 1B, that is interesting and must be reported,
   not smoothed.

**Watch system RAM, not just VRAM.** The machine has 13.7 GiB. CPU evaluation of 1B in FP32 needs
**4.81 GiB with 1.70 GiB headroom** — it fits, but `batch_size: 1` may be necessary.

**If a model download stalls:** the weight blobs sit at **exactly 0 bytes** while the log stays
quiet. Delete the stale `*.incomplete` blobs and retry with `HF_HUB_DISABLE_XET=1`. A stalled
download and a slow one are indistinguishable from the log — watch bytes on disk.

---

### Task 3 — The reduced S6 mechanistic control (A1 §5.4, 12 runs)

**2 models × 2 arms × 3 draws.** S6 is **40% + W8** — quality-matched to the aggressive budget by a
*different recipe*. The question it answers: is the joint gain **precision-specific**, or just a
compression-severity effect?

[F-23](findings_log.md#f-23) already points at the answer for free: S5 (30% + W4) and S6 (40% + W8)
retain 56.7% and 54.4% — comparable quality — yet the joint gain is **+1.08 pp at W4 and −0.23 pp at
W8**. Among all three eligible budgets, both W8 budgets give gains indistinguishable from zero and
only W4 gives one. The control's job is to make that comparison properly, with replicates.

**Be honest about S6's provenance.** A1 records it as the weakest of the five decisions on the
"would this have been justified before seeing results" test — it became interesting *because* we saw
40% + W8 land near 30% + W4, which is a result. It is defensible because it tests a mechanism rather
than a headline, and A1 says so explicitly. **Keep that framing in the paper.** 1B gets diagnostics
only.

---

---

## ⚠️ Read this before starting anything: who is doing what

**On 2026-08-01 we independently did the same three tasks on the same day** — per-block offload, the
1B selection config, and GPU quality evaluation. Two people, one day, one result. Nobody was at
fault; there was no protocol.

There is one now. **Claim a task in `docs/STATUS.md` before you start it**, in the "Who is on what"
table, and push that claim immediately. A one-line commit is cheap; a duplicated day is not.

| Task | Owner | State |
| --- | --- | --- |
| Per-block GPU offload | main | ✅ done — [F-31](findings_log.md#f-31) |
| 1B budgets + order selection | main | ✅ done — [F-32](findings_log.md#f-32) |
| S6 mechanistic control | main | ✅ done — [F-33](findings_log.md#f-33) |
| **A5 — prefill/decode split** | **main** | 🔵 **in progress, do not start** |
| **A4 — downstream tasks** | **unclaimed — yours if you want it** | ⬜ not started |
| Steps 9–10 — freeze and confirm | unclaimed | ⬜ blocked on A4 and A5 |

### Your parallel run was not wasted — it is now evidence

Your Colab 1B numbers replicate ours on **different hardware, torch and Python**: dense 17.9432
identical to four decimals, Q→P favoured at moderate, P→Q at aggressive, and a joint gain inside
our three-draw range. Four qualitative conclusions, independently reproduced. That is worth more
than a fourth draw on the same machine would have been, and it is going in the paper as a
cross-host replication.

Two things to know about how it differed from what landed on `main`:

* **Your capture design was right** — staging only the pre-block modules and capturing on the
  device. We arrived there on the third attempt, after a host-side capture flipped a mask through
  rotary-embedding numerics (B-34). Yours never had that bug.
* Two gaps kept it off `main`: recorded quantisation grids do not follow their block back to the
  host (B-35, which kills `convert` after the whole compression is spent), and
  `evaluation.device: cuda` has no effect on your branch because the runner change that reads it
  landed separately.

**Do not merge `phase7-close-phase8-setup`.** It would reintroduce a runner without the GPU-evaluation
wiring and lacks the B-35 fix, and its `F-31`/`B-34` entries collide with different content on
`main` in an append-only log. Branch fresh from `main` instead.

---

### Task 4 — A4 and A5, the two required gaps nobody has started

Both are **§-required**, not optional, and neither is blocked by anything. They have simply been
behind the correctness work.

#### A4 — downstream tasks (§4.3). **This one is yours. Here is everything you need.**

HellaSwag, PIQA, ARC-Easy via lm-eval-harness, **pinned** (§2.7 freezes the environment, and §4.8
requires the *task versions* logged too — task definitions change between harness releases and an
unversioned accuracy cannot be compared to a published one). Perplexity alone does not establish
that a compressed model is still *useful*, and a reviewer will ask.

**What to build**

1. Pin `lm-eval` in `pyproject.toml` and `requirements-dev.txt`. Record the resolved version.
2. A thin adapter. Our compressed models are still `GPTNeoXForCausalLM` subclasses, so
   `HFLM(pretrained=model, tokenizer=tokenizer, batch_size=...)` should take one directly —
   **verify that on a packed model before building anything on top of it**, the same way the
   packed-on-CUDA path was checked before the 1B grid (`tests/test_arms.py`,
   `test_a_packed_layer_survives_a_move_to_cuda`).
3. A `downstream` config section: task list, batch size, and an optional `limit`.
4. Record fields: per-task accuracy, per-task **version**, harness version, and the evaluation
   device. Add them to `RESULT_CSV_COLUMNS` so they reach the flat table too.
5. **Offline tests.** Every test in this repo runs with no network. lm-eval downloads datasets, so
   the tests must stub the harness at the boundary — assert that our adapter passes the right model
   and records the right fields, not that HellaSwag scores anything.
6. A `scripts/run_downstream.py` driver, and a `configs/experiments/downstream.yaml`.

**The one decision to make first, because it sets the cost**

The three tasks are ~14,250 examples, but multiple-choice scoring is one forward *per choice* —
roughly **53,000 forwards, ~8M tokens, about 32× a perplexity evaluation**.

| Per 1B cell | Perplexity | Downstream |
| --- | --- | --- |
| GPU | 3.5 min | ~1.9 h |
| CPU | ~38 min | **~20 h** |

Across ~15 cells that is **~15–20 h on GPU against ~150 h on CPU**. CPU is not feasible.

**Recommendation: run downstream on GPU and declare it.** `check_evaluation_device` says reported
numbers must come from CPU, but that is *our* convention; the plan's §4.6 restricts **deployment**
measurements, and downstream accuracy is a quality metric, not a latency claim. Accuracy is
device-invariant far below the ~1 pp differences being reported. The alternative — `--limit`
subsampling — weakens comparability with published numbers, which is the exact thing lm-eval was
chosen to preserve. **Whichever you pick, write the reasoning into the config before you run it.**

**Traps specific to this task**

* A packed model's `lm_head` is *not* compressed (§2.6 excludes embeddings and the head), so
  logits come from an FP32 layer. That is correct and worth stating, because a reader may expect
  quantisation to affect the scoring path.
* Accuracy has a floor: random is 25% on HellaSwag and ARC-Easy, 50% on PIQA. A compressed model
  at chance is a *broken* model, not a weak one — check against the floor before reporting a
  degradation.
* Report **measured against dense**, the same way retention works for perplexity. An absolute
  accuracy without its dense reference is not interpretable.

#### A5 — prefill vs decode (§4.7). **Being done on `main`. Do not start it.**

Built in `benchmarking/phases.py`, driven by `scripts/run_prefill_decode.py`, configured by
`configs/experiments/prefill_decode.yaml`. Recorded here so you know what exists rather than
rebuilding it:

* **decode is timed against a primed cache** — the prompt forward runs once, untimed, and the timed
  region is a single-token step. Timing `generate` would fold the prefill into every repetition and
  report the sum under the decode label. Two tests pin it: decode must emit logits for exactly one
  position, and the cache must not grow across repetitions.
* **IQR** (`p25_ms`, `p75_ms`, `iqr_ms`) added to `LatencyStatistics` — §4.7 requires it and it was
  simply missing.
* **`rotate()`** for model-order rotation, with arms rebuilt inside each round so the rotation is
  real and two full-size FP32 models never sit in RAM at once.
* **FP32 arms only** (dense, pruning-only). Per D1 a packed layer dequantises on every forward, so
  timing it measures unpacking. The exclusion is written into the record rather than left as a gap.

At prompt lengths **128 and 512**, reporting **IQR**, with **model-order rotation** so thermal drift
does not load onto one arm.
**CPU-only, 4 threads, as pinned, on the designated benchmark host** — this is a tier-3 deployment
measurement, so both the CPU rule and the one-machine rule are absolute. You can *write* it
anywhere and test it against `tiny_causal_lm`; only the measured numbers are host-bound. Read
[benchmarking_protocol.md](benchmarking_protocol.md) before writing a line of it.

Note the standing limitation from **D1**: the sole latency backend is PyTorch native CPU **INT8**,
engine **`onednn`** (not `x86` — on the pinned torch, `supported_engines` is `['onednn']` only). W4
keeps quality and size but **never appears in a latency table**, because a packed 4-bit CPU linear
would measure the dequantisation kernel. RQ4 is answerable from the pruning-only arm, whose
weights stay FP32 -- but only at the sparsities actually benchmarked. [F-34](findings_log.md#f-34)
measured 30% at three scales and found no commensurate speedup; that is one point, not a curve.

---

### Task 5 — Freeze the confirmatory configuration, then run it **once** — ✅ **DONE 2026-08-10**

> **COMPLETE. Do not run this.** The freeze executed at `cbe2098` (re-frozen `e0c06ac` for the 1B
> offload pin, [B-44](findings_log.md#4-bugs-found-that-would-have-invalidated-results)) and the
> grid ran once: **171/171 cells, 42/42 pairs, 0 failures**, `AUDIT PASSED`. The result is
> **[F-37](findings_log.md#f-37)**.
>
> **What it cost, against what this section estimated:** ~60 h of compute over six days, not the
> ~38 h below. Measured per-cell: 160M 7.8 min, 410M 17.1, 1B 29.3 rising to ~47–54 for the arms
> that pack.
>
> **Three faults surfaced during execution, none numerical** —
> [B-45](findings_log.md#4-bugs-found-that-would-have-invalidated-results) (dense reference chosen
> across splits), [B-46](findings_log.md#4-bugs-found-that-would-have-invalidated-results) (reload
> guard demanded an unreachable sparsity, killing every `sequential` and `joint` cell while leaving
> controls green), and [B-48](findings_log.md#4-bugs-found-that-would-have-invalidated-results) (the
> runner never releases memory between cells; ~4 GiB of commit per 1B cell). **Fix B-48 before any
> future long grid** — with `continue_on_error` on, the `MemoryError` it causes drops cells silently
> rather than stopping.
>
> **The one operational lesson worth carrying:** B-46 removed both arms of every comparison while
> leaving every control green, so 36 records looked healthy and **zero comparisons existed**. A
> record count is not a progress measure. `audit_confirmatory_run.py` now fails closed on exactly
> that.

**The original instructions, kept because they document what was frozen and why:**

**This is last, and it is one-way.** Do not start it until tasks 1–3 are done and every selection
is frozen.

1. **Freeze everything** — budgets, orders per cell, coverage, calibration draws, evaluation window,
   thread count. Record the freeze in [protocol_freeze.md](protocol_freeze.md) with the commit SHA.
2. Run the confirmatory grid on the **test** split. The configs are already switched:
   `main_scale_sweep.yaml`, `extended_scale_sweep.yaml`, `qwen_validation.yaml`.
3. **R = 8 at 160M, 8 at 410M, 5 at 1B** — set by `sweep.replicates` and
   `sweep.replicates_by_model`. Roughly **38 hours**. The split is deliberate: at R=5 the *best
   possible* outcome (every replicate agreeing) is p = 0.0625, so **no significance claim exists at
   any effect size**, while R=8 reaches 0.008. The extra hours buy that transition, spent only on the
   two models carrying most of the scale-trend evidence.
4. **CPU evaluation, on the benchmark host, no GPU shortcut.** That is what the ~38 hours are, and
   the whole table has to come from one machine.
5. **R must be reported per cell** — A1 §5.1 makes it a hard requirement.
6. **No tuning after this point.** Not a threshold, not a window, not a coverage list. If something
   is wrong, it gets reported as a limitation, or the whole confirmatory stage is re-declared and
   re-run — not adjusted.

The analysis code is already built: `metrics/replicates.py` has `summarise_replicates`, an exact
`sign_test_p_value`, `paired_block_bootstrap` over whole evaluation windows, and `compare_scales`
(replicate-by-replicate). Per-window NLL and token counts are stored per run, which is what makes
the paired bootstrap possible.

---

## How to record what you do

**Non-negotiable, both of them.**

**Every number goes in [findings_log.md](findings_log.md)** with the conditions that produced it:
model revision SHA, budget, calibration fingerprint, evaluation window, replicate index, machine,
`METHOD_VERSION`. Append-only — **never silently delete a superseded number**; mark it retracted and
say why. A number that exists only in a commit message will be re-derived later under different
conditions, and a number without its conditions cannot go in the paper.

Use the next free `F-` id and the existing heading format:

```markdown
### F-30 - <one-line claim> {#f-30}

*<date> - <model> `<revision>` - <eval window> - `METHOD_VERSION = N`*
```

Bugs get a `B-` row in the same file. We are at **B-32**.

**At the end of any session that changed something meaningful, update
[STATUS.md](STATUS.md)** — current phase, what moved, what is next, any new open decision — and
commit. That file is the handoff, and it goes stale faster than you expect.

---

## Traps, gotchas, and things that have already cost us time

* **Do not `sys.modules.pop("torch")` in tests.** Re-importing raises
  `Only a single TORCH_LIBRARY can be used`. Use the `imported_after` /
  `environment_after_import` fixtures in `tests/conftest.py`, which check in a subprocess.
* **Heavy imports are lazy on purpose.** `import scale_aware_compression` must not pull in torch or
  transformers; tests enforce it. Import them inside functions.
* **Patch where a symbol is defined, not where it is used.** Several functions import their
  dependencies locally, so patching the importing module misses.
* **On Windows, do not round-trip UTF-8 through PowerShell `Get-Content`/`Set-Content`** — it reads
  as ANSI and mangles em-dashes. Use your editor or the file tools.
* **`.gitattributes` pins LF.** Files written by script often come out CRLF and then fail
  `ruff format --check` on line endings alone, with a diff that looks like a full-file rewrite.
* **`get_decoder_blocks` returns `list(current)` — a copy.** Assigning `blocks[0] = Catcher(...)`
  rebinds nothing the model will ever call, and silently captures zero activations (**B-29**). Use a
  forward pre-hook on the real module.
* **Run IDs must not collide across arms.** `make_experiment_id` follows §5.6's
  `<family>_<size>_<method>_<sparsity>_<bits>_<seed>` and appends `_rep{N}`. A collision means a
  compressed run overwrites the dense record it needs for retention.
* **`exists_valid` now compares the evaluation device** (**B-32**). Without that, switching a grid to
  GPU evaluation would reuse CPU records and mix devices inside one comparison at ~1e-5.
* **`skip_existing: true` makes sweeps resumable.** An interrupted sweep re-runs only cells with no
  record. You can shut the laptop down mid-grid and pick up where you left off — verified.

### Testing conventions

* Every test is offline: no downloads, no training, no real benchmarks.
* `tiny_causal_lm` is the key fixture — a randomly initialised 2-layer `GPTNeoXForCausalLM` from a
  config object. Same architecture class as Pythia, no download, milliseconds. Use it for anything
  touching a real model. `fake_tokenizer` is a byte-level stand-in.
* Markers: `requires_torch` runs in CI; `requires_model` and `slow` are excluded.
* **A test asserting a fairness invariant is worth more than one asserting a return type.**

### Code conventions

Python 3.11+, `src/` layout, full type hints, Google-style docstrings on every public symbol. Ruff
for lint and format, 100 columns. **British spelling in domain terms** — quantisation, tokenise,
normalise, optimisation. Comments explain *why*, not *what*. No new abstractions without a second
caller.

---

## What not to do

* **Do not reopen the frozen budgets.** §6.3 forbids it once results exist.
* **Do not move to W2 to chase a larger effect.** The mechanism is more active there, and selecting a
  precision because it yields a positive result is exactly what §6.3 forbids.
* **Do not re-propose clipping scale search or keep-benefit scoring.** Both were implemented,
  measured, and rejected because they made the layer objective *worse* — keep-benefit by 16 pp at
  W4, and analytically so.
* **Do not report a single draw as a point estimate.** That is what forced the F-25 → F-26
  retraction.
* **Do not run a confirmatory number on GPU**, and do not run any latency, throughput, memory or
  checkpoint-size measurement anywhere but CPU on the designated benchmark host.
* **Do not mix machines inside one comparison or one table.** Running compression on your own GPU
  is fine and expected; splitting a cell's two arms across two machines is not.
* **Do not claim a scaling law.** Three points cannot fit one, and the plan says so. We report a
  *direction*.
* **Do not tune anything after the confirmatory freeze.**

---

## The one-paragraph version

We are testing whether joint pruning-and-quantisation beats sequential, and whether that changes
with scale. The code now passes three independent correctness anchors, and the effect is **+1.69 pp
at 160M (3/3 draws positive) and indistinguishable from zero at 410M** — so it **shrinks** with
scale, against the motivating hypothesis. Everything so far is exploratory: validation split, three
draws, no significance claim available. **Your job, in order:** get per-block GPU offload working so
1B runs at all, confirm the budgets and freeze the sequential order at 1B, run the 12-run S6
mechanistic control, build the two required-but-missing pieces (downstream tasks and the
prefill/decode split), then freeze everything and run the test-split confirmation **once**. Verify
every step against the named reference values, log every number with its conditions, and remember
that **every fault found in this project so far has flattered the joint arm.**
