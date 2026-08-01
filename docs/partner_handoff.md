# Partner handoff — read this before writing any code

**Date:** 2026-07-31 · **Repo state:** `main` at the commit that added this file ·
**Audience:** the second author, and whatever LLM assistant they are using

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

### Task 1 — Per-block GPU offload. This is what unblocks Pythia-1B.

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

---

### Task 2 — 1B budget confirmation and order selection

Only after task 1's gates pass.

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

### Task 4 — A4 and A5, the two required gaps nobody has started

Both are **§-required**, not optional, and neither is blocked by anything. They have simply been
behind the correctness work.

**A4 — downstream tasks (§4.3).** HellaSwag, PIQA, ARC-Easy via lm-eval-harness, **pinned** (the
version is frozen by §2.7). Every arm at every budget at every scale. Perplexity alone does not
establish that a compressed model is still useful, and a reviewer will ask.

**A5 — prefill vs decode timed separately (§4.7).** At prompt lengths **128 and 512**, reporting
**IQR**, with **model-order rotation** so thermal drift does not load onto one arm.
**CPU-only, 4 threads, as pinned, on the designated benchmark host** — this is a tier-3 deployment
measurement, so both the CPU rule and the one-machine rule are absolute. You can *write* it
anywhere and test it against `tiny_causal_lm`; only the measured numbers are host-bound. Read
[benchmarking_protocol.md](benchmarking_protocol.md) before writing a line of it.

Note the standing limitation from **D1**: the sole latency backend is PyTorch native CPU **INT8**,
engine **`onednn`** (not `x86` — on the pinned torch, `supported_engines` is `['onednn']` only). W4
keeps quality and size but **never appears in a latency table**, because a packed 4-bit CPU linear
would measure the dequantisation kernel. RQ4's sparsity→latency curve comes free from the
pruning-only arm, whose weights stay FP32.

---

### Task 5 — Freeze the confirmatory configuration, then run it **once**

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
