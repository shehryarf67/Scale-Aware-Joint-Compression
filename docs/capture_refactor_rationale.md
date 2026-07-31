# Why we are changing how activations are captured

**Date:** 2026-07-31 · **Status:** equivalence proven on Pythia-160M, refactor not yet applied ·
**Audience:** anyone picking this up, in particular the partner working on Phase 8

This document exists because the change it describes touches
`compression/layerwise.py` — the most safety-critical file in the project, and the one three
independent anchors currently vouch for ([F-19](findings_log.md#f-19),
[F-20](findings_log.md#f-20), [F-22](findings_log.md#f-22)). Nobody should have to reverse-engineer
the reasoning from a diff.

---

## The problem, in one line

`_compress_group` captures activations by running the **entire model** forward, once per dependency
group:

```python
with torch.inference_mode():
    for batch in batches:
        model(batch.to(next(model.parameters()).device))
```

That is correct. It is also doing roughly **16× more forward work than necessary at Pythia-1B**, and it
forces the whole model to be resident on the capture device — which is what makes 1B unrunnable on this
hardware.

## Why it is wasteful

To capture the activations entering block *k*, you need the output of blocks 0…*k*−1. You do **not**
need blocks *k*+1…*N* — they cannot influence an earlier block's inputs.

Running the full model per group means:

- blocks after the current one run and their outputs are discarded — pure waste;
- blocks before it are recomputed from the embedding for **every group**, instead of having their
  output cached once.

| | Current | Block-sequential |
| --- | --- | --- |
| Block-forwards per cell (1B: 16 blocks × 2 groups) | **O(blocks × groups)** ≈ 512 | **O(blocks)** ≈ 48 |
| Model residency on the capture device | **whole model** — 3.77 GiB at 1B | **one block** — ~0.2 GiB |

## Why it blocks Pythia-1B specifically

Measured on this machine (RTX 4050, 6.00 GiB, ~4.95 GiB free):

| | |
| --- | --- |
| 1B FP32 weights resident on GPU | **3.77 GiB** |
| Widest layer's inverse-Cholesky temporaries (`in_features` 8192) | **~2.5 GiB** |
| **Measured peak** | **6.31 GiB** — free memory hit exactly 0.00 |

It "completed", but only by spilling into host memory: the widest-layer solve took **33.9 s against
4.8 s** standalone, a 7× slowdown. The real run also holds calibration activations and the previous
block's outputs, so it would thrash or fail outright.

**`block_size` does not help.** Tested at 128 / 64 / 32: peak was 6.31 / 6.37 / 6.37 GiB. The dominant
allocation is the Gram factorisation, not the block loop, so shrinking the loop changes nothing.

> **Documentation correction found on the way.** `sweep_reconstruct`'s docstring claims `block_size` is
> "purely a memory/throughput knob; it does not change the result." The three block sizes gave three
> distinct losses — 1.983672e+07, 1.983672e+07, 1.983673e+07. The difference is ~5e-7 relative, smaller
> than the thread-configuration sensitivity already recorded in [F-23](findings_log.md#f-23), so it is
> negligible in effect. But the docstring overstates the guarantee and should say "does not
> meaningfully change the result" instead.

## The fix

The standard approach, used by both SparseGPT and Wanda, and already implemented in this repository in
`scripts/run_sparsegpt_external_anchor.py`:

1. capture the hidden states entering block 0 **once**;
2. for each block: move it to GPU, run **only that block** over the cached states to capture, compress
   its groups, then run it once more to advance the cache to the next block, and move it back;
3. one block resident at a time — ~0.2 GiB of weights plus the Gram and its temporaries.

## The equivalence claim, and the evidence for it

**Claim.** Blocks after *k* cannot affect block *k*'s inputs, so both strategies capture the same
activations, hence the same Gram, hence the same mask, scales and reconstruction.

**Verified.** `scripts/verify_block_sequential_capture.py` captures the Gram both ways on real
Pythia-160M with the real calibration draw and compares every targeted module:

```
modules compared            : 48
worst relative Gram error   : 0.000e+00
worst relative norm error   : 0.000e+00
modules disagreeing > 1e-5  : 0

VERDICT: EQUIVALENT
```

**Bit-identical, not merely close.** Block-sequential capture is not an approximation of the current
path — it computes the same quantity, which is why the extra work buys nothing.

## What still has to pass before the refactor is trusted

Gram equivalence is necessary, not sufficient. The remaining gates, in order:

| Gate | Requirement |
| --- | --- |
| 1. Gram equivalence | ✅ **passed** — bit-identical on 48 modules |
| 2. Full 160M cell | must reproduce [F-23](findings_log.md#f-23): sequential **65.261**, joint **64.041**, to three decimals |
| 3. Wanda anchor | must still pass — 0 differing mask positions |
| 4. Exact-optimum anchor | must still pass — 0 rows below the optimum, 0 worse than naive |
| 5. 410M cell | must reproduce [F-25](findings_log.md#f-25): sequential **37.851**, joint **37.415** |
| 6. `METHOD_VERSION` | bump **if any number moves**, and re-run the affected grids |

**Why verification happens on 160M and not 1B**, since it is the natural question: 160M is the only
scale where the right answer is already known. F-23's values have been reproduced exactly four separate
times, the anchors run there, and a verification cycle costs 9 minutes against 1B's ~90. 1B is the
*worst* place to validate a refactor, because the current code does not work properly there — a new
number would be indistinguishable between "the refactor is correct", "the refactor is buggy", and "the
old path was thrashing". At 160M any difference is attributable to the change and nothing else.

## The honest cost

If gate 2 or 5 fails — if the numbers move at all — then the ~50 run records taken on 2026-07-30/31
become stale and the screening and replicate work needs redoing. That is a few hours of recompute.

Against that: a permanent ~16× reduction in capture work, which matters because the confirmatory stage
is budgeted at ~38 hours, and **Pythia-1B becoming runnable at all**, which is the difference between a
two-point and a three-point scale study.

## What is *not* being changed

- the reconstruction solver, the saliency rule, the mask construction, the packing path;
- the dependency-group scheme itself — capture still happens once per group, after the previous group
  is written back, which is the [B-19](findings_log.md) fix and is what makes reconstruction genuinely
  layerwise;
- anything about the arms, the budgets, or the evaluation.

Only *how the activations are gathered* changes. The equivalence check above is precisely the claim that
this is a distinction without a difference in the numbers.

## Related

- [findings_log.md](findings_log.md) — F-19, F-20, F-22 are the anchors this must not break
- [protocol_freeze.md](protocol_freeze.md) — D2 explains why the sweep solver was chosen, and why the
  Gram factorisation is the memory-dominant step
- `scripts/verify_block_sequential_capture.py` — the equivalence check, re-runnable
- `scripts/run_sparsegpt_external_anchor.py` — a working block-sequential driver for GPT-NeoX already
  in this repository, and the model for the refactor
