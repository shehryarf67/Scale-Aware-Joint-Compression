# Why we are changing how activations are captured

**Date:** 2026-07-31 · **Status:** ✅ **applied and all gates passed** — see [F-29](findings_log.md#f-29) ·
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

## The gates — all passed

Gram equivalence is necessary, not sufficient, so the full reproductions were run:

| Gate | Requirement | Result |
| --- | --- | --- |
| 1. Gram equivalence | bit-identical on 48 modules | ✅ **0.000e+00** |
| 2. **Full 160M cell** | reproduce [F-23](findings_log.md#f-23): **65.261 / 64.041** | ✅ **exact**, gain +1.08 pp |
| 3. Exact-optimum anchor | 0 rows below optimum, 0 worse than naive | ✅ 0.6409, every module identical |
| 4. **410M cell** | reproduce [F-25](findings_log.md#f-25): **37.851 / 37.415** | ✅ **exact** |
| 5. `METHOD_VERSION` | bump if any number moves | **not needed — nothing moved** |

**Caveat on gate 3, because it is easy to over-read.** Both anchor scripts capture activations with
their own hooks and call `sweep_reconstruct` directly — they never enter `compress_model_layerwise`. Their
passing confirms the solver is untouched, which it is, and says **nothing** about the capture change.
**Gates 2 and 4 are the meaningful ones**, and they are exact.

Because nothing moved, the ~50 existing run records stay valid and no recompute is needed. That was the
main risk of touching this file and it did not materialise.

**Why verification happens on 160M and not 1B**, since it is the natural question: 160M is the only
scale where the right answer is already known. F-23's values have been reproduced exactly four separate
times, the anchors run there, and a verification cycle costs 9 minutes against 1B's ~90. 1B is the
*worst* place to validate a refactor, because the current code does not work properly there — a new
number would be indistinguishable between "the refactor is correct", "the refactor is buggy", and "the
old path was thrashing". At 160M any difference is attributable to the change and nothing else.

## What it actually bought, measured

The stage decomposition, taken from a real 160M joint cell rather than estimated:

| Stage | Before | After |
| --- | --- | --- |
| compress | ~170 s | **62 s** — 2.7× |
| evaluate quality (CPU) | ~377 s | 377 s |

**Which revealed the more important thing: compression was only 14% of a cell.** Perplexity evaluation
on CPU was the other 86%, and nobody had looked. `evaluation.device` has always been a config field, and
`check_evaluation_device` warns rather than errors off CPU — *"Exploratory evaluation on GPU is fine, but
any number reported in the write-up must be produced on CPU."*

| | CPU | GPU |
| --- | --- | --- |
| Perplexity, 160M dense | 36.974099 | 36.974405 |
| Time, 493 × 512 | **345.6 s** | **15.4 s** — 22.5× |
| Relative difference | — | 8.3e-06 |

That drift is the same magnitude as the thread-configuration sensitivity in
[F-23](findings_log.md#f-23), three orders of magnitude below the ~1e-2 effects being measured.

**Together: an exploratory 160M cell goes from ~9.3 min to ~1.3 min, about 7×.** The 13-cell screening
grid drops from 2 h 08 m to roughly 20 minutes. The **confirmatory test-split run keeps CPU evaluation
and its ~38 hours** — that is the rule and it is not being touched.

One guard was needed to make GPU evaluation safe: `exists_valid` did not compare the evaluation device,
so switching would have let `skip_existing` reuse CPU records inside a GPU grid and mix devices within a
single comparison. It compares it now (B-32).

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
