# Confirmatory timing pilot

**Status:** complete, 2/2 cells successful

**Date:** 2026-08-05

**Commit:** `0f05b9e3138b01b0399a55c3699f193d1ccd72d5` (`confirmatory-freeze-v2`)

**Config:** `configs/experiments/confirmatory_timing_pilot.yaml`

**Data:** validation, 493 sequences × 512 tokens

**Purpose:** runtime and operational validation only; quality values are intentionally omitted

| Cell | Total minutes | Compression | Checkpoint verification | CPU quality | CPU benchmark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pythia-1B dense | 25.12 | — | cached dense snapshot | 23.98 | 0.62 |
| Pythia-1B joint, aggressive, replicate 0 | 121.40 | 65.67 | 0.42 | 55.13 | excluded by D1 |

The joint checkpoint was 1.145 GiB, independently reloaded with maximum logit difference 0.0, and
had SHA-256 `6ea850eac1d66caf7c127b9c95577a331eb8cdac8b2166f2eff79bdf5d9ee665`.

## Finding

The joint record resolved `compression.reconstruction.offload_blocks: false`. This is the default,
not the verified 1B path used by `screening_1b.yaml`. Its 65.67-minute compression stage is 14.3×
F-31's 4 min 34 s offloaded measurement. The v2 confirmatory freeze must therefore not launch.

No full-run estimate is derived from the 121.40-minute cell. It combines the real packed-W4 CPU
evaluation cost with an accidental non-offloaded compression cost. Freeze `offload_blocks: true`,
re-freeze the manifest, and rerun only the joint timing cell before updating the total.
