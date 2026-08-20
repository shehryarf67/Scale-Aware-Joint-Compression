# Confirmatory timing pilot

**Status:** complete; initial two-cell pilot plus corrected joint-only rerun successful

**Date:** 2026-08-05

**Commit:** `0f05b9e3138b01b0399a55c3699f193d1ccd72d5` (`confirmatory-freeze-v2`)

**Config:** `configs/experiments/confirmatory_timing_pilot.yaml`

**Data:** validation, 493 sequences × 512 tokens

**Purpose:** runtime and operational validation only; quality values are intentionally omitted

| Cell | Total minutes | Compression | Checkpoint verification | CPU quality | CPU benchmark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pythia-1B dense | 25.12 | — | cached dense snapshot | 23.98 | 0.62 |
| Pythia-1B joint, aggressive, replicate 0 | 121.40 | 65.67 | 0.42 | 55.13 | excluded by D1 |
| Pythia-1B joint, aggressive, replicate 0, **offload corrected** | **53.51** | **11.70** | **0.37** | **40.98** | excluded by D1 |

The joint checkpoint was 1.145 GiB, independently reloaded with maximum logit difference 0.0, and
had SHA-256 `6ea850eac1d66caf7c127b9c95577a331eb8cdac8b2166f2eff79bdf5d9ee665`.

## Finding

The joint record resolved `compression.reconstruction.offload_blocks: false`. This is the default,
not the verified 1B path used by `screening_1b.yaml`. Its 65.67-minute compression stage is 14.3×
F-31's 4 min 34 s offloaded measurement. The v2 confirmatory freeze must therefore not launch.

Amendment A3 froze `offload_blocks: true`, the v3 manifest passed every check, and the corrected cell
completed at commit `e0c06ac17fb95ee27156508ed114f145d0111784`. It independently reloaded with
maximum logit difference 0.0 and reproduced the same artefact hash, closing B-44 without changing a
scientific condition.

## Revised estimate

Using executable rather than logical-grid counts gives approximately **65.3 hours** before retries:

| Scale | Executable cells | Cost assumption | Total |
| --- | ---: | ---: | ---: |
| 160M | 65 | 7.5 min/cell | 8.1 h |
| 410M | 65 | 19.5 min/cell | 21.1 h |
| 1B | 1 dense + 40 compressed | 25.12 min dense; 53.51 min compressed upper bound | 36.1 h |

The W4 joint cell is used as the conservative 1B compressed-cell bound. Coverage remains the full
210-slot five-arm grid represented by 171 executable records.
