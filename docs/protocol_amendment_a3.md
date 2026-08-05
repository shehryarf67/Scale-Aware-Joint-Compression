# Protocol Amendment A3: freeze the verified 1B residency path

**Recorded before any held-out test result was inspected.**

The validation-only timing pilot in F-36 showed that `confirmatory-freeze-v2` inherited
`compression.reconstruction.offload_blocks: false`. That default is not the Pythia-1B path already
verified in F-31 and used for 1B screening. It made the joint apply stage take 65 min 24 s, compared
with 4 min 34 s under per-block offload.

Amendment A3 explicitly sets `offload_blocks: true` in the main confirmatory and timing-pilot
configs, makes manifest generation reject any other value, and includes the field in resume
validation. The runner, manifest and existing checkpoint records therefore cannot silently disagree
about residency policy.

This is an operational correction, not a scientific-condition change. F-31 established the
resident and offloaded paths as bit-identical: 0 of 148 targeted tensors disagreed and the reproduced
160M sequential/joint cell was exact. Models, weights, budgets, methods, calibration draws,
evaluation split, metrics, frozen orders and replicate counts are unchanged.

`confirmatory-freeze-v2` is superseded before launch. A successor manifest must be built from a
clean commit and the corrected joint timing cell must complete before the full runtime estimate is
updated.
