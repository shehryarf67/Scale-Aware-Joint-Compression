# Phase 7 screening audit

**Status:** completed 2026-07-31. This is a curated audit of the raw JSON run records retained in
`outputs/metrics/`; it is not a confirmatory result table.

## Decision

| Budget | Definition | 160M sequential / joint retention | 410M sequential / joint retention | Decision |
| --- | --- | --- | --- | --- |
| `moderate` | 30% sparsity, W8 | 80.28% / 80.25% | 76.15% / 76.15% | Freeze as control |
| `aggressive` | 30% sparsity, W4 | 57.65% / 58.95% | 57.07% / 59.03% | Freeze as headline condition |

The other 160M candidates were rejected: S2, S3, and S4 were catastrophic; S6 was eligible but
W8 is the near-inert control precision and therefore cannot test the joint mechanism.

## Scope and provenance

- All cells used one screening replicate and a 493-sequence × 512-token WikiText-2 validation
  window. These values choose budgets only; they are not paper findings.
- Each selected 410M pair used the same calibration fingerprint and solver budget within its
  sequential-versus-joint comparison, as checked by the screening summariser.
- The imported records report `method_version: 4` and a Colab source identifier of
  `aec509917226874e7f6872b768e8d82d9cd74e33-dirty`. That commit is not present in this local Git
  history, so the Colab source archive must be retained alongside the raw metrics for exact
  reproduction.
- Phase 8 uses three calibration-subset replicates and the WikiText-2 **test** split. It must not
  combine these Phase 7 validation numbers with final reporting.
