# Protocol Amendment A2: resilient confirmatory execution

**Recorded before any held-out test result was inspected.**

Amendment A2 changes failure handling from fail-fast to continue-and-audit and corrects the
manifest from 210 logical grid slots to 171 executable cells by representing dense once per model.
No scientific condition or analysis rule changes.

The logical grid remains all three scales, all five arms, both frozen budgets, and every declared
calibration replicate. It contains 210 Cartesian slots. Dense evaluation is independent of budget
and calibration draw, so 39 identical dense aliases are not executed. The executable set is:

| Scale | Dense | Compressed | Total |
| --- | ---: | ---: | ---: |
| Pythia-160M | 1 | 64 | 65 |
| Pythia-410M | 1 | 64 | 65 |
| Pythia-1B | 1 | 40 | 41 |
| **Total** | **3** | **168** | **171** |

There are 42 required paired sequential-versus-joint comparisons: 16 each at 160M and 410M and 10
at 1B. The shared `executable_cells()` function is authoritative for runner, manifest and audit.

`continue_on_error: true` is an operational resilience policy. A failed cell is recorded but is not
a valid resume candidate. Relaunching the same command retries failed, missing, unreadable or stale
cells and skips only valid successes. No analysis may begin until `scripts/audit_confirmatory_run.py`
exits successfully for all 171 cells and all 42 pairs.

Checkpoint artefacts are retained throughout execution and the completeness audit. They may be
pruned only under a separately recorded retention procedure after hashes and reload verification
have been preserved; records must never point at an artefact known to have been deleted.
