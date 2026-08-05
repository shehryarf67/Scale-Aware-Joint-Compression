# Response to the external review of `4575482`

**Date:** 2026-08-04 · **Reviewed commit:** `4575482` · **Response commits:** `712e2eb`, `e293c90`,
`5672ea6`, and this one

An external reviewer read `main` and produced a findings list. This file records the response
**item by item**, including the three categories that are easiest to lose: where the reviewer was
**more right than they claimed**, where I **deviated** from what they proposed, and what is **still
not done**.

Written because a review answered only in commit messages is a review whose open items disappear.
The bug rows are in [findings_log.md](findings_log.md); this is the audit trail of the response.

---

## 1. Where the reviewer was right, and the fix went further

### The frozen sequential order — worse than reported

The reviewer flagged that `scripts/run_downstream.py` maps every sequential arm to P→Q and never
consults the frozen table, and judged it *"currently harmless"* for the shipped downstream config.

**That judgement was correct for that file and wrong about the blast radius.** Checking the same
question one file over found that **`configs/experiments/main_scale_sweep.yaml` — the confirmatory
config — listed `sequential` for every cell and encoded no order resolution at all.** At
`pythia-1b`/`moderate` the frozen order is **Q→P** ([F-32](findings_log.md#f-32)). Both would have
run P→Q there: the weaker baseline, inflating the joint gain. That is
[B-30](findings_log.md#f-24) recurring, in the one run that cannot be redone.

Logged as **B-42**. Fixed with `scale_aware_compression.protocol` as the single machine-readable
source, resolution behind `sweep.use_frozen_order`, and a test asserting every test-split config
sets it.

**Consequence accepted deliberately:** `extended_scale_sweep.yaml` and `qwen_validation.yaml` now
**refuse to build a plan**, because no order has ever been frozen for `pythia-1.4b` or
`qwen2.5-0.5b`. A config that refuses is a better signal than one that quietly runs P→Q and calls
the result a joint gain. Documented in both files.

### Sign-test ties — real, and verified latent

Logged as **B-40**. Confirmed exactly as described. Then checked whether it had *fired*: no gain in
any committed record is exactly zero, and the smallest recorded 1B gain is **+0.0044 pp**, which
rounds to +0.00 in a two-decimal table but is genuinely positive. **No published figure moved.** The
size of the error is pinned as a test rather than described: three positives and one tie gave
p = 0.625 and now gives p = 0.25.

---

## 2. Where I deviated from what the reviewer proposed

### Downstream provenance — neither of the two offered options

The reviewer proposed either (a) reuse the standard `ExperimentRecord`/`ExperimentTracker` schema for
downstream runs, or (b) embed an immutable reference to a standard run record in every row.

**I did neither.** The thirteen provenance fields are written **directly onto each downstream row**:
git commit (with `-dirty`), model revision SHA, `METHOD_VERSION`, budget label, sparsity, bits,
resolved sequential order, calibration replicate, calibration fingerprint, targeted parameter count,
task split, timestamp, status.

Reasoning: (a) would need `RESULT_SCHEMA_VERSION` bumped and a `downstream` section added to a schema
that ~70 committed records already use, for a secondary endpoint — a schema migration is a larger
risk than the problem. (b) requires a standard run record to *exist* for each downstream cell, and
downstream runs its own compression rather than reusing a sweep cell, so there is often nothing to
point at. Writing the fields directly gets the auditability without either cost.

**Judge this deviation on the property that matters**: can a row be reproduced from what it carries?
Yes — commit, revision, budget, order, draw and fingerprint fully determine it.

### Downstream uncertainty — option 1 of the three offered

The reviewer offered: (1) descriptive secondary endpoints with no formal claim, (2) paired
calibration replicates for downstream, (3) predeclared item-level confidence intervals.

**Chose (1), and did part of (3) without letting it be mistaken for (2).** The harness standard
error is recorded and drives the chance verdict, but the policy declared in
`configs/experiments/downstream.yaml` states explicitly that this quantifies **task-item sampling
only** and says nothing about calibration-draw variance — which [F-26](findings_log.md#f-26) measured
at a 1.47 pp swing, wider than either arm's own spread. An item-level interval must not be presented
as bounding draw-to-draw variation.

(2) was not chosen on **scope, not cost**: it is affordable (~4 min per 160M evaluation) but it is 27
evaluations and §4.3 asks whether the compared arms stay usable, not for a second significance test.
Recorded as a declared possible addition rather than a reinterpretation of this run.

### "Above chance" thresholds — chosen, and made non-load-bearing

The reviewer warned against choosing statistical thresholds after seeing results. The three-way
verdict uses **± 2 standard errors**, which *is* a threshold chosen now.

Defence, stated rather than assumed: 2σ is the conventional ~95% default rather than one selected to
make a row read a particular way, **and the labelling is descriptive** — the primary downstream
comparison is accuracy retention against dense, so no claim depends on which side of the line a row
falls. If a threshold has to be chosen late, the right mitigation is to ensure nothing rests on it.

By contrast, tie handling in the sign test uses **exact equality with no tolerance**, precisely
because that threshold *would* be load-bearing and every candidate value is now visible in the
results.

---

## 3. Where the reviewer was wrong or imprecise

* **"HellaSwag's 39% smoke result is unexplained."** Right to flag, but it is now settled and was not
  a scoring defect: the full-task value is **0.2816** against a published ~0.29. `--limit` takes the
  **first N examples, not a random sample**, so a 200-example prefix is biased. Recorded in
  [F-35](findings_log.md#f-35).
* **"The partner branch is eight commits ahead and eight behind."** Not verified, and the counts move
  with every push to `main`, so the figure is not durable. What matters and *is* verified: its
  `F-31`/`B-34` entries collide with different content on `main`, so it must not be merged wholesale.
* **The reviewer could not confirm the test count from CI.** True, and see the open items below — I
  have not investigated CI either, so their inability to verify it stands as a real gap rather than
  a misreading.

---

## 4. Still not done

Recorded here rather than left implicit. **None of these can change a number**; all of them affect
auditability or presentation, and the first two must land before A1 step 9.

| Item | Status | Why it matters |
| --- | --- | --- |
| **Compact recomputable evidence artefacts** | ✅ **done** | `results/evidence/` is now committed, with a `.gitignore` exception and an enumerated allowance in the no-artefacts invariant test. Four plain-text files, **1.9 MB**: `cells.csv` (85 normalised rows with commit, revision, `METHOD_VERSION`, host, draw, fingerprint and window), `joint_gains.csv` (26 per-replicate gains against **best-of**, with the chosen order as an explicit column, because B-30 was measuring against the weaker one), `windows.csv` (27,608 per-window NLL and token rows — the only way to reproduce a paired block bootstrap rather than take the interval on trust), and `MANIFEST.json` (sha256 of every source record, so excluded artefacts stay identifiable, plus a stated exclusion list). **Verified**: the committed set recomputes the headline exactly — +1.689 / +0.387 / +0.200 pp against F-27's +1.69 / +0.39 and F-32's +0.20. `--check` mode plus a test guard the staleness failure mode |
| **CI status** | 🟡 **partly addressed** | I cannot observe CI runs from this machine — `gh` is not installed — so the reviewer's inability to confirm the test count **stands as a real gap**. What I could do is remove the likeliest breaker: lm-eval had been added as a **core** dependency, so CI would have installed ~20 extra packages (nltk, sacrebleu, rouge_score) for tests that never import it. It is now the `[downstream]` extra, with the exact pin kept in both places and five tests asserting the two cannot drift or soften to a range. The workflow installs `pip install -e . -r requirements-dev.txt`, so it no longer pulls the harness. **Test counts remain self-reported until someone with `gh` access confirms a green run on `main`.** |
| **F-35 downstream results** | 🔵 **run in flight** | A4 is implemented but incomplete. Step 9 must wait for all nine evaluations, full tasks, comparison against dense, and a committed summary |
| **A confirmatory manifest** | ✅ **done, and the freeze is executed** | `results/evidence/confirmatory_manifest.json`, built by `scripts/build_confirmatory_manifest.py`, `valid_for_freeze: true`, `checks_failed: []` at clean commit `cbe2098`. 210 cells fully resolved with per-cell frozen-order evidence; every revision a 40-char SHA; split=test, both devices CPU; R=8/8/5 with 1B's unreachable significance carried as an explicit warning; the amended importance rule with its withdrawn clause; and the four exclusion rules stated. The builder **refuses a dirty tree** rather than warning. Ten tests guard it. A1 step 9 recorded in `protocol_freeze.md` |
| Downstream at the moderate budget | ⬜ deliberate omission | Only the aggressive budget is run. Reasoning is in `downstream.yaml`: W8 retains 96% perplexity at 1B, so a downstream difference would be below what three tasks at these sample sizes resolve. A defensible addition, not a substitute |
| **pythia-1.4b — the extended fourth scale point** | ⬜ **tracked, deferred** | A commitment in the plan that was **missing from my own open-items list until the author asked** — recorded now rather than left invisible. Registered, revision pinned (`fedc38a1`), config exists, GPTNeoX adapter shared with the other Pythias. Not downloaded. Blocked on: download, order selection (its config **refuses to plan** without a frozen order — B-42's guard working), and a VRAM measurement. Feasibility improved materially and nobody updated the assessment: `protocol_freeze.md` D2 records 1B and 1.4B as **the same width** (hidden 2048, intermediate 8192), and per-block offload makes peak memory width-bound rather than depth-bound, so ~4.29 GiB is the expectation — **to be measured, not extrapolated**. `methodology.md` calls the extended sweep optional and hardware-dependent, and §8.2 forbids it consuming the primary sweep's time, so it follows step 10 |
| **qwen2.5-0.5b — external validity** | ⬜ **tracked, deferred** | Same omission from my list. Registered, revision pinned (`060db649`), and the **Qwen2 adapter is implemented** including the part that is easy to get wrong: Qwen2's *sequential* residual gives four dependency groups per block against Pythia's two, which if mishandled makes reconstruction blockwise while still reporting as layerwise. Not downloaded; needs order selection. Same §8.2 rule |
| 1B S6 control | ⬜ deliberate omission | A1 §5.4 makes it conditional on the smaller-model control producing a useful distinction. It did at 160M ([F-33](findings_log.md#f-33)), so this is now defensible if wanted |

---

## 5. The interpretation the reviewer proposed, and whether it is adopted

They proposed this as the most defensible current statement:

> On validation data, joint compression shows a positive W4 advantage at 160M that decreases at 410M
> and 1B, while W8 behaves as an approximately inert control. These results replicate across three
> paired calibration draws but remain exploratory because model selection and budget selection used
> validation, and three draws cannot support a conventional significance claim.

**Adopted.** It matches what [F-27](findings_log.md#f-27), [F-32](findings_log.md#f-32) and
[F-33](findings_log.md#f-33) support, and it is narrower than anything the STATUS summaries claim.

Their list of things **not** to claim is adopted verbatim as a limitations checklist: not that the
hypothesis is confirmed; not that scale *causes* the shrinkage; not that pruning has no latency
effect in general; not that downstream usefulness is preserved until F-35 is complete; and no
significance at 1B under R=5.

One addition of my own: **not that the arms agree across endpoints.** The first downstream numbers
show joint *behind* sequential on all three 160M tasks while joint *wins* by +1.69 pp on perplexity
at the same scale. One draw, one scale, and a declared secondary endpoint — but a divergence between
endpoints belongs in the paper rather than being smoothed over.
