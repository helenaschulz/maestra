# Results — the measurement ledger

> Every claim in this project traces back to a row here: a graded run against a real answer key,
> LLM vs. the deterministic `--no-llm` AutoGluon baseline under the same budget and seed, or a
> decision experiment with a held-out truth. Includes the negative results and the corrections —
> that is the point. Lower-is-better metrics are marked; otherwise higher is better. Raw machine
> logs: `runs.jsonl` / `benchmark.jsonl` (local, not committed).

## MLE-bench tasks

| Date | Competition | Metric (mode) | Baseline | Maestra | Winner | Medal | CV↔LB gap | Note |
|---|---|---|---|---|---|---|---|---|
| 2026-06-14 | tabular-playground-series-dec-2021 | accuracy (aligned) | 0.9592 | **0.9607** | Maestra (+0.0015) | gold\* | < 0.001 ✓ | \*thresholds degenerate → "gold" is cheap; memory-bound (RF/ExtraTrees OOM on 8 GB) |
| 2026-06-14 | leaf-classification | log_loss ↓ (proba) | 0.0737 | 0.0783 | Baseline (−0.0046) | none | ~0.82 (genuine) | uncalibrated. OOF clean post-fix (no NaN, sums to 1); residual gap real (under-confident bulk + 15/891 tail rows). |
| 2026-06-14 | leaf-classification + `--calibrate` | log_loss ↓ (proba) | 0.1078 | **0.0453** | Maestra (+0.0625) | none | gap 1.15→0.69 (CV cal) | T≈0.3 (sharpen). CV-side calibration cuts the gap ~40%; submission calibration helped Maestra but **hurt baseline** (0.074→0.108) — OOF-T does not transfer to the full-data model. |
| 2026-06-14 | leaf-classification + `--hybrid` | log_loss ↓ (proba) | 0.0737 | 0.0553 | — (noise) | none | 0.79 | Gate verified fair: 5 candidates, **all rejected `no_improvement`** (`hybrid_kept=0`), no timeouts. The 0.078→0.055 swing is **run nondeterminism**, not hybrid (no features kept). |

**Reading the two so far:**
- **tps-dec-2021** — Maestra beats the baseline by a real (if small) margin, and the CV↔LB gap
  is < 0.001 in both directions → the leakage-free CV is **trustworthy** on real competition data.
  But "gold" here is not an achievement (gold ≈ silver ≈ bronze ≈ 0.9566), and the run was
  memory-bound, so half the model portfolio was skipped.
- **leaf-classification** — uncalibrated, the LLM conductor **hurt** (0.0783 > 0.0737, lower is
  better), no medal. The OOF assembly bug (which had inflated the gap further) is fixed and verified.
  The residual CV↔LB gap (~0.82) is **genuine**: the fold models are under-confident on the bulk
  (calibration fits T≈0.3, sharpen, cutting the gap ~40%) plus a tail of ~15/891 rows at P(true)≈0.
  **Calibration finding:** CV-side temperature scaling reliably narrows the gap, but applying it to
  the *submission* is unreliable — it helped Maestra's LB (0.078→0.045, flipping it ahead of the
  baseline) yet hurt the baseline's (0.074→0.108), because the fold-OOF temperature doesn't transfer
  to the full-data final model. Lesson: trust calibration for *reporting*, not as a blanket LB lever.

## Kaggle submissions (pre-benchmark)

| Date | Competition | Metric | Score | Note |
|---|---|---|---|---|
| earlier | Playground S6E6 "Predicting Stellar Class" | balanced_accuracy | public 0.95045 | ≈ holdout 0.9516 → pipeline empirically leakage-clean |

## Local mini-benchmark (`maestra-bench`, answer key carved from train)

| Competition | Metric | Baseline | Maestra | Winner |
|---|---|---|---|---|
| Titanic | balanced_accuracy | **0.7933** | 0.7322 | Baseline — single seed; **superseded 2026-07-04 by the K1 5-seed verdict: *undecided*** (this seed was a downward outlier, see K1) |
| House Prices, seed 42 | rmse ↓ | 26 453 | **25 828** (−625) | Maestra |
| House Prices, seed 7 | rmse ↓ | 25 745 | **24 343** (−1 402) | Maestra |
| House Prices + `--hybrid`, seed 42 | rmse ↓ | 26 453 | 25 390 / 26 350 | (0/5 features kept — noise) |
| House Prices + `--hybrid`, seed 7 | rmse ↓ | 25 745 | 27 925 (+2 181) | Baseline (0/5 kept — pure nondeterminism) |

**The decision experiment (2026-07-02).** All prior tasks (tps, leaf, titanic) had anonymous/
semantics-free columns — terrain where an LLM is structurally blind, and it never beat the baseline.
House Prices (43 meaningful text columns, 19 with NaNs, ordinal quality ratings, years) is the first
task with an LLM information edge. Findings across two seeds:

1. **Plain Maestra beats the baseline on both seeds** (−625, −1 402) — direction replicates, **H2
   supported**: the conductor pays off where column semantics exist. But the margin sits *inside* the
   noise band — three near-identical seed-42 Maestra runs span ~960 rmse, so the −625 is within one
   nondeterminism swing; the −1 402 at seed 7 is more convincing.
2. **The `--hybrid` layer adds nothing** — the CV gate kept **0/5** generated features on both seeds
   (correctly). The apparent hybrid scores are pure run-to-run noise: at seed 7, "maestra+hybrid" even
   *lost* to the baseline (+2 181) while keeping zero features. Single-run hybrid comparisons are
   worthless.
3. **The provenance is the sharp finding.** The LLM proposed exactly the sensible domain features
   (`age_of_house` = YrSold−YearBuilt, `remodel_age`, `garage_age`, `total_bathrooms`,
   `total_porch_area`, `quality_condition_index`) — and the gate rejected all of them, because
   AutoGluon's trees already extract that signal from the raw columns (`age_of_house`/`remodel_age`
   had cv_delta *exactly* 0.0 → the new `no_effect` verdict).

**Refined thesis:** the conductor's value is in **cleaning/encoding judgment** (plain Maestra), *not*
in feature generation — even semantically-correct engineered features don't beat AutoGluon, which
does FE-equivalent work internally. Onboarding bonus: the first regression task exposed (and fixed) an
integer-target stratification bug.

## Methodology — what the seed sweep showed (correcting an earlier note)

A 3-seed sweep (`--seed 1/2/3`) on leaf settled it:
- **The LB is seed-invariant.** `--seed` reseeds the CV folds but not the final model (the CV-path
  final fit doesn't consume our seed), so Maestra is **0.0783** and the baseline **0.0737** on every
  seed, std = 0. **Maestra reproducibly loses by 0.0046** — a solid negative result, not noise. (An
  earlier note here claimed "within noise"; that was wrong — it conflated the stable LB with two
  cross-invocation outliers: 0.081 on the very first run and 0.055 on the hybrid run, both from
  LLM-plan drift across separate invocations / the hybrid path, not the seed.)
- **The CV log_loss is what's highly variable** — it swings 1.23–1.73 (±0.5) with the fold seed.
  On a tiny 99-class task, k-fold log_loss is an unstable, pessimistic estimator.
- **So the large CV↔LB gap means "don't trust the CV here," not "the model is bad."** The gap is
  the meta-signal working as designed: tiny on tps (trust CV), huge on leaf (don't).

## M1 decision experiment — group leakage (2026-07-02)

Synthetic grouped data (150 customers × 8 rows; label = per-customer coin flip; features = noisy
per-customer fingerprint → nothing generalizable to learn), answer key = whole customers held out.
Identical pipeline, one variable (`--fold-advisor`):

| Arm | Folds chosen | CV | Truth (graded) | CV↔truth gap |
|---|---|---|---|---|
| random-folds | random | 0.992 | 0.493 | **+0.499** |
| fold-advisor | **group by `customer_id`** (113 entities, detected by the agent) | 0.488 | 0.493 | **−0.006** |

The Strategist's own rationale (verbatim): *"A random split would place the same customer in both
training and validation sets, leading to an optimistic estimate … Grouping by 'customer_id'
ensures … a more honest evaluation of the model's ability to generalize to new customers."*

**Reading:** this is the strongest result of the project. Cleaning/FE judgment moves scores by
~±0.005 (within noise); fixing the validation design removed a **+0.499** CV lie — two orders of
magnitude more. It confirms the reframed thesis empirically: the LLM's value concentrates where
the engine is structurally blind. Caveat: synthetic, maximally adversarial construction — a real
grouped dataset (patients, repeated customers) is the natural follow-up; the mechanism, however,
is not synthetic (it is the standard silent killer of deployed models).

## M2 decision experiment — ordinal encoding (2026-07-02)

The last open FE hypothesis: ordinal *order* (`KitchenQual` Po<Fa<TA<Gd<Ex) is the one thing
trees cannot infer from unordered labels, so encoding it should INJECT information. House Prices,
baseline vs ordinal-only (no other cleaning/FE), two seeds, using the competition's
`data_description.txt`. Float codes so absent values stay missing (a first run with int codes was
biased — AutoGluon imputed nulls to rank 0 = "worst"; fixed, this is the fair run):

| Seed | Baseline rmse | Ordinal rmse | Δ (ordinal − baseline) |
|---|---|---|---|
| 42 | 26 453 | 25 940 | −513 (marginal win, within noise) |
| 7 | 25 745 | 28 324 | **+2 579 (loses badly)** |

**Verdict: ordinal encoding does NOT reliably beat AutoGluon** — mean-negative (~+1 030), huge
variance. Two structural reasons: (1) a single monotonic rank is **lossy** versus native
categorical handling (which captures non-monotonic effects and treats missingness as signal);
(2) the LLM **over-applies** ordinality to borderline-nominal columns (`LandContour`, `Utilities`),
forcing false orders. So even the FE type designed to beat the engine doesn't.

## M1 on real data — Grunfeld & MathAchieve (2026-07-02)

The synthetic result replicated on two classic, genuinely grouped datasets (public Rdatasets
mirror; truth = whole entities held out). The Strategist detected the entity column on **both**
(`firm`, `School`) with a correct rationale, unaided:

| Dataset | random-CV rmse | group-CV rmse (advisor) | Truth | Optimism removed |
|---|---|---|---|---|
| Grunfeld (200 rows, 10 firms) | 41.5 | 143.0 | 236.1 | ~52% (5.7× → 1.7× too optimistic) |
| MathAchieve (7 185 rows, 160 schools) | 6.17 | 6.33 | 6.49 | ~53% (gap −0.32 → −0.15) |

**Reading:** on real data the random-fold CV was up to **5.7× too optimistic**; the advisor cut
the lie roughly in half on both datasets. Honest caveat (Grunfeld): even group-CV understates the
error there — with only 10 firms, the truth (3 held-out firms) is itself dominated by entity
heterogeneity, which no fold scheme can fix. The mechanism and the agent's detection are no
longer synthetic-only. (This run also flushed out two real bugs: the Strategist was wrongly gated
on `use_llm`, and AutoGluon's negative-rmse CV mean was compared unsigned — both fixed.)

## M1 (time) on real data — economics (2026-07-02)

The second validation blind spot, on the classic `economics` dataset (ggplot2; 574 months of US
macro data, predict the unemployment level; truth = the last 30% of months, i.e. the future):

| Arm | Folds chosen | CV rmse | Truth (future) | Optimism |
|---|---|---|---|---|
| random-folds | random | 282 | 4 304 | **15.3× too optimistic** |
| fold-advisor | **time by `date`** (detected unaided) | 1 764 | 4 304 | 2.4× |

**Reading:** a random-fold CV interpolates between known months and reports a fantasy error; the
Strategist detected `date` from the semantics ("the task is to forecast…"), validated on the
future only, and cut a 15× lie to 2.4×. Honest caveat on the residual: the truth window contains
the 2008 financial crisis — an unprecedented regime that no validation scheme can anticipate from
pre-2001 data. Time-CV tells you extrapolation is hard; it cannot predict a crisis.

## M3 probe — the Skeptic when there is nothing to catch (2026-07-02)

Bait dataset recreating the Stellar failure (a continuous, unique-per-row measurement `flux` is
the only signal — exactly what the id-heuristic trap eats). Result: the **hardened cleaner kept
`flux` unaided** in both arms (the post-Stellar prompt fix holds), dropped only the genuine
`sample_id`; the Skeptic rated that drop low-risk and spent **zero** CV checks. Both arms: truth
accuracy 0.867. The safety net had nothing to catch and raised no false alarm — which is the
designed behaviour: cheap when the cleaner is sane, decisive (arbiter veto) when it is not.

## M6 evidence run — House Prices over 5 seeds (2026-07-03)

The one n=2 headline claim, hardened. Five seeds (42, 7, 1, 2, 3), each re-carving the answer key
and re-splitting folds; baseline and Maestra share each seed's carve, so the deltas are paired:

| Seed | Baseline rmse | Maestra rmse | Δ (improvement) |
|---|---|---|---|
| 42 | 26 453 | 25 828 | +625 |
| 7 | 25 745 | 24 343 | +1 402 |
| 1 | 31 655 | 31 278 | +377 |
| 2 | 44 222 | 40 569 | +3 653 |
| 3 | 25 640 | 25 270 | +370 |

**Verdict: Maestra — 5/5 seeds ahead, mean improvement 1 285 rmse; passes the paired 2-SEM +
majority rule (SEM 621, threshold 1 243) — narrowly, and we say so.** Note how much the carve
difficulty varies between seeds (baseline 25 640 → 44 222): exactly why the comparison must be
paired per seed, and why single-seed results were never trustworthy. The claim upgrades from
"directional (n=2)" to supported under the pipeline's own strict arbiter rule.

**Superseded 2026-07-05 by N1 (see below): under the Nadeau-Bengio-corrected accept rule, this
verdict flips to *undecided*** (corrected threshold 2 029 > mean 1 285). The "narrowly" above was
the honest tell — this is exactly the kind of narrow pass a harder, more defensible variance
estimate was expected to catch. The directional fact (5/5 seeds ahead) stands; the aggregate
statistical claim does not, and the README/STRATEGY tables are corrected accordingly.

## E1 — Strategist detection benchmark (2026-07-03, gpt-4o, v1 prompt)

17 classic datasets with known structure truth (6 grouped, 5 temporal, 6 iid incl. a deliberate
trap), acceptable-answer sets for panel data, detection from the column profile alone:

| Metric | Result |
|---|---|
| Overall acceptable | **14/17 (82%)** |
| **Group recall** | **6/6** — the flagship claim, now quantified (correct column every time) |
| Time recall | 3/5 — both misses are minimal `{time, value}` schemas; identical-schema datasets got opposite verdicts (airpassengers/ukgas OK, nottem/lynx MISS) → the judgment is *unstable at the decision boundary*, not systematically wrong |
| False alarms on iid | **1/6** — the designed trap fired: PlantGrowth's `group` column (a 3-level treatment factor) was read as an entity |

**Risk asymmetry worth stating:** a false alarm makes the CV *conservative* (pessimistic estimate)
— the safe direction; a miss produces an optimistic lie. The agent errs mostly on the safe side.

**v2 (same day): prompt hardened → re-measured → 17/17 (100%).** Group 6/6, time 5/5 (nottem and
lynx now detected), false alarms 0/6 (the PlantGrowth trap correctly read as a treatment factor).
The fixes encode general principles (few-balanced-levels ≠ entity; a numeric monotone time axis is
temporal), not dataset-specific strings. **Honest caveats:** a perfect score on the benchmark that
motivated the fixes carries Goodhart risk — generalization to unseen datasets is not yet shown —
and a single run does not prove boundary stability (v1 demonstrated instability). Both get fresh
evidence from M9 (same catalog, different models). Before/after: `detection_benchmark.jsonl`.

## M9 — model-robustness matrix, cross-provider (2026-07-03)

Same 17-dataset catalog, same v2 prompt, five backbones across two providers:

| Model | Overall | group | time | random | False alarms (iid) | Misses |
|---|---|---|---|---|---|---|
| anthropic/claude-opus-4-8 | **17/17** | 6/6 | 5/5 | 6/6 | 0/6 | — |
| anthropic/claude-sonnet-4-5 | **17/17** | 6/6 | 5/5 | 6/6 | 0/6 | — |
| anthropic/claude-haiku-4-5 | **17/17** | 6/6 | 5/5 | 6/6 | 0/6 | — |
| gpt-4o | **17/17** | 6/6 | 5/5 | 6/6 | 0/6 | — |
| gpt-4o-mini | 14/17 | **4/6** | 5/5 | 5/6 | 1/6 | emplUK, mathachieve (group→random); the PlantGrowth trap |

**Reading — two findings, one against expectation.**

*(1) The judgment is provider-robust.* All four current frontier models — across OpenAI and
Anthropic — hit a full 17/17, including the `PlantGrowth` trap (a column literally named `group`
that is a treatment factor, not an entity). So the v2 prompt's principles are not a lucky fit to
one backbone; they transfer across the provider boundary. That is a stronger portfolio claim than
"works on gpt-4o".

*(2) The boundary is "expensive vs. cheap", not "big vs. small" — and it is model-specific.* Only
gpt-4o-mini degrades, and it degrades exactly on the flagship capability (group 4/6), with misses
in the **dangerous direction** (`group→random` = a silently optimistic CV), unlike false alarms
which only cost conservatism. But **Haiku 4.5 — a small, cheap model — matches the flagships
perfectly.** The pre-run hypothesis that a toy single-sentence probe suggested (Haiku would false-
alarm) did not survive the real profile-driven benchmark: given the actual column profile, Haiku
applies the same principles as Opus. So the failure is specific to gpt-4o-mini, not a general
property of cheap models.

**Consulting takeaway:** the capability "recognize when a random CV would leak" is present in
today's frontier families and is *not* the exclusive property of the largest model — but at least
one deliberately-stripped small model (gpt-4o-mini) loses it, and loses it in the direction that
silently inflates your validation score. The honest recommendation is therefore not "buy the
biggest model" but "verify this specific judgment on your specific model — a cheap model may or
may not have it, and the failure mode is invisible without a benchmark like this one." Limits: one
run per model (temperature 0, so deterministic-ish but not variance-quantified); mini has no v1
baseline; catalog is 17 classic datasets.

## M9-extend — model robustness of the CLEANING node (2026-07-04)

M9's sibling benchmark for the other blind-spot agent: does the *drop* judgment hold across
models? Same method as E1 — profile-only, one structured call per dataset, no AutoGluon, scored
against known ground truth (`scripts/cleaning_robustness_benchmark.py`, 6 datasets: 5 controlled
synthetics + raw Rdatasets diamonds). Two directions scored separately, because their costs
differ enormously: **drop-recall** (catching genuine junk — running ids, constants, a
name-flagged leak) is a nuisance if missed; **keep-violations** (dropping a genuine feature) is
the *dangerous* direction — the exact Stellar bug (`u,g,r,i,z` bands dropped as "unique per row").
The 10 **trap** columns are high-cardinality *float* measurements (5 photometric bands, lat/lon,
diamonds x/y/z); since the deterministic `id_like` flag exempts floats, any trap drop is pure
model over-eagerness.

| Model | drop-recall | keep-violations | trap-drops (dangerous) |
|---|---|---|---|
| gpt-4o | 7/7 | 0 | **0/10** |
| gpt-4o-mini | 6/7 | 0 | **0/10** |
| claude-opus-4-8 | 7/7 | 0 | **0/10** |
| claude-sonnet-4-5 | 7/7 | 0 | **0/10** |
| claude-haiku-4-5 | 7/7 | 0 | **0/10** |

**The dangerous direction is solved across every current model: 0/10 float traps dropped.** The
failure that motivated this whole project's "validate against a baseline" philosophy no longer
reproduces on any frontier model — they respect that a high-cardinality float is a measurement,
not an id. The only recall miss is gpt-4o-mini failing to flag the leak by *name*
(`target_snapshot`) — consistent with M9 (the small model degrades on judgment), but here in the
*safe* direction (a surviving leak is caught by the separate correlation-based audit scan, M7; a
missed group in M9 was a silent CV lie).

**The benchmark also found — and closed — a real weakness in this pipeline's own heuristic.** The
first run (v1 prompt) had 2 keep-violations: gpt-4o and haiku both dropped `population` (a genuine
integer feature). Traced to the deterministic `id_like` flag, which flags any high-cardinality
non-float — so a scattered real integer (`population`, unique_frac 0.996) is flagged exactly like
a running id, and those two models deferred to it. The flag *cannot* be fixed univariately: a
sparse high-card integer is statistically indistinguishable from a sampled sparse id (e.g.
diamonds' `rownames` after subsampling) — which is precisely why it is a judgment call for the
LLM. So the fix was prompt-level (mirroring E1's v2 hardening): a general principle that
high-cardinality *integers* can be legitimate features (counts, amounts, populations, years) and
must be judged by name/meaning, with `id_like` demoted from "guidance" to "a hint to scrutinise".
**v2 result: keep-violations 2 → 0, with no regression on recall or traps** (table above is v2).
The v1→v2 loop is the E1 story again in miniature: the benchmark exposed a specific weakness, one
targeted general-principle edit fixed it, and a re-run confirmed it.

## E2 — task battery, complete (2026-07-04)

`maestra-bench --seeds 42 7 1 2 3` over a semantic spectrum; three-way paired verdict (M8). Δ is
`maestra − baseline`; for rmse lower is better (a negative Δ favours Maestra), for
balanced_accuracy higher is better (a positive Δ favours Maestra).

| Task | Semantics | Metric | Baseline | Maestra | Δ | Verdict |
|---|---|---|---|---|---|---|
| credit | rich | rmse ↓ | 72.832 | 44.284 | **−28.548 (−39%)** | **maestra** |
| wage | rich | rmse ↓ | 34.059 | 33.685 | **−0.374 (−1.1%)** | **maestra** |
| heart | rich | bal-acc ↑ | 0.788 | 0.811 | +0.023 | undecided |
| insurance | rich | rmse ↓ | 4484.1 | 4550.6 | +66.5 | undecided |
| loan-grade | rich | bal-acc ↑ | 0.221 | 0.221 | −0.001 | undecided |
| diamonds (leak-free rerun) | rich | rmse ↓ | 609.51 | 609.20 | −0.31 | undecided |
| abalone | mixed | rmse ↓ | 2.459 | 2.500 | +0.041 | undecided |
| wine-quality | mixed | rmse ↓ | 0.658 | 0.651 | −0.007 | undecided |
| wine-quality-anon | poor (anon. twin) | rmse ↓ | 0.658 | 0.666 | +0.008 | undecided |
| friedman-synth | poor (synthetic) | rmse ↓ | 1.205 | 1.203 | −0.001 | undecided |

**The pattern (10 valid tasks): 2 decided wins, 8 undecided, 0 decided losses — and the wins are
exactly where the thesis puts them.**

- **Both decided wins are rich-semantics tasks** (credit −39%, wage −1.1%), human-domain columns
  (`Student`, `Married`, `Income`; `education`, `jobclass`, `health`). The other rich tasks land
  *undecided*: AutoGluon is already strong there, and the paired test refuses to call noise a win.
- **The poor-semantics controls are as close to zero as measurement gets:** friedman Δ −0.001,
  anonymized twin Δ +0.008. Without column semantics, Maestra's judgment layer is **inert** — it
  neither helps nor hurts. That is the cleanest possible support for "the semantics are the
  mechanism": remove them and the effect vanishes, not just shrinks.
- **The anonymized-twin control worked as designed** (identical baseline 0.658 on both twins —
  semantics was the only variable; the isolated effect ~0.015 rmse in the predicted direction).

### The diamonds verdict was a harness leak — found *because* the verdict pattern flagged it

diamonds initially returned the battery's only decided-**baseline** verdict (baseline 217 vs
Maestra 602 — ~3× worse, consistent across all 5 seeds). That anomaly did not survive scrutiny:

1. Ablations cleared the suspects one by one: the cleaning plan drops only id columns; the FE
   plan is purely additive; the ratio ops guard division by zero. Cleaning-only Maestra was
   *still* 4× behind — so the gap was never caused by what Maestra *did*.
2. The cause was what the **baseline was allowed to keep**: Rdatasets CSVs carry the source
   frame's row index as `rownames`, and ggplot2's diamonds is ordered in price blocks —
   spearman(`rownames`, `price`) = **−0.40**, a target leak. Maestra's cleaning had correctly
   dropped it as id-like; the baseline exploited it.
3. **Proof:** baseline *with* the leak column: rmse 156. Baseline *without* it: rmse **681** —
   behind Maestra's honest 588 on the same seed. The entire "loss" was the leak.

The battery loader now strips `rownames` unconditionally (E1's loader always did). **The
leak-free rerun confirms the diagnosis end to end:** the baseline fell from 217 to 609.5 rmse
(its entire advantage *was* the leak), Maestra is statistically unchanged (602 → 609.2 — it never
had the leak), and the verdict is a clean *undecided* with per-seed deltas scattered around zero
(−10.4 … +11.9). Two lessons worth the space: (a) the **anomaly-shaped verdict is what exposed
the leak** — a battery that only reported means would have shipped a false conclusion; (b) this
is the project's core claim playing out *inside its own harness*: leakage silently corrupts
comparisons, and honest cleaning **looks like losing** until the leak is found.

## M11 — target framing: the predicted setup win, confirmed the hard way (2026-07-03/04)

The target-framing agent proposes a `log1p` reframing for a skewed regression target; a paired CV
on identical folds, **scored in original units** (predictions inverted before scoring, so base and
trial are comparable), is the arbiter. (Implementation verified end-to-end: each fold trains in
log space, inverts via `expm1`, rescores against the original-space truth held aside before
transforming.)

**The final evidence — 5 seeds, House Prices / `SalePrice`, plain RMSE, 3 folds each:**

| Seed | log1p proposed | Δ rmse (improvement) | Per-run 3-fold gate |
|---|---|---|---|
| 42 | yes | +3 634.6 | reject |
| 7 | yes | +3 221.5 | **accept** |
| 1 | yes | +2 079.7 | reject |
| 2 | yes | +1 214.8 | reject |
| 3 | yes | +1 212.4 | reject |

**log1p genuinely helps: 5/5 seeds improve, mean +2 273 rmse (≈ −8%).** At the seed level the
project's own paired rule is unambiguous — mean 2 273 > 2×SEM (≈1 004), improvement in 5/5 seeds.
**This is the setup win the thesis predicted**: target framing is a decision AutoGluon never makes,
and the LLM's textbook judgment (skewness 1.88, long right tail) was correct.

Three harder-won findings sit underneath that headline:

1. **An earlier single run said the opposite, and we published it too early.** The first evidence
   run (2026-07-03) returned REJECT at Δ +124 and was recorded as "the arbiter overrules the
   textbook", with a plausible invariance story attached. That run turns out to have been doubly
   compromised: its cleaning plan had been silently discarded by a tool-argument double-encoding
   bug (see the M8 receipt below), and its +124 was a 10–30× downward outlier against the
   multi-seed deltas. The correction chain is the project working as designed — the regression
   receipt caught the bug, the fix enabled a clean rerun, the flipped verdict demanded multi-seed,
   and multi-seed settled it.
2. **The per-run 3-fold gate is conservative enough to miss a real ~2 300-rmse effect** (1/5
   accepts): with n=3 folds and ±5 900 fold spread, 2 SEM is a high bar. The error is in the safe
   direction — a missed improvement, never a false adoption — but the miss rate is now measured,
   not assumed. Practical consequence: for adopt-decisions of this size, judge framing at the
   seed level (M8) or raise the fold count; the per-run gate alone under-accepts.
3. **Precision holds on both controls:** `none` on the symmetric friedman target (skew 0.048) and
   `none` on a classification target (SMS spam) — the agent fires only where the textbook says it
   should, and the false-positive direction stays clean.

(Still not worth running: M11 under RMSLE — near-tautological, since RMSLE(y, ŷ) =
RMSE(log1p y, log1p ŷ), and AutoGluon has no native RMSLE scorer.)

## M4 receipt — the refactor holds; the receipt itself caught a real bug (2026-07-04)

After the M4 intervention-core refactor (one `run_counterfactual` primitive replacing three ad-hoc
gate loops, plus the `CVBudget`), the regression evidence is two-layered:

* **Code level:** all pre-refactor gate tests pass unchanged (behaviour-neutral by test suite);
  the M10 run and the M11 multi-seed runs exercised the new path end-to-end on real data
  (`cv_budget: {limit, trials_spent}` now in every CV-path ledger record).
* **What the replication receipt actually found:** re-running the pre-M4 House-Prices framing
  command surfaced not a refactor regression but a **tool-argument double-encoding bug** —
  claude-sonnet-5 occasionally returns a schema-declared array as a JSON string (re-wrapping the
  whole arguments object), and iterating that string char-by-char silently discarded every
  planned column drop. Fixed schema-guided in `call_structured` (string-typed fields are never
  decoded; unparseable values left to the total processors). The pre-M4 reference run itself
  turned out to carry this bug, which is what had made its REJECT verdict look reproducible.

The honest net of the receipt: no M4 regression found; one real robustness bug found and fixed;
and a measured demonstration that **single-run 3-fold verdicts on a high-variance target are not
stable across runs** — the finding that forced M11 to multi-seed and settled it properly.

## M10 — free-text featurization: the FE thesis dies in its last lane (2026-07-04)

The one FE lane the negative results did not yet cover: free text, where the LLM can *read* the
column and write semantic extractors an n-gram model supposedly cannot represent. Evidence run:
UCI SMS Spam (5 574 rows, one prose column), 3-fold stratified CV, `--text-features` isolated
(`--no-fe`), every candidate judged by the paired counterfactual gate against AutoGluon's own
n-gram handling of the raw text.

| Candidate (all `source=text`) | Δcv (accuracy) | Verdict |
|---|---|---|
| informal_language_ratio | −0.0014 | drop |
| exclamation_density | +0.0005 | drop |
| question_density | −0.0014 | drop |
| currency_mention_count | −0.0002 | drop |
| time_mention_count | +0.0000 | drop |

**0/5 kept.** The candidates are exactly the semantic features a domain expert would propose for
spam — and none of them moves a 0.9864 ± 0.001 n-gram baseline beyond noise. The engine's n-grams
already carry the same signal (a currency mention *is* an n-gram; register *is* a token
distribution). With this, all three FE lanes are measured and closed: arithmetic (hybrid 0/5),
ordinal (mean-negative), and now semantic text extraction (0/5). **Feature engineering against a
strong engine is a wash even where the LLM can read prose** — the thesis' FE-null hypothesis
survives its most favourable test. (The run also exercised the M4 intervention core end-to-end
on real data: `cv_budget: {limit: null, trials_spent: 5}` in the ledger, and the framing agent
correctly declined the classification target — a third precision data point for M11.)

## K1 — the Kaggle battery, complete (2026-07-04/05; real competition data)

E2's instrument (5 seeds, paired three-way verdict) on REAL Kaggle competition data — messier
columns, competition metrics, known leaks handled up front (bike's `casual`+`registered` sum to
the target and are dropped in the loader: the diamonds lesson applied proactively).
`scripts/kaggle_battery.py`; `--make-submission` produces real leaderboard submissions with the
CV estimate attached — the CV↔LB gap on live competitions.

| Task | Semantics | Metric | Baseline | Maestra | Δ | Verdict |
|---|---|---|---|---|---|---|
| titanic | mixed (891 rows) | bal-acc ↑ | 0.814 | 0.806 | −0.007 | undecided |
| spaceship-titanic | rich | accuracy ↑ | 0.796 | 0.795 | −0.001 | undecided |
| **bike-sharing** | rich + temporal | rmse ↓ | 124.03 | **36.06** | **−87.97 (−71%)** | **maestra** |
| allstate | poor (anonymized) | mae ↓ | 1246.1 | 1255.2 | +9.1 | undecided |

**titanic corrects an early claim — the same lesson as M11, again.** The project's single-seed
result ("LLM hurts": 0.793 vs 0.732, Δ −0.061) had stood since the first benchmark runs. Over 5
seeds the per-seed deltas swing from −0.052 to +0.052 and the paired verdict is *undecided* with
a small negative mean (−0.007). The original number was a downward-outlier seed — the third time
multi-seed has overturned a single-run conclusion (after M11's REJECT and the 3-fold framing
flip). spaceship-titanic and the anonymized allstate control are both undecided, consistent with
the thesis (rich-but-noisy, and no-semantics-to-exploit, respectively).

### bike-sharing: a three-bug hunt, from a decided loss to a 3.4× win

bike-sharing first returned a **decided LOSS** for Maestra (baseline 124.0 vs maestra 143.5 rmse)
— scrutinised the same way the diamonds leak was, rather than reported as-is. Three separate,
real defects were found and fixed in sequence, each verified before moving to the next:

1. **Cleaning dropped the raw `datetime` column** as "unique per row" — correctly id-like, but
   its own stated rationale said it "can be decomposed into more useful features", and never
   acted on that: feature engineering runs *after* cleaning and never sees a column cleaning
   already dropped. No agent in the pipeline could say "keep this, someone else will use it."
   **Fix:** the cleaning prompt now distinguishes a genuine timestamp from a plain running id —
   keep it, since `date_parts` parses raw strings itself and drops the raw column afterward, so
   keeping it here has no downside.
2. **Target framing said `none`** for `count` (822 distinct values, right-skewed) with a
   self-contradicting rationale: "count is a small-range count variable". The prompt's caution
   clause for small-range counts ("a handful of distinct integer values") was being misapplied to
   a target with hundreds of values. **Fix:** the caution now explicitly contrasts "a handful
   (under ~20-30)" against a wide-range count, which should be treated as continuous.
3. **The `date_parts` vocabulary had no `hour`.** Even after fixes 1–2, Maestra still trailed the
   baseline (debug CV rmse 136.7 vs 131.8): the baseline sees the *raw* datetime string directly
   and AutoGluon's own datetime auto-feature-generation captures sub-day granularity, while our
   fixed vocabulary (year/month/weekday) had no way to express hour-of-day at all — structurally
   unavailable regardless of judgment. For hourly bike demand, the commute-peak hour is close to
   the dominant signal. **Fix:** added `hour` to `_DATE_PARTS`.

Each fix was verified cheaply (LLM-only calls, then a single debug seed) before spending a full
5-seed receipt. The `hour` fix alone took the debug seed from rmse 136.7 to **38.8** — a result
large enough that the full battery was rerun to confirm: **verdict flips from decided baseline
(Δ +19.55) to decided maestra (Δ −87.97, ≈ −71%), consistent across all 5 seeds (−86 to −90).**
This is the project's philosophy exercised on itself three times in one task: an unexpected
decided result is a bug hunt, not a headline, and the fixes it produces (a cross-agent
coordination gap, a misapplied caution clause, a missing vocabulary entry) are all now permanent,
tested improvements to every future run — not just this one dataset.

### The submissions: real leaderboard receipts

`--make-submission` produced five files, submitted to their live competitions. Each CV estimate is
replayed in the metric's *actual* units (never a naive cross-metric comparison — RMSE vs RMSLE is
a units error, not a finding) using the exact logged, deterministic cleaning/feature plan:

| Task | CV estimate (LB units) | Public LB | Gap | Direction |
|---|---|---|---|---|
| **house-prices** | 0.1307 RMSLE (OOF, log-space) | **0.12544** | +0.0053 (≈4%) | pessimistic (safe) |
| spaceship-titanic | 0.7909 accuracy | **0.79214** | −0.0012 | ≈exact |
| titanic | 0.8137 accuracy | **0.75598** | +0.058 | pessimistic (safe; small-data variance, 891 rows) |
| allstate | 1897.06 mae | **1141.97** | +755 (≈40%) | pessimistic (safe; carved 8k-row subsample vs the full real test set) |
| **bike-sharing** | 0.372 RMSLE (random folds) | **0.48758** | **−0.116** | **optimistic (dangerous)** |

**house-prices, spaceship-titanic, titanic and allstate all land safely pessimistic** — the CV
either matches closely or over-estimates the error, never under-estimates it. For comparison: the
project's synthetic/Rdatasets CV↔LB gaps ran 5.7×–15.3× when the validation *structure* was wrong
(Grunfeld, economics) and <0.001 when it was right (TPS); these real-competition gaps sit
comfortably inside that trusted range.

### bike-sharing's CV↔LB gap: the project's thesis, replayed live — and sharpened

bike-sharing's battery/submission CV used **random folds** (the library default; `--make-submission`
does not pass `--fold-advisor`) on a task whose real Kaggle split is temporal: train is days 1–19
of every month, test is days 20–end of every month — a repeating within-month future-prediction
task. Random folds are exactly the blind spot M1 exists to catch, and here it shows up on a real
leaderboard: **CV 0.372 vs LB 0.48758, a −0.116 gap in the dangerous, optimistic direction.**

The obvious next step — turn on the Strategist — is only half the story. `propose_fold_strategy`
correctly detects the structure unaided (`strategy: time, time_column: datetime`, reasoning "may
involve predicting future counts based on past data"). But a plain global time-ordered CV (3
sequential chronological folds over the full ~2-year span) does **not** close the gap — it
**overshoots the other way**: RMSLE 0.593, a +0.105 gap, now *pessimistic*. The fold scores expose
why (rmse per fold: 46 / 123 / 65) — classic expanding-window bias: with only 3 folds over two
years of strong seasonality and year-over-year growth, an early fold trains on very little data
and validates on a large, distributionally different future block (a whole season/trend shift).
The real competition's test set is a much *milder* extrapolation than that: 11 days ahead within a
month whose first 19 days the model has already seen, repeated across the whole timeline — closer
to a fine-grained, repeated local holdout than one large global future chunk.

**Neither vocabulary entry matches the competition's actual split rule.** Random folds ignore time
entirely (optimistic). A single global chronological split captures a distribution shift far
larger than what the real test set contains (pessimistic). The honest, sharper thesis this
sharpens to: *validation-design judgment is not just "detect group/time structure exists" — it is
matching the fold **granularity** to how the deployment split actually happens*, and today's
`validation_strategist.py` only distinguishes `random`/`group`/`time`, not the shape of the time
split itself. This is a well-scoped, concrete backlog item (a repeating/local time-split fold
strategy — e.g. holding out the same relative window within every period), not a same-night fix,
and it is a stronger, more specific finding than "turn on fold-advisor" would have been on its own.
**Addressed in N2 below** (`time_local`) — mechanism shipped and confirmed on a second task, though
closing the loop on bike-sharing's own raw-timestamp shape surfaced a further, still-open gap.
Note this does not affect the submitted predictions' quality — the final model is trained on the
full data regardless of which CV variant is used to *estimate* it; only the CV number's
trustworthiness for this specific task was in question, and now precisely characterised.

### A second robustness fix, found along the way

`run_multi_seed` had no per-seed error handling: allstate's battery hit a rare AutoGluon-internal
fragility (an inhomogeneous-shape error deep in AutoGluon's own ensemble code, not a maestra
defect) on one of 5 seeds, and the unhandled exception discarded all 4 good results. Fixed:
a per-seed exception is now caught, recorded in a new `failed_seeds` field (seed + error), and the
remaining seeds still produce a verdict; if every seed fails, that raises rather than reporting a
silent "undecided". Separately, spaceship-titanic's `Transported` column (native `bool`) exposed a
pandas gotcha: `.loc[rows, [False, True]]` is read as a boolean column MASK, not labels, silently
selecting the wrong shape — none of the curated E2/Rdatasets targets were ever native `bool`. Fixed
with position-based assignment, immune to the label/mask ambiguity regardless of class dtype.

## N1 — arbiter hardening: the Nadeau-Bengio variance correction (2026-07-05)

Every decided verdict in this document rests on `paired_delta_test` (validation.py):
mean paired delta beyond `sigma_mult` standard errors AND a strict majority of pairs improving.
The naive SEM (`std(deltas, ddof=1) / sqrt(n)`) treats the n paired replications — folds of the
same k-fold split, or seeds re-carving the same pool — as i.i.d. samples. They are not: adjacent
folds/seeds share overlapping training data, so the true variance is higher than the naive
estimate reports (Nadeau & Bengio 2003, "Inference for the generalization error"; no unbiased
estimator of k-fold CV variance exists — Bengio & Grandvalet 2004 — so this is a conservative
heuristic, not an unbiased fix). `paired_delta_test` and `improves_beyond_noise` now inflate the
variance by `1/n + test_train_ratio` instead of the naive `1/n`: `test_train_ratio = 1/(k-1)` for
the per-fold gate (derived from `CVResult.n_folds`, always known), and
`holdout_frac / (1 - holdout_frac)` for the per-seed (M8) verdict (derived from the answer-key
carve fraction). A new `paired_delta_mde` reports the minimum mean delta that would have cleared
the bar at a given n/spread, so an "undecided" `MultiSeedResult` is now interpretable as "no
effect at least this large" rather than a bare non-result (revision point 3). The correction only
ever raises the bar — it cannot turn a rejection into an acceptance.

**Every existing decided verdict was recomputed against the corrected rule** (script over the
committed `benchmark.jsonl`; per-seed rows matched positionally/by-seed, M6 hand-matched since it
predates the `seed` field):

| Task | Old verdict | New (NB) verdict | Mean Δ | Old threshold | New threshold |
|---|---|---|---|---|---|
| **House Prices, plain cleaning (M6, 5 seeds)** | **maestra** | **undecided** | 1 285 | 1 242 | **2 029** |
| House Prices, target framing (M11, 5 seeds) | maestra | maestra | 2 273 | 1 004 | 1 639 |
| E2 credit (5 seeds) | maestra | maestra | 28.5 | 13.6 | 22.2 |
| E2 wage (5 seeds) | maestra | maestra | 0.374 | 0.149 | 0.244 |
| E2 diamonds, leak-affected (5 seeds, superseded) | baseline | baseline | −385 | 40.5 | 66.1 |
| E2 diamonds, leak-free rerun (5 seeds) | undecided | undecided | 0.31 | 10.3 | 16.9 |
| K1 bike-sharing, final (5 seeds) | maestra | maestra | 88.0 | 1.40 | 2.29 |
| K1 bike-sharing, pre-fix runs (5 seeds, superseded) | baseline | baseline | −8.6 to −19.6 | 0.57–0.62 | 0.94–1.01 |
| All other E2/K1 tasks (7: heart, insurance, loan-grade, abalone, wine-quality ×2, friedman-synth, titanic, spaceship-titanic, allstate) | undecided | undecided | — | — | — |

**One real flip, and it is the honest headline: M6 (House Prices, plain LLM cleaning/encoding,
no target framing) goes from a narrow "maestra" pass to undecided.** It was the one verdict in
the whole ledger that passed the old rule "narrowly" (mean 1 285 vs. threshold 1 242, a 3.5%
margin) — exactly the kind of narrow pass a harder, more defensible variance estimate exists to
catch. The directional fact stands (5/5 seeds ahead) and is kept in RESULTS/README as such; the
statistical "decided win" claim does not survive and is marked superseded above, not deleted.
**Every other currently-live decided claim holds comfortably** under the corrected rule — target
framing (M11), credit and wage (E2), and bike-sharing (K1) all clear the harder bar with room to
spare, so the thesis's substantive claims (setup wins, semantics-gated cleaning wins, FE null)
are unaffected. README and STRATEGY.md updated to match; `docs/RESULTS.md`'s M6 section carries
the superseded note in place rather than rewriting history.

**Deferred, not fixed:** the docstring's other admitted gap — no correction for greedy multiple
testing across a candidate sequence (the `--hybrid`/`--text-features` gates test several
candidates against the same base CV in turn) — is left open. Those FE lanes are measured-null and
frozen (no further development), so the engineering cost of a proper alpha-spending correction is
not worth spending there; it would matter again if run memory (N4) or a wider candidate pool ever
reopens that lane.

## N2 — the fold-granularity fix, and the integration gap it surfaced (2026-07-05)

The backlog item from K1's bike-sharing gap: `--fold-advisor`'s vocabulary gains a fourth
strategy, `time_local` (`validation_strategist.py`, `validation.py::_time_local_folds`) — blocked,
within-period folds (fold i trains on early time-blocks and validates on the next block, WITHIN
every period, unioned across all periods) for a deployment split that repeats locally rather than
cutting the timeline once globally. Fully offline-tested (fold construction, verify branch,
pipeline wiring — 7 new tests, `test_validation_strategist.py`).

**Gate: a second repeating-period task, to confirm the mechanism isn't a one-off** — met, cleanly:

`scripts/time_local_experiment.py` (synthetic, no LLM calls — this isolates the FOLD CONSTRUCTION
mechanism from the Strategist's separately-tested detection capability, M1/E1/M9). A monthly-demand
series with a strong across-period trend (mirroring bike-sharing's year-over-year growth) plus a
smooth within-period ramp; truth = the real local deployment shape (last third of every period
from the first two-thirds, repeated across all 10 periods):

| Arm | CV rmse | Truth rmse | Gap | Direction |
|---|---|---|---|---|
| random folds | 5.008 | 5.631 | −0.623 | optimistic (dangerous) |
| global time split | 10.385 | 5.631 | +4.755 | pessimistic (overshoot) |
| **time_local** | 6.142 | 5.631 | **+0.511** | pessimistic (mild) |

**time_local cuts the global-split gap by >9× (4.755 → 0.511)** and lands on the same (safe) side
as bike-sharing's own result — the mechanism is confirmed on independent data, not a fit to one
dataset's noise.

**Rerunning bike-sharing itself did NOT close cleanly — and the reason is itself a finding.**
`scripts/kaggle_battery.py --make-submission bike-sharing --fold-advisor` logged:
`FOLDS time-ordered by 'datetime' -- ...`, i.e. the Strategist proposed plain `time`, not
`time_local`. This is not a prompt failure: `propose_fold_strategy` runs on the RAW column
profile, *before* cleaning/feature engineering. Bike-sharing's only period-shaped signal is a raw
`datetime` string; the month it would need to name as `period_column` does not exist as a column
at that point (`season` exists but is far too coarse — 4 values spanning 2 years). The Strategist
correctly refused to name a column that isn't there yet (its own instruction: "only name columns
that exist in the profile") and fell back to `time`. **This is the same "decided before
decomposed" timing gap the original bike-sharing bug hunt already found once** (cleaning dropped
the raw `datetime` column before FE could decompose it) — now recurring one layer up, at fold-
strategy time instead of cleaning time. (This run also flushed out a real harness gap:
`maestra-bench`'s `run_task`/`run_multi_seed` and `kaggle_battery.py`'s `main()`/`make_submission`
never threaded `fold_advisor` through at all — the K1 battery/submission runs never exercised the
Strategist on either arm. Fixed: `fold_advisor` is now a first-class parameter on `run_task`,
`maestra-bench --fold-advisor`, and `kaggle_battery.py --fold-advisor`.)

**Honest scope: shipped vs. open.** `time_local` is real, tested, and production-ready for any
task whose period is already a materialized column (a patient-visit index, a pre-existing
month/week field) — not a hypothetical. Closing the loop for raw-timestamp tasks like
bike-sharing needs one more piece, not attempted here: surfacing derived period candidates
(month-of/week-of a timestamp) during *profiling*, so the Strategist has something to name before
cleaning/FE ever runs. Scoped follow-up, gated the same way as this one: confirm on a second task
before calling it done.

## K2 — the structure battery, 8 real Kaggle tasks (2026-07-05/06)

E2/K1's instrument (5 seeds, paired three-way verdict) extended from K1's 5 tasks to 13, adding
8 real Kaggle competitions chosen for temporal/group structure — store-sales, rossmann, walmart
(all Store-repeating retail forecasting), ieee-fraud (time + wide anonymized), santander-
transaction (a third anonymized control), restaurant-revenue (137 rows, a small-n stress test),
airbnb (rich semantics + weak time), two-sigma-rental (manager_id/building_id repeat). All 8
completed 5/5 seeds, `failed_seeds: []` on every task — no crashes.

| Task | Semantics | Metric | Baseline | Maestra | Δ | MDE | Verdict |
|---|---|---|---|---|---|---|---|
| rossmann | rich+time+group | rmse ↓ | 2395.02 | **2322.66** | **−72.36** | 31.20 | **maestra** |
| walmart | rich+time+group | rmse ↓ | 11070.98 | **8447.11** | **−2623.87** | 2500.34 | **maestra** |
| store-sales | rich+time+group | rmse ↓ | 438.79 | 441.91 | +3.12 | 8.44 | undecided |
| ieee-fraud | poor+time | bal-acc ↑ | 0.6279 | 0.6287 | +0.0009 | 0.0434 | undecided |
| santander-transaction | poor | bal-acc ↑ | 0.5623 | 0.5634 | +0.0011 | 0.0279 | undecided |
| restaurant-revenue | rich+time (n=137) | rmse ↓ | 2 403 116 | 2 457 062 | +53 946 | 504 874 | undecided |
| airbnb | rich+time | bal-acc ↑ | 0.1684 | 0.1653 | −0.0030 | 0.0073 | undecided |
| two-sigma-rental | rich+group | bal-acc ↑ | 0.4593 | 0.4700 | +0.0107 | 0.0364 | undecided |

**The pattern (8 tasks): 2 decided wins, 6 undecided, 0 decided losses** — both wins
(rossmann, walmart) are 5/5-seeds-consistent retail-forecasting tasks; every other task lands
honestly undecided, including the anonymized control and the tiny-n task, exactly as the
thesis predicts. No decided loss anywhere in the battery.

**santander-transaction confirms as inert, as predicted.** A third anonymized control (after
friedman-synth/E2 and allstate/K1): Δ +0.0011, an order of magnitude below its own MDE
(0.0279) — no effect, matching the mechanism-not-correlate thesis (E2) on modern Kaggle data.

**restaurant-revenue confirms as underpowered, as predicted.** 137 rows, per-seed deltas swing
from −309 306 to +615 910 — the MDE (504 874) is nearly 10× the mean delta; this is exactly what
"undecided" should mean on a dataset this small, not a null result about the LLM's judgment.

### The anomaly this battery was designed to catch: the advisor fires, but not always as expected

The plan's stated purpose for rossmann/walmart/store-sales/two-sigma-rental was "the first real
test of `--fold-advisor` on EXISTING group columns" (Store, manager_id) — unlike bike-sharing,
these datasets carry a genuine repeating entity column already, so N2's raw-timestamp gap should
not apply. **`benchmark.jsonl` does not log which fold strategy was chosen per run — this is
itself a gap** (`BenchResult`/`MultiSeedResult` carry no `fold_strategy` field, and
`kaggle_battery.py`'s battery loop never printed or persisted it either). To check what actually
fired, `propose_fold_strategy`/`validate_fold_strategy` were re-run directly against the exact
materialized, cached datasets (`data/kbattery_<task>.csv`, the same files the battery trained on;
gpt-4o, same as the battery's default model) — the Strategist's proposal depends only on the
static column profile, not the per-seed row sample, so this reproduces the decision the battery
itself would have made without re-spending the full CV:

| Task | Proposed | Verified | Note |
|---|---|---|---|
| store-sales | `time` | `time` on `date` | NOT `group` on `store_nbr`, despite it repeating 15000/54 ≈ 278× |
| rossmann | `time` | `time` on `Date` | NOT `group` on `Store` (1115 stores, repeats densely) |
| walmart | `time` | `time` on `Date` | NOT `group` on `Store`/`Dept` |
| ieee-fraud | `time` | `time` on `TransactionDT` | expected — no group_column intended in the catalog design |
| santander-transaction | `random` | `random` | correct — no structure exists, matches the control's design |
| restaurant-revenue | `random` | `random` | correct — no repeat structure on 137 rows |
| airbnb | `time` | `time` on `date_account_created` | matches the "weak time" catalog description |
| two-sigma-rental | `group` | `group` on `building_id` (4079 entities) | **the only group pick** — chose `building_id`, not the `manager_id` anticipated in the catalog note; also correctly flagged `created` as a leakage warning (advisory, not acted on) |

**The finding: `--fold-advisor` fired on every task (not a silent no-op — the K1-era bug this
battery was explicitly built to re-check for did not recur), but it chose `group` only once out
of four tasks that carry a real, dense repeating entity.** On store-sales/rossmann/walmart, a
`Date` column sits next to an equally valid `Store`/`store_nbr` group column, and the Strategist
picked time every time. This is a genuinely untested edge case: M1's synthetic/real-data
experiments and E1's 17-dataset detection benchmark each tested group-only or time-only
structure in isolation — none tested a column profile carrying BOTH a real group axis and a real
time axis at once. The current `FOLD_STRATEGY_SCHEMA` is a single enum choice (no combined
"group+time" / panel-CV strategy exists), so even a correct read of "both exist" cannot express
the textbook-correct choice for repeated-panel forecasting data (group folds, or a nested
group-then-time split). **Not fixed here, not silently worked around — flagged as a precise
open question**: does the prompt need an explicit priority rule for competing structures, or
does the vocabulary need a fifth strategy for panel data? Either answer needs its own gated
experiment, not a same-session guess.

**Does the `time`-over-`group` choice explain rossmann/walmart's wins?** Unknown, and not
resolved here — the wins could come from cleaning/encoding judgment layered on top of a
time-ordered (still non-random, still leakage-conscious) CV, independent of whether group would
have scored differently. Answering this needs a controlled rerun with the fold strategy forced
to `group` on the same data, which is future work, not part of this battery.

### K2 submissions — real leaderboard receipts (2026-07-06)

Two SEPARATE claims, do not conflate them: the **verdict table above** is the battery
(battery presets: `--time-limit 60 --cv 3`, per-seed N1 rule, answer key carved from train).
The table below is the **real-competition submissions**, a different pipeline configuration —
AutoGluon `best_quality` (multi-layer stacking + bagging, `--time-limit 900 --cv 2`), each task's
own real metric, and the competition's own test set graded on the public/private leaderboard.
CV estimate = the submission run's logged CV in the LB metric; gap = `public LB − CV` in the
metric's direction. LB scores are Helena's, read from Kaggle 2026-07-06 (verbatim).

| Task | LB metric | Preset | CV estimate | Public LB | Private LB | Gap (LB−CV) | Direction |
|---|---|---|---|---|---|---|---|
| rossmann | RMSPE ↓ | best_quality | 0.3586 | **0.20860** | 0.22002 | **−0.150** | pessimistic (safe) |
| santander-transaction | AUC ↑ | best_quality | 0.8928 | **0.89657** | 0.89436 | **+0.0038** | pessimistic (safe, ≈exact) |
| walmart | WMAE ↓ | best_quality | 1441.6 | **2958.56** | 3057.83 | **+1516.9** | **optimistic (dangerous)** |
| restaurant-revenue\* | RMSE ↓ | standard (NOT bq) | 2 627 861 | **1 869 510** | 1 891 848 | −758 350 | pessimistic (safe) |
| store-sales | RMSLE ↓ | best_quality | — (see note) | **0.79002** | pending | — | not comparable |
| airbnb | NDCG@5 ↑ | best_quality | — (see note) | **0.85354** | 0.85671 | — | not comparable |
| ieee-fraud | AUC ↑ | high_quality, 50k\*\* | 0.8965 | **0.91427** | 0.89402 | **+0.0178** | pessimistic (safe) |
| two-sigma-rental | multi-class log loss ↓ | — | — | — | — | — | no submission (now wired, below) |

\* restaurant-revenue is the **standard** (non-best_quality) run from the first submission pass
(its own printed CV −2 627 861), and it was submitted **twice** — so it is NOT a like-for-like
best_quality receipt; kept here for completeness, labelled as such.
\*\* ieee-fraud's best_quality run OOM'd (393 kept columns × 200k rows × bagging on ~10 GB); made
memory-safe (`submit_sample` 200k→50k, `high_quality`) and re-run 2026-07-06 — completed, graded,
row above.

**Four clean, comparable receipts, and one of them is the dangerous kind.**
- **rossmann (RMSPE), santander (AUC) and ieee-fraud (AUC) land safely pessimistic** — the CV
  over-estimated the error / under-estimated the score, the safe direction, comfortably inside the
  project's trusted CV↔LB range (cf. K1: house-prices +4%, spaceship ≈exact). ieee-fraud's
  +0.0178 AUC is the receipt that the kept 393-column table (V-block for signal, LLM off) plus a
  memory-safe `high_quality` still produces a trustworthy CV.
- **walmart is the one optimistic, dangerous gap: CV WMAE 1441.6 vs LB 2958.6, ≈2× too
  optimistic.** The most likely cause is the same blind spot the whole project exists to catch:
  `--make-submission` does NOT pass `--fold-advisor`, so the submission CV used **random folds**
  on a temporal retail-forecast task, interpolating within the training window while the real
  test set is a future block. This is bike-sharing's K1 lesson replayed on a second temporal
  competition — a live, real-leaderboard instance of "a random CV lies optimistically on time
  data". (Not fixed here: turning on the Strategist for submissions is a scoped follow-up.)
- **store-sales and airbnb have no computable gap — a metric-units mismatch, left empty, not
  guessed.** store-sales' CV is logged in RMSE of original sales units (~310, framing scores in
  original space) while the LB is RMSLE (0.79) — not comparable without a log-space OOF replay
  (the K1 house-prices manoeuvre; not run here). airbnb's CV is top-1 accuracy (0.873) while the
  LB is NDCG@5 (0.854) — different metrics, different scales; the CV never scored the ranked top-5
  the submission actually emits.

### two-sigma-rental produced no submission — diagnosed and wired (not run)

`--make-submission two-sigma-rental` wrote no file because the catalog spec had **`test_path:
None`** — deliberately battery-only last session — so `make_submission` returned early with "not
submittable (no test_path wired)". Not a crash. Making it submittable needed three pieces, all
now in place (the expensive submission run itself is Helena's call, deliberately NOT started):
1. the competition test ships as `test.json` with the same list-valued `features`/`photos`
   columns as train — flattened once to `data/kaggle_twosigma/test_flat.csv` (74 659 rows);
2. a **3-class probability** submission (`listing_id,high,medium,low`, the sample's exact column
   order) — a new `submit_proba_columns` spec field routes through the existing, already-tested
   multiclass-wide proba path (`test_multiclass_proba_submission_per_class_in_order`);
3. `test_path` + `submit_id="listing_id"` + `submit_eval_metric="log_loss"` wired in the spec.
Offline-verified (catalog loads, columns resolve to `[high, medium, low]`, suite green); the
real submission awaits Helena's go.

### K2 anomalies & open questions (for Helena, not silently resolved)

1. **The battery's actual fold strategy is unverifiable from the record.** `benchmark.jsonl`
   carries no `fold_strategy` field (schema checked: `BenchResult`/`MultiSeedResult` have none),
   and the battery loop neither prints nor persists it. The verdict-table §'s proposal re-check
   shows what the Strategist *proposes* on the cached data, but whether the K2 **battery run
   itself passed `--fold-advisor`** (vs. plain random folds) cannot be confirmed from records or
   any surviving log. **Resolved forward (2026-07-06, Follow-up A):** `BenchResult`/
   `MultiSeedResult` now carry a `fold_strategy` field, logged in every record — so this is never
   a post-hoc reconstruction again. K2 itself stays honestly unverifiable (the field is
   forward-only; the existing records are not back-filled).
2. **walmart's submission CV↔LB gap is optimistic/dangerous (≈2×).** Almost certainly random-fold
   submission CV on temporal data (`--make-submission` didn't pass `--fold-advisor`) — the K1
   bike-sharing lesson on a second competition. **Addressed forward (2026-07-06, Follow-up B):**
   `--make-submission` now defaults the Validation Strategist ON (tri-state `--fold-advisor`), so
   a future submission CV reflects the deployment split. Still open — the confirming measurement:
   re-run walmart with the advisor and check the ≈2× gap closes (a best_quality run; Helena's call).
3. **store-sales & airbnb have no CV↔LB gap** — CV logged in a different metric/units than the LB
   (rmse-original vs RMSLE; top-1 accuracy vs NDCG@5). Left empty per instruction. Open: replay
   store-sales OOF in log space (K1 house-prices method) and score an NDCG@5 CV for airbnb, if a
   real gap number is wanted.
4. **ieee-fraud — resolved.** best_quality OOM'd; the memory-safe re-run (`high_quality`, 50k
   rows, all 393 columns kept, LLM off) completed and graded: CV AUC 0.8965 vs LB 0.91427,
   **+0.0178 pessimistic (safe)** — row in the submission table above.
5. **restaurant-revenue** is a standard (non-best_quality) run and was submitted twice — not a
   clean best_quality receipt; kept labelled.
6. **The group-vs-time competing-structure edge case** (verdict-table § above) stands: on the 3
   Store-repeating tasks the Strategist proposed `time`, not `group`, and whether that changed the
   wins is unresolved — a controlled forced-`group` rerun is future work.

## P2b generalprobe — the MCP tools' own thesis, replayed live (2026-07-07)

The P2b demo rehearsal (`docs/examples/demo/SCRIPT.md`) ran all three MCP tools for real
(`gpt-4o`, real AutoGluon, no mocks) against `docs/examples/demo/demand.csv` — Kaggle Bike Sharing
Demand's raw training data, kept deliberately unclean so the tools find its known issues live
rather than on pre-scrubbed input.

`audit_csv`: risk_level `high` — `time` fold strategy (datetime-ordered rows) + a deterministic
leak (`registered`, |corr| 0.971 with `count`; `count = casual + registered`, both absent from the
real competition test set).

`check_validation`: naive random-split CV `root_mean_squared_error` ≈ **5.29**; recommended
time-ordered CV ≈ **20.27** — **optimism_gap = 14.98, "optimistic (dangerous)"**. The project's
own CV↔LB-gap thesis (K1/K2), replayed live through the MCP surface rather than the CLI/battery
path, on the same dataset K1 already used.

`feasibility`: achievable quality (time-ordered CV, leak-cleaned by the LLM's own cleaning plan)
`root_mean_squared_error` ≈ **77.6 (± 31.1)**; strongest drivers `datetime_hour` (178.1),
`datetime_year` (54.2), `workingday` (51.9).

**Two real bugs found and fixed by this run** (`p2-mcp-server` branch, fixup commit
2026-07-07, ahead of `docs/MCP.md`'s writing): `feasibility`'s feature-importance call was fed the
raw input schema and crashed with a `KeyError` on AutoGluon's own derived columns (fixed:
`feature_stage="transformed"`, no external data needed — scores the model's post-processed
features against its own internal validation split); `check_validation`'s and `feasibility`'s
wall-clock backstops had almost no headroom over their nominal AutoGluon time
(`check_validation` hit its original 90s backstop for real on this file's ~10.9k rows) — both
loosened (150s/360s, less nominal time per fold) using the measured numbers above as the
calibration basis.

## Where LLM judgment pays off — the whole map

The systematic answer to the project's question, across every layer a conductor could touch:

| Layer | Does LLM judgment beat the AutoGluon baseline? | Evidence |
|---|---|---|
| **Setup / validation** (fold strategy, leakage) | **Yes — decisively, with one open edge case** | M1: removed a **+0.499** CV lie (synthetic); real data: cut a **5.7×** (Grunfeld, group) and a **15.3×** (economics, time) optimism roughly in half or better; detection 17/17 with 0 false alarms, provider-robust (M9). **Open (K2):** M1/E1's benchmark tested group-only or time-only structure in isolation; on real data carrying BOTH (rossmann/walmart/store-sales' Store + Date), the Strategist picked `time` over `group` in 3/4 cases — an untested competing-structure edge case, not yet resolved |
| Setup / target framing (M11) | **Yes — the predicted setup win** | House Prices, 5 seeds: log1p improves rmse in **5/5** (mean +2 273, ≈ −8%), seed-level paired test unambiguous; agent correctly silent on symmetric and classification targets. Caveat: the per-run 3-fold gate under-accepts (1/5) — conservative in the safe direction |
| Cleaning / encoding | Yes, modestly — on semantic-rich data, and **only** there; and **provider-robust** | House Prices: ahead 5/5 seeds (+1 285 rmse) but **undecided** under the N1-corrected variance rule (was a narrow "decided" pass pre-correction); E2 battery (10 tasks): 2 decided wins, both rich-semantics (credit −39%, wage −1.1%, both hold under N1), 8 undecided, 0 decided losses; poor-semantics controls **inert** (Δ −0.001 / +0.008); drop judgment 0/10 dangerous trap-drops across 5 models (M9-extend); real Kaggle data (K1): 1 decided win (bike-sharing, holds under N1), 3 undecided, 0 decided losses; **K2 (8 more real competitions): 2 decided wins (rossmann, walmart — both retail forecasting), 6 undecided, 0 decided losses**, including a third anonymized control confirmed inert (santander-transaction) and a 137-row task confirmed underpowered (restaurant-revenue) — but the group-column detection anomaly (below) means the wins' mechanism isn't fully attributed |
| **Feature engineering** (arithmetic, ordinal *and* free-text) | **No — across the board, all three lanes** | hybrid kept 0/5; ordinal mean-negative; text extractors 0/5 vs the engine's own n-grams (M10) |
| Temporal decomposition (`date_parts`) — a distinct mechanism from arithmetic FE | **Yes, decisively, once the vocabulary could express it** | K1 bike-sharing: **rmse 124.0 → 36.1 (−71%)**, driven by keeping+decomposing a raw datetime column (year/month/weekday/**hour**). Not "clever domain feature invented from numerics" (the null FE story) — the engine cannot parse a raw timestamp string into hour-of-day itself; this is closer to a structural setup capability than to CAAFE-style FE |

**The publishable conclusion:** the feature-engineering layer — where most LLM-for-AutoML work
concentrates (CAAFE, MALMAS, LLM-FE) — is a wash against a strong engine. The LLM's value is
real but concentrated where the engine is structurally **blind**: validation design and setup.
E2 sharpens the cleaning row into a causal claim: the anonymized-twin control and the synthetic
control show the effect **vanishes** (not shrinks) when column semantics are removed — semantics
is the mechanism, and the arbiter is what converts that mechanism into only-upside (2 wins, 0
losses, noise refused).

## P3 — the arbiter as a generic DS tool, and two spikes (2026-07-07)

`compare()` (`src/maestra/compare.py`), the P3 public API, was verified end to end on real
sklearn estimators (no LLM, no AutoGluon): `compare(Lasso(alpha=10), LinearRegression(), df, "y")`
on a clean linear synthetic signal correctly calls `LinearRegression` "improved" (mean paired
delta +0.13 R², `n_folds=4`) — the same `paired_delta_test`/Nadeau-Bengio machinery every
internal gate uses (N1), now reachable without training a single AutoGluon model.

**AutoGluon-as-optional-dependency investigation.** Verified via grep: exactly two files in
`src/maestra` import `autogluon.tabular` at module level (`engine.py`, `validation.py`) — both
now import it LAZILY, inside the 3 functions that actually construct a `TabularPredictor`.
Verified live: `sys.modules["autogluon"] = None; import maestra; from maestra import compare` and
a real `compare()` call both succeed with AutoGluon completely absent. The `pyproject.toml`
DEPENDENCY itself was deliberately NOT made optional — that would mean `pip install -e .` no
longer yields an immediately-runnable `maestra`/`maestra-audit`/`-bench`/`-mlebench`/`-mcp` CLI
without an extra install step, breaking the project's existing zero-friction install contract
(a Stopp-Trigger case, not mine to decide unilaterally). The Colab notebook
(`docs/examples/compare_quickstart.ipynb`) works around this with `pip install --no-deps`
instead, which needs no packaging change.

**TabPFN spike — aborted, setup problem.** `pip install tabpfn` installs cleanly (8.0.8), but
`TabPFNClassifier.fit()` requires a ONE-TIME interactive license acceptance (browser login at
priorlabs.ai + a personal `TABPFN_TOKEN` API key) before it will download model weights — not
obtainable non-interactively. No `TabPFNEngine` was built; this is exactly the "setup problem,
note the finding, don't force it" case the task explicitly allows for.

**A real, environment-only finding, not a code defect:** running the full 293-test offline
suite as a single `pytest` process intermittently stalled partway through (reproducibly at
`test_pipeline.py::test_cv_path_submission_also_clips_negative_predictions`, 0% CPU, no
progress for minutes) on this specific long-lived local dev machine — traced to `AutogluonModels/`
having grown to **91 GB** over the session's many real AutoGluon runs (disk was at 97%,
15 GB free). Cleared (it is gitignored, purely regenerable model cache; nothing tracked was
lost). The suite passes cleanly and quickly in every other configuration checked: every
individual test file alone, and the full 293 tests split into two ~150-test halves as separate
processes (151 + 142 passed, ~9s each) — strong evidence the code is correct and this was pure
single-process resource accumulation on a heavily-used machine, not a bug introduced by P3.

## P4 — README reframe, case study, architecture writeup, and one open ledger gap (2026-07-07)

`docs/case_studies/bike_sharing.md` and `docs/ARCHITECTURE.md` were written; the README was
reordered around the agreed "what is Maestra" opening, a 10-minute path, and a market-vocabulary
mapping. A subagent then cross-checked every numeric claim in README.md and the new case study
against this ledger: every graded-experiment number (M1, M2, M6, M9, M9-extend, M10, M11, E1, E2,
K1, N1, N2) matches exactly.

**One real, pre-existing discrepancy surfaced — not introduced this session, not silently fixed.**
The README's existing "Case study: Maestra caught its own mistake" section (Playground S6E6, the
dropped-photometric-bands leak) states the baseline comparison exposed damage "0.955 → 0.919".
This ledger's only S6E6 entry (Kaggle submissions, above) records a different number pair: public
`0.95045` ≈ holdout `0.9516` — the final, fixed submission's score, not the before/after
comparison the README describes. The 0.955/0.919 pair has no traceable source here (it survives only
as a narrative note in the now-untracked `CHANGELOG.md`; `runs.jsonl` has no Stellar/S6E6 run, so it
is not reproducible from what the repo holds). **Resolved 2026-07-07:** the unbacked pair was removed
from the README; the case study keeps its qualitative point ("only the baseline comparison exposed the
damage") and the ledger-backed submission score (`0.95045` public ≈ `0.9516` holdout). Restoring a
real number would require Cody to re-run the with/without-photometric-bands experiment on S6E6 and add
a proper ledger line — optional strengthening, not required for the claim to be honest. The Titanic CLI-transcript example elsewhere in the README
(`accuracy: 0.826`, etc.) is an illustrative single-command output, not a dated/seeded claim, and
is not treated as a ledger discrepancy.

## F1 — backtest audit, real forecasting competitions (2026-07-07)

`scripts/backtest_audit_battery.py` ran `backtest_audit.py` (`gpt-4o`, real AutoGluon, no mocks)
against three real Kaggle forecasting competitions, reusing K1/K2's already-local data: Rossmann,
Walmart, and store-sales (Favorita), each a random 15 000-row sample (matching
`kaggle_battery.py`'s own sampling convention), 3 rolling origins, 10s per fit.

**A real backtest lie, found and quantified — Rossmann's `Customers` column.** The LLM flagged
`Customers` as a future-leaking feature ("only known after the sales period has ended, as it
reflects actual customer visits"); the deterministic correlation check corroborated it at
**|corr| = 0.892** with the target `Sales`. Confirmed structurally: `Customers` is **absent from
Rossmann's real `test.csv`** — the same definitive absent-from-test-set evidence standard this
project already uses for bike-sharing's `casual`/`registered` leak, not merely a correlation
guess. This is F1's Done condition met: a naive setup that kept `Customers` as a feature would
train on information genuinely unavailable at real forecast time. (The battery script drops
`Customers` up front for its main 3-task run, mirroring the bike-sharing precedent — the
above detection ran as a deliberate SEPARATE check with `Customers` still present, specifically
to verify the future-feature check catches a real, known leak rather than only synthetic ones.)
Report: [`docs/examples/reports/rossmann-backtest-audit.html`](examples/reports/rossmann-backtest-audit.html).

**Split-design check (naive vs. embargoed backtest): no measurable gap on any of the 3 tasks** at
this budget — Rossmann mean_gap −2.27 (MDE 159), Walmart +13.05 (MDE 1126), store-sales +52.38
(MDE 154), all "undecided" (gap far below its own MDE, i.e. genuinely underpowered at 3 origins /
10s fits / 15k-row samples, not evidence of "no lie"). Open, not claimed as a finding either way —
more origins and/or a larger sample would sharpen this.

**Series-leak check: near-perfect adversarial AUC (~1.0) on all 3 tasks — a real result, but with
an important, honestly-flagged confound.** `series_leak_check` splits the data at the naive time
boundary and asks whether a classifier can tell "before" from "after" with the series column
removed. All three real, genuinely-trending retail time series scored ~1.0 AUC. This is expected
for ANY real time series with a strong trend/seasonality — ordinary time-correlated signal (not
series-identity leakage specifically) is enough for a classifier to separate before/after
perfectly, since the check does not currently control for plain temporal drift. **Open
methodological question for a future iteration:** the check as designed cannot currently
distinguish "this global model can exploit series identity across the boundary" from "this is
just a normal time series with a trend" — it needs a control (e.g. compare against the AUC of a
model trained on RANDOMLY shuffled series-assignment across the same time split) before a high
AUC here can be read as a series-specific warning rather than an expected property of temporal
data. Not silently resolved — noted here for F2/a future backtest_audit revision.

**Follow-up (2026-07-07, PR#6 review): the series-leak AUC no longer drives the verdict.** The
ledger above was honest about the confound, but the shipped artifact was not: `risk_level`
escalated to **"high"** on `series_leak_auc > 0.75`, so the HTML/MCP verdict a non-DS sees would
have labelled every one of these ordinary trending retail series as a high-risk series leak — a
false alarm baked into the product, contradicting this very ledger and the "verdicts, not
build-buttons" invariant. Fixed: `series_leak_auc` is now **diagnostic only** and does NOT feed
`risk_level` at all (neither "high" nor "elevated"). The number is still computed and shown in the
report, but under an explicit caveat ("does not yet control for ordinary time trend; a high value
is expected for any trending series and is not evidence of series leakage — real separation needs
the shuffled-series control, planned for F2"). `series_leak_check`'s docstring was corrected to say
the same (the old "beyond just time itself" claim, which the code never delivered, is gone). The
committed `rossmann-backtest-audit.html` was re-rendered from the same measured values through the
corrected renderer — its RED verdict still rests on the `Customers` future-feature finding (the
real leak, unchanged), and the series-boundary section is now a caveated diagnostic rather than a
"STRONG shift / leaking series identity" verdict block. The full shuffled-series-assignment
control that would let a high AUC actually mean something remains F2 scope, not built here.

**Target framing: store-sales' skewed `sales` target correctly flagged** (`log1p`, skewness
7.60, mean 364.5 vs. median 10.0) — consistent with M11/K1's established retail-sales pattern
(house-prices, bike-sharing). Rossmann's `Sales` target was correctly NOT flagged (skewness 0.60,
mean/median close) — the framing check is not indiscriminately proposing log1p on every
regression target.

## Recurring pattern

On 2 of 3 graded comparisons (leaf, titanic) the LLM cleaning/FE **underperformed** plain
AutoGluon; on tps it helped slightly. The baseline comparison is the point — it keeps us honest.
Raw records (every run, all fields): `runs.jsonl` (`kind: mlebench`) and `benchmark.jsonl`.
