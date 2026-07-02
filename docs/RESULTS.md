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
| Titanic | balanced_accuracy | **0.7933** | 0.7322 | Baseline (LLM hurts — expected negative; what the harness is for) |
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

## Where LLM judgment pays off — the whole map

The systematic answer to the project's question, across all three layers a conductor could touch:

| Layer | Does LLM judgment beat the AutoGluon baseline? | Evidence |
|---|---|---|
| **Setup / validation** (fold strategy, leakage) | **Yes — decisively** | M1: removed a **+0.499** CV lie (synthetic); real data: cut a **5.7×** (Grunfeld, group) and a **15.3×** (economics, time) optimism roughly in half or better |
| Cleaning / encoding | Marginally, and only on semantic-rich data | House Prices plain −2.4% (seed 42), within noise elsewhere |
| **Feature engineering** (arithmetic *and* ordinal) | **No — across the board** | hybrid kept 0/5; ordinal mean-negative |

**The publishable conclusion:** the feature-engineering layer — where most LLM-for-AutoML work
concentrates (CAAFE, MALMAS, LLM-FE) — is a wash against a strong engine. The LLM's value is
real but concentrated where the engine is structurally **blind**: validation design and setup.

## Recurring pattern

On 2 of 3 graded comparisons (leaf, titanic) the LLM cleaning/FE **underperformed** plain
AutoGluon; on tps it helped slightly. The baseline comparison is the point — it keeps us honest.
Raw records (every run, all fields): `runs.jsonl` (`kind: mlebench`) and `benchmark.jsonl`.
