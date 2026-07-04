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

## M11 — target framing agent: the arbiter overrules the textbook (2026-07-03)

The target-framing agent proposes a `log1p` reframing for a skewed regression target; a paired CV
on identical folds, **scored in original units** (predictions inverted before scoring, so base and
trial are comparable), is the arbiter. First evidence run, House Prices / `SalePrice`, plain RMSE,
3 folds:

| Step | Outcome |
|---|---|
| LLM proposal | `log1p` — correctly reasoned (skewness 1.88, mean 180.9k ≫ median 163k, max 4.6× median) |
| Arbiter | **REJECT** — CV −28 860 → −28 735 rmse, Δ +124.66 (≈0.43%), within the 3-fold paired noise band |

**Reading — the arbiter overruled a plausible, textbook-correct suggestion, and was right to.** The
project's own prior expectation was that House Prices ("the textbook log-transform case") would be a
*win* for M11. It was not — and the reason is instructive. The classic advice is textbook for
**RMSLE + linear models**: a log-scale metric and a model class sensitive to target skew. This run
used **plain RMSE + AutoGluon's gradient-boosted trees**, which are ~invariant to a monotone target
transform and already absorb the tail. So the transform moved the score by less than half a percent
— indistinguishable from fold noise — and the arbiter declined it. A system that applied "the LLM
said log1p, and it's the textbook case" on faith would have added a transform for nothing. This is
the empirical arbiter earning its place: the LLM contributes a sound hypothesis, the measurement
decides, and a wrong-for-this-setup convention is caught before it ships.

**Precision control (2026-07-03).** The complementary test: a *symmetric* target, where the agent
should decline. On friedman-synth / `y` (skewness 0.048, mean 14.088 ≈ median 14.069, max only 2×
median):

| Step | Outcome |
|---|---|
| LLM proposal | **`none`** — correctly reasoned (low skew, mean ≈ median, no long tail), no transform |
| Arbiter | not invoked (no verified proposal) — zero cost |

Together the two runs characterise M11 on both sides: it **fires on a genuinely skewed target** (and
the arbiter then gates it on measurement) and it **does not fire on a symmetric one** (no false
positive). The agent has precision, and the arbiter has the final say — exactly the intended split.

**The one open question worth a run** is whether *any* realistic target is skewed enough that log1p
helps even on a neutral absolute-error metric with boosted trees — the real question SalePrice
answered "no" for. (Explicitly *not* worth running: M11 under RMSLE. That test is near-tautological —
RMSLE(y, ŷ) = RMSE(log1p y, log1p ŷ), so log1p training minimises it by construction — and AutoGluon
has no native RMSLE metric anyway.) The plain-RMSE result does not falsify M11; it correctly reports
that log1p does not help *this metric on this engine*. (Implementation verified end-to-end: each fold
trains in log space, inverts predictions via `expm1`, and rescores against the original-space truth
held aside before transforming.)

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

## Where LLM judgment pays off — the whole map

The systematic answer to the project's question, across every layer a conductor could touch:

| Layer | Does LLM judgment beat the AutoGluon baseline? | Evidence |
|---|---|---|
| **Setup / validation** (fold strategy, leakage) | **Yes — decisively** | M1: removed a **+0.499** CV lie (synthetic); real data: cut a **5.7×** (Grunfeld, group) and a **15.3×** (economics, time) optimism roughly in half or better; detection 17/17 with 0 false alarms, provider-robust (M9) |
| Setup / target framing (M11) | The *judgment* is sound; the win depends on metric × engine | House Prices: LLM correctly proposed `log1p` (skew 1.88), arbiter correctly **rejected** it (plain RMSE + trees are transform-invariant); correctly silent on a symmetric target |
| Cleaning / encoding | Yes, modestly — on semantic-rich data, and **only** there | House Prices 5/5 seeds (+1 285 rmse); E2 battery (10 tasks): 2 decided wins, both rich-semantics (credit −39%, wage −1.1%), 8 undecided, 0 decided losses; poor-semantics controls **inert** (Δ −0.001 / +0.008) |
| **Feature engineering** (arithmetic, ordinal *and* free-text) | **No — across the board, all three lanes** | hybrid kept 0/5; ordinal mean-negative; text extractors 0/5 vs the engine's own n-grams (M10) |

**The publishable conclusion:** the feature-engineering layer — where most LLM-for-AutoML work
concentrates (CAAFE, MALMAS, LLM-FE) — is a wash against a strong engine. The LLM's value is
real but concentrated where the engine is structurally **blind**: validation design and setup.
E2 sharpens the cleaning row into a causal claim: the anonymized-twin control and the synthetic
control show the effect **vanishes** (not shrinks) when column semantics are removed — semantics
is the mechanism, and the arbiter is what converts that mechanism into only-upside (2 wins, 0
losses, noise refused).

## Recurring pattern

On 2 of 3 graded comparisons (leaf, titanic) the LLM cleaning/FE **underperformed** plain
AutoGluon; on tps it helped slightly. The baseline comparison is the point — it keeps us honest.
Raw records (every run, all fields): `runs.jsonl` (`kind: mlebench`) and `benchmark.jsonl`.
