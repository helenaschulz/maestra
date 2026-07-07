# Case study: bike-sharing demand — a three-bug hunt, and a CV↔LB gap that told the truth

**Task:** Kaggle's [Bike Sharing Demand](https://www.kaggle.com/c/bike-sharing-demand) —
predict hourly rental `count` from weather, calendar, and a raw `datetime` column. A real,
temporal demand-forecasting problem, not a synthetic one: the actual leaderboard is the ground
truth throughout this case study, not an internal proxy. All numbers below trace to
[`docs/RESULTS.md`](../RESULTS.md)'s K1 section.

## The setup

Maestra ran its standard battery (5 seeds, LLM cleaning + feature engineering vs. a
`--no-llm` AutoGluon baseline, paired verdict) against bike-sharing alongside three other
Kaggle tasks. Two known traps were handled up front, deterministically, before any LLM saw the
data: `casual` and `registered` sum exactly to the target `count` and are absent from the real
test set — dropped in the loader, the same "diamonds" leak-hygiene lesson applied proactively
rather than left for the arbiter to stumble over.

## The first result was a loss — and that's exactly when the real work started

The first full run returned a **decided LOSS** for Maestra: baseline 124.0 rmse vs. Maestra
143.5 rmse. Nothing here demanded action — a plausible LLM-hurts result on one of four tasks is
exactly the kind of thing an unverified pipeline would report and move on from. Maestra's own
design principle says otherwise: an unexpected decided result is scrutinized like the earlier
"diamonds" harness leak was, not reported as-is. Three separate, real defects were found and
fixed in sequence — each verified cheaply (a single debug seed) before the next was pursued.

**Bug 1 — a coordination gap between agents.** The cleaning agent dropped the raw `datetime`
column as "unique per row" (correctly id-like) — but its own stated rationale said the column
"can be decomposed into more useful features", and then never acted on that, because feature
engineering runs *after* cleaning and never sees a column cleaning already discarded. No agent
in the pipeline could say "keep this one, someone else downstream needs it." Fixed: the cleaning
prompt now distinguishes a genuine timestamp from a plain running ID and keeps it — the
`date_parts` feature step parses the raw string itself and drops the column afterward, so keeping
it here costs nothing.

**Bug 2 — a misapplied caution clause.** The target-framing agent declined to reframe `count`
(822 distinct values, strongly right-skewed) with a self-contradicting rationale: *"count is a
small-range count variable."* A caution clause meant for genuinely small-range counts ("a handful
of distinct integer values") was firing on a target with hundreds of values. Fixed: the prompt
now explicitly contrasts "a handful (under ~20–30)" against a wide-range count, which should be
treated as continuous and is a `log1p` candidate.

**Bug 3 — a structurally missing vocabulary entry.** Even after fixes 1–2, Maestra still trailed
the baseline (debug CV: 136.7 vs. 131.8 rmse). The reason wasn't judgment — it was capability: the
baseline sees the *raw* datetime string directly, and AutoGluon's own automatic datetime handling
captures sub-day granularity from it. Maestra's fixed feature vocabulary (`date_parts`) only
extracted year/month/weekday — no hour-of-day at all. For *hourly* bike demand, the commute-peak
hour is close to the dominant predictive signal, and no amount of LLM judgment can propose a
feature the vocabulary has no way to express. Fixed: added `hour` to the vocabulary.

**The `hour` fix alone took the debug seed from rmse 136.7 to 38.8** — large enough to justify
re-running the full 5-seed battery. The verdict flipped: **from a decided baseline win
(Δ +19.55) to a decided Maestra win (Δ −87.97, ≈ −71%), consistent across all 5 seeds (−86 to
−90).**

## What this is actually evidence of

It would be easy to read "−71%" as "the LLM is a great feature engineer." That reading doesn't
survive contact with the mechanism: two of the three fixes are cross-agent coordination gaps and
prompt-wording bugs, not new domain insight, and the third is a vocabulary gap the engine itself
cannot close (AutoGluon can't parse an hour out of a string it never decomposes). The honest
framing, and the one this project holds itself to elsewhere (see the README's feature-engineering
findings): this result measures *fixing defects in Maestra's own setup logic*, not LLM cleverness
beating a strong engine at feature invention. It is a real, reproducible, −71% win — earned by
comparing against a baseline and refusing to accept an unexplained result, the same discipline
that caught the Playground-S6E6 leak elsewhere in this project.

## The CV↔LB gap: the project's central thesis, replayed live

Once bike-sharing reached the real Kaggle leaderboard, a second, independent test became
possible: does the internal cross-validation estimate predict the real public score?

The submission CV used random folds (the library default — `--make-submission` did not pass
`--fold-advisor` for this run): **CV 0.372 RMSLE vs. public LB 0.48758 — a −0.116 gap, in the
dangerous, optimistic direction.** Random folds are exactly the blind spot Maestra's Validation
Strategist exists to catch, and here it showed up on a real leaderboard, not a synthetic
benchmark.

The obvious next step — turn the Strategist on — turned out to be only half the story. It
correctly detected the temporal structure unaided (`strategy: time, time_column: datetime`,
reasoning: "may involve predicting future counts based on past data"). But a plain global
time-ordered CV (3 sequential chronological folds over the full ~2-year span) **overshot the
other way**: RMSLE 0.593, a +0.105 gap, now pessimistic. The fold scores explain why (per-fold
rmse: 46 / 123 / 65) — classic expanding-window bias: with only 3 folds over two years of strong
seasonality and year-over-year growth, an early fold trains on very little data and validates on
a large, distributionally different future block.

The real competition's actual split is much milder than that: **train is days 1–19 of every
month, test is days 20 through the end of every month — a repeating, local, within-month
prediction task**, not one large future block. Neither `random` nor a single global `time` split
matches that shape. This sharpened the project's own thesis: validation-design judgment isn't
just "detect whether group/time structure exists" — it's **matching the fold's granularity to
how the real deployment split actually happens.**

## What this taught about backtest honesty

Bike-sharing's gap directly motivated a new fold-strategy vocabulary entry, `time_local`: blocked,
within-period folds (each fold trains on early blocks and validates on the next block, *within
every period*, pooled across all periods) for a deployment split that repeats locally rather than
cutting the timeline once. Confirmed on an independent synthetic task with a strong across-period
trend plus a local within-period ramp: random folds gap −0.62 (optimistic), a global time split
gap +4.76 (pessimistic overshoot), `time_local` gap +0.51 — a **>9× reduction** versus the global
split.

Closing the loop on bike-sharing itself surfaced one more honest, precisely-scoped gap rather
than a clean win: the Validation Strategist decides fold strategy from the *raw* column profile,
before feature engineering has decomposed `datetime` into a month/period column it could name.
Bike-sharing's only time signal at that point is the raw timestamp string — there is no
`period_column` yet to propose. This is a real, well-understood integration gap (surfacing derived
period candidates during profiling, before the Strategist decides), not a same-session fix, and
it is documented as open rather than silently closed.

**The lesson this case study leaves, stated plainly:** a leaderboard-beating result and a
validation design that lies about its own reliability are two *independent* claims. Bike-sharing
earned both a real accuracy win (via bug fixes, honestly attributed) and a real, quantified,
still-partially-open lesson about how validation folds must match the shape of a deployment split
— not just whether time matters, but how. Neither claim substitutes for the other, and the
project's own ledger keeps them separate for exactly that reason.
