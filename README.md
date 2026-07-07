<div align="center">

# 🎼 Maestra

### Trustworthy validation design and a measured, empirical arbiter — for tabular ML that doesn't lie to itself.

![Python](https://img.shields.io/badge/python-3.9–3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Engine](https://img.shields.io/badge/engine-AutoGluon-FF6F00)
![LLM](https://img.shields.io/badge/LLM-LiteLLM%20·%20model--agnostic-7E57C2)
![MCP](https://img.shields.io/badge/frontend-MCP%20server-4B8BBE)
![CI](https://github.com/helenaschulz/maestra/actions/workflows/ci.yml/badge.svg)

</div>

---

## What is Maestra

**Maestra is an agentic AutoML system for tabular data.** Give it a dataset and a
target, and it delivers predictive models together with a trustworthy estimate of
the achievable performance. Specialized LLM agents read the semantics of your data
and surface the risks that sink real-world ML projects: data leakage, temporal and
group structure, flawed validation design. Every agent decision must pass an
empirical gate — only interventions that measurably improve results in a controlled
experiment are adopted; the LLM itself never decides. The result is a model backed
by auditable evidence — or a reasoned refusal when the data cannot support the
question.

Maestra doesn't automate model building — modern AutoML engines already do that.
It automates the senior data scientist's judgment around it: risk detection,
validation design, honest expectation-setting. That is the blind spot of every
AutoML pipeline, and the estimates' reliability is demonstrated against external
ground truth, including real Kaggle leaderboards.

**What a run actually does:** a column profile goes to specialized LLM agents (validation
strategist, cleaning, feature engineering), each of which proposes a structured JSON plan —
never raw code, never a final decision. A leakage-free k-fold cross-validation — re-fitting every
transform per fold — then measures whether each proposal beats a deterministic `--no-llm`
baseline beyond fold noise (Nadeau-Bengio-corrected, so overlapping folds don't fool the accept
rule). Only measured improvements survive. What comes out is a verdict — a risk-flagged audit, a
quantified optimism gap, an achievable-quality estimate, or a full model — never a bare number to
trust on faith.

**The differentiator is not the agent, it's the empirical arbiter around it.** Conflicts are
settled by measurement, never by one model judging another — most LLM-for-AutoML work resolves
disagreement with an LLM judge instead. The findings below, including the negative ones, come
from that harness.

## The 10-minute path

No installation needed to see the core claims:

1. **Read this README** — the evidence table and findings below.
2. **Open a clickable example report** — verdict-first, the full DS evidence collapsible
   underneath (what was tried, measured, and *rejected*, with numbers):
   - [bike-sharing run dossier](docs/examples/reports/bike-sharing.html) — temporal demand, fold-advisor + framing
   - [House Prices run dossier](docs/examples/reports/house-prices.html) — rich-semantics regression
   - [Grunfeld data-risk audit](docs/examples/reports/grunfeld.html) — group leakage caught before modeling
3. **Watch the 3-minute demo** — a live Claude session asking "can we forecast this, and can I
   trust the number?", answered by Maestra's [MCP tools](docs/MCP.md)
   (script and a real, measured rehearsal: [docs/examples/demo/SCRIPT.md](docs/examples/demo/SCRIPT.md);
   recorded video: pending).
4. **Run the arbiter yourself, no install** — [`compare_quickstart.ipynb`](docs/examples/compare_quickstart.ipynb)
   in Colab: two sklearn pipelines, one honest verdict, `pip install --no-deps` (no AutoGluon
   needed for this one).
5. **Check any number** — every claim traces to a line in [`docs/RESULTS.md`](docs/RESULTS.md),
   the project's measurement ledger, negative results included.

Reports/notebook are generated with `maestra --dossier out.html …` / `maestra-audit --html out.html …`
/ `scripts/build_example_reports.py`.

**Start building:** `maestra-audit --csv data.csv --target y` — a client-ready, pre-modeling data-risk
report (validation-design recommendation, leakage scan, structural traps). Trains no model.

```bash
pip install -e ".[dev]"
echo "OPENAI_API_KEY=sk-..." > .env
maestra --csv data/titanic.csv --target Survived
```

```
Loaded data/titanic.csv: rows=891, columns=12

=== LLM cleaning plan (gpt-4o) ===
{ "columns_to_drop": [ ... ], "imputations": [ ... ] }   # validated JSON, drawn from a fixed vocabulary

=== Applied ===
  DROP 'PassengerId'  -- ID-like, unique per row
  DROP 'Name'         -- high-cardinality free text
  DROP 'Cabin'        -- 77% missing
  IMPUTE 'Age' [median] fit on train (missing=140) -> 28.0
  IMPUTE 'Embarked' [most_frequent] -> 'S'
Columns after cleaning: 9 (from 12)

Problem type (inferred by AutoGluon): binary
Eval metric: accuracy

=== Best-model metrics on holdout ===
  accuracy: 0.826
  roc_auc: 0.884
```

---

## What we measured (the honest part)

Every claim below is a graded run against a real answer key: LLM vs. the deterministic
`--no-llm` AutoGluon baseline, under the same budget and seed.

| Task | Semantics | Metric | Baseline | Maestra | Verdict |
|---|---|---|---|---|---|
| Titanic | mixed (891 rows) | balanced_acc | 0.814 | 0.806 | undecided over 5 seeds (an earlier single-seed "LLM hurts" was an outlier — corrected) |
| TPS Dec-2021 (MLE-bench, 3.6M rows) | none (anonymous) | accuracy | 0.9592 | **0.9607** | marginal win, CV↔LB gap < 0.001 |
| Leaf classification (MLE-bench, 99 classes) | none (anonymous) | log_loss ↓ | **0.0737** | 0.0783 | LLM hurts (reproducible over 3 seeds) |
| **House Prices** (43 semantic text columns) | **rich** | rmse ↓ | mean 30 743 | **mean 29 458** | ahead in 5/5 seeds, mean −1 285 rmse — **undecided** under the corrected variance rule (N1, 2026-07-05; passed narrowly before the correction — see RESULTS.md) |
| **10-task battery** (5 seeds each, paired verdict) | rich → anonymous | mixed | — | — | **2 decided wins (both rich: credit −39%, wage −1.1%), 8 undecided, 0 decided losses**; anonymous controls inert (Δ ≈ 0); both wins hold under the N1-corrected rule |
| **Kaggle battery** (4 real competitions, 5 seeds) | rich → anonymous, real data | mixed | — | — | **1 decided win** (bike-sharing rmse 124.0→**36.1**, −71% — driven by fixing 3 of Maestra's own bugs, not a clean baseline comparison; [full case study](docs/case_studies/bike_sharing.md)), 3 undecided, **0 decided losses** |
| **Target framing** (House Prices, `--target-framing`) | setup decision | rmse ↓ | raw target | **log1p, 5/5 seeds** | **mean −2 273 rmse (≈ −8%)** — a setup win AutoGluon cannot make itself |
| **House Prices submission** (real Kaggle leaderboard) | rmse ↓ (LB: RMSLE) | CV↔LB gap | CV 0.1307 (log-space) | **LB 0.12544** | **gap +0.0053 (≈4%), CV pessimistic — trustworthy on a live leaderboard** |
| **Grouped data** (entity leakage, synthetic) | structural | CV↔truth gap | **+0.499** (random folds) | **−0.006** (`--fold-advisor`) | **Strategist removes a 50-point CV lie** |

Four findings that shape the design — in the order the thesis is argued, not the order they were
found:

1. **Where AutoML is structurally blind, the LLM's contribution is two orders of magnitude
   larger than anything else it does.** On grouped data (several rows per customer, per-entity
   labels) a random-fold CV reported **0.99** against a true score of **0.49** — the classic
   silent killer of deployed models. The Validation Strategist (`--fold-advisor`) read the column
   semantics, detected the entity column, switched to group folds, and reported **0.488 vs 0.493
   truth** (gap −0.006). Cleaning/FE judgment moves scores by ~0.005; fixing the validation design
   moved it by ~0.5. **Replicated on real data, for both blind spots:** on Grunfeld (10 firms) the
   random-fold CV was **5.7× too optimistic** (rmse 41.5 vs 236.1 truth); the Strategist detected
   `firm` unaided and cut the lie in half (group-CV 143.0) — likewise on MathAchieve (160
   schools). On the `economics` time series it was **15.3× too optimistic** (rmse 282 vs 4 304
   future truth); the Strategist detected `date`, switched to time-ordered folds, and cut it to
   2.4×. Reproduce: `scripts/group_leakage_experiment.py`,
   `scripts/real_group_leakage_experiment.py`, `scripts/time_leakage_experiment.py`.
   **Detection quantified** on a 17-dataset benchmark with ground truth
   (`scripts/strategist_detection_benchmark.py`), including iid datasets and a deliberate trap
   column: first run 14/17 — the benchmark exposed two weaknesses, one targeted prompt iteration
   later it scores **17/17** (group 6/6, time 5/5, **0/6 false alarms**). The judgment is
   **provider-robust**: claude-opus-4-8, claude-sonnet-4-5, claude-haiku-4-5 and gpt-4o all score
   17/17 on the same catalog. Only gpt-4o-mini degrades (group recall **4/6, misses in the
   dangerous direction** — a missed group means a silently optimistic CV). Notably the boundary is
   *model-specific, not price-tier*: Haiku 4.5, a small cheap model, matches the flagships. The
   honest consulting takeaway is not "buy the biggest model" but **"verify this specific judgment
   on your specific model — the failure mode is invisible without a benchmark like this one."**
   A real Kaggle leaderboard sharpened this further: on bike-sharing-demand the Strategist
   correctly detected the temporal axis, but a plain global time-split *overshot* into an overly
   pessimistic CV (expanding-window bias over 2 years of seasonality), because the competition's
   actual test split is a repeating LOCAL window (last days of every month), not one big future
   block — detecting "is it temporal" is necessary but not sufficient; the fold's *granularity*
   has to match how the real split happens too. **`--fold-advisor`'s vocabulary now includes
   `time_local`** (blocked, within-period folds pooled across every period) for exactly this
   shape — confirmed on independent synthetic data, but rerunning bike-sharing itself surfaced a
   further, precisely-scoped gap rather than a clean close: see
   [N2 below](#n2--the-fold-granularity-fix-and-the-integration-gap-it-surfaced-2026-07-05) and
   the [full bike-sharing case study](docs/case_studies/bike_sharing.md) for the whole arc.
2. **The LLM pays off where column *semantics* exist, and nowhere else — now shown causally.**
   On House Prices (`Neighborhood`, `KitchenQual`, `YearBuilt`, …) its cleaning/encoding judgment
   was ahead of the baseline on **all 5 seeds** (mean −1 285 rmse) — directionally consistent,
   though the aggregate verdict is **undecided** under the harder, variance-corrected accept rule
   (N1). Over a 10-task battery both decided wins are rich-semantics tasks (credit −39%, wage
   −1.1% rmse) with zero decided losses, and both hold under the corrected rule. The causal half:
   an **anonymized-twin control** (identical bytes, names stripped to `x1..xn`) and a synthetic
   control show the effect *vanishes* (Δ −0.001/+0.008) when semantics are removed — semantics is
   the mechanism, not a correlate. This independently reproduces what
   [CAAFE](https://arxiv.org/abs/2305.03403) and [LATTEArena](https://arxiv.org/pdf/2606.09004)
   report. (The battery also caught its own harness leak: a Rdatasets index column ordered by the
   outcome made the baseline look 3× better on one task — the anomaly-shaped verdict flagged it,
   and the leak-free rerun landed on *undecided*. Honest cleaning looks like losing until the
   leak is found.)
3. **The feature-engineering layer doesn't beat AutoGluon — not arithmetic, not ordinal, not even
   free-text. Measured null, and frozen: the flags stay for reproducibility, but get no further
   development.** The `--hybrid` layer let the LLM write real feature code (sandboxed, CV-gated);
   it proposed exactly the right domain features (`age_of_house = YrSold − YearBuilt`, …) and the
   gate rejected **all of them** — trees already extract that from raw columns. `--ordinal`
   encoding was mean-negative across seeds. And on free text (SMS spam), five textbook semantic
   extractors — currency mentions, exclamation density, informality — all failed to move a 0.986
   n-gram baseline beyond noise (`--text-features`, 0/5 kept): a currency mention *is* an n-gram.
   Where CAAFE/MALMAS/LLM-FE concentrate their effort, a strong engine has already closed the
   gap — an independent finding LATTEArena also reports. The null is itself an asset: it is what
   turns "the LLM's value is in setup/validation" from a slogan into a measured claim. The one
   FE-adjacent decision that **does** pay is one level up, in *setup*: training on `log1p` of a
   skewed target (`--target-framing`) improved House-Prices rmse in **5/5 seeds (mean ≈ −8%)** —
   a reframing AutoGluon never performs on its own.
4. **The CV↔LB gap works as a trust meta-signal.** Near zero on TPS (trust the CV), huge on
   leaf-classification, where 3-fold log-loss over 99 classes is an unstable estimator (don't).

> **The lesson, baked into the design:** never trust an LLM's judgment blind — make it beat a
> baseline, and make the validation's own trustworthiness measurable.

### N2 — the fold-granularity fix, and the integration gap it surfaced (2026-07-05)

`--fold-advisor` has a fourth vocabulary entry, `time_local` — blocked, within-period folds (each
fold trains on early blocks and validates on the next block, *within every period*, pooled across
all periods) for a deployment split that **repeats locally** rather than cutting the timeline
once. Status: the mechanism is built, tested, and confirmed on independent data — but closing the
loop on bike-sharing itself surfaced a real, precisely-scoped gap rather than a clean win.

- **The mechanism is confirmed on a second, independent, synthetic repeating-period task**
  (`scripts/time_local_experiment.py`, no LLM involved — this isolates fold construction from the
  Strategist's separately-tested detection capability): a monthly-demand series with a strong
  across-period trend plus a local within-period ramp, truth = the real deployment shape (last
  third of every period from the first two-thirds, repeated). Random folds: gap **−0.62**
  (optimistic/dangerous). Global time split: gap **+4.76** (pessimistic overshoot). **`time_local`:
  gap +0.51 — a >9× reduction** versus the global split.
- **The bike-sharing rerun did NOT pick up `time_local`** — and the reason is structural, not a
  prompt bug: the Validation Strategist decides fold strategy from the RAW column profile, before
  cleaning/feature engineering runs. Bike-sharing's only time signal is a raw `datetime` string;
  the month it would need as `period_column` doesn't exist as a column yet at that point (`season`
  exists but is far too coarse — 4 values over 2 years). The Strategist correctly named only
  `time_column=datetime` and fell back to plain `time`, per its own instruction not to name columns
  that don't exist — the same **"decided before decomposed"** timing gap the K1 bug hunt already
  hit once (cleaning dropping a timestamp before FE could decompose it). Rerunning with the same
  `--fold-advisor` flag confirmed the log line: `FOLDS time-ordered by 'datetime'`, no
  `period_column` proposed.
- **Honest scope of what's shipped vs. what's next:** `time_local` is production-ready for data
  that already carries an explicit period column (patient visit index, a pre-existing month/week
  field) — this is a real, tested capability, not vaporware. Closing the loop for raw-timestamp
  tasks like bike-sharing needs one more piece: surfacing derived period candidates (month/week
  from a timestamp) during *profiling*, before the Strategist decides — a well-scoped follow-up,
  not a same-session fix, and not yet attempted.

### Case study: Maestra caught its own mistake

On the open Kaggle competition
[Playground S6E6](https://www.kaggle.com/competitions/playground-series-s6e6), the cleaning
agent confidently dropped the photometric bands `u, g, r, i, z` — real features — reasoning
*"unique per row → not useful."* The holdout metric still looked fine; only the **baseline
comparison** exposed the damage (0.955 → 0.919). The fix was deterministic (an `id_like`
profile signal that never flags continuous floats), and the final submission scored **0.95045**
public — within 0.001 of the local estimate, confirming the leakage-safe pipeline gives honest
numbers.

## Vocabulary, in market terms

Maestra's own language (agent, arbiter, gate) maps onto more familiar terms if you're scanning
for specific capabilities:

| Market term | What it is in Maestra |
|---|---|
| Structured outputs | [`llm.py`](src/maestra/llm.py) — every LLM call is forced tool-calling against a fixed JSON schema, `temperature=0` |
| Retrieval-augmented generation | [`research.py`](src/maestra/research.py) + [`websearch.py`](src/maestra/websearch.py) — opt-in, web-grounded strategy hypotheses that inform planning, never bypass validation |
| Multi-agent, with empirical conflict resolution | Skeptic / Validation Strategist / diagnosis agents don't defer to each other's judgment — every disagreement is settled by a CV measurement, not an LLM-judge vote |
| Guardrails | The `--hybrid`/`--text-features` sandbox (no network, secrets stripped, CPU/memory caps), `_is_row_independent` (blocks context-dependent generated features), `CVBudget` (caps counterfactual trial spend per run) |
| Eval harness | The arbiter itself: paired per-fold tests with Nadeau-Bengio variance correction, multi-seed replications, anonymized-twin/synthetic control experiments |
| MCP (agentic frontend) | [`maestra-mcp`](docs/MCP.md) — three tools (`audit_csv`, `check_validation`, `feasibility`) for Claude Desktop/Code, each a verdict record, never a model |

## Architecture

**Specialized LLM agents propose. A measurement arbiter disposes.** Every agent's output is a
*proposal*, not a decision — a leakage-free CV, a `--no-llm` baseline, or a per-candidate gate has
the final word. That inverts the usual multi-agent design, where one LLM judges another. Full
write-up (the gate design, why no agent framework, the layer separation): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

![Maestra architecture](assets/architecture.png)

Orchestrated by a plain Python loop — **no agent framework** — so the whole control flow reads top
to bottom in [`pipeline.py`](src/maestra/pipeline.py). The three planes:

| 🎼 LLM agents **propose** | 🎻 The engine **executes** | ⚖️ The arbiter **decides** |
|---|---|---|
| Validation strategy, cleaning, encoding, features, diagnosis | AutoGluon model & HP search, every metric | Leakage-free k-fold CV |
| Structured JSON only, from fixed vocabularies | Deterministic fit/transform (fit on train, replay per fold) | `--no-llm` baseline, paired tests |
| The Skeptic attacks proposals adversarially | Any generated code runs sandboxed | CV↔LB gap, per-candidate gate, JSONL audit trail |

Cleaning, encoding and feature engineering follow strict **fit/transform separation** — every
parameter is fitted on train (per fold, under CV) and replayed on holdout/test, so scores are honest.

## Features

- **`maestra-mcp`** — three MCP tools for agentic frontends (Claude Desktop/Code):
  `audit_csv` (the data-risk audit), `check_validation` (fold-strategy recommendation +
  a MEASURED naive-split optimism gap, not just an LLM assertion), `feasibility` (achievable
  quality, strongest drivers, biggest risks — from one conservative internal run). Every
  return is a verdict record, never a model. See [`docs/MCP.md`](docs/MCP.md).
- **`compare()`** — the arbiter as a generic DS tool: `from maestra import compare` honestly
  compares two arbitrary sklearn-compatible estimators/pipelines via the same paired,
  Nadeau-Bengio-corrected test every internal gate uses. No LLM call, no AutoGluon needed —
  see [`docs/examples/compare_quickstart.ipynb`](docs/examples/compare_quickstart.ipynb).
- **`maestra-audit`** — a standalone data-risk report to run *before* building a model:
  an executive summary with an overall risk verdict, the recommended validation strategy, LLM-flagged
  *and* deterministically-detected leakage (near-copies of the target by correlation), structural
  traps (id-like, constant, high-missing, free-text columns), and an optional train/test shift
  check — every finding with a recommended action. English or German output (`--lang de`); reads
  CSV, Parquet or Excel. Trains no model — one LLM call plus a profile.
- **Skeptic** (`--skeptic`) — a second LLM in an adversarial role attacks the cleaning plan's
  drops; each high-risk drop is put to the CV arbiter (keep vs drop) and vetoed **only if keeping
  the column measurably helps** — a safety net against dropping real signal, ruled by measurement,
  never by one model overriding another
- **Validation Strategist** (`--fold-advisor`) — the LLM decides how CV folds must be built
  (random / group / time) from the column semantics, the one validation decision AutoML cannot
  make; every proposal is verified deterministically and falls back to random on any defect
- **Target framing** (`--target-framing`) — the LLM proposes `log1p` for a skewed regression
  target (a setup decision AutoGluon never makes: it fits the target exactly as given); the
  transform is adopted **only if a paired CV, scored in original units, beats the untransformed
  base beyond noise**
- **Dataset-description context** (`--description`) — feed the provider's data description to
  every judgment node, so the LLM knows what columns *mean* (units, ordinal orders, entities)
- **Ordinal encoding** (`--ordinal`) — the LLM maps ordinal categoricals (quality/condition
  ratings, sizes) to a worst→best rank the trees cannot infer from unordered labels; verified
  and applied leakage-free (the map is the LLM's knowledge, not a data statistic)
- **Agentic cleaning & feature engineering** — constrained JSON plans from fixed vocabularies
- **Hybrid feature generation** (`--hybrid`) — LLM-written feature code in a resource-bounded
  sandbox (network blocked, secrets stripped from the environment, CPU/memory caps, target
  stripped; file *reads* are not blocked — it bounds execution, it is not a security boundary),
  kept only if it beats a paired per-fold CV test; full provenance (kept/rejected/why) in the run log
- **Free-text featurization** (`--text-features`) — detects prose columns, shows the LLM real
  sample text, and has it write *deterministic* extraction code (semantic keyword groups, numbers
  parsed out of prose) — no per-row LLM calls; same sandbox and CV gate as `--hybrid`, so an
  extractor is kept only if it beats the engine's own n-grams beyond fold noise
- **Leakage-safe by construction** — fit on train only, replayed per fold
- **Trustworthy validation** — leakage-free k-fold CV (`--cv`), out-of-fold predictions *and*
  probabilities, adversarial train/test shift check
- **Probability calibration** — temperature scaling fitted on OOF probabilities; CV-side
  effect always logged, submission reshaping opt-in (`--calibrate`)
- **Self-healing** — bounded diagnose-and-retry loop on failures (`--max-attempts`)
- **Strategy research** (`--research`) — web-grounded, competition-rules-aware hypotheses
  that inform planning but never bypass validation
- **Kaggle-ready** — label *and* probability submissions (binary + multiclass), shaped from
  the sample submission
- **Benchmark harnesses** — `maestra-bench` (local answer-key carving; `--seeds 1 2 3 …` runs
  genuine replications and settles the comparison with a paired test whose third verdict,
  *undecided-within-noise*, is a first-class outcome) and `maestra-mlebench` (real MLE-bench
  grading with medal thresholds and the CV↔LB gap)
- **Model-agnostic** — any [LiteLLM](https://docs.litellm.ai/) backbone via one `--model` string
- **Extensively tested** — the decision logic, gates and wiring are covered by a fast,
  fully offline suite (LLM *and* AutoGluon mocked); `engine.py`/`cli.py` are thin wrappers
  exercised mainly through integration runs

## Install

Requires Python 3.9–3.12 (CI tests 3.12; 3.11 exercised locally for the MLE-bench extra).
AutoGluon's install is large (pulls in PyTorch).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add the key for your --model backbone
```

Optional extras: `pip install -e ".[research]"` (web research) ·
`pip install -e ".[mlebench]"` (MLE-bench grading; needs Python ≤ 3.11 and Kaggle credentials).

> **MLE-bench + kaggle ≥ 2.x note:** mlebench pins `kaggle<1.7`, which cannot read the new
> `access_token` credential format. With kaggle 2.x installed instead, mlebench imports
> `kaggle.rest.ApiException`, which no longer exists — create a one-line shim
> `site-packages/kaggle/rest.py` re-exporting `ApiException` (any `pip install kaggle`
> wipes it, so re-create it after upgrades).

## Usage

```bash
# basic run (any LiteLLM backbone)
maestra --csv data/titanic.csv --target Survived --model gpt-4o

# leakage-free cross-validation + hybrid feature generation
maestra --csv data/train.csv --target class --cv 5 --hybrid

# build a Kaggle submission
maestra --csv data/train.csv --target class --test data/test.csv --submission sub.csv

# benchmark Maestra vs. the no-LLM baseline: 5 seeded replications, paired verdict
maestra-bench --csv data/titanic.csv --target Survived \
              --metric balanced_accuracy --id-col PassengerId --cv 3 --seeds 1 2 3 4 5 --name titanic

# run + grade a prepared MLE-bench task (medals, CV↔LB gap)
maestra-mlebench --task /path/to/prepared/public:leaf-classification \
                 --data-dir ~/.cache/mle-bench/data --metric log_loss --cv 3

# audit a dataset BEFORE modelling — risk verdict, validation strategy, leakage, actions
maestra-audit --csv data/train.csv --target churn --test data/test.csv --lang de --out audit.md
```

<details>
<summary><b>All <code>maestra</code> options</b></summary>

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gpt-4o` / `$AUTOML_MODEL` | LiteLLM model string |
| `--time-limit` | `120` | AutoGluon training budget (seconds) |
| `--test-size` | `0.2` | Holdout fraction |
| `--seed` | `42` | Split/fold seed |
| `--no-llm` | off | Baseline run (always worth comparing against) |
| `--no-fe` | off | Skip LLM feature engineering |
| `--cv` | — | Leakage-free K-fold CV instead of a single holdout (K ≥ 2) |
| `--cv-time-limit` | `--time-limit` | Budget per CV fold |
| `--fold-advisor` / `--no-fold-advisor` | **on with `--cv`** | Validation Strategist: LLM-chosen fold strategy, verified deterministically. Default-on whenever CV is active (0 false alarms across all frontier models, cross-provider); `--no-fold-advisor` opts out |
| `--ordinal` | off | Ordinal encoding: LLM-chosen worst→best rank for ordinal categoricals |
| `--skeptic` | off | Skeptic reviews cleaning drops; the CV arbiter vetoes a drop only if keeping helps (needs `--cv`) |
| `--target-framing` | off | LLM proposes `log1p` for a skewed regression target; adopted only if a paired CV in original units beats the base (needs `--cv`) |
| `--description` | — | Path to a provider-written dataset description, fed to every judgment node |
| `--hybrid` | off | LLM-generated feature code, sandboxed + CV-gated (needs `--cv`) |
| `--text-features` | off | Free-text lane: LLM-written deterministic text extractors, same sandbox + CV gate (needs `--cv`) |
| `--cv-budget` | unlimited | Cap on counterfactual trial CVs across all intervention gates; exhausted trials are recorded as skipped |
| `--hybrid-max-candidates` | `5` | Max generated-feature candidates |
| `--hybrid-threshold` | `1.0` | Keep threshold in fold-noise sigmas |
| `--research` | off | Web-grounded strategy brief feeding the planners |
| `--rules-mode` | `offline` | `live` forbids external-data recommendations (competition rules) |
| `--max-attempts` | `1` | `>1` enables the failure-diagnosis loop |
| `--revise-below` | — | Internal-val floor triggering one plan revision |
| `--test` / `--submission` | — | Unlabeled test CSV → submission file |
| `--id-col` | `id` | Identifier column for the submission |
| `--report` | — | LLM Markdown report of the run |
| `--runs-log` | `runs.jsonl` | Append-only run log |
| `--compare` | off | Print the latest LLM-vs-baseline diff, then exit |

</details>

### As a library

```python
import pandas as pd
from maestra import run_pipeline

df = pd.read_csv("data/titanic.csv")
result = run_pipeline(df, "Survived", model="gpt-4o", test_size=0.2,
                      time_limit=120, seed=42, model_dir="AutogluonModels")

result.training.metrics   # {'accuracy': 0.826, 'roc_auc': 0.884, ...}
result.cleaning_log       # every drop/impute decision, auditable
result.cv                 # CVResult with OOF predictions/probabilities (with cv_folds=...)
result.hybrid             # generated-feature provenance (with hybrid=True)
```

`compare()` needs neither an LLM key nor AutoGluon installed — it is the arbiter alone, over any
sklearn-compatible estimator:

```python
from maestra import compare
from sklearn.linear_model import LinearRegression, Ridge

result = compare(LinearRegression(), Ridge(alpha=1.0), df, "SalePrice", cv=5, seeds=3)
print(result.summary())   # verdict: improved | no_improvement | underpowered, + Markdown detail
```

## Module map

| Module | Responsibility |
|--------|----------------|
| [`pipeline.py`](src/maestra/pipeline.py) | The conductor loop; holdout & CV paths, bounded retry |
| [`profiling.py`](src/maestra/profiling.py) | Deterministic column profile — the LLM's view of the data |
| [`llm.py`](src/maestra/llm.py) | Thin LiteLLM wrapper; structured JSON via function-calling |
| [`cleaning.py`](src/maestra/cleaning.py) | Cleaning plan schema + leakage-safe fit/transform |
| [`feature_engineering.py`](src/maestra/feature_engineering.py) | Fixed feature vocabulary + fit/transform |
| [`encoding.py`](src/maestra/encoding.py) | Ordinal-encoding agent: LLM worst→best order + deterministic, leakage-free apply |
| [`skeptic.py`](src/maestra/skeptic.py) | Skeptic agent: adversarial review of cleaning drops, each veto ruled by a CV measurement |
| [`hybrid_features.py`](src/maestra/hybrid_features.py) | LLM-written feature code: sandbox, row-independence check, greedy CV gate |
| [`text_features.py`](src/maestra/text_features.py) | Free-text lane: detects prose columns; the LLM reads sample text and writes deterministic extractors — same sandbox and CV gate |
| [`_sandbox_worker.py`](src/maestra/_sandbox_worker.py) | Locked-down subprocess (no network, rlimits, whitelisted builtins) |
| [`intervention.py`](src/maestra/intervention.py) | The intervention core: one counterfactual primitive (base vs. trial on identical folds) shared by every gate, plus the per-run CV budget |
| [`validation.py`](src/maestra/validation.py) | Leakage-free k-fold CV (random/group/time folds, OOF preds + probas) + adversarial validation |
| [`validation_strategist.py`](src/maestra/validation_strategist.py) | Validation Strategist: LLM fold-strategy proposal + deterministic verification; also the public, DataFrame-input `check_validation()` |
| [`target_framing.py`](src/maestra/target_framing.py) | Target framing agent: LLM `log1p` proposal for skewed regression targets, CV-arbitrated in original units |
| [`audit.py`](src/maestra/audit.py) | `maestra-audit`: standalone data-risk report (validation / leakage / structural / shift) |
| [`dossier.py`](src/maestra/dossier.py) | Shared HTML rendering: verdict-first, DS evidence collapsible — used by both the run dossier and the audit report |
| [`calibration.py`](src/maestra/calibration.py) | Temperature scaling on OOF probabilities |
| [`engine.py`](src/maestra/engine.py) | AutoGluon training, metrics, predict / predict_proba; the `Engine` fit/predict/score protocol (`SklearnEngine`/`LightGBMEngine`/`AutoGluonEngine`) |
| [`compare.py`](src/maestra/compare.py) | Public API: `compare()` — the paired arbiter over any sklearn-compatible estimator, no LLM/AutoGluon needed |
| [`mcp_server.py`](src/maestra/mcp_server.py) | `maestra-mcp`: the three MCP tools (`audit_csv`/`check_validation`/`feasibility`) for agentic frontends |
| [`diagnosis.py`](src/maestra/diagnosis.py) | LLM failure diagnosis → bounded recovery actions |
| [`research.py`](src/maestra/research.py) / [`websearch.py`](src/maestra/websearch.py) | Opt-in web research → non-binding strategy brief (cached) |
| [`run_memory.py`](src/maestra/run_memory.py) | The project's own past verdicts, retrieved as non-binding context for future planning |
| [`benchmark.py`](src/maestra/benchmark.py) | Local benchmark: answer-key carving, grading metrics, scoreboard |
| [`mlebench_runner.py`](src/maestra/mlebench_runner.py) | MLE-bench adapter: real grading, medals, CV↔LB gap, metric modes |
| [`report.py`](src/maestra/report.py) | LLM Markdown report grounded in the run's real numbers |
| [`runlog.py`](src/maestra/runlog.py) | Append-only run log + baseline comparison |
| [`cli.py`](src/maestra/cli.py) / [`config.py`](src/maestra/config.py) | Arg parsing & output / shared env loading |

## Design decisions

- **Constrained JSON, not executed code** — with one audited exception: `--hybrid` runs
  LLM-written feature code, but only inside a sandbox and only past a CV gate.
- **The baseline is part of the product.** `--no-llm` exists so every agentic claim can be
  falsified; it has caught real regressions (see case study) and real null results (hybrid).
- **Validation is the only arbiter.** No LLM judges another LLM here; disagreements are settled
  by measurement.
- **Library returns data; the CLI does I/O.** `run_pipeline` returns a dataclass, which is what
  makes the whole flow — including retry and the hybrid gate — unit-testable offline.
- **No agent framework.** Deterministic control flow in plain Python, by choice.

## Known limitations

- **Run-to-run nondeterminism.** LLM plans vary between runs (even at temperature 0) and
  AutoGluon under a wall-clock budget varies with timing; on House Prices the swing (~960 rmse)
  is the same order as the LLM-vs-baseline effect. Comparisons need multiple seeds —
  `maestra-bench --seeds …` runs genuine replications and reports a paired verdict for exactly this.
- **Feature generation rarely beats a strong engine.** Measured and expected: keep `--hybrid`
  for semantic long-shots, not as a default.
- **Submission-side calibration is opt-in for a reason** — a temperature fitted on out-of-fold
  probabilities does not always transfer to the final full-data model; measured on the same task,
  it improved one submission and degraded another.

## Development

```bash
python -m pytest    # fast & offline — LLM and AutoGluon are mocked
```

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>An LLM that conducts, an engine that plays, and a scoreboard that keeps both honest.</sub>
</div>
