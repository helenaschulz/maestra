<div align="center">

# 🎼 Maestra

### Agentic AutoML for tabular data, where every decision has to earn its place through evidence.

LLM agents propose, measurement decides, and every number ships with the evidence that earned it.

![Python](https://img.shields.io/badge/python-3.9–3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Engine](https://img.shields.io/badge/engine-AutoGluon-FF6F00)
![LLM](https://img.shields.io/badge/LLM-LiteLLM%20·%20model--agnostic-7E57C2)
![MCP](https://img.shields.io/badge/frontend-MCP%20server-4B8BBE)
![CI](https://github.com/helenaschulz/maestra/actions/workflows/ci.yml/badge.svg)

</div>

---

## What Maestra is

Give it a dataset and a target. Maestra returns a predictive model backed by auditable evidence, a trustworthy estimate of the achievable performance, or a reasoned refusal when the data cannot carry the question. Specialized LLM agents read the *meaning* of your columns and surface the risks that quietly sink real ML projects: data leakage, temporal and group structure, flawed validation design.

Classic AutoML builds and tunes models, then hands the judgment back to a senior data scientist: which validation is honest, which score to trust, whether to ship at all. Maestra makes that judgment layer part of the system, and proves its estimates against external ground truth, including real Kaggle leaderboards.

The core rule is simple and never bent: **the agents propose, measurement decides.** Every agent output is a structured JSON plan, never raw code, never a final call. A leakage-free k-fold cross-validation, re-fitting every transform per fold, then checks whether each proposal beats a deterministic `--no-llm` baseline beyond fold noise (Nadeau-Bengio corrected, so overlapping folds cannot fool the accept rule). Only measured improvements survive. Most LLM-for-AutoML work resolves disagreement with an LLM judge. Maestra resolves it with an experiment. That inversion is the product.

## Where Maestra sits

|  | Classic AutoML (AutoGluon, H2O) | LLM-for-AutoML (CAAFE-style) | Commercial AutoML (DataRobot-style) | **Maestra** |
|---|:---:|:---:|:---:|:---:|
| Model building and HP search | ● | ◐ | ● | ● |
| Reads column semantics (units, entities, ordinality) | — | ● | ◐ | ● |
| Every LLM proposal gated by measurement, no LLM-judge | n/a | — | — | ● |
| Leakage-safe by construction (per-fold refit) | ◐ | ◐ | ◐ | ● |
| Validation strategy chosen from structure (group / time folds) | — | — | ◐ | ● |
| Trustworthiness of the estimate itself measured (CV↔LB gap) | — | — | — | ● |
| Reasoned refusal as a first-class result | — | — | — | ● |
| Consumable verdict artifact, no install | — | — | ● | ● |
| Open source and model-agnostic | ● | ◐ | — | ● |

<sub>● full · ◐ partial · — not the focus</sub>

Maestra does not reinvent the modeling engine. It borrows a strong one (AutoGluon) and wraps it in the layer that decides whether its output deserves your trust.

## See it in action, no install

- **[Example run dossiers](docs/examples/reports/):** verdict-first HTML reports, the full evidence trail collapsible underneath (what was tried, measured, and *rejected*, with numbers). [bike-sharing](docs/examples/reports/bike-sharing.html) (temporal demand), [House Prices](docs/examples/reports/house-prices.html) (rich-semantics regression), [Grunfeld](docs/examples/reports/grunfeld.html) (group leakage caught before modeling), [Rossmann backtest audit](docs/examples/reports/rossmann-backtest-audit.html) (a future-leaking feature caught in a forecasting setup).
- **[Arbiter quickstart notebook](docs/examples/compare_quickstart.ipynb):** the empirical arbiter as a standalone tool, in Colab. Two sklearn pipelines, one honest verdict, no AutoGluon, no API key.
- **[Measurement ledger](docs/RESULTS.md):** every number in this README traces to a line there, negative results included.

## Quickstart

```bash
pip install -e ".[dev]"
echo "OPENAI_API_KEY=sk-..." > .env

# a client-ready data-risk report, BEFORE any model is built:
# validation-design recommendation, leakage scan, structural traps
maestra-audit --csv data.csv --target y

# the full pipeline: clean, validate, model, verdict
maestra --csv data/titanic.csv --target Survived
```

```
Loaded data/titanic.csv: rows=891, columns=12

=== Applied cleaning (validated JSON, fixed vocabulary) ===
  DROP 'PassengerId'  -- ID-like, unique per row
  DROP 'Cabin'        -- 77% missing
  IMPUTE 'Age' [median] fit on train -> 28.0
Columns after cleaning: 9 (from 12)

Problem type (inferred): binary   Eval metric: accuracy

=== Best-model metrics on holdout ===
  accuracy: 0.826   roc_auc: 0.884
```

## Evidence

### Tested against real leaderboards

Maestra's promise is a *trustworthy* estimate, so the estimate is what gets tested: submit to real Kaggle competitions, compare what the internal CV predicted against what the public leaderboard returned. Nine comparable receipts (K1 and K2 in the [ledger](docs/RESULTS.md)):

| Competition | Metric | CV estimate | Public LB | Direction of error |
|---|---|---|---|---|
| house-prices | RMSLE ↓ | 0.1307 | 0.12544 | pessimistic (safe, ≈4%) |
| spaceship-titanic | accuracy ↑ | 0.7909 | 0.79214 | ≈ exact |
| santander-transaction | AUC ↑ | 0.8928 | 0.89657 | ≈ exact |
| ieee-fraud | AUC ↑ | 0.8965 | 0.91427 | pessimistic (safe) |
| rossmann | RMSPE ↓ | 0.3586 | 0.20860 | pessimistic (safe) |
| allstate | MAE ↓ | 1897.1 | 1141.97 | pessimistic (safe) |
| titanic | accuracy ↑ | 0.8137 | 0.75598 | optimistic (small-data variance, 891 rows) |
| **bike-sharing** | RMSLE ↓ | 0.372 | 0.48758 | **optimistic (random folds on temporal data)** |
| **walmart** | WMAE ↓ | 1441.6 | 2958.56 | **optimistic (random folds on temporal data)** |

Six of nine land exact or on the safe, pessimistic side. Titanic misses within small-sample noise. The two structural misses are the most instructive rows: both ran with the Validation Strategist off (random folds on temporal competitions), the exact failure mode `--fold-advisor` exists to catch, and it detects both structures when enabled ([bike-sharing case study](docs/case_studies/bike_sharing.md)). A blind spot, demonstrated live on real leaderboards, is a stronger argument for the product than a clean chart would be. Follow-up: a rerun of bike-sharing with the time-aware (`time_local`) folds the Strategist now proposes scored **0.43660** on the live leaderboard (public = private, no LB overfitting), a ~10.5% improvement over the 0.48758 above. Honest caveat, straight from the ledger: this was a whole-pipeline rerun (model, target framing, and fold strategy all changed), so the gain is not attributable to the fold fix alone.

### LLM vs. the deterministic baseline

Every claim is a graded run against a real answer key: LLM vs. the `--no-llm` AutoGluon baseline, same budget and seed.

| Task | Semantics | Metric | Baseline | Maestra | Verdict |
|---|---|---|---|---|---|
| TPS Dec-2021 (MLE-bench, 3.6M rows) | none | accuracy | 0.9592 | **0.9607** | marginal win, CV↔LB gap < 0.001 |
| Leaf classification (99 classes) | none | log_loss ↓ | **0.0737** | 0.0783 | LLM hurts (reproducible, 3 seeds) |
| House Prices (43 text columns) | **rich** | rmse ↓ | 30 743 | **29 458** | ahead 5/5 seeds, undecided under the corrected variance rule |
| 10-task battery (5 seeds, paired) | rich → anon | mixed | — | — | **2 decided wins (both rich), 8 undecided, 0 decided losses** |
| Kaggle battery (4 comps, 5 seeds) | rich → anon | mixed | — | — | **1 decided win** (bike-sharing −71%), 3 undecided, **0 decided losses** |
| Target framing (`--target-framing`) | setup | rmse ↓ | raw | **log1p, 5/5** | **≈ −8% rmse**, a setup win AutoGluon cannot make itself |
| Grouped data (entity leakage) | structural | CV↔truth gap | **+0.499** | **−0.006** | **Strategist removes a 50-point CV lie** |

Four findings shape the design, and all of them are measurements, not opinions:

1. **Validation design is where the LLM's leverage is ~100× everything else.** Fixing fold strategy moved a grouped-data score by ~0.5 (a CV of 0.99 masking a true 0.49). Cleaning and FE judgment move scores by ~0.005. Structure detection benchmarks at **17/17, 0/6 false alarms**, stable across four providers.
2. **The LLM pays off only where column semantics exist, shown causally.** Both decided wins over the 10-task battery are rich-semantics tasks. An anonymized-twin control (names stripped to `x1..xn`) makes the effect vanish. Independently reproduces [CAAFE](https://arxiv.org/abs/2305.03403).
3. **The feature-engineering layer is a measured null, and frozen.** `--hybrid` proposed the right domain features and the CV gate rejected all of them (a strong engine already extracts them). Turning a slogan ("the LLM's value is in setup and validation") into a measurement.
4. **The CV↔LB gap works as a trust meta-signal.** Near zero when the CV is trustworthy, large when it is not.

> **The lesson, baked into the design:** never trust an LLM's judgment blind. Make it beat a baseline, and make the validation's own trustworthiness measurable.

### It caught its own mistake

On the open [Playground S6E6](https://www.kaggle.com/competitions/playground-series-s6e6) competition, the cleaning agent confidently dropped the photometric bands `u, g, r, i, z` (real features), reasoning *"unique per row → not useful."* The holdout metric still looked fine. Only the baseline comparison exposed the damage. The fix was deterministic (an `id_like` signal that never flags continuous floats), and the final submission scored **0.95045** public, within 0.001 of the local estimate.

## How it works

**Specialized LLM agents propose. A measurement arbiter disposes.** No agent framework: the whole control flow reads top to bottom in [`pipeline.py`](src/maestra/pipeline.py), a plain Python loop, by choice. Three planes, cleanly separated:

| 🎼 LLM agents **propose** | 🎻 The engine **executes** | ⚖️ The arbiter **decides** |
|---|---|---|
| Validation strategy, cleaning, encoding, features, diagnosis | AutoGluon model & HP search, every metric | Leakage-free k-fold CV |
| Structured JSON only, from fixed vocabularies | Deterministic fit on train, replay per fold | `--no-llm` baseline, paired tests |
| The Skeptic attacks proposals adversarially | Generated code runs sandboxed | CV↔LB gap, per-candidate gate, JSONL audit trail |

Full write-up (gate design, why no agent framework, the layer separation): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Capabilities

**Understand** the data before touching a model.

- **`maestra-audit`:** a standalone data-risk report to run *before* modeling. Executive summary with an overall risk verdict, the recommended validation strategy, LLM-flagged *and* deterministically-detected leakage, structural traps (id-like, constant, high-missing, free-text), an optional train/test shift check. Trains no model. English or German (`--lang de`), reads CSV, Parquet, Excel.
- **Backtest audit for forecasting** (`maestra-audit --backtest --time-col COL [--series-col COL]`): the same idea for an *existing* time-series setup. Detects future-leaking features (it caught Rossmann's `Customers`, absent from the real test set, at |corr| 0.892 with the target), measures the naive-vs-embargoed backtest gap, and runs a null-controlled series-boundary shift check. Also exposed as an MCP tool.
- **Semantic context** (`--description`, `--ordinal`): feed the provider's data description to every judgment node, and let the LLM map ordinal categoricals to a worst→best rank the trees cannot infer. The map is the LLM's knowledge, applied leakage-free, never a data statistic.

**Validate** so the score is honest.

- **Validation Strategist** (`--fold-advisor`): the LLM chooses how folds are built (random / group / time / `time_local` rolling-origin) from column semantics, the one validation decision AutoML cannot make. Every proposal is verified deterministically and falls back to random on any defect. Default-on with `--cv`, 0 false alarms across frontier models. The `time_local` strategy (a repeating local split, e.g. each month's early days predicting its later days) ships as a standalone rolling-origin splitter, reachable from the raw column profile.
- **Leakage-safe by construction:** fit on train only, replayed per fold, out-of-fold predictions *and* probabilities, adversarial train/test shift check.
- **Trust meta-signal:** the CV↔LB gap that told you which leaderboard rows above to trust.

**Judge**, and refuse when the data does not carry the question.

- **The arbiter is the only decider.** No LLM judges another LLM. `compare()` (`from maestra import compare`) exposes it as a generic DS tool over any sklearn-compatible estimator, no LLM key, no AutoGluon needed.
- **`maestra-mcp`:** four MCP tools for agentic frontends (Claude Desktop/Code): `audit_csv`, `check_validation` (fold recommendation + a *measured* naive-split optimism gap), `feasibility` (achievable quality, drivers, risks from one conservative run), and `audit_backtest` (the forecasting backtest audit above). Every return is a verdict record, never a model.
- **Verdicts, not build buttons.** A refusal ("the data does not support this question") is a first-class result with a reason, not an error.

**Extend**, all measurement-gated.

- **Skeptic** (`--skeptic`): a second LLM attacks the cleaning plan's drops; a drop is vetoed only if keeping the column measurably helps.
- **Target framing** (`--target-framing`): proposes `log1p` for skewed regression targets, adopted only if a paired CV in original units beats the base.
- **Hybrid features** (`--hybrid`) and **free-text featurization** (`--text-features`): LLM-written *deterministic* extraction code in a resource-bounded sandbox (network blocked, secrets stripped, target stripped), kept only if it beats a paired per-fold CV. Full provenance (kept/rejected/why) in the run log.
- **Strategy research** (`--research`): web-grounded, rules-aware hypotheses that inform planning but never bypass validation.
- **Benchmark harnesses:** `maestra-bench` (seeded replications, paired verdict with *undecided-within-noise* as a first-class outcome) and `maestra-mlebench` (real MLE-bench grading, medals, CV↔LB gap).
- **Model-agnostic:** any [LiteLLM](https://docs.litellm.ai/) backbone via one `--model` string.

## Install

Requires Python 3.9–3.12. AutoGluon's install is large (pulls in PyTorch).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add the key for your --model backbone
```

Optional extras: `pip install -e ".[research]"` (web research), `pip install -e ".[mlebench]"` (MLE-bench grading, needs Python ≤ 3.11 and Kaggle credentials).

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

# audit a dataset BEFORE modeling: risk verdict, validation strategy, leakage, actions
maestra-audit --csv data/train.csv --target churn --test data/test.csv --lang de --out audit.md

# audit an existing FORECASTING setup: future-leaking features, backtest-design gap
maestra-audit --backtest --csv data/sales.csv --target sales --time-col date --series-col store_id
```

Full flag reference and the module map live in [`docs/`](docs/). The library entry point:

```python
from maestra import run_pipeline, compare

result = run_pipeline(df, "Survived", model="gpt-4o", cv_folds=5, seed=42)
result.training.metrics   # {'accuracy': 0.826, 'roc_auc': 0.884, ...}
result.cleaning_log       # every drop/impute decision, auditable

# the arbiter alone, no LLM key or AutoGluon needed:
compare(LinearRegression(), Ridge(alpha=1.0), df, "SalePrice", cv=5, seeds=3).summary()
```

## Design decisions

- **Constrained JSON, not executed code**, with one audited exception: `--hybrid` runs LLM-written feature code, but only inside a sandbox and only past a CV gate.
- **The baseline is part of the product.** `--no-llm` exists so every agentic claim can be falsified. It has caught real regressions and real null results.
- **Validation is the only arbiter.** No LLM judges another. Disagreements are settled by measurement.
- **No agent framework.** Deterministic control flow in plain Python, by choice.

## Known limitations

- **Run-to-run nondeterminism.** LLM plans vary between runs (even at temperature 0) and AutoGluon under a wall-clock budget varies with timing. Comparisons need multiple seeds; `maestra-bench --seeds …` runs genuine replications for exactly this.
- **Feature generation rarely beats a strong engine.** Measured and expected. Keep `--hybrid` for semantic long-shots, not as a default.
- **Submission-side calibration is opt-in for a reason.** A temperature fitted on out-of-fold probabilities does not always transfer to the final full-data model.

## License

MIT, see [LICENSE](LICENSE).

<div align="center">
<sub>An LLM that conducts, an engine that plays, and a scoreboard that keeps both honest.</sub>
</div>
