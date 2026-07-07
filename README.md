<div align="center">

# 🎼 Maestra

### Agentic AutoML for tabular data, where every decision has to earn its place through evidence.

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
target, and it delivers predictive models backed by auditable evidence, together
with a trustworthy estimate of the achievable performance, or a reasoned refusal
when the data cannot support the question. Specialized LLM agents read the
semantics of your data and surface the risks that sink real-world ML projects:
data leakage, temporal and group structure, flawed validation design. Every agent
proposal must pass an empirical gate. Only changes that measurably improve results
in a controlled experiment make it into the pipeline: the agents propose,
measurement decides.

Maestra covers the full arc of a modeling project. It builds and tunes models on a
state-of-the-art AutoML engine, and it automates the judgment that normally
requires a senior data scientist: risk detection, validation design, honest
expectation-setting. Classic AutoML leaves that judgment layer to the human
expert; Maestra makes it part of the system, and proves the reliability of its
estimates against external ground truth, including real Kaggle leaderboards.

**What a run actually does:** a column profile goes to specialized LLM agents (validation
strategist, cleaning, feature engineering), each of which proposes a structured JSON plan,
never raw code, never a final decision. A leakage-free k-fold cross-validation, re-fitting
every transform per fold, then measures whether each proposal beats a deterministic `--no-llm`
baseline beyond fold noise (Nadeau-Bengio-corrected, so overlapping folds don't fool the accept
rule). Only measured improvements survive. What comes out is a verdict: a risk-flagged audit, a
quantified optimism gap, an achievable-quality estimate, or a full model. Never a bare number
to trust on faith.

**The differentiator is not the agent, it's the empirical arbiter around it.** Conflicts are
settled by measurement, never by one model judging another (most LLM-for-AutoML work resolves
disagreement with an LLM judge instead). All findings below, including the negative ones, come
from that harness.

## See it in five minutes, no install

- **[Example run dossiers](docs/examples/reports/)**: verdict-first HTML reports, the full
  evidence trail collapsible underneath (what was tried, measured, and *rejected*, with numbers).
  [bike-sharing](docs/examples/reports/bike-sharing.html) (temporal demand),
  [House Prices](docs/examples/reports/house-prices.html) (rich-semantics regression),
  [Grunfeld](docs/examples/reports/grunfeld.html) (group leakage caught before modeling).
- **[3-minute demo](docs/examples/demo/SCRIPT.md)**: a live Claude session asking "can we
  forecast this, and can I trust the number?", answered by Maestra's [MCP tools](docs/MCP.md).
- **[Arbiter quickstart notebook](docs/examples/compare_quickstart.ipynb)**: the empirical
  arbiter as a standalone tool, in Colab. Two sklearn pipelines, one honest verdict, no
  AutoGluon, no API key.
- **[Measurement ledger](docs/RESULTS.md)**: every number in this README traces to a line
  there, negative results included.

## Quickstart

![Maestra demo](assets/demo.gif)

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

## Evidence

### Tested against real leaderboards

Maestra's core promise is a *trustworthy* estimate, so the estimate itself is what gets tested:
submit to real Kaggle competitions and compare what the internal CV predicted against what the
public leaderboard returned. Nine comparable receipts from two submission batteries
([K1 and K2 in the ledger](docs/RESULTS.md)):

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

Six of nine estimates land exact or on the safe, pessimistic side; titanic misses within
small-sample noise. The two structural misses are the most instructive rows in the table: both
submissions ran with the Validation Strategist off (random folds on temporal competitions),
which is exactly the failure mode `--fold-advisor` exists to catch, and it detects both
structures when enabled ([bike-sharing case study](docs/case_studies/bike_sharing.md)). The
blind spot, demonstrated live on real leaderboards, is the argument for the product. (Two
further submissions have no computable gap due to metric-unit mismatches, one non-comparable
rerun is excluded; details in the ledger.)

### LLM vs. the deterministic baseline

Every claim below is a graded run against a real answer key: LLM vs. the deterministic
`--no-llm` AutoGluon baseline, under the same budget and seed.

| Task | Semantics | Metric | Baseline | Maestra | Verdict |
|---|---|---|---|---|---|
| Titanic | mixed (891 rows) | balanced_acc | 0.814 | 0.806 | undecided over 5 seeds (an earlier single-seed "LLM hurts" was an outlier, corrected) |
| TPS Dec-2021 (MLE-bench, 3.6M rows) | none (anonymous) | accuracy | 0.9592 | **0.9607** | marginal win, CV↔LB gap < 0.001 |
| Leaf classification (MLE-bench, 99 classes) | none (anonymous) | log_loss ↓ | **0.0737** | 0.0783 | LLM hurts (reproducible over 3 seeds) |
| **House Prices** (43 semantic text columns) | **rich** | rmse ↓ | mean 30 743 | **mean 29 458** | ahead in 5/5 seeds, mean −1 285 rmse; **undecided** under the corrected variance rule ([ledger](docs/RESULTS.md)) |
| **10-task battery** (5 seeds each, paired verdict) | rich → anonymous | mixed | — | — | **2 decided wins (both rich semantics), 8 undecided, 0 decided losses**; anonymous controls inert |
| **Kaggle battery** (4 real competitions, 5 seeds) | rich → anonymous, real data | mixed | — | — | **1 decided win** (bike-sharing −71%, [case study](docs/case_studies/bike_sharing.md)), 3 undecided, **0 decided losses** |
| **Target framing** (House Prices, `--target-framing`) | setup decision | rmse ↓ | raw target | **log1p, 5/5 seeds** | **mean −2 273 rmse (≈ −8%)**, a setup win AutoGluon cannot make itself |
| **Grouped data** (entity leakage, synthetic) | structural | CV↔truth gap | **+0.499** (random folds) | **−0.006** (`--fold-advisor`) | **Strategist removes a 50-point CV lie** |

### Four findings that shape the design

1. **Validation design is where the LLM's leverage is ~100× everything else.**
   Fixing fold strategy moved a grouped-data score by ~0.5 (a CV of 0.99 masking a
   true 0.49); cleaning/FE judgment moves scores by ~0.005. Structure detection is
   benchmarked at **17/17, 0/6 false alarms**, stable across four model providers.
2. **The LLM pays off only where column *semantics* exist, shown causally.** Both
   decided wins over a 10-task battery are rich-semantics tasks; an anonymized-twin
   control (names stripped to `x1..xn`) makes the effect vanish. Independently
   reproduces [CAAFE](https://arxiv.org/abs/2305.03403) and [LATTEArena](https://arxiv.org/pdf/2606.09004).
3. **The feature-engineering layer is a measured null, and frozen.** `--hybrid`
   proposed the right domain features and the CV gate rejected all of them (a strong
   engine already extracts them). The one setup-level win is `--target-framing`
   (`log1p`, **5/5 seeds, ≈ −8% rmse**). The null turns "the LLM's value is in setup
   and validation" from slogan into measurement.
4. **The CV↔LB gap works as a trust meta-signal.** Near zero when the CV is
   trustworthy, large when it isn't (3-fold log-loss over 99 classes).

Full numbers, the fold-granularity follow-up, and reproduction scripts: [the
ledger](docs/RESULTS.md) and the [bike-sharing case study](docs/case_studies/bike_sharing.md).

> **The lesson, baked into the design:** never trust an LLM's judgment blind: make it beat a
> baseline, and make the validation's own trustworthiness measurable.

### Case study: Maestra caught its own mistake

On the open Kaggle competition
[Playground S6E6](https://www.kaggle.com/competitions/playground-series-s6e6), the cleaning
agent confidently dropped the photometric bands `u, g, r, i, z` (real features), reasoning
*"unique per row → not useful."* The holdout metric still looked fine; only the **baseline
comparison** exposed the damage. The fix was deterministic (an `id_like`
profile signal that never flags continuous floats), and the final submission scored **0.95045**
public, within 0.001 of the local estimate, confirming the leakage-safe pipeline gives honest
numbers.

## Architecture

**Specialized LLM agents propose. A measurement arbiter disposes.** Every agent's output is a
*proposal*, not a decision: a leakage-free CV, a `--no-llm` baseline, or a per-candidate gate has
the final word. That inverts the usual multi-agent design, where one LLM judges another. Full
write-up (the gate design, why no agent framework, the layer separation): [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

![Maestra architecture](assets/architecture.png)

Orchestrated by a plain Python loop, **no agent framework**, so the whole control flow reads top
to bottom in [`pipeline.py`](src/maestra/pipeline.py). The three planes:

| 🎼 LLM agents **propose** | 🎻 The engine **executes** | ⚖️ The arbiter **decides** |
|---|---|---|
| Validation strategy, cleaning, encoding, features, diagnosis | AutoGluon model & HP search, every metric | Leakage-free k-fold CV |
| Structured outputs: JSON only, from fixed vocabularies | Deterministic fit/transform (fit on train, replay per fold) | `--no-llm` baseline, paired tests |
| The Skeptic attacks proposals adversarially | Any generated code runs sandboxed | CV↔LB gap, per-candidate gate, JSONL audit trail |

Cleaning, encoding and feature engineering follow strict **fit/transform separation**: every
parameter is fitted on train (per fold, under CV) and replayed on holdout/test, so scores are honest.

## Features

- **`maestra-mcp`**: three MCP tools for agentic frontends (Claude Desktop/Code):
  `audit_csv` (the data-risk audit), `check_validation` (fold-strategy recommendation +
  a MEASURED naive-split optimism gap, not just an LLM assertion), `feasibility` (achievable
  quality, strongest drivers, biggest risks, from one conservative internal run). Every
  return is a verdict record, never a model. See [`docs/MCP.md`](docs/MCP.md).
- **`compare()`** (the eval harness as a generic DS tool): `from maestra import compare` honestly
  compares two arbitrary sklearn-compatible estimators/pipelines via the same paired,
  Nadeau-Bengio-corrected test every internal gate uses. No LLM call, no AutoGluon needed;
  see [`docs/examples/compare_quickstart.ipynb`](docs/examples/compare_quickstart.ipynb).
- **`maestra-audit`**: a standalone data-risk report to run *before* building a model:
  an executive summary with an overall risk verdict, the recommended validation strategy, LLM-flagged
  *and* deterministically-detected leakage (near-copies of the target by correlation), structural
  traps (id-like, constant, high-missing, free-text columns), and an optional train/test shift
  check; every finding with a recommended action. English or German output (`--lang de`); reads
  CSV, Parquet or Excel. Trains no model: one LLM call plus a profile.
- **Skeptic** (`--skeptic`): a second LLM in an adversarial role attacks the cleaning plan's
  drops; each high-risk drop is put to the CV arbiter (keep vs drop) and vetoed **only if keeping
  the column measurably helps**, a safety net against dropping real signal, ruled by measurement,
  never by one model overriding another
- **Validation Strategist** (`--fold-advisor`): the LLM decides how CV folds must be built
  (random / group / time) from the column semantics, the one validation decision AutoML cannot
  make; every proposal is verified deterministically and falls back to random on any defect
- **Target framing** (`--target-framing`): the LLM proposes `log1p` for a skewed regression
  target (a setup decision AutoGluon never makes: it fits the target exactly as given); the
  transform is adopted **only if a paired CV, scored in original units, beats the untransformed
  base beyond noise**
- **Dataset-description context** (`--description`): feed the provider's data description to
  every judgment node, so the LLM knows what columns *mean* (units, ordinal orders, entities)
- **Ordinal encoding** (`--ordinal`): the LLM maps ordinal categoricals (quality/condition
  ratings, sizes) to a worst→best rank the trees cannot infer from unordered labels; verified
  and applied leakage-free (the map is the LLM's knowledge, not a data statistic)
- **Agentic cleaning & feature engineering**: constrained JSON plans from fixed vocabularies
- **Hybrid feature generation** (`--hybrid`): LLM-written feature code in a resource-bounded
  sandbox with guardrails (network blocked, secrets stripped from the environment, CPU/memory caps, target
  stripped; file *reads* are not blocked; it bounds execution, it is not a security boundary),
  kept only if it beats a paired per-fold CV test; full provenance (kept/rejected/why) in the run log
- **Free-text featurization** (`--text-features`): detects prose columns, shows the LLM real
  sample text, and has it write *deterministic* extraction code (semantic keyword groups, numbers
  parsed out of prose), no per-row LLM calls; same sandbox and CV gate as `--hybrid`, so an
  extractor is kept only if it beats the engine's own n-grams beyond fold noise
- **Leakage-safe by construction**: fit on train only, replayed per fold
- **Trustworthy validation**: leakage-free k-fold CV (`--cv`), out-of-fold predictions *and*
  probabilities, adversarial train/test shift check
- **Probability calibration**: temperature scaling fitted on OOF probabilities; CV-side
  effect always logged, submission reshaping opt-in (`--calibrate`)
- **Self-healing**: bounded diagnose-and-retry loop on failures (`--max-attempts`)
- **Strategy research / RAG** (`--research`): web-grounded, competition-rules-aware hypotheses
  that inform planning but never bypass validation
- **Kaggle-ready**: label *and* probability submissions (binary + multiclass), shaped from
  the sample submission
- **Benchmark harnesses**: `maestra-bench` (local answer-key carving; `--seeds 1 2 3 …` runs
  genuine replications and settles the comparison with a paired test whose third verdict,
  *undecided-within-noise*, is a first-class outcome) and `maestra-mlebench` (real MLE-bench
  grading with medal thresholds and the CV↔LB gap)
- **Model-agnostic**: any [LiteLLM](https://docs.litellm.ai/) backbone via one `--model` string
- **Extensively tested**: the decision logic, gates and wiring are covered by a fast,
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
> `kaggle.rest.ApiException`, which no longer exists; create a one-line shim
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

# audit a dataset BEFORE modelling: risk verdict, validation strategy, leakage, actions
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

`compare()` needs neither an LLM key nor AutoGluon installed; it is the arbiter alone, over any
sklearn-compatible estimator:

```python
from maestra import compare
from sklearn.linear_model import LinearRegression, Ridge

result = compare(LinearRegression(), Ridge(alpha=1.0), df, "SalePrice", cv=5, seeds=3)
print(result.summary())   # verdict: improved | no_improvement | underpowered, + Markdown detail
```

## Module map

<details>
<summary><b>One module per responsibility: the full map</b></summary>

| Module | Responsibility |
|--------|----------------|
| [`pipeline.py`](src/maestra/pipeline.py) | The conductor loop; holdout & CV paths, bounded retry |
| [`profiling.py`](src/maestra/profiling.py) | Deterministic column profile, the LLM's view of the data |
| [`llm.py`](src/maestra/llm.py) | Thin LiteLLM wrapper; structured JSON via function-calling |
| [`cleaning.py`](src/maestra/cleaning.py) | Cleaning plan schema + leakage-safe fit/transform |
| [`feature_engineering.py`](src/maestra/feature_engineering.py) | Fixed feature vocabulary + fit/transform |
| [`encoding.py`](src/maestra/encoding.py) | Ordinal-encoding agent: LLM worst→best order + deterministic, leakage-free apply |
| [`skeptic.py`](src/maestra/skeptic.py) | Skeptic agent: adversarial review of cleaning drops, each veto ruled by a CV measurement |
| [`hybrid_features.py`](src/maestra/hybrid_features.py) | LLM-written feature code: sandbox, row-independence check, greedy CV gate |
| [`text_features.py`](src/maestra/text_features.py) | Free-text lane: detects prose columns; the LLM reads sample text and writes deterministic extractors, same sandbox and CV gate |
| [`_sandbox_worker.py`](src/maestra/_sandbox_worker.py) | Locked-down subprocess (no network, rlimits, whitelisted builtins) |
| [`intervention.py`](src/maestra/intervention.py) | The intervention core: one counterfactual primitive (base vs. trial on identical folds) shared by every gate, plus the per-run CV budget |
| [`validation.py`](src/maestra/validation.py) | Leakage-free k-fold CV (random/group/time folds, OOF preds + probas) + adversarial validation |
| [`validation_strategist.py`](src/maestra/validation_strategist.py) | Validation Strategist: LLM fold-strategy proposal + deterministic verification; also the public, DataFrame-input `check_validation()` |
| [`target_framing.py`](src/maestra/target_framing.py) | Target framing agent: LLM `log1p` proposal for skewed regression targets, CV-arbitrated in original units |
| [`audit.py`](src/maestra/audit.py) | `maestra-audit`: standalone data-risk report (validation / leakage / structural / shift) |
| [`dossier.py`](src/maestra/dossier.py) | Shared HTML rendering: verdict-first, DS evidence collapsible; used by both the run dossier and the audit report |
| [`calibration.py`](src/maestra/calibration.py) | Temperature scaling on OOF probabilities |
| [`engine.py`](src/maestra/engine.py) | AutoGluon training, metrics, predict / predict_proba; the `Engine` fit/predict/score protocol (`SklearnEngine`/`LightGBMEngine`/`AutoGluonEngine`) |
| [`compare.py`](src/maestra/compare.py) | Public API: `compare()`, the paired arbiter over any sklearn-compatible estimator, no LLM/AutoGluon needed |
| [`mcp_server.py`](src/maestra/mcp_server.py) | `maestra-mcp`: the three MCP tools (`audit_csv`/`check_validation`/`feasibility`) for agentic frontends |
| [`diagnosis.py`](src/maestra/diagnosis.py) | LLM failure diagnosis → bounded recovery actions |
| [`research.py`](src/maestra/research.py) / [`websearch.py`](src/maestra/websearch.py) | Opt-in web research → non-binding strategy brief (cached) |
| [`run_memory.py`](src/maestra/run_memory.py) | The project's own past verdicts, retrieved as non-binding context for future planning |
| [`benchmark.py`](src/maestra/benchmark.py) | Local benchmark: answer-key carving, grading metrics, scoreboard |
| [`mlebench_runner.py`](src/maestra/mlebench_runner.py) | MLE-bench adapter: real grading, medals, CV↔LB gap, metric modes |
| [`report.py`](src/maestra/report.py) | LLM Markdown report grounded in the run's real numbers |
| [`runlog.py`](src/maestra/runlog.py) | Append-only run log + baseline comparison |
| [`cli.py`](src/maestra/cli.py) / [`config.py`](src/maestra/config.py) | Arg parsing & output / shared env loading |

</details>

## Design decisions

- **Constrained JSON, not executed code**: with one audited exception: `--hybrid` runs
  LLM-written feature code, but only inside a sandbox and only past a CV gate.
- **The baseline is part of the product.** `--no-llm` exists so every agentic claim can be
  falsified; it has caught real regressions (see case study) and real null results (hybrid).
- **Validation is the only arbiter.** No LLM judges another LLM here; disagreements are settled
  by measurement.
- **Library returns data; the CLI does I/O.** `run_pipeline` returns a dataclass, which is what
  makes the whole flow (including retry and the hybrid gate) unit-testable offline.
- **No agent framework.** Deterministic control flow in plain Python, by choice.

## Known limitations

- **Run-to-run nondeterminism.** LLM plans vary between runs (even at temperature 0) and
  AutoGluon under a wall-clock budget varies with timing; on House Prices the swing (~960 rmse)
  is the same order as the LLM-vs-baseline effect. Comparisons need multiple seeds;
  `maestra-bench --seeds …` runs genuine replications and reports a paired verdict for exactly this.
- **Feature generation rarely beats a strong engine.** Measured and expected: keep `--hybrid`
  for semantic long-shots, not as a default.
- **Submission-side calibration is opt-in for a reason**: a temperature fitted on out-of-fold
  probabilities does not always transfer to the final full-data model; measured on the same task,
  it improved one submission and degraded another.

## Development

```bash
python -m pytest    # fast & offline: LLM and AutoGluon are mocked
```

## License

MIT, see [LICENSE](LICENSE).

<div align="center">
<sub>An LLM that conducts, an engine that plays, and a scoreboard that keeps both honest.</sub>
</div>
