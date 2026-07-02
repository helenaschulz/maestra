<div align="center">

# 🎼 Maestra

### Agentic AutoML for tabular data — with honest measurement built in.

*An LLM conductor over AutoGluon: the LLM decides, the engine computes,
and every "smart" decision has to beat doing nothing.*

![Python](https://img.shields.io/badge/python-3.9–3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Engine](https://img.shields.io/badge/engine-AutoGluon-FF6F00)
![LLM](https://img.shields.io/badge/LLM-LiteLLM%20·%20model--agnostic-7E57C2)
![CI](https://github.com/helenaschulz/maestra/actions/workflows/ci.yml/badge.svg)

</div>

---

**Maestra** points a large language model at a CSV and lets it *conduct* an AutoML run:
it reads a profile of your data, decides how to clean it, writes a structured plan, and hands
the numbers to [AutoGluon](https://auto.gluon.ai/). When something breaks, it reads the
traceback and tries again. It never does arithmetic — and any code it *does* write runs in a
sandbox and survives only if it measurably improves cross-validation.

What makes Maestra different is not the agent — it's the **measurement discipline around it**:
every LLM intervention is compared against a `--no-llm` baseline, graded against real answer
keys, and validated with a leakage-free CV whose trustworthiness is itself measured (the
**CV↔LB gap**). The findings below, including the negative ones, come from that harness.

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
| Titanic | poor | balanced_acc | **0.793** | 0.732 | LLM hurts |
| TPS Dec-2021 (MLE-bench, 3.6M rows) | none (anonymous) | accuracy | 0.9592 | **0.9607** | marginal win, CV↔LB gap < 0.001 |
| Leaf classification (MLE-bench, 99 classes) | none (anonymous) | log_loss ↓ | **0.0737** | 0.0783 | LLM hurts (reproducible over 3 seeds) |
| **House Prices** (43 semantic text columns) | **rich** | rmse ↓ | 26 453 / 25 745 | **25 828 / 24 343** | **LLM wins on both seeds** |
| **Grouped data** (entity leakage, synthetic) | structural | CV↔truth gap | **+0.499** (random folds) | **−0.006** (`--fold-advisor`) | **Strategist removes a 50-point CV lie** |

Four findings that shape the design:

1. **The LLM pays off where column *semantics* exist, and nowhere else.** On anonymous numeric
   data an LLM is structurally blind; on House Prices (`Neighborhood`, `KitchenQual`,
   `YearBuilt`, …) its cleaning/encoding judgment beat the baseline on both seeds. This
   independently reproduces what [CAAFE](https://arxiv.org/abs/2305.03403) and
   [LATTEArena](https://arxiv.org/pdf/2606.09004) report.
2. **The feature-engineering layer doesn't beat AutoGluon — not arithmetic, not ordinal.** The
   `--hybrid` layer let the LLM write real feature code (sandboxed, CV-gated); it proposed exactly
   the right domain features (`age_of_house = YrSold − YearBuilt`, …) and the gate rejected **all
   of them** — trees already extract that from raw columns. Even `--ordinal` encoding (mapping
   `KitchenQual` to a rank, the one FE type that *injects* information) was mean-negative across
   seeds: a monotonic rank is lossy versus native categorical handling. Where CAAFE/MALMAS/LLM-FE
   concentrate their effort, a strong engine has already closed the gap.
3. **The CV↔LB gap works as a trust meta-signal.** Near zero on TPS (trust the CV), huge on
   leaf-classification, where 3-fold log-loss over 99 classes is an unstable estimator (don't).
4. **Where AutoML is structurally blind, the LLM's contribution is two orders of magnitude
   larger.** On grouped data (several rows per customer, per-entity labels) a random-fold CV
   reported **0.99** against a true score of **0.49** — the classic silent killer of deployed
   models. The Validation Strategist (`--fold-advisor`) read the column semantics, detected the
   entity column, switched to group folds, and reported **0.488 vs 0.493 truth** (gap −0.006).
   Cleaning/FE judgment moves scores by ~0.005; fixing the validation design moved it by ~0.5.
   **Replicated on real data:** on Grunfeld (10 firms) the random-fold CV was **5.7× too
   optimistic** (rmse 41.5 vs 236.1 truth); the Strategist detected `firm` unaided and cut the
   lie in half (group-CV 143.0) — likewise on MathAchieve (160 schools). Reproduce:
   `python scripts/group_leakage_experiment.py` and `scripts/real_group_leakage_experiment.py`.

> **The lesson, baked into the design:** never trust an LLM's judgment blind — make it beat a
> baseline, and make the validation's own trustworthiness measurable.

### Case study: Maestra caught its own mistake

On the open Kaggle competition
[Playground S6E6](https://www.kaggle.com/competitions/playground-series-s6e6), the cleaning
agent confidently dropped the photometric bands `u, g, r, i, z` — real features — reasoning
*"unique per row → not useful."* The holdout metric still looked fine; only the **baseline
comparison** exposed the damage (0.955 → 0.919). The fix was deterministic (an `id_like`
profile signal that never flags continuous floats), and the final submission scored **0.95045**
public — within 0.001 of the local estimate, confirming the leakage-safe pipeline gives honest
numbers.

## How it works

```
        ┌──────────────────────────────────────────────────────────────────┐
        │                     🎼  MAESTRA  ·  the LLM                       │
        │     profile → clean → engineer → [research] → diagnose failures   │
        └──────────────────────────────────────────────────────────────────┘
              │ structured JSON only        │ sandboxed feature code (--hybrid)
              ▼                             ▼  kept only if CV improves
        ┌──────────────────────────────────────────────────────────────────┐
        │                  🎻  AUTOGLUON  ·  the engine                     │
        │            split · train · tune · score · predict                 │
        └──────────────────────────────────────────────────────────────────┘
              ▼
        ⚖️  the arbiter: leakage-free CV · --no-llm baseline · CV↔LB gap
```

One function per step, orchestrated by a plain Python loop — **no agent framework**; the whole
flow reads top to bottom in [`pipeline.py`](src/maestra/pipeline.py).

| 🎼 The LLM **decides** | 🎻 The engine **computes** | ⚖️ The arbiter **judges** |
|---|---|---|
| Cleaning plan (drop / impute) | Model & HP search | Leakage-free k-fold CV |
| Feature plan (fixed vocabulary) | Every metric | Baseline comparison |
| Failure diagnosis & recovery | Training & prediction | CV↔LB gap vs. real answer keys |
| Research brief (opt-in, non-binding) | Probability calibration math | Per-candidate feature gate |

Cleaning and feature engineering follow strict **fit/transform separation** — every parameter
is fitted on train (per fold, under CV) and replayed on holdout/test, so scores are honest.

## Features

- **`maestra-audit`** — a standalone data-risk report to run *before* building a model: the
  recommended validation strategy, LLM-flagged leakage, deterministic structural traps (id-like,
  constant, high-missing, free-text columns), and an optional train/test shift check. Trains no
  model — one LLM call plus a profile.
- **Skeptic** (`--skeptic`) — a second LLM in an adversarial role attacks the cleaning plan's
  drops; each high-risk drop is put to the CV arbiter (keep vs drop) and vetoed **only if keeping
  the column measurably helps** — a safety net against dropping real signal, ruled by measurement,
  never by one model overriding another
- **Validation Strategist** (`--fold-advisor`) — the LLM decides how CV folds must be built
  (random / group / time) from the column semantics, the one validation decision AutoML cannot
  make; every proposal is verified deterministically and falls back to random on any defect
- **Dataset-description context** (`--description`) — feed the provider's data description to
  every judgment node, so the LLM knows what columns *mean* (units, ordinal orders, entities)
- **Ordinal encoding** (`--ordinal`) — the LLM maps ordinal categoricals (quality/condition
  ratings, sizes) to a worst→best rank the trees cannot infer from unordered labels; verified
  and applied leakage-free (the map is the LLM's knowledge, not a data statistic)
- **Agentic cleaning & feature engineering** — constrained JSON plans from fixed vocabularies
- **Hybrid feature generation** (`--hybrid`) — LLM-written feature code in a locked-down
  sandbox (no network, CPU/memory caps, target stripped), kept only if it beats CV fold noise;
  full provenance (kept/rejected/why) in the run log
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
- **Benchmark harnesses** — `maestra-bench` (local answer-key carving) and `maestra-mlebench`
  (real MLE-bench grading with medal thresholds and the CV↔LB gap)
- **Model-agnostic** — any [LiteLLM](https://docs.litellm.ai/) backbone via one `--model` string
- **Fully tested** — fast, offline test suite (LLM *and* AutoGluon mocked)

## Install

Requires Python 3.9–3.12. AutoGluon's install is large (pulls in PyTorch).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # add the key for your --model backbone
```

Optional extras: `pip install -e ".[research]"` (web research) ·
`pip install -e ".[mlebench]"` (MLE-bench grading; needs Python ≤ 3.11 and Kaggle credentials).

## Usage

```bash
# basic run (any LiteLLM backbone)
maestra --csv data/titanic.csv --target Survived --model gpt-4o

# leakage-free cross-validation + hybrid feature generation
maestra --csv data/train.csv --target class --cv 5 --hybrid

# build a Kaggle submission
maestra --csv data/train.csv --target class --test data/test.csv --submission sub.csv

# benchmark Maestra vs. the no-LLM baseline on a carved answer key
maestra-bench --csv data/titanic.csv --target Survived \
              --metric balanced_accuracy --id-col PassengerId --cv 3 --name titanic

# run + grade a prepared MLE-bench task (medals, CV↔LB gap)
maestra-mlebench --task /path/to/prepared/public:leaf-classification \
                 --data-dir ~/.cache/mle-bench/data --metric log_loss --cv 3

# audit a dataset BEFORE modelling — validation strategy, leakage, structural traps
maestra-audit --csv data/train.csv --target churn --test data/test.csv --out audit.md
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
| `--fold-advisor` | off | Validation Strategist: LLM-chosen fold strategy, verified deterministically (needs `--cv`) |
| `--ordinal` | off | Ordinal encoding: LLM-chosen worst→best rank for ordinal categoricals |
| `--skeptic` | off | Skeptic reviews cleaning drops; the CV arbiter vetoes a drop only if keeping helps (needs `--cv`) |
| `--description` | — | Path to a provider-written dataset description, fed to every judgment node |
| `--hybrid` | off | LLM-generated feature code, sandboxed + CV-gated (needs `--cv`) |
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

## Architecture

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
| [`_sandbox_worker.py`](src/maestra/_sandbox_worker.py) | Locked-down subprocess (no network, rlimits, whitelisted builtins) |
| [`validation.py`](src/maestra/validation.py) | Leakage-free k-fold CV (random/group/time folds, OOF preds + probas) + adversarial validation |
| [`validation_strategist.py`](src/maestra/validation_strategist.py) | Validation Strategist: LLM fold-strategy proposal + deterministic verification |
| [`audit.py`](src/maestra/audit.py) | `maestra-audit`: standalone data-risk report (validation / leakage / structural / shift) |
| [`calibration.py`](src/maestra/calibration.py) | Temperature scaling on OOF probabilities |
| [`engine.py`](src/maestra/engine.py) | AutoGluon training, metrics, predict / predict_proba |
| [`diagnosis.py`](src/maestra/diagnosis.py) | LLM failure diagnosis → bounded recovery actions |
| [`research.py`](src/maestra/research.py) / [`websearch.py`](src/maestra/websearch.py) | Opt-in web research → non-binding strategy brief (cached) |
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
  is the same order as the LLM-vs-baseline effect. Comparisons need multiple seeds — `--seed`
  exists for exactly that.
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
