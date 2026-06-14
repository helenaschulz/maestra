<div align="center">

# 🎼 Maestra

### An LLM conductor for tabular AutoML.

*The model decides. The engine computes. The two never blur.*

![Python](https://img.shields.io/badge/python-3.9–3.12-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![Engine](https://img.shields.io/badge/engine-AutoGluon-FF6F00)
![LLM](https://img.shields.io/badge/LLM-LiteLLM%20·%20model--agnostic-7E57C2)
![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen)
![Kaggle](https://img.shields.io/badge/Kaggle%20S6E6-balanced__acc%200.950-20BEFF?logo=kaggle&logoColor=white)

</div>

---

**Maestra** points a large language model at a CSV and lets it *conduct* an AutoML run:
it reads a profile of your data, decides how to clean it, writes a structured plan, and
hands the numbers to [AutoGluon](https://auto.gluon.ai/). When something breaks, it reads
the traceback and tries again. It never does arithmetic — and it never runs code it wrote.

```bash
pip install -e ".[dev]"
echo "OPENAI_API_KEY=sk-..." > .env
maestra --csv data/titanic.csv --target Survived
```

```
=== LLM cleaning plan (gpt-4o) ===
  DROP 'PassengerId' -- ID-like, no predictive signal
  DROP 'Cabin'       -- 77% missing
  IMPUTE 'Age' [median] fit on train (missing=140) -> 28.0
Columns after cleaning: 8 (from 12)

=== Best-model metrics on holdout ===
  accuracy: 0.826   roc_auc: 0.884
```

---

## How it works

```
        ┌─────────────────────────────────────────────────────────────────┐
        │                     🎼  MAESTRA  ·  the LLM                        │
        │                read  ·  decide  ·  write the plan                 │
        └─────────────────────────────────────────────────────────────────┘
              │ profile(train)     │ cleaning plan        │ diagnose failure
              ▼                    ▼  (structured JSON)    ▼  (retry, bounded)
        ┌─────────────────────────────────────────────────────────────────┐
        │                  🎻  AUTOGLUON  ·  the orchestra                  │
        │           split · train · tune · score · predict                 │
        └─────────────────────────────────────────────────────────────────┘

  CSV ─▶ split ─▶ profile(train) ─▶ LLM plan ─▶ fit + apply ─▶ train ─▶ metrics ─▶ submission
                                   └─ decides ─┘           └──────── computes ───────┘
```

One function per step, orchestrated by a plain Python loop — **no agent framework**, the
whole flow reads top to bottom in [`pipeline.py`](src/maestra/pipeline.py).

## The split that matters

| 🎼 The LLM **decides** | 🎻 The engine **computes** |
|---|---|
| Reads a compact column profile | Splits train / holdout |
| Picks columns to drop & impute | Searches models & hyperparameters |
| Diagnoses failures, picks a fix | Calculates every metric |
| Emits **validated JSON** (function-calling) | Trains, scores, predicts |

The plan is drawn from a **fixed vocabulary** and applied by deterministic pandas code —
**no LLM-generated code is ever executed**, so every run stays auditable. Decisions arrive
as structured JSON, never parsed out of free text.

## 🔍 Case study: Maestra caught its own mistake

On a real, open Kaggle competition
([Playground S6E6 — Predicting Stellar Class](https://www.kaggle.com/competitions/playground-series-s6e6)),
the cleaning agent confidently dropped the photometric bands `u, g, r, i, z` — **real
features** — reasoning *"unique per row → not useful."* It had over-generalized the
ID-column heuristic to continuous measurements.

The holdout metric still looked fine — the mistake was *masked* by a strong remaining
signal (`redshift`). Only a baseline comparison exposed it:

| run | balanced accuracy |
|---|---|
| `--no-llm` baseline (all features) | **0.955** |
| LLM cleaning (bands dropped) | 0.919 ❌ |
| after the fix | **0.952** ✓ |

The fix: a deterministic `id_like` profile signal that *never* flags continuous floats —
so the agent can't mistake a measurement for an identifier. The final submission scored
**0.95045** on the public leaderboard — within **0.001** of the holdout estimate,
confirming the leakage-safe pipeline gives honest numbers.

> **The lesson, baked into the design:** never trust an LLM's cleaning blind — validate
> against a baseline, and make correctness deterministic wherever you can.

## Features

- 🧠 **Agentic cleaning** — LLM proposes a drop/impute plan as constrained JSON
- 🔒 **Leakage-safe** — cleaning is *fitted on train only* (scikit-learn-style fit/transform)
- 🔁 **Self-healing** — a bounded diagnosis loop retries failures with structured recovery actions
- 🔌 **Model-agnostic** — any [LiteLLM](https://docs.litellm.ai/) backbone via one `--model` string
- 🏆 **Kaggle-ready** — produces a submission file in one command
- 🧪 **Fully tested** — 26 fast, offline tests (LLM *and* AutoGluon mocked)

## Install

Requires Python 3.9–3.12. AutoGluon's `[all]` extra is large (pulls in PyTorch); the first
install takes a while.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # then add the key for your --model backbone
```

## Usage

```bash
# basic run
maestra --csv data/titanic.csv --target Survived

# swap the backbone (any LiteLLM model string)
maestra --csv data/titanic.csv --target Survived --model claude-3-5-sonnet-latest

# self-healing: diagnose failures and retry up to 3 times
maestra --csv data/train.csv --target class --max-attempts 3

# build a Kaggle submission
maestra --csv data/train.csv --target class \
        --test data/test.csv --submission submission.csv
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gpt-4o` / `$AUTOML_MODEL` | LiteLLM model string (`claude-3-5-sonnet-latest`, `ollama/qwen2.5`, …) |
| `--time-limit` | `120` | AutoGluon training budget in seconds |
| `--test-size` | `0.2` | Holdout fraction |
| `--seed` | `42` | Split seed |
| `--no-llm` | off | Skip cleaning — baseline run (always worth comparing against) |
| `--max-attempts` | `1` | `>1` enables the failure-diagnosis loop |
| `--test` | — | Unlabeled test CSV to predict on (for a submission) |
| `--submission` | — | Output path for the submission CSV (requires `--test`) |
| `--id-col` | `id` | Identifier column carried into the submission |

### As a library

```python
import pandas as pd
from maestra import run_pipeline

df = pd.read_csv("data/titanic.csv")
result = run_pipeline(df, "Survived", model="gpt-4o",
                      test_size=0.2, time_limit=120, seed=42, model_dir="AutogluonModels")

print(result.training.metrics)     # {'accuracy': 0.826, 'roc_auc': 0.884, ...}
print(result.cleaning_log)         # every drop/impute decision, auditable
```

## The self-healing loop

With `--max-attempts > 1`, a failed attempt is handed back to the LLM, which reads the
truncated traceback and picks a **bounded** recovery action:

```
attempt ─▶ fail ─▶ LLM diagnoses ─▶ { revise_plan | increase_time_limit | give_up } ─▶ retry
```

The decision is structured JSON (no executed code) and the loop can't spin past
`--max-attempts`. AutoGluon is robust on clean data, so this rarely fires on the happy
path — it exists for genuine failures (e.g. a plan that drops every feature) and is
verified deterministically by the test suite.

## Architecture

| Module | Responsibility |
|--------|----------------|
| [`profiling.py`](src/maestra/profiling.py) | Deterministic column profile — the LLM's only view of the data |
| [`llm.py`](src/maestra/llm.py) | Thin LiteLLM wrapper; structured JSON via function-calling |
| [`cleaning.py`](src/maestra/cleaning.py) | Plan schema + defensive, leakage-safe fit/transform |
| [`diagnosis.py`](src/maestra/diagnosis.py) | LLM failure diagnosis; structured recovery actions |
| [`engine.py`](src/maestra/engine.py) | AutoGluon training, metrics & prediction — the *only* number-crunching |
| [`pipeline.py`](src/maestra/pipeline.py) | The conductor loop + bounded diagnosis/retry; returns plain data |
| [`cli.py`](src/maestra/cli.py) | Argument parsing, `.env` loading, output formatting |

## Design decisions

- **Constrained JSON, not executed code.** The LLM chooses from a fixed op-vocabulary
  (drop / impute); deterministic code applies it. Safer and fully auditable.
- **One model string, not a config zoo.** The backbone is a single `--model` flag — no
  speculative configuration system.
- **Library returns data; the CLI does I/O.** `run_pipeline` returns a dataclass; that
  separation is what makes the whole pipeline — including the retry loop — unit-testable
  with mocks, no network and no AutoGluon needed.

## Development

```bash
pytest      # 26 tests, fast & offline — LLM and AutoGluon are mocked
```

## Known limitations

- **Reproducibility.** The split is seeded, but AutoGluon's `fit` has no single global
  seed, so trained models vary slightly run to run. Fine for experiments.
- **Submission model.** A submission uses the model trained on the train split; a maximal
  leaderboard score would refit on all labeled rows.

## License

MIT — see [LICENSE](LICENSE).

<div align="center">
<sub>Built as a learning project — an LLM that conducts, an engine that plays.</sub>
</div>
