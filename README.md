# automl-agent

An **LLM conductor** over [AutoGluon](https://auto.gluon.ai/) as the **workhorse** for
tabular AutoML on CSV data.

The guiding split is strict:

- **The LLM decides.** It inspects a compact column profile and returns a structured
  *cleaning plan* (which columns to drop, how to impute missing values) as validated
  JSON via function-calling — never free text we have to parse.
- **The engine computes.** All model search, hyperparameter tuning and metric
  calculation happen inside AutoGluon. The LLM never does arithmetic.

The LLM's plan is drawn from a fixed vocabulary and applied by deterministic pandas code
— **no LLM-generated code is ever executed**, so every run stays auditable.

## Pipeline

```
split  →  profile(train)  →  LLM cleaning plan  →  fit on train + apply to both  →  train  →  evaluate
```

One function per step, orchestrated by a plain Python loop ([`pipeline.py`](src/automl_agent/pipeline.py)).
No agent framework — the whole flow is readable top to bottom.

The split happens **first**, and the cleaning plan is *fitted on the training rows only*
(scikit-learn style): imputation values are train statistics applied unchanged to the
holdout. Computing them over the full dataset would leak test information and inflate the
reported metrics.

### Failure-diagnosis loop (`--max-attempts`)

With `--max-attempts > 1` the pipeline becomes agentic: if an attempt fails, the LLM
reads the truncated traceback plus run context and picks a **bounded recovery action**
from a fixed vocabulary — `revise_plan`, `increase_time_limit`, or `give_up` — and the
loop retries. The decision is structured JSON (no executed code); the loop is bounded by
`--max-attempts` so it can't spin.

```
attempt → fail → LLM diagnoses → {revise_plan | increase_time_limit | give_up} → retry
```

> **Note.** AutoGluon is robust on clean tabular data, so the happy path rarely triggers
> this loop. It exists for genuine failures — e.g. a plan that drops every feature
> column, or a time/resource shortfall. The loop's behaviour is verified deterministically
> by the test suite (engine fails on attempt 1, LLM diagnoses, attempt 2 succeeds).

## Install

Requires Python 3.9–3.12. AutoGluon's `[all]` extra is large (pulls in PyTorch); the
first install takes a while.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"      # editable install incl. test deps
```

Set the API key for your chosen backbone in a local `.env` (auto-loaded, git-ignored):

```bash
cp .env.example .env
# edit .env -> OPENAI_API_KEY=sk-...
```

## Usage

```bash
automl-agent --csv data/titanic.csv --target Survived
```

Example output (abridged):

```
=== LLM cleaning plan (gpt-4o) ===
{ "columns_to_drop": [ {"column": "PassengerId", "reason": "ID-like, no signal"}, ... ],
  "imputations":     [ {"column": "Age", "strategy": "median", ...} ], ... }

=== Applied ===
  DROP 'PassengerId' -- ID-like column ...
  IMPUTE 'Age' [median] 177 Werte -> 28.0 -- ...
Columns after cleaning: 8 (from 12)

=== Best-model metrics on holdout ===
  accuracy: 0.826
  roc_auc:  0.884
```

### Kaggle submission

Pass an unlabeled test set and an output path to also produce a submission file. The
test set is cleaned with the *same* fitted transform as training, and its id column is
carried through even though cleaning drops it from the features:

```bash
automl-agent --csv data/train.csv --target class \
  --test data/test.csv --submission data/submission.csv
```

This writes `id,<target>` rows (Kaggle format). The model used is the one trained on the
train split — for a maximal leaderboard score you'd refit on all labeled rows.

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--model` | `gpt-4o` / `$AUTOML_MODEL` | LiteLLM model string (`claude-3-5-sonnet-latest`, `ollama/qwen2.5`, …) |
| `--time-limit` | `120` | AutoGluon training budget in seconds |
| `--test-size` | `0.2` | Holdout fraction |
| `--seed` | `42` | Split seed |
| `--no-llm` | off | Skip the cleaning step (baseline run) |
| `--max-attempts` | `1` | Attempts before giving up; `>1` enables the failure-diagnosis loop |
| `--test` | — | Unlabeled test CSV to predict on (for a submission) |
| `--submission` | — | Output path for the submission CSV (requires `--test`) |
| `--id-col` | `id` | Identifier column carried into the submission |

The backbone is **model-agnostic** via [LiteLLM](https://docs.litellm.ai/) — switch
provider with `--model`; only the matching API key needs to be set.

### As a library

```python
import pandas as pd
from automl_agent import run_pipeline

df = pd.read_csv("data/titanic.csv")
result = run_pipeline(df, "Survived", model="gpt-4o",
                      test_size=0.2, time_limit=120, seed=42, model_dir="AutogluonModels")
print(result.training.metrics)
```

## Development

```bash
pytest          # fast, offline — LLM and AutoGluon are mocked
```

## Project layout

| Module | Responsibility |
|--------|----------------|
| `profiling.py` | Deterministic column profile (the LLM's input) |
| `llm.py` | Thin LiteLLM wrapper; structured JSON via function-calling |
| `cleaning.py` | Plan schema + defensive, leakage-safe fit/transform |
| `diagnosis.py` | LLM failure diagnosis; structured recovery actions |
| `engine.py` | AutoGluon training, holdout metrics + prediction (the only number-crunching) |
| `pipeline.py` | The conductor loop + bounded diagnosis/retry; returns structured results |
| `cli.py` | Argument parsing, `.env` loading, output formatting |

## Known limitations

- **Run-to-run reproducibility.** The split is seeded (`--seed`), but AutoGluon's `fit`
  has no single global seed, so trained models can vary slightly between runs. Fine for
  a learning experiment; a production setup would pin per-model seeds.

## License

MIT — see [LICENSE](LICENSE).
