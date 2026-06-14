# automl-agent

Agentisches AutoML für Tabellendaten. **Schritt 1 (dieser Stand): Skelett ohne LLM** —
CSV + Zielspalte rein, AutoGluon trainiert, Metriken auf einem Holdout-Set raus.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lauf

```bash
python train.py --csv data/titanic.csv --target Survived --time-limit 120
```

Flags: `--test-size` (default 0.2), `--time-limit` Sekunden (default 120), `--seed` (default 42),
`--model-dir`. AutoGluon inferiert Problemtyp und Metrik selbst.
