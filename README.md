# automl-agent

Agentisches AutoML für Tabellendaten: ein LLM als **Dirigent** über AutoGluon als **Arbeitspferd**.
Das LLM **entscheidet** (Cleaning-Plan als strukturiertes JSON), gerechnet wird nur
deterministisch (profiling/cleaning) und in AutoGluon (Modelle/Metriken).

**Stand: Schritt 2** — LLM-Cleaning-Plan + AutoGluon-Training.

## Pipeline (`run.py`)

```
CSV laden → profilieren → LLM-Cleaning-Plan → Plan anwenden → trainieren → Holdout-Metriken
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

LLM-Key in eine `.env` (gitignored) im Projektordner legen:

```
OPENAI_API_KEY=sk-...
```

## Lauf

```bash
python run.py --csv data/titanic.csv --target Survived
```

Flags: `--model` (LiteLLM-String, default `gpt-4o` oder `$AUTOML_MODEL`; z.B.
`claude-3-5-sonnet-latest`, `ollama/qwen2.5`), `--no-llm` (Cleaning überspringen, Baseline),
`--test-size`, `--time-limit`, `--seed`, `--model-dir`.

## Dateien

- `profiling.py` — deterministisches Spalten-Profil (Input fürs LLM)
- `llm.py` — dünner LiteLLM-Wrapper, strukturiertes JSON via Function-Calling
- `cleaning.py` — Plan-Schema + deterministische Anwendung (festes Op-Vokabular)
- `automl.py` — AutoGluon-Training + Holdout-Metriken
- `run.py` — Orchestrator (die schlichte Schleife)
