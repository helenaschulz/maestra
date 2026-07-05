# CLAUDE.md — Maestra

Wird jede Session komplett in den Kontext geladen. Knapp halten.

## Was Maestra ist
Ein LLM-Conductor über AutoGluon für tabulares AutoML auf CSV. Das LLM macht Vorschläge
(Cleaning, Encoding, Features, Fold-Strategie, Target-Framing), **entschieden wird per
CV-Messung, nie durch das LLM**. Topologie bewusst linear (ein Python-Loop, eine Funktion
pro Schritt), kein Agent-Framework. Python >=3.9,<3.13.

## Nicht verhandelbare Invarianten
- **Kein Leakage.** Fit nur auf Train, per-Fold-Refit aller Transforms, generierte Features
  müssen zeilenunabhängig sein (`_is_row_independent`). Bei jeder Transform-Änderung zuerst prüfen.
- **LLM entscheidet nie.** Jeder Vorschlag geht durch ein deterministisches Gate, das per CV
  misst. Der zentrale Weg dafür ist `intervention.py` (M4): jede gemessene Änderung ist eine
  Intervention. Nichts geht ungemessen in die Pipeline.
- **temperature=0** überall (`llm.py::call_structured`, forced tool-calling gegen festes Schema).
- **Holdout unantastbar.** Retries/Diagnose gaten auf internen Val-Score, nie auf Holdout.
- **`docs/RESULTS.md` ist das Mess-Ledger.** Jede Zahl-Behauptung führt auf eine Zeile dort
  zurück, inkl. negativer Ergebnisse. Neuer Claim → neue Zeile mit Beleg.

## Orientierung im Code
Die Architektur ist gerade in Bewegung. Verlass dich auf keine gecachte Modul-Liste.
Einstieg ist `pipeline.py::run_pipeline()`, der lineare Top-to-bottom-Loop. Für den Rest
`ls src/maestra/` plus den Ein-Zeilen-Docstring jedes Moduls lesen. Die Docstrings sind
die Wahrheit über Modul-Zweck, nicht dieses Dokument. Wer ein Modul ändert, hält seinen
Docstring aktuell (wandert im Diff mit).

## Commands
- Tests (offline, LLM + AutoGluon gemockt): `pytest`
- Lint: `ruff check .`
- Pipeline: `maestra --csv <file> --target <col> [flags]`  (Entry: `maestra.cli:main`)
- Audit-Deliverable: `maestra-audit ...`   Benchmark: `maestra-bench ...`   MLE-bench: `maestra-mlebench ...`
- Häufige Flags: `--hybrid`, `--skeptic`, `--fold-advisor`, `--ordinal`, `--target-framing`,
  `--text-features`, `--research`, `--no-llm`, `--no-fe`, `--cv`, `--seed`, `--compare`,
  `--report`, `--max-attempts`, `--test`, `--submission`.

## Arbeitskonventionen
- src-Layout, ein Modul pro Verantwortung. Schichten getrennt: Entscheidung / Validierung /
  Ausführung / Bewertung. Neue Logik in die passende Schicht, Grenzen nicht aufweichen.
- Tests bleiben offline: LLM und AutoGluon immer mocken. CI (`.github/workflows/ci.yml`) läuft
  bei jedem Push auf main/master.
- Optionale deps in eigenen groups (`research`, `mlebench`), Core-Install bleibt schlank.
- `.env` enthält API-Keys, nie committen/zeigen.
- Für Suche/Exploration im Code Subagents nutzen, damit große File-Dumps nicht im Hauptkontext landen.
