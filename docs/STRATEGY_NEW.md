# STRATEGY_NEW — Maestra Produktstrategie ab 2026-07

Status: beschlossen 2026-07-05. Ersetzt die Roadmap-Teile von `STRATEGY.md` (bleibt als
historisches Dokument liegen; die dort dokumentierten Thesen und Evidenz gelten weiter).

Dieses Dokument ist das Arbeitsdokument für Cody (Claude Code). Es ist so geschrieben,
dass mehrere Aufgaben nacheinander ohne Rückfrage abgearbeitet werden können. Es gibt
bewusst keine Zeitangaben — die Reihenfolge ist verbindlich, das Tempo nicht.

---

## 0. Arbeitsanweisung für Cody

**Reihenfolge:** Meilensteine strikt top-to-bottom (P0 ✅ → 2b Übergang → P1 → P2 →
P2b-Video → P3 → P4 → F1 → F2 → F3). Innerhalb eines Meilensteins die Checkboxen in
der angegebenen Reihenfolge.
Erledigte Checkboxen in diesem Dokument abhaken (`- [x]`), das Dokument wandert im Diff mit.

**Pro Aufgabe, immer:**
1. Zuerst die betroffenen Module lesen (Docstring + Code), nicht aus dem Gedächtnis
   arbeiten. Modul-Docstrings sind die Wahrheit über Modul-Zweck.
2. Kleine Schritte: eine Checkbox = ein abgeschlossener, getesteter Zustand. Nie mehrere
   Checkboxen in einem ungetesteten Wurf.
3. Nach jeder Checkbox: `pytest` (offline, LLM + AutoGluon gemockt) und `ruff check .`
   müssen grün sein. Rot = nicht weitermachen, erst fixen.
4. Docstrings geänderter Module aktuell halten. Neue Module bekommen einen
   Ein-Zeilen-Docstring nach dem Muster der bestehenden.
5. Jede neue Zahl-Behauptung → Zeile in `docs/RESULTS.md` mit Beleg (Invariante).
6. Ein Meilenstein = ein Branch/PR. Commit-Messages nach bestehendem Stil (siehe `git log`).

**Invarianten (aus CLAUDE.md, gelten uneingeschränkt):** kein Leakage (Fit nur auf Train,
per-Fold-Refit, `_is_row_independent`), das LLM entscheidet nie (jeder Vorschlag durch
deterministisches CV-Gate via `intervention.py`), `temperature=0`, Holdout unantastbar,
Tests offline.

**Stopp-und-Fragen-Trigger (hier anhalten und Helena fragen, sonst nicht):**
- Eine Aufgabe verlangt, eine Invariante aufzuweichen.
- Ein bestehender öffentlicher API-/CLI-Vertrag müsste brechen (Flag entfernen,
  Signatur inkompatibel ändern), außer es steht explizit in der Aufgabe.
- Zwei Anweisungen in diesem Dokument widersprechen sich.
- Eine Abhängigkeit lässt sich nicht installieren oder kollidiert mit `pyproject.toml`.

**Explizit erlaubt ohne Rückfrage:** neue Module anlegen, neue optionale dep-groups,
neue Tests, neue CLI-Flags (additiv), Refactorings die alle Tests grün lassen.

---

## 1. Strategische Entscheidung (Kontext, kein Task)

**Produktidentität (S1):** Maestra ist eine Verifikationsschicht über Modellbau, kein
AutoML-Conductor. Der Pitch: "Maestra sagt dir, ob du dieser Zahl glauben darfst."
Modellbau (AutoGluon) bleibt Mittel, nicht Produkt.

**Gewichtung:** S2 (Thought Leadership) und S4 (MCP-first, agentisches Frontend über
gemessenem Backend) sind vorgezogen. S3 (DS-Komponenten-API) ist Substrat und wird
mitgebaut, wo es auf dem Weg liegt. S5 (Beratungsinstrument) ist späterer Kanal, keine
Roadmap-Position. S6 (Conductor-Ausbau) ist depriorisiert (siehe Abschnitt 11).

**Erfolgsmaßstab:** Portfolio-Wirkung (Sichtbarkeit, Gespräche, Vorführbarkeit),
nicht Nutzerwachstum. Operatives Kriterium: der **10-Minuten-Pfad** — eine unbeteiligte
Person erlebt ohne Installation alle Kernbehauptungen (README → klickbarer
Beispiel-Report → Demo-Video → optional Colab/Ledger) und kann Maestra danach in zwei
Sätzen erklären.

**Kategorienerweiterung:** genau eine — Time-Series-/Demand-Forecasting, Einstieg
verifikations-first über Backtest-Audit (F-Serie). Brücke: bike-sharing (K1) ist bereits
Demand Forecasting. Verworfen: Vision/NLP/Multimodal, Unsupervised (kein Ground Truth →
kein Arbiter), Multi-Engine-für-Accuracy. Erlaubt: Pluggability (Proxy-Engine,
User-Estimatoren, TabPFN-Spike).

Die zugehörigen Arbeitsregeln (Artefakt-Pflicht, Ledger-Bindung von Content,
Verdikte-statt-Bau-Knöpfe) stehen in `CLAUDE.md` bei den Invarianten.

---

## 2. P0 — Härtung ✅ erledigt (als N-Serie, 2026-07-05)

Die ursprünglichen P0-Aufgaben wurden als N0/N1 in `STRATEGY.md` abgearbeitet —
Status hier nur nachgeführt, Evidenz in `docs/RESULTS.md`:

- [x] **P0.1 Nadeau-Bengio-Korrektur** — erledigt als N1: `paired_delta_test` mit
      Varianz-Inflation (`1/n + test_train_ratio`), beide Aufrufer (per-Fold-Gate und
      per-Seed-M8-Verdikt), als konservative Heuristik gelabelt. Recompute gegen
      `benchmark.jsonl` ausgeführt: M6 House Prices kippt auf undecided, alles andere
      hält.
- [x] **P0.2 MDE-Ausweis** — erledigt als N1: `paired_delta_mde` +
      `MultiSeedResult.mde`; "undecided" ist von "underpowered" unterscheidbar.
- [x] **P0.3 teilweise hinfällig (N0-Prüfung):** `AutogluonModels/`/`cache/` waren
      bereits gitignored; Flag-Validierung lebt NUR in `pipeline.py` (keine
      CLI-Duplikation); `proba`/`proba_columns` sind bewusst nur im
      mlebench-Entry-Point exponiert — kein Handlungsbedarf. Diese drei Aufgaben
      entfallen.

**Noch offen aus P0** (übernommen in Abschnitt 2b): `CLAUDE.md` wieder aus dem Index
nehmen (N0 hatte es committet, Entscheidung danach: bleibt intern) und die Frage der
Top-Level-JSONL-Ablage.

---

## 2b. Übergang zu P1: N3-Abschluss & Repo-Hygiene

Dieser Abschnitt muss VOR P1 abgeschlossen sein. Kontext: Die N-Arbeit (N1, N2, N4,
N5) liegt uncommitted im Working Tree, und die K2-Battery (N3, 8 Tasks × 5 Seeds)
läuft gerade in einem separaten Terminal.

**Teil A — sofort möglich (während die Battery läuft):**  ✅ erledigt 2026-07-05/06

- [x] **Working Tree committen, in kohärenten Commits.** Erledigt: N1 (7a67df3),
      N2 (bf02a1a), N4 (a986697), N5/Docs (57e0284). `runs.jsonl`/`benchmark.jsonl`
      blieben draußen.
- [x] **`CLAUDE.md` internalisieren:** `git rm --cached` + `.gitignore`-Eintrag
      (aa7ccbb); `git ls-files | grep CLAUDE` leer, Datei auf Platte erhalten.
- [x] **`.gitignore` verifizieren:** `x/`, `mlebench_out/`, `__pycache__/` waren alle
      bereits ignoriert — nichts zu ergänzen (Annahme geprüft, wie gefordert).
- [x] **`docs/STRATEGY.md` Kopfvermerk:** eingefügt (Datei bleibt gitignored, daher
      lokal, nicht committet — als Auffälligkeit an Helena gemeldet).

**Teil B — nach Ende der Battery (alle 8 Tasks durchgelaufen):**

- [x] **N3 auswerten:** 8 Multi-Seed-Blöcke gelesen (alle 5 Seeds, `failed_seeds: []`):
      2 wins (rossmann, walmart), 6 undecided, 0 losses; santander inert (Kontrolle),
      restaurant underpowered (137 Zeilen). Fold-Advisor: `benchmark.jsonl` hat KEIN
      `fold_strategy`-Feld → aus den Records nicht verifizierbar, ob die Battery mit
      `--fold-advisor` lief; als offene Frage im Ledger notiert.
- [x] **Ledger schreiben:** K2-Abschnitt in `docs/RESULTS.md` erweitert um die
      Submission/LB-Tabelle (best_quality, getrennt von den Battery-Verdikten), die
      two-sigma-Diagnose+Wiring, und eine Anomalien/offene-Fragen-Liste. Verdikt-
      Tabelle aus der letzten Session unangetastet.
- [ ] **Schritt 6 (freigegeben 2026-07-06):** ieee-fraud-LB-Zeile ergänzen
      (Public 0.914271 / Private 0.894022, AUC, best_quality) + CV↔LB-Gap aus
      runs.jsonl; dann `runs.jsonl` + `benchmark.jsonl` + finale RESULTS-Ergänzung
      in EINEM Commit. **JSONL-Umzug: GESTRICHEN** (Entscheidung 2026-07-06: 7+
      Code/Skript/Doc-Referenzen, Kosten > Nutzen; die JSONL bleiben Wurzel-Ledger,
      Ablage wird in P4/ARCHITECTURE.md dokumentiert).
- [x] **Follow-up A: `fold_strategy`-Feld in `benchmark.jsonl`-Records** — erledigt:
      `BenchResult`/`MultiSeedResult` tragen `fold_strategy` (z. B. `"time:Date"`,
      `"group:building_id"`, `None` wenn Advisor aus), `_fold_strategy_label` +
      Logging in beiden append-Funktionen, 2 neue Tests. Nur vorwärts wirksam; die
      K2-Unverifizierbarkeit bleibt ehrlich im Ledger, keine Rekonstruktion.
- [x] **Follow-up B: Fold-Advisor in `--make-submission` durchreichen** — erledigt:
      `--fold-advisor` ist jetzt tri-state (`BooleanOptionalAction`), **default-ON für
      Submissions** (ehrliche CV↔LB-Gap), default-OFF für die Battery (unverändert),
      beides via `--fold-advisor`/`--no-fold-advisor` überschreibbar. KEIN teurer
      Re-Run gestartet — der walmart-Gap-schließt-sich-Receipt ist Helenas separate
      Entscheidung.
- [x] **`STRATEGY.md`:** N3-Zeile als done markiert (Verweis auf K2 in RESULTS.md);
      Datei ist gitignored, daher lokal.

**2b Done:** Working Tree sauber (nur laufende Battery-Artefakte offen bzw. nach
Teil B gar nichts), N3 im Ledger, `pytest`/`ruff` grün. Danach beginnt P1.

---

## 3. P1 — Evidenz-Dossier + Audit-Report als HTML

**Ziel:** Der wertvollste Output eines Runs (was wurde probiert, gemessen, abgelehnt
und warum) wird ein klickbares Artefakt. Gleiches Rendering für den Audit-Report.

**Gestaltungsregeln (gelten für beide Reports):**
- Verdikt zuerst: Ampel (grün/gelb/rot) + ein Satz in Stakeholder-Sprache, darunter
  aufklappbar (`<details>`) die DS-Evidenz. Zwei Lesetiefen in einem Dokument.
- Abgelehnte Interventionen gleichrangig sichtbar wie angenommene (inkl. Delta, MDE,
  reason).
- Kein Metrikwert ohne Übersetzung in Zieleinheiten. Die Übersetzungssätze erzeugt die
  bestehende LLM-Schicht in `report.py` (LLM erklärt, entscheidet nicht); in Tests
  gemockt.
- Ein einziges statisches HTML-File pro Report, keine externen Assets, kein
  JS-Framework (inline CSS; `<details>` reicht fürs Aufklappen).

- [ ] Neues Modul `src/maestra/dossier.py`: `render_dossier(result: PipelineResult,
      run_record: dict | None = None) -> str` (HTML-String) und
      `write_dossier(result, path)`. Datenquellen: `PipelineResult` (cv, hybrid,
      fold_strategy, skeptic, target_framing, cv_budget), InterventionOutcome-Records,
      `runlog.py`-Record. Template-Ansatz: f-Strings oder `string.Template`, KEINE
      neue Pflicht-Dependency.
- [ ] Abschnitte des Dossiers: (1) Verdikt-Kopf, (2) Setup (Datensatz, Target, Metrik,
      Fold-Strategie inkl. Advisor-Begründung), (3) Interventionen-Tabelle (Name,
      Kind, proposed_by, Delta, MDE, reason, angenommen ja/nein), (4) CV-Ergebnis mit
      per-Fold-Scores, (5) CV-Budget, (6) Limitierungen (automatisch: was war
      deaktiviert, was ist underpowered).
- [ ] `audit.py` auf dieselbe Rendering-Schicht heben: `write_audit_html(...)`.
      Stakeholder-Satz oben (Beispiel-Ton: "Eure CV-Schätzung ist vermutlich ~5x zu
      optimistisch, weil Gruppen über Folds leaken"), Evidenz darunter.
- [ ] CLI: `--report html [pfad]` für `maestra`, `--html [pfad]` für `maestra-audit`.
- [ ] Offline-Tests: Rendering aus einem fixen, synthetischen `PipelineResult` (LLM
      gemockt); Assertions auf Schlüsselinhalte (Verdikt-Satz vorhanden, abgelehnte
      Intervention gelistet, kein roher Metrikname ohne Übersetzungssatz).
- [ ] Skript `scripts/build_example_reports.py`: erzeugt Beispiel-Reports für
      bike-sharing (K1-Setup), House Prices (M6/M11) und einen Group-Leakage-Fall
      (Grunfeld aus M1-Real), Ablage unter `docs/examples/reports/`. Die Läufe
      brauchen echte LLM/AutoGluon-Ausführung — Skript bereitstellen und testen;
      die eigentliche Generierung führt Helena mit API-Keys aus.
- [ ] GitHub-Pages-Workflow (`.github/workflows/pages.yml`), der `docs/examples/`
      publiziert. README bekommt die drei Links (Platzhalter, bis P4 das README
      umbaut).

**P1 Done:** drei klickbare Beispiel-Reports; mindestens einer zeigt eine abgelehnte
Intervention mit Begründung; Tests grün.

---

## 4. P2 — MCP-Server

**Ziel:** Maestra als Werkzeug-Backend für LLM-Frontends (Claude Desktop/Code). Der
Nicht-DS-Kanal: Verdikte konsumieren, nie Modelle bauen.

**Designregeln:** meinungsstarke Defaults, keine wählbaren Parameter außer Pfad und
Target. Konservative CV-Einstellungen, hartes Zeit- und CV-Budget. Ablehnung bei zu
kleinen Daten oder untragbarem Target ist ein reguläres Ergebnis mit Begründung.
Rückgaben sind strukturierte Verdikt-Records (dict), nie Modelle.

- [ ] Optionale dep-group `mcp` in `pyproject.toml` (FastMCP / `mcp`-SDK). Core-Install
      bleibt schlank.
- [ ] Neues Modul `src/maestra/mcp_server.py` mit Entry-Point `maestra-mcp` in
      `pyproject.toml`. Tool-Beschreibungen sind Prompts: präzise formulieren, mit
      je einem Beispiel; sie entscheiden, ob das Frontend die Tools richtig einsetzt.
- [ ] Tool 1 `audit_csv(path: str) -> dict`: führt den bestehenden Audit aus, liefert
      Verdikt-Record (Ampel, Stakeholder-Satz, Befunde mit Evidenz) + Pfad zum
      HTML-Report (P1-Rendering).
- [ ] Tool 2 `check_validation(path: str, target: str) -> dict`:
      Fold-Strategie-Empfehlung (nutzt `validation_strategist` + adversarial
      validation), quantifizierter Optimismus des naiven Splits, empfohlener Splitter
      als benennbare Konfiguration (z. B. `{"strategy": "group",
      "column": "customer_id"}`).
- [ ] Tool 3 `feasibility(path: str, target: str) -> dict`: intern konservativer
      Pipeline-Lauf (feste Flags, hartes Budget), Rückgabe ist die Antwort, nicht das
      Modell: erreichbare Güte in Zieleinheiten übersetzt, stärkste Treiber
      (Feature-Importance der Engine), größte Risiken (aus Audit), was die Güte
      verbessern würde. Bei untragbarem Setup: strukturierte Ablehnung mit Begründung.
- [ ] Guardrails als Code: maximale Laufzeit pro Tool-Call (konfigurierbar; Default
      z. B. 300s für `feasibility`, 60s für die anderen), Mindest-Zeilenzahl, Abbruch
      mit verständlicher Meldung statt Traceback.
- [ ] Offline-Tests für alle drei Tools (LLM/AutoGluon gemockt): Happy Path,
      Ablehnungsfall, Guardrail-Fall.
- [ ] Doku `docs/MCP.md`: Installation, Konfiguration in Claude Desktop/Code,
      Beispiel-Dialog.

**P2 Done:** Server startet lokal, drei Tools mit Tests, Doku vorhanden.

---

## 5. P2b — Demo-Video (eigener Meilenstein, Helena-geführt)

**Ziel:** ~3 Minuten Screencast: Demand-CSV in Claude, Frage "können wir die Nachfrage
vorhersagen, und darf ich der Zahl trauen?", Maestra-Tools liefern das gemessene
Verdikt, Claude erklärt. Das Video ist das zentrale Vorführ-Artefakt des Portfolios.

Cody-Anteile:
- [ ] Demo-Datensatz vorbereiten: bike-sharing-Ausschnitt als handliche CSV
      (`docs/examples/demo/demand.csv`, wenige MB, mit README-Zeile zur Herkunft).
- [ ] Drehbuch `docs/examples/demo/SCRIPT.md`: exakte Prompts, erwartete Tool-Calls,
      erwartete Verdikte, Timing-Hinweise. So konkret, dass Helena beim Aufnehmen nur
      ablesen muss.
- [ ] Generalprobe: die Drehbuch-Prompts einmal gegen den laufenden MCP-Server
      ausführen und prüfen, dass die Verdikte den Erwartungen entsprechen;
      Abweichungen im Drehbuch korrigieren.

Helena-Anteile (nicht Cody): Aufnahme, Schnitt, Hosting (z. B. YouTube unlisted),
Link ins README.

**P2b Done:** Drehbuch verifiziert, Video aufgenommen und im README verlinkt.

---

## 6. P3 — `compare()` + engine-agnostischer Arbiter + Colab

**Ziel:** Der Arbiter als generisches DS-Werkzeug: zwei beliebige sklearn-kompatible
Pipelines ehrlich vergleichen. Dafür wird die Engine-Schicht pluggable.

- [ ] Engine-Protokoll definieren (`engine.py`): minimales Interface `fit(X, y)` /
      `predict(X)` (+ `predict_proba` optional), plus Adapter `AutoGluonEngine`
      (bestehendes Verhalten, unverändert Default der Pipeline) und
      `SklearnEngine(estimator)` (wrappt ein beliebiges Estimator-Objekt via
      `sklearn.base.clone` pro Fold).
- [ ] `validation.py::cross_validate` nimmt einen Engine-Parameter statt hart
      AutoGluon. Alle bestehenden Aufrufer explizit auf `AutoGluonEngine` setzen →
      Verhalten identisch. **Regressionskriterium: alle bestehenden Tests bleiben ohne
      inhaltliche Anpassung grün.**
- [ ] Leichte Proxy-Engine: `LightGBMEngine` als Default für schnelle Checks
      (`check_validation`, künftige Gates optional). Vorher prüfen, ob LightGBM
      bereits transitiv über AutoGluon vorhanden ist — dann direkt nutzen, sonst
      optionale dep-group `fast`. Befund im Commit dokumentieren.
- [ ] Public API in `src/maestra/__init__.py`: `compare(estimator_a, estimator_b,
      df: pd.DataFrame, target: str, *, cv: int = 5, seeds: int = 1,
      metric: str | None = None) -> CompareResult`. `CompareResult`: Verdikt
      (`improved | no_improvement | underpowered`), mean_delta,
      per-Fold/Seed-Deltas, MDE, menschenlesbares `summary()` (Markdown-String,
      geeignet zum Einfügen in eine PR-Beschreibung). Nutzt `paired_delta_test` mit
      Nadeau-Bengio (P0.1) und MDE (P0.2). Kein LLM-Call nötig.
- [ ] `check_validation(df, target)` und `audit(df, target)` ebenfalls als Public API
      mit DataFrame-Input (dünne Wrapper um Bestehendes; CSV-Laden bleibt CLI-Sache).
- [ ] Colab-Notebook `docs/examples/compare_quickstart.ipynb`: zwei sklearn-Pipelines
      auf einem kleinen öffentlichen Datensatz, `compare()` in ~5 Zellen, kurze
      Laufzeit, ohne AutoGluon-Installation. Notebook wird in CI nicht ausgeführt
      (Vermerk im Notebook-Kopf); stattdessen Smoke-Test `tests/test_public_api.py`
      mit zwei sklearn-Dummies.
- [ ] Prüfen, ob AutoGluon zur optionalen Dependency werden kann, ohne bestehende
      Einstiege zu brechen (Import-Struktur, Fehlermeldung bei fehlender
      Installation). Wenn ja: umsetzen. Wenn nein: Befund dokumentieren und an Helena
      geben — nicht auf eigene Faust brechen (Stopp-Trigger API-Vertrag).
- [ ] Optionaler, streng timeboxter Spike: TabPFN als Gate-Engine für kleine
      Datensätze (`TabPFNEngine`). Ergebnis als Ledger-Zeile in `docs/RESULTS.md`,
      egal wie es ausgeht. Bei Setup-Problemen: abbrechen, Befund notieren.

**P3 Done:** `from maestra import compare` funktioniert mit reinen sklearn-Estimatoren;
Colab-Notebook läuft; alle Alt-Tests unverändert grün.

---

## 7. P4 — README-Reframe + Case Study + Architektur-Writeup

**Ziel:** Die Substanz lesbar machen. Cody entwirft, Helena redigiert.

- [ ] README umbauen, Reihenfolge: (1) These in zwei Sätzen ("gemessenes Urteil statt
      LLM-Meinung"; das LLM entscheidet nie), (2) der 10-Minuten-Pfad (Report-Links
      aus P1, Video aus P2b, Colab aus P3), (3) Kern-Evidenz als kompakte Tabelle mit
      Ledger-Verweisen, (4) Vokabular-Mapping in Marktbegriffe: structured outputs
      (`llm.py`), retrieval-augmented research (`research.py` + `websearch.py`),
      Multi-Agent mit empirischer Konfliktlösung (Skeptic/Strategist/Diagnosis),
      Guardrails (Sandbox, `_is_row_independent`, CVBudget), Eval-Harness (Arbiter,
      Multi-Seed, Kontrollexperimente), MCP (P2), (5) FE-Ergebnisse ehrlich als
      "measured null, kept for reproducibility", (6) Quickstart/Install.
      Beim bike-sharing-Ergebnis (−71%) den Kausalitäts-Caveat direkt an die Zahl:
      der Gewinn entstand durch das Beheben dreier per Anomalie gefundener Defekte,
      nicht durch "LLM-Intelligenz".
- [ ] Case Study `docs/case_studies/bike_sharing.md` (~2 Seiten), erzählt als
      Demand-Forecasting-Fall: Ausgangslage, der 3-Bug-Hunt, der CV↔LB-Gap als
      Wahrheitssignal, was das über Backtest-Ehrlichkeit lehrt. Alle Zahlen mit
      Ledger-Verweis.
- [ ] `docs/ARCHITECTURE.md` (~2 Seiten): der lineare Loop, das Gate-Design
      (`intervention.py` als einziges Mess-Primitiv), warum kein Agent-Framework
      (Auditierbarkeit, Debugbarkeit, triviale Topologie), Schichtentrennung
      Entscheidung/Validierung/Ausführung/Bewertung. Bestehendes Architekturdiagramm
      aus `assets/` einbinden.
- [ ] Konsistenz-Pass: jede Zahl in README/Case Study gegen `docs/RESULTS.md` prüfen
      (Invariante). Diskrepanzen nicht stillschweigend fixen, sondern auflisten und
      an Helena geben.

**P4 Done:** 10-Minuten-Pfad vollständig; Testlauf mit einer unbeteiligten Person
(Helena organisiert); die Person kann Maestra in zwei Sätzen erklären.

### Begleitspur S2 (parallel, kein Code, Helena)

Post 1 (ab P1): "Empirischer Arbiter statt LLM-Judge" — setzt die These.
Post 2 (nach P4): "Wie Single-Runs lügen" — die drei Multi-Seed-Flips.
Post 3 (vor F1): "Wo LLMs nichts beitragen" — FE-Nullresultate + LATTEArena.
Talk-Einreichung ("Empirischer Arbiter, Forecasting als Testfall") sobald zwei Posts
draußen sind. Jedes Stück verlinkt seine Ledger-Zeilen.

---

## 8. F1 — Backtest-Audit (Forecasting, verifikations-first)

**Ziel:** Temporale Leakage-Detektion und Backtest-Design-Prüfung für bestehende
Forecasting-Setups. Kein Modellbau. Die These ("Zahlen lügen dort am überzeugendsten,
wo das Validierungsdesign falsch ist") wird auf Forecasting getestet. Volle
Forecasting-Pipeline erst, wenn F1/F2 den Transfer belegen — ein Nein wäre ebenfalls
publizierbares Ergebnis.

**LLM-Rolle wie überall:** liest Spaltensemantik und schlägt Verdachtsmomente vor;
entschieden wird per Messung (naiver Backtest vs. korrigierter Backtest, gleiche
Arbiter-Regel aus `intervention.py`).

- [ ] Neues Modul `src/maestra/backtest_audit.py`. Input: DataFrame, Zeitspalte,
      Target, optional Serien-ID-Spalte. Checks:
      (a) **Zukunfts-Features:** je Spalte prüfen, ob Werte zum Forecast-Zeitpunkt
      verfügbar wären (LLM klassifiziert Verfügbarkeit aus Spaltensemantik, ein
      deterministischer Timing-/Korrelations-Check validiert; Muster von
      `validation_strategist.py` übernehmen);
      (b) **Split-Design:** fehlender Gap/Embargo zwischen Train und Test; bei
      Serien-ID: leakt ein globales Modell über Serien (adversarial validation über
      die Zeitgrenze, bestehende Maschinerie aus `validation.py` nutzen);
      (c) **Target-Framing bei Counts:** log1p-Prüfung aus M11/K1 wiederverwenden
      (`target_framing.py`).
- [ ] Messprimitiv `quantify_backtest_lie(...)`: Differenz zwischen naivem
      Backtest-Score und korrigiertem Backtest-Score, mit Unsicherheit (mehrere
      Origins = Paare für `paired_delta_test`).
- [ ] Datensätze: M5 (Walmart), Rossmann, Favorita — alle mit Kaggle-LB als Ground
      Truth. Download/Grading über die bestehende K1-Infrastruktur
      (`scripts/kaggle_battery.py`) — wiederverwenden, nicht duplizieren. Neues
      Skript `scripts/backtest_audit_battery.py`.
- [ ] MCP-Tool 4 `audit_backtest(path, time_column, target, series_column=None) ->
      dict` im P2-Server ergänzen; HTML-Report über P1-Rendering.
- [ ] CLI: `maestra-audit --backtest --time-col <col> [--series-col <col>]`.
- [ ] Offline-Tests: synthetische Datensätze mit eingebauten Lügen (ein
      Zukunfts-Feature, ein fehlender Gap, ein Serien-Leak) → Audit findet alle drei;
      ein sauberer synthetischer Datensatz → Audit meldet nichts
      (False-Alarm-Kontrolle).

**F1 Done:** Auf mindestens einem öffentlichen Datensatz eine Backtest-Lüge gefunden
und quantifiziert, die ein naives Setup übersieht, mit Kaggle-LB als Beleg →
Ledger-Zeilen + Post. Wenn auf keinem der drei Datensätze etwas Substanzielles
gefunden wird: ebenfalls Ledger-Zeile + ehrlicher Post, und F2-Scope wird mit Helena
neu bewertet.

---

## 9. F2 — Rolling-Origin-CV + lokale/wiederholende Time-Splits

**Ziel:** Den offenen K1-Faden schließen: bike-sharing zeigt, dass globaler Time-Split
den Pessimismus überschätzt (+0.105 RMSLE), weil der echte Split lokal pro Monat ist
(Tage 20–Ende). Das Fold-Vokabular bekommt die fehlende Granularität.

**Stand nach N2 (2026-07-05, siehe STRATEGY.md/RESULTS.md):** `time_local` ist
GEBAUT und mechanisch bestätigt (synthetisch: Gap 4.755 → 0.511). Offen ist die
Integrationslücke: der Strategist entscheidet auf dem ROH-Profil, bevor FE
`datetime` in eine Perioden-Spalte zerlegt — bike-sharing kann `time_local` deshalb
nie vorschlagen. F2 baut time_local NICHT neu, sondern schließt diese Lücke.

- [ ] **Derived period candidates beim Profiling** (der präzise Backlog-Eintrag in
      STRATEGY.md): abgeleitete Perioden-Features (month-of/week-of/day-of-week aus
      jeder datetime-artigen Spalte) als PROFILE-ONLY-Hints in
      `profiling.py`/`validation_strategist.py` sichtbar machen — sichtbar für den
      Strategist-Vorschlag, NICHT auf den DataFrame angewandt, bis `date_parts`-FE
      sie materialisiert. Deterministische Verifikation wie bisher.
- [ ] `validation.py`: `RollingOriginSplit(n_origins, horizon, gap=0)` als
      zusätzlicher Splitter mit sklearn-kompatiblem Interface
      (`get_n_splits`/`split`), damit er standalone über die P3-API nutzbar ist.
      Prüfen, ob `time_local` dasselbe Interface schon erfüllt; wenn nein,
      nachziehen.
- [ ] Messung 1: bike-sharing re-run — Erwartung: `time_local` wird jetzt
      vorgeschlagen UND der CV↔LB-Gap (−0.116) schrumpft Richtung 0.
      K1-Infrastruktur nutzen, Ergebnis als Ledger-Zeilen.
- [ ] Messung 2: mindestens ein F1-Datensatz mit demselben Vorher/Nachher.
- [ ] Regression: der erweiterte Advisor bleibt false-alarm-frei auf den bestehenden
      M9-Fällen (vorhandene Tests/Skripte erneut laufen lassen).

**F2 Done:** CV↔LB-Gap-Verbesserung mit LB-Beleg im Ledger; M9-Regression sauber; Post.

---

## 10. F3 — Demand-spezifisches Judgment (Scope nach F1/F2)

Setup-Entscheidungen, bei denen die Engine blind ist: intermittierende Nachfrage
(Metrik-/Framing-Wahl), Promotions-/Kalender-Features als **Verfügbarkeitsfrage**
(nicht FE — die FE-These ist gemessen tot), Hierarchie-Ebene der Vorhersage. Engine:
AutoGluon-TimeSeries. Bewusst noch nicht detailliert — Scope wird nach den
F1/F2-Ergebnissen mit Helena geschnitten. **Cody: hier nicht ohne Abstimmung anfangen.**

---

## 11. Depriorisiert (nicht verworfen)

- **K-Serie / Kaggle-Batteries:** weiterhin wichtig als externer Beweis (LB = Ground
  Truth); aktuell laufende Läufe werden zu Ende geführt und ins Ledger geschrieben.
  Neue K-Läufe opportunistisch und als Messinfrastruktur für die F-Serie — nur keine
  eigenständige Ausbau-Roadmap mehr.
- **Web-UI:** Etwas Visuelles zeigen zu können ist wichtig — kurzfristig übernehmen
  das die HTML-Reports (P1) und das Video (P2b). Ein eigenes Web-UI kommt auf die
  Roadmap, sobald der 10-Minuten-Pfad steht und klar ist, welche Interaktion es
  braucht, die die Reports nicht leisten.
- **FE-Lanes (`--hybrid`, `--text-features`, `--ordinal`):** eingefroren (kein Ausbau,
  Flags bleiben funktionsfähig), im README als measured null markiert. Entscheidung
  ggf. als ADR dokumentieren — Helenas Call.
- **LangGraph-Vergleichs-Spike:** optional, streng timeboxt, Ergebnis wäre eine
  halbseitige dokumentierte Bewertung. Niedrigste Priorität.
- **S5-Deliverables (Feasibility-Report als Beratungsprodukt):** entsteht als
  Nebenprodukt von P1/P2, keine eigene Roadmap-Position.

**Verworfen:** Vision/NLP/Multimodal, Unsupervised (kein Ground Truth → kein Arbiter),
Multi-Engine-für-Accuracy, Self-Serve-Modellbau für Laien, Deployment/Serving.

---

## 12. Abhängigkeiten

Abschnitt 2b (N3-Abschluss, Repo-Hygiene) → P1.
P0.1/P0.2 (✅ als N1) → P3 (Verdikt-Vokabular, MDE) und F1 (Arbiter-Regel).
P1 (Rendering) → P2 (Report-Pfade in Tool-Antworten) → P2b (Demo braucht Server) →
P4 (README verlinkt P1/P2b/P3-Artefakte).
F1 → F2 (Datensätze, Messinfrastruktur) → F3 (Scope-Entscheidung).
S2-Posts (Helena) laufen parallel, gebunden an Ledger-Zeilen, nie an Pläne.

Bei jedem Meilenstein vor dem Merge: `pytest` (offline), `ruff check .`, Docstrings
der geänderten Module aktuell, neue Zahlen als Zeile in `docs/RESULTS.md` mit Beleg.
