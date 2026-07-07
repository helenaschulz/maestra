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
- [x] **Schritt 6 (freigegeben 2026-07-06):** ieee-fraud-LB-Zeile ergänzt
      (Public 0.914271 / Private 0.894022, AUC, high_quality memory-safe; CV 0.8965
      aus runs.jsonl → Gap +0.0178 pessimistisch/sicher); `runs.jsonl` +
      `benchmark.jsonl` + finale RESULTS-Ergänzung in EINEM Commit. **JSONL-Umzug:
      GESTRICHEN** (Entscheidung 2026-07-06: 7+ Code/Skript/Doc-Referenzen, Kosten >
      Nutzen; die JSONL bleiben Wurzel-Ledger, Ablage wird in P4/ARCHITECTURE.md
      dokumentiert).
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

- [x] **`src/maestra/dossier.py`** — `render_dossier(result, *, run_record, verdict_sentence,
      metric_notes) -> str` + `write_dossier`. Pur, duck-typed auf `result` (kein Import-Zyklus),
      f-Strings + `html`-Escaping, keine neue Dependency. Die Interventionen kommen aus
      `hybrid`/`skeptic`/`target_framing` (via `collect_interventions`); MDE ist ein
      Multi-Seed-Konzept und wird — ehrlich — nur auf Run-/Verdikt-Ebene aus `run_record` gezeigt,
      nicht pro Intervention erfunden.
- [x] **Die sechs Dossier-Abschnitte** — Verdikt-Kopf (Ampel, deterministisch), Setup (inkl.
      Advisor-Begründung), Interventionen-Tabelle (abgelehnte gleichrangig sichtbar), CV mit
      per-Fold-Scores (+ MDE aus run_record), CV-Budget, auto-abgeleitete Limitierungen.
- [x] **`audit.py` auf dieselbe HTML-Schicht** — `render_audit` (in dossier.py) + `write_audit_html`;
      Risk-Level → Ampel, deterministischer Stakeholder-Satz aus dem schlimmsten Befund.
- [x] **CLI** — `maestra --dossier PATH` (LLM schreibt nur die Prosa via `dossier_narrative`,
      Fallback deterministisch), `maestra-audit --html PATH`. (Wortlaut leicht abweichend von
      "--report html": `--dossier`/`--html` sind additiv und brechen den bestehenden
      `--report`-Markdown-Vertrag nicht.)
- [x] **Offline-Tests** (`test_dossier.py`, `test_build_example_reports.py`): Verdikt-Satz da,
      abgelehnte Intervention gelistet, kein roher Metrikname ohne Übersetzung, Ampel-Farbe
      deterministisch trotz LLM-Satz. 12 Dossier- + 2 Builder-Tests.
- [x] **`scripts/build_example_reports.py`** — bike-sharing/House-Prices-Dossier + Grunfeld-Audit
      nach `docs/examples/reports/`; `--dry-run` (offline, synthetisch, getestet). Die echte
      Generierung braucht LLM/AutoGluon → **Helena führt sie mit API-Keys aus.**
- [x] **`.github/workflows/pages.yml`** (publiziert `docs/examples/`, nur committetes HTML, keine
      Secrets) + README-Platzhalterlinks.

**P1 Done — abgeschlossen (2026-07-06/07).** Helena hat die drei echten Beispiel-Reports generiert
und committet (`aca898a`): bike-sharing (GREEN, `target:log1p` abgelehnt mit Δ+Begründung sichtbar),
House Prices (GREEN, ebenfalls ein abgelehntes `target:log1p`), Grunfeld-Audit (YELLOW). Die
P1-Done-Bedingung ("mind. ein Report zeigt eine abgelehnte Intervention mit Begründung") ist damit
live erfüllt, nicht nur getestet.

**Nebenbefund beim Sichten der echten Reports, gefixt:** Grunfeld zeigte nicht den ursprünglich
geplanten reinen Group-Fall — der Strategist wählte real `time_local` (`year` + wiederkehrend pro
`firm`), ein weiterer, dritter Beleg für den K2-"competing structure"-Befund (Strategist zieht bei
gleichzeitig vorhandener Group- UND Time-Achse eine zeitbasierte Strategie vor). Kein Bug, aber
`_audit_verdict`'s Top-Zeile behandelte `time_local` wie simples `time` und unterschlug die
Perioden-Spalte — gefixt (eigener Satz, nennt beide Spalten), das bereits generierte
`grunfeld.html` NICHT überschrieben (echter bezahlter Lauf).

---

## 4. P2 — MCP-Server

**Ziel:** Maestra als Werkzeug-Backend für LLM-Frontends (Claude Desktop/Code). Der
Nicht-DS-Kanal: Verdikte konsumieren, nie Modelle bauen.

**Designregeln:** meinungsstarke Defaults, keine wählbaren Parameter außer Pfad und
Target. Konservative CV-Einstellungen, hartes Zeit- und CV-Budget. Ablehnung bei zu
kleinen Daten oder untragbarem Target ist ein reguläres Ergebnis mit Begründung.
Rückgaben sind strukturierte Verdikt-Records (dict), nie Modelle.

- [x] Optionale dep-group `mcp` in `pyproject.toml` (FastMCP / `mcp`-SDK). Core-Install
      bleibt schlank.
- [x] Neues Modul `src/maestra/mcp_server.py` mit Entry-Point `maestra-mcp` in
      `pyproject.toml`. Tool-Beschreibungen sind Prompts: präzise formulieren, mit
      je einem Beispiel; sie entscheiden, ob das Frontend die Tools richtig einsetzt.
- [x] Tool 1 `audit_csv(path: str, target: str, model="gpt-4o") -> dict`: führt den
      bestehenden Audit aus, liefert Verdikt-Record (Risk-Level, Befunde mit Evidenz)
      + Pfad zum HTML-Report (P1-Rendering). (Signatur weicht von der Spec ab: `audit()`
      selbst braucht zwingend einen Target-Namen — ohne Target ist Leak-Scan/Fold-Advisor
      nicht möglich — daher `target` als Pflichtparameter, nicht nur `path`.)
- [x] Tool 2 `check_validation(path: str, target: str, model="gpt-4o") -> dict`:
      Fold-Strategie-Empfehlung (`validation_strategist`), plus — nur wenn eine
      Nicht-Random-Strategie erkannt wird — zwei echte, gepaarte `cross_validate()`-Läufe
      (naiver Random-Split vs. empfohlene Folds, gleiche Daten/Seed) für einen
      quantifizierten, vorzeichenrichtigen Optimismus-Gap
      (`"optimistic (dangerous)"`/`"pessimistic (safe)"`/`"negligible"`). Adversarial
      Validation aus der Spec-Klammer entfällt bewusst: die Tool-Signatur nimmt nur eine
      CSV entgegen, adversarial validation braucht aber zwingend ein zweites (Test-)Set.
- [x] Tool 3 `feasibility(path: str, target: str, model="gpt-4o") -> dict`: intern
      konservativer Pipeline-Lauf (feste Flags: `cv_folds=3`, `fold_advisor=True`, kurzes
      Zeitbudget), Rückgabe ist die Antwort, nicht das Modell: erreichbare Güte in
      Zieleinheiten, stärkste Treiber (`predictor.feature_importance`, budget-begrenzt),
      größte Risiken (aus `audit()`). Bei untragbarem Setup (keine CV-Schätzung):
      strukturierte Ablehnung mit Begründung.
- [x] Guardrails als Code: `_MIN_ROWS = 50`, `_with_budget` (ThreadPoolExecutor-Backstop,
      Default 60s/90s/300s für audit_csv/check_validation/feasibility — bewusst eigene
      Werte, da STRATEGY_NEW nur Beispielzahlen "z. B." nennt), strukturierte Ablehnung
      (`{"verdict": "rejected", "reason": ...}`) statt Traceback für fehlende Datei,
      fehlendes Target, zu wenig Zeilen, Budget-Überschreitung. Ehrlich dokumentierte
      Grenze: der Backstop ist best-effort (kein Hard-Kill) — der eigentliche Kostenrahmen
      bleibt AutoGluons eigenes `time_limit`.
- [x] Offline-Tests für alle drei Tools (LLM/AutoGluon gemockt): Happy Path,
      Ablehnungsfall (fehlende Datei/Target/zu wenig Zeilen), Guardrail-Fall (Timeout via
      Toy-Funktion), plus Tool-2-spezifische Tests für beide Optimismus-Richtungen und
      einen Test, dass `main()` `.env` lädt. 12 Tests in `test_mcp_server.py`, übersprungen
      ohne die optionale `mcp`-Gruppe (CI installiert nur `.[dev,research]`).
- [x] Doku `docs/MCP.md`: Installation, Konfiguration in Claude Desktop/Code,
      Beispiel-Dialog.

**P2 Done — abgeschlossen (2026-07-07).** `maestra-mcp` startet lokal und bleibt auf stdio
laufen (smoke-getestet), 12 Tests grün, `docs/MCP.md` vorhanden. Nicht gemacht: P2b (Demo-Video,
eigener Meilenstein, Helena-geführt) — kein Teil dieses Abschnitts.

---

## 5. P2b — Demo-Video (eigener Meilenstein, Helena-geführt)

**Ziel:** ~3 Minuten Screencast: Demand-CSV in Claude, Frage "können wir die Nachfrage
vorhersagen, und darf ich der Zahl trauen?", Maestra-Tools liefern das gemessene
Verdikt, Claude erklärt. Das Video ist das zentrale Vorführ-Artefakt des Portfolios.

Cody-Anteile:
- [x] Demo-Datensatz vorbereiten: bike-sharing-Ausschnitt als handliche CSV
      (`docs/examples/demo/demand.csv`, wenige MB, mit README-Zeile zur Herkunft). Bewusst
      unverändert (roh) übernommen, inkl. des bekannten `casual`+`registered`→`count`-Leaks
      und der Zeitstruktur — genau das soll die Demo live von den Tools finden lassen.
- [x] Drehbuch `docs/examples/demo/SCRIPT.md`: exakte Prompts, erwartete Tool-Calls,
      erwartete Verdikte, Timing-Hinweise. So konkret, dass Helena beim Aufnehmen nur
      ablesen muss.
- [x] Generalprobe: alle drei Tools live gegen `docs/examples/demo/demand.csv` ausgeführt
      (echter `gpt-4o`-Key, echtes AutoGluon, 2026-07-07). Zwei echte Bugs im MCP-Server
      selbst gefunden und gefixt (Fixup-Commit auf `p2-mcp-server`): `feasibility`s
      Feature-Importance-Aufruf crashte auf dem rohen Input-Schema (jetzt
      `feature_stage="transformed"`, keine externen Daten nötig); `check_validation`s und
      `feasibility`s Zeit-Backstops hatten fast keinen Puffer über ihrer eigenen nominalen
      AutoGluon-Zeit (check_validation lief real in den 90s-Backstop; beide gelockert:
      150s/360s, weniger nominale Zeit pro Fold). Drehbuch mit den ECHTEN Zahlen aus dem
      korrigierten Lauf geschrieben (kein Korrekturbedarf danach — die Zahlen sind live
      reproduziert).

Helena-Anteile (nicht Cody): Aufnahme, Schnitt, Hosting (z. B. YouTube unlisted),
Link ins README.

**P2b (Cody-Anteile) — abgeschlossen (2026-07-07).** Drehbuch verifiziert gegen einen echten
Lauf. Offen: Aufnahme, Schnitt, Hosting, README-Link (Helenas Teil).

**P2b Done:** Drehbuch verifiziert, Video aufgenommen und im README verlinkt.

---

## 6. P3 — `compare()` + engine-agnostischer Arbiter + Colab

**Ziel:** Der Arbiter als generisches DS-Werkzeug: zwei beliebige sklearn-kompatible
Pipelines ehrlich vergleichen. Dafür wird die Engine-Schicht pluggable.

- [x] Engine-Protokoll definieren (`engine.py`): minimales Interface `fit(X, y)` /
      `predict(X)` (+ `predict_proba` optional, + `score(X, y)` mit fixiertem
      Higher-is-better-Vorzeichen), plus Adapter `AutoGluonEngine` (API-Symmetrie,
      standalone nutzbar; `cross_validate`s eigener AutoGluon-Pfad routet NICHT darüber, s.u.)
      und `SklearnEngine(estimator)` (wrappt ein beliebiges Estimator-Objekt via
      `sklearn.base.clone` pro Fold).
- [x] `validation.py::cross_validate` nimmt einen Engine-Parameter statt hart
      AutoGluon. Alle bestehenden Aufrufer explizit auf `AutoGluonEngine` setzen →
      Verhalten identisch. **Regressionskriterium: alle bestehenden Tests bleiben ohne
      inhaltliche Anpassung grün.** (Umsetzung: `engine=None` UND eine `AutoGluonEngine`-Instanz
      dispatchen beide auf den unveränderten Alt-Code — bestehende Aufrufer wurden NICHT
      einzeln auf ein explizites `engine=AutoGluonEngine(...)` umgestellt, da das für Dutzende
      Call-Sites ein rein kosmetisches No-Op mit echtem Regressionsrisiko wäre; ein neuer
      expliziter Regressionstest belegt die Äquivalenz stattdessen.)
- [x] Leichte Proxy-Engine: `LightGBMEngine` als Default für schnelle Checks
      (`check_validation`, künftige Gates optional). LightGBM ist bereits transitiv über
      `autogluon.tabular[all]` vorhanden (geprüft 2026-07-07: `import lightgbm` klappt ohne
      weitere Installation) — direkt genutzt, keine neue dep-group nötig.
- [x] Public API in `src/maestra/__init__.py`: `compare(estimator_a, estimator_b,
      df: pd.DataFrame, target: str, *, cv: int = 5, seeds: int = 1,
      metric: str | None = None) -> CompareResult`. `CompareResult`: Verdikt
      (`improved | no_improvement | underpowered`), mean_delta,
      per-Fold/Seed-Deltas, MDE, menschenlesbares `summary()` (Markdown-String,
      geeignet zum Einfügen in eine PR-Beschreibung). Nutzt `paired_delta_test` mit
      Nadeau-Bengio (P0.1) und MDE (P0.2). Kein LLM-Call nötig.
- [x] `check_validation(df, target)` und `audit(df, target)` ebenfalls als Public API
      mit DataFrame-Input (dünne Wrapper um Bestehendes; CSV-Laden bleibt CLI-Sache).
      **Abweichung:** `audit` wird NICHT als `maestra.audit` re-exportiert — `audit.py` ist
      selbst ein Submodul, und `from maestra.audit import audit` in `__init__.py` würde das
      Submodul verdecken (`from maestra import audit as audit_mod`, das bestehende Muster im
      Testsuite, bekäme dann die Funktion statt des Moduls) — ein echter, per Testlauf
      verifizierter Regressionsfund, nicht nur Vorsicht. `audit()` ist bereits public und
      DataFrame-Input via `from maestra.audit import audit`; nur `check_validation` (neu,
      keine Namenskollision) ist unter `maestra.check_validation` erreichbar.
- [x] Colab-Notebook `docs/examples/compare_quickstart.ipynb`: zwei sklearn-Pipelines
      auf einem kleinen öffentlichen Datensatz (`sklearn.datasets.load_diabetes`, kein
      Download), `compare()` in ~5 Zellen, kurze Laufzeit, ohne AutoGluon-Installation
      (`pip install --no-deps`). Notebook wird in CI nicht ausgeführt (Vermerk im
      Notebook-Kopf); Smoke-Test `tests/test_public_api.py` mit zwei sklearn-Dummies deckt
      denselben Importpfad (`from maestra import compare`) offline ab.
- [x] Geprüft, ob AutoGluon zur optionalen Dependency werden kann — **teils umgesetzt, teils
      an Helena zurückgegeben.** Der IMPORT ist jetzt verzögert (`TabularPredictor` wird lazy
      innerhalb der 3 Funktionen importiert, die ihn brauchen: `engine.py::train_and_evaluate`/
      `fit_predictor`, `validation.py::adversarial_validation` — die einzigen Stellen im ganzen
      Package, verifiziert per Grep). Das reicht bereits, damit `import maestra`/`compare()`
      ohne installiertes AutoGluon laufen (per echtem Test verifiziert:
      `sys.modules["autogluon"] = None` vor dem Import). Die DEPENDENCY selbst (in
      `pyproject.toml`) NICHT optional gemacht — das würde das bestehende
      "`pip install -e .` gibt eine sofort lauffähige CLI"-Versprechen für `maestra`/
      `maestra-audit`/-`bench`/-`mlebench`/-`mcp` brechen, ein Stopp-Trigger-Fall
      (API-/CLI-Vertrag). Befund an Helena: Colab braucht deshalb `pip install --no-deps`
      (im Notebook so gelöst) statt eines normalen Installs.
- [x] Optionaler, streng timeboxter Spike: TabPFN als Gate-Engine (`TabPFNEngine`) —
      **abgebrochen, Setup-Problem.** `pip install tabpfn` installiert sauber (8.0.8), aber
      `TabPFNClassifier.fit()` verlangt beim ersten Aufruf eine interaktive Lizenz-Annahme
      (Browser-Login bei priorlabs.ai + persönlicher API-Key als `TABPFN_TOKEN`) — nicht
      autonom herstellbar. Kein `TabPFNEngine` gebaut. Ledger-Zeile in `docs/RESULTS.md`.

**P3 Done:** `from maestra import compare` funktioniert mit reinen sklearn-Estimatoren;
Colab-Notebook läuft; alle Alt-Tests unverändert grün.

---

## 7. P4 — README-Reframe + Case Study + Architektur-Writeup

**Ziel:** Die Substanz lesbar machen. Cody entwirft, Helena redigiert.

- [x] README umbauen, Reihenfolge: (1) "What is Maestra" in drei Zoom-Stufen
      (Beschreibung → Ablauf eines Runs → Einordnung "automatisiert die Urteilsarbeit,
      nicht den Modellbau"; abgestimmter Text vom 2026-07-06 verbatim übernommen), (2) der
      10-Minuten-Pfad (Report-Links aus P1, Demo-Skript aus P2b — Video-Link als "pending"
      markiert, da Aufnahme Helenas Teil ist, Colab aus P3), (3) Kern-Evidenz-Tabelle mit
      Ledger-Verweisen (unverändert, bereits vorhanden), (4) neuer Abschnitt
      "Vocabulary, in market terms" (structured outputs, RAG, Multi-Agent mit empirischer
      Konfliktlösung, Guardrails, Eval-Harness, MCP), (5) FE-Ergebnisse waren bereits als
      "measured null" gerahmt (Finding 3) — unverändert, (6) Quickstart/Install (unverändert
      Position). Der Kausalitäts-Caveat beim bike-sharing-Ergebnis war schon vorhanden
      ("driven by fixing 3 of Maestra's own bugs, not a clean baseline comparison") und
      bleibt. Zusätzlich `mcp_server.py`/`compare.py`/`dossier.py`/`run_memory.py` in die
      Modul-Tabelle nachgetragen (fehlten dort seit P1/P2/P3 — ein eigener Konsistenzfund).
- [x] Case Study `docs/case_studies/bike_sharing.md` (~2 Seiten), erzählt als
      Demand-Forecasting-Fall: Ausgangslage, der 3-Bug-Hunt, der CV↔LB-Gap als
      Wahrheitssignal, was das über Backtest-Ehrlichkeit lehrt. Alle Zahlen mit
      Ledger-Verweis, gegengeprüft (siehe Konsistenz-Pass unten).
- [x] `docs/ARCHITECTURE.md` (~2 Seiten): der lineare Loop, das Gate-Design
      (`intervention.py` als einziges Mess-Primitiv), warum kein Agent-Framework
      (Auditierbarkeit, Debugbarkeit, triviale Topologie), Schichtentrennung
      Entscheidung/Validierung/Ausführung/Bewertung. Bestehendes Architekturdiagramm
      aus `assets/` eingebunden.
- [x] Konsistenz-Pass (Subagent, vollständig über README.md + die neue Case Study gegen
      `docs/RESULTS.md`): **alle** graded-experiment-Zahlen (M1/M2/M6/M9/M9-extend/M10/M11/
      E1/E2/K1/N1/N2) stimmen exakt überein. **Eine echte, vorbestehende Diskrepanz
      gefunden** (nicht durch diese Session eingeführt, nicht stillschweigend gefixt — an
      Helena): der bestehende "Case study: Maestra caught its own mistake"-Abschnitt
      (Playground S6E6) nennt "0.955 → 0.919" als Vorher/Nachher-Zahlen — `docs/RESULTS.md`
      führt zu S6E6 nur eine einzige Zeile ("public 0.95045 ≈ holdout 0.9516", ein anderes
      Zahlenpaar, der FINALE Submission-Stand, nicht der Vorher/Nachher-Vergleich). Die
      0.955/0.919-Zahlen sind unbelegt — entweder in RESULTS.md nachtragen (falls aus
      `runs.jsonl` rekonstruierbar) oder im README korrigieren/entfernen. Nicht selbst
      entschieden. Der illustrative CLI-Transkript-Auszug (Titanic-Beispiel, `accuracy: 0.826`
      etc.) ist bewusst kein Ledger-Claim (kein Seed/Datum genannt, reine Beispielausgabe) —
      nicht als Diskrepanz gewertet.

### Textvorlage "What is Maestra" (abgestimmt 2026-07-06, verbatim verwenden)

README-Kopf (englisch):

> **Maestra is an agentic AutoML system for tabular data.** Give it a dataset and a
> target, and it delivers predictive models together with a trustworthy estimate of
> the achievable performance. Specialized LLM agents read the semantics of your data
> and surface the risks that sink real-world ML projects: data leakage, temporal and
> group structure, flawed validation design. Every agent decision must pass an
> empirical gate — only interventions that measurably improve results in a controlled
> experiment are adopted; the LLM itself never decides. The result is a model backed
> by auditable evidence — or a reasoned refusal when the data cannot support the
> question.
>
> Maestra doesn't automate model building — modern AutoML engines already do that.
> It automates the senior data scientist's judgment around it: risk detection,
> validation design, honest expectation-setting. That is the blind spot of every
> AutoML pipeline, and the estimates' reliability is demonstrated against external
> ground truth, including real Kaggle leaderboards.

Abgestimmte Positionierungssätze (für Posts/LinkedIn/Gespräche, nicht README-pflichtig):
- DS/technisch: "Maestra ist ein agentisches ML-System über AutoGluon, in dem jede
  LLM-Entscheidung ein empirisches Gate passieren muss: paired CV auf identischen
  Folds, Multi-Seed-Verdikte, ausgewiesener minimal detektierbarer Effekt."
- Executive/Consulting: "Maestra beurteilt vor der Modellentwicklung, ob Daten eine
  Fragestellung tragen, und liefert eine belastbare Schätzung der erreichbaren Güte.
  Die Verlässlichkeit dieser Schätzungen ist an externen Referenzdaten nachgewiesen,
  unter anderem auf echten Kaggle-Wettbewerben."
- Hero-Zeile: "Measured judgment, not model opinion: an agentic ML system where
  every decision has to earn its place through evidence."

**P4 (Cody-Anteile) — abgeschlossen (2026-07-07).** README/Case-Study/Architektur-Writeup stehen,
Konsistenz-Pass gelaufen (ein Fund, an Helena). Offen (Helenas Teil, nicht Cody): der
10-Minuten-Pfad ist inhaltlich vollständig, aber der Testlauf mit einer unbeteiligten Person
braucht Helena zu organisieren — kann erst nach dem P2b-Video (noch nicht aufgenommen) sinnvoll
stattfinden.

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

- [x] Neues Modul `src/maestra/backtest_audit.py`. Input: DataFrame, Zeitspalte,
      Target, optional Serien-ID-Spalte. Checks:
      (a) **Zukunfts-Features:** je Spalte prüfen, ob Werte zum Forecast-Zeitpunkt
      verfügbar wären (LLM klassifiziert Verfügbarkeit aus Spaltensemantik, ein
      deterministischer Timing-/Korrelations-Check validiert; Muster von
      `validation_strategist.py` übernommen);
      (b) **Split-Design:** fehlender Gap/Embargo zwischen Train und Test (naiver vs.
      embargoed Backtest über mehrere rollierende Origins); bei Serien-ID: leakt ein
      globales Modell über Serien (adversarial validation über die Zeitgrenze,
      bestehende Maschinerie aus `validation.py` genutzt);
      (c) **Target-Framing bei Counts:** log1p-Prüfung aus M11/K1 wiederverwendet
      (`target_framing.py`, propose+verify direkt aufgerufen, CV-Arbiter bewusst NICHT
      neu implementiert — der existiert schon in `pipeline.py --target-framing`).
- [x] Messprimitiv `quantify_backtest_lie(...)`: Differenz zwischen naivem
      Backtest-Score und korrigiertem Backtest-Score, mit Unsicherheit (mehrere
      Origins = Paare für `paired_delta_test`, Nadeau-Bengio-korrigiert wie überall).
- [x] Datensätze: Walmart, Rossmann, Favorita (`store-sales`) — alle bereits lokal aus
      K1/K2, wiederverwendet, kein neuer Download. Neues Skript
      `scripts/backtest_audit_battery.py` (15k-Zeilen-Sample pro Task, 3 Origins).
- [x] MCP-Tool 4 `audit_backtest(path, target, time_column, series_column=None) ->
      dict` im P2-Server ergänzt; HTML-Report über P1-Rendering
      (`render_backtest_audit` in dossier.py).
- [x] CLI: `maestra-audit --backtest --time-col <col> [--series-col <col>]`.
- [x] Offline-Tests: synthetische Datensätze mit eingebauten Lügen (ein
      Zukunfts-Feature, ein fehlender Gap, ein Serien-Leak) → Audit findet alle drei;
      ein sauberer synthetischer Datensatz → Audit meldet nichts
      (False-Alarm-Kontrolle). 20 Tests in `test_backtest_audit.py` + 6 in
      `test_dossier.py` (Rendering) + 2 in `test_mcp_server.py` + 2 in `test_audit.py`
      (CLI).

**F1 Done (2026-07-07):** Eine echte Backtest-Lüge gefunden und quantifiziert, die ein
naives Setup übersieht — Rossmanns `Customers`-Spalte: vom LLM als Zukunfts-Feature
erkannt, deterministisch mit |corr| 0.892 zum Target bestätigt, UND strukturell
bestätigt (`Customers` fehlt in Rossmanns echter `test.csv` — derselbe
Beweis-Standard wie beim bike-sharing-`casual`/`registered`-Leak, kein Kaggle-LB
nötig für diese spezifische Art von Beweis). Ledger-Zeile + committetes Beispiel
(`docs/examples/reports/rossmann-backtest-audit.html`). Split-Design-Check: auf
keinem der 3 Tasks eine messbare Lüge bei diesem Budget (ehrlich als "underpowered",
nicht als "sauber" gemeldet). Series-Leak-Check: ~1.0 AUC auf allen 3 Tasks, aber mit
einem offenen methodischen Vorbehalt (Confound mit gewöhnlichem Zeit-Trend, nicht
zwingend Serien-spezifisches Leck) — nicht stillschweigend als sauberer Fund verkauft,
sondern als offene Frage für F2/eine künftige Iteration im Ledger notiert.

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
