"""maestra-audit — a data-risk report to run BEFORE anyone builds a model.

This is the standalone form of Maestra's most valuable finding: the errors that kill deployed
models live in the *setup*, not the algorithm, and AutoML cannot see them. The audit answers the
questions a strong engine silently gets wrong:

  * How must this data be validated?  (random / group / time — a wrong choice inflates every
    later metric; the Validation Strategist decides from the column semantics.)
  * What looks like target leakage?   (LLM-flagged proxy/post-outcome columns PLUS a
    deterministic scan for features that are numerically near-copies of the target.)
  * What structural traps are present? (id-like columns, near-constant columns, extreme
    missingness, high-cardinality free text — computed deterministically, no LLM.)
  * Does the test set match the training distribution? (adversarial validation, when --test given.)

It trains no predictive model (only, optionally, a throwaway adversarial classifier), so it is
fast and cheap: one structured LLM call plus a profile. The report opens with an executive
summary and an overall risk verdict, every finding carries a recommended action, and the output
is available in English and German (``--lang de``) — the LLM's own rationale sentences stay in
the pipeline language.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import pandas as pd

from maestra.config import load_dotenv
from maestra.profiling import description_context, profile_dataframe
from maestra.validation import adversarial_validation
from maestra.validation_strategist import propose_fold_strategy, validate_fold_strategy

_HIGH_MISSING = 0.5      # fraction of missing values that is a real modelling risk
_HIGH_CARD_TEXT = 0.9    # unique fraction above which an object column is likely free text / noise
_LEAK_CORRELATION = 0.9  # |pearson r| with the target above which a feature smells like a leak


@dataclass
class AuditReport:
    csv: str
    n_rows: int
    n_cols: int
    target: str
    fold_strategy: dict
    fold_log: list[str]
    leakage_warnings: list[dict]                                # LLM-flagged (semantic)
    target_leaks: list[tuple] = field(default_factory=list)     # deterministic: (column, |corr|)
    id_like: list[str] = field(default_factory=list)
    constant: list[str] = field(default_factory=list)
    high_missing: list[tuple] = field(default_factory=list)     # (column, missing_frac)
    high_card_text: list[str] = field(default_factory=list)
    adversarial_auc: float | None = None

    @property
    def risk_level(self) -> str:
        """Overall verdict: any leak evidence -> high; a validation-design change or strong
        train/test shift -> elevated; otherwise low."""
        if self.target_leaks or self.leakage_warnings:
            return "high"
        if self.fold_strategy.get("strategy") != "random":
            return "elevated"
        if self.adversarial_auc is not None and self.adversarial_auc >= 0.8:
            return "elevated"
        return "low"


def _structural_flags(profile: dict, target: str) -> dict:
    """Deterministic, LLM-free risk flags read straight off the column profile."""
    id_like, constant, high_missing, high_card_text = [], [], [], []
    for col in profile["columns"]:
        if col["is_target"]:
            continue
        name = col["name"]
        if col["id_like"]:
            id_like.append(name)
        if col["n_unique"] <= 1:
            constant.append(name)
        if col["missing_frac"] > _HIGH_MISSING:
            high_missing.append((name, col["missing_frac"]))
        if col["dtype"] == "object" and not col["id_like"] and col["unique_frac"] > _HIGH_CARD_TEXT:
            high_card_text.append(name)
    return {"id_like": id_like, "constant": constant,
            "high_missing": high_missing, "high_card_text": high_card_text}


def _target_leak_scan(df: pd.DataFrame, target: str) -> list[tuple]:
    """Deterministic leak scan: numeric features whose |correlation| with the target exceeds
    ``_LEAK_CORRELATION``. A feature that is a near-copy of the target is almost always recorded
    at or after the outcome — the classic silent leak the LLM can only guess at from names, but
    the data states outright. Binary targets are encoded 0/1; non-numeric targets with more
    classes are skipped (correlation is not meaningful there)."""
    y = df[target]
    if y.dtype.kind not in "iufb":
        if y.dropna().nunique() != 2:
            return []
        y = (y == y.dropna().unique()[0]).astype(float)
    leaks = []
    for col in df.columns:
        if col == target or df[col].dtype.kind not in "iufb":
            continue
        x = df[col]
        if x.dropna().nunique() <= 1:
            continue
        r = x.astype(float).corr(y.astype(float))
        if pd.notna(r) and abs(r) > _LEAK_CORRELATION:
            leaks.append((col, round(float(abs(r)), 3)))
    return sorted(leaks, key=lambda t: -t[1])


def audit(train_df: pd.DataFrame, target: str, *, model: str, test_df: pd.DataFrame | None = None,
          description: str | None = None, time_limit: int = 30, csv: str = "data") -> AuditReport:
    """Produce an :class:`AuditReport` for ``train_df``. One LLM call (the Validation Strategist);
    the adversarial train/test check runs only when ``test_df`` is given."""
    if target not in train_df.columns:
        raise ValueError(f"Target column {target!r} not in CSV. Columns: {list(train_df.columns)}")
    profile = profile_dataframe(train_df, target)
    context = description_context(description)

    proposal = propose_fold_strategy(model, profile, target, context)
    verified, fold_log = validate_fold_strategy(proposal, train_df, target)
    flags = _structural_flags(profile, target)

    adv = None
    if test_df is not None:
        adv = adversarial_validation(train_df, test_df, target, cleaning_plan=None,
                                     model_dir="AutogluonModels/audit_adversarial", time_limit=time_limit)

    return AuditReport(
        csv=csv, n_rows=len(train_df), n_cols=len(train_df.columns), target=target,
        fold_strategy=verified, fold_log=fold_log, leakage_warnings=verified.get("leakage_warnings", []),
        target_leaks=_target_leak_scan(train_df, target), adversarial_auc=adv, **flags,
    )


# --- rendering (English / German) ----------------------------------------------------

_STRINGS = {
    "en": {
        "title": "Data-risk audit",
        "meta": "{rows:,} rows x {cols} columns  ·  target: `{target}`",
        "summary": "Executive summary",
        "risk": {"high": "**Overall risk: HIGH** — leak evidence found; scores from a naive setup "
                         "will not survive deployment.",
                 "elevated": "**Overall risk: ELEVATED** — the validation design must change "
                             "before any reported score can be trusted.",
                 "low": "**Overall risk: LOW** — no leak evidence, standard validation applies."},
        "s1": "1. How to validate this data",
        "fold_group": "**Group folds by `{col}`.** A random split would leak an entity across train "
                      "and validation — the CV would over-state real accuracy.",
        "fold_group_action": "→ Action: use GroupKFold / `maestra --cv K --fold-advisor`; hold out "
                             "whole entities for any final test.",
        "fold_time": "**Time-ordered folds by `{col}`.** Validate on the future, never train on it.",
        "fold_time_action": "→ Action: use expanding-window splits; the final test must be the most "
                            "recent period.",
        "fold_random": "**Random folds are appropriate** — no grouping or temporal structure detected.",
        "s2": "2. Leakage risks",
        "leak_det": "Deterministic scan (|correlation| with the target > {thr}):",
        "leak_det_item": "- **`{col}`** — |r| = {r}. A near-copy of the target is almost always "
                         "recorded at/after the outcome.",
        "leak_det_action": "→ Action: verify WHEN each flagged column is recorded; drop everything "
                           "not available at prediction time, then re-run this audit.",
        "leak_llm": "Semantically flagged (LLM, from column meaning):",
        "leak_llm_action": "→ Action: confirm with the data owner; a proxy column invalidates every "
                           "metric computed with it.",
        "leak_none": "_None flagged._",
        "s3": "3. Structural flags (deterministic)",
        "id_like": "- **ID-like columns** (drop from features): {cols}",
        "constant": "- **Constant columns** (no signal): {cols}",
        "missing": "- **High missingness** (>50%): {cols}",
        "text": "- **High-cardinality text** (likely free text / noise): {cols}",
        "none": "_None._",
        "s4": "4. Train/test distribution shift",
        "shift_skip": "_Not checked (no `--test` set provided)._",
        "shift": "Adversarial AUC **{auc:.3f}** — {verdict}.",
        "shift_low": "no meaningful train/test shift — a random split is representative",
        "shift_mid": "MODERATE shift — some columns differ between train and test; check them and "
                     "consider adversarial-weighted validation",
        "shift_high": "STRONG shift — train and test are easily told apart; your validation will "
                      "over-estimate test performance",
        "footer": "_Generated by `maestra-audit` (no predictive model trained). The fold "
                  "recommendation and semantic leak flags come from an LLM and are deterministically "
                  "verified against the data; correlation and structural flags are fully "
                  "deterministic. Suitable as an annex to model-risk documentation._",
    },
    "de": {
        "title": "Datenrisiko-Audit",
        "meta": "{rows:,} Zeilen x {cols} Spalten  ·  Zielvariable: `{target}`",
        "summary": "Management-Zusammenfassung",
        "risk": {"high": "**Gesamtrisiko: HOCH** — Leak-Evidenz gefunden; Kennzahlen aus einem "
                         "naiven Setup werden den Produktivbetrieb nicht überleben.",
                 "elevated": "**Gesamtrisiko: ERHÖHT** — das Validierungsdesign muss geändert "
                             "werden, bevor berichteten Kennzahlen zu trauen ist.",
                 "low": "**Gesamtrisiko: NIEDRIG** — keine Leak-Evidenz, Standard-Validierung "
                        "ist angemessen."},
        "s1": "1. Wie diese Daten validiert werden müssen",
        "fold_group": "**Gruppen-Folds nach `{col}`.** Ein zufälliger Split würde eine Entität über "
                      "Training und Validierung verteilen — die CV überschätzt die echte Güte.",
        "fold_group_action": "→ Maßnahme: GroupKFold / `maestra --cv K --fold-advisor` verwenden; "
                             "für jeden finalen Test ganze Entitäten zurückhalten.",
        "fold_time": "**Zeitlich geordnete Folds nach `{col}`.** Auf der Zukunft validieren, nie "
                     "auf ihr trainieren.",
        "fold_time_action": "→ Maßnahme: Expanding-Window-Splits verwenden; der finale Test muss "
                            "der jüngste Zeitraum sein.",
        "fold_random": "**Zufällige Folds sind angemessen** — keine Gruppen- oder Zeitstruktur "
                       "erkannt.",
        "s2": "2. Leakage-Risiken",
        "leak_det": "Deterministischer Scan (|Korrelation| mit der Zielvariable > {thr}):",
        "leak_det_item": "- **`{col}`** — |r| = {r}. Eine Fast-Kopie der Zielvariable wird fast "
                         "immer beim/nach dem Ereignis erfasst.",
        "leak_det_action": "→ Maßnahme: prüfen, WANN jede markierte Spalte erfasst wird; alles "
                           "entfernen, was zum Vorhersagezeitpunkt nicht vorliegt, dann Audit "
                           "wiederholen.",
        "leak_llm": "Semantisch markiert (LLM, aus der Spaltenbedeutung):",
        "leak_llm_action": "→ Maßnahme: mit dem Daten-Owner bestätigen; eine Proxy-Spalte "
                           "entwertet jede damit berechnete Kennzahl.",
        "leak_none": "_Keine markiert._",
        "s3": "3. Strukturelle Auffälligkeiten (deterministisch)",
        "id_like": "- **ID-artige Spalten** (aus den Features entfernen): {cols}",
        "constant": "- **Konstante Spalten** (kein Signal): {cols}",
        "missing": "- **Hoher Fehlwert-Anteil** (>50%): {cols}",
        "text": "- **Hochkardinaler Text** (vermutlich Freitext / Rauschen): {cols}",
        "none": "_Keine._",
        "s4": "4. Verteilungs-Shift zwischen Train und Test",
        "shift_skip": "_Nicht geprüft (kein `--test`-Datensatz übergeben)._",
        "shift": "Adversarial-AUC **{auc:.3f}** — {verdict}.",
        "shift_low": "kein nennenswerter Train/Test-Shift — ein zufälliger Split ist repräsentativ",
        "shift_mid": "MODERATER Shift — einige Spalten unterscheiden sich zwischen Train und Test; "
                     "prüfen und ggf. adversarial gewichtete Validierung erwägen",
        "shift_high": "STARKER Shift — Train und Test sind leicht unterscheidbar; die Validierung "
                      "wird die Test-Güte überschätzen",
        "footer": "_Erstellt mit `maestra-audit` (kein Vorhersagemodell trainiert). Fold-Empfehlung "
                  "und semantische Leak-Hinweise stammen von einem LLM und werden deterministisch "
                  "gegen die Daten verifiziert; Korrelations- und Strukturprüfungen sind vollständig "
                  "deterministisch. Geeignet als Anlage zur Modellrisiko-Dokumentation._",
    },
}


def render_report(r: AuditReport, lang: str = "en") -> str:
    """Render the audit as Markdown (``lang`` in {'en', 'de'}). LLM rationale stays verbatim."""
    t = _STRINGS[lang]
    out = [f"# {t['title']}: `{r.csv}`", "",
           t["meta"].format(rows=r.n_rows, cols=r.n_cols, target=r.target), ""]

    out += [f"## {t['summary']}", t["risk"][r.risk_level], ""]

    out += [f"## {t['s1']}"]
    strat = r.fold_strategy["strategy"]
    if strat == "group":
        out.append(t["fold_group"].format(col=r.fold_strategy["group_column"]))
        out.append(t["fold_group_action"])
    elif strat == "time":
        out.append(t["fold_time"].format(col=r.fold_strategy["time_column"]))
        out.append(t["fold_time_action"])
    else:
        out.append(t["fold_random"])
    if r.fold_strategy.get("rationale"):
        out.append(f"> {r.fold_strategy['rationale']}")

    out += ["", f"## {t['s2']}"]
    if not r.target_leaks and not r.leakage_warnings:
        out.append(t["leak_none"])
    if r.target_leaks:
        out.append(t["leak_det"].format(thr=_LEAK_CORRELATION))
        out += [t["leak_det_item"].format(col=c, r=v) for c, v in r.target_leaks]
        out.append(t["leak_det_action"])
    if r.leakage_warnings:
        out.append(t["leak_llm"])
        out += [f"- **`{w.get('column')}`** — {w.get('reason')}" for w in r.leakage_warnings]
        out.append(t["leak_llm_action"])

    out += ["", f"## {t['s3']}"]
    any_flag = False
    if r.id_like:
        any_flag = True
        out.append(t["id_like"].format(cols=", ".join(f"`{c}`" for c in r.id_like)))
    if r.constant:
        any_flag = True
        out.append(t["constant"].format(cols=", ".join(f"`{c}`" for c in r.constant)))
    if r.high_missing:
        any_flag = True
        out.append(t["missing"].format(cols=", ".join(f"`{c}` ({f:.0%})" for c, f in r.high_missing)))
    if r.high_card_text:
        any_flag = True
        out.append(t["text"].format(cols=", ".join(f"`{c}`" for c in r.high_card_text)))
    if not any_flag:
        out.append(t["none"])

    out += ["", f"## {t['s4']}"]
    if r.adversarial_auc is None:
        out.append(t["shift_skip"])
    else:
        auc = r.adversarial_auc
        verdict = t["shift_low"] if auc < 0.6 else t["shift_mid"] if auc < 0.8 else t["shift_high"]
        out.append(t["shift"].format(auc=auc, verdict=verdict))

    out += ["", t["footer"]]
    return "\n".join(out) + "\n"


def _load_table(path: str) -> pd.DataFrame:
    """Load a tabular file by extension: .csv (default), .parquet, .xlsx."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def write_audit_html(report: AuditReport, path: str, *, verdict_sentence: str | None = None) -> None:
    """Render the audit on the shared HTML layer (:func:`maestra.dossier.render_audit`) and write
    it to ``path`` — a verdict-first, dependency-free static file, the clickable twin of the
    Markdown report."""
    from maestra.dossier import render_audit
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_audit(report, verdict_sentence=verdict_sentence))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-audit",
                                description="Data-risk report (validation strategy, leakage, "
                                            "structural traps) — run before building a model.")
    p.add_argument("--csv", required=True, help="Training table to audit (.csv, .parquet or .xlsx).")
    p.add_argument("--target", required=True, help="Target column.")
    p.add_argument("--test", help="Optional test table, for a train/test distribution-shift check.")
    p.add_argument("--description", help="Optional dataset-description file (column semantics).")
    p.add_argument("--model", default="gpt-4o", help="LiteLLM model for the Validation Strategist.")
    p.add_argument("--lang", choices=["en", "de"], default="en", help="Report language.")
    p.add_argument("--time-limit", type=int, default=30, help="Budget for the adversarial classifier.")
    p.add_argument("--out", help="Write the Markdown report here (default: stdout).")
    p.add_argument("--html", metavar="PATH", help="Also write a clickable HTML audit to PATH.")
    args = p.parse_args(argv)
    load_dotenv()

    train_df = _load_table(args.csv)
    test_df = _load_table(args.test) if args.test else None
    description = open(args.description).read() if args.description else None

    report = audit(train_df, args.target, model=args.model, test_df=test_df,
                   description=description, time_limit=args.time_limit, csv=args.csv)
    text = render_report(report, lang=args.lang)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"Audit written to {args.out}")
    else:
        print(text)
    if args.html:
        write_audit_html(report, args.html)
        print(f"HTML audit written to {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
