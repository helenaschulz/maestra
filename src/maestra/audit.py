"""maestra-audit — a data-risk report to run BEFORE anyone builds a model.

This is the standalone form of Maestra's most valuable finding: the errors that kill deployed
models live in the *setup*, not the algorithm, and AutoML cannot see them. The audit answers the
questions a strong engine silently gets wrong:

  * How must this data be validated?  (random / group / time — a wrong choice inflates every
    later metric; the Validation Strategist decides from the column semantics.)
  * What looks like target leakage?   (LLM-flagged proxy/post-outcome columns.)
  * What structural traps are present? (id-like columns, near-constant columns, extreme
    missingness, high-cardinality free text — computed deterministically, no LLM.)
  * Does the test set match the training distribution? (adversarial validation, when --test given.)

It trains no predictive model (only, optionally, a throwaway adversarial classifier), so it is
fast and cheap: one structured LLM call plus a profile.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field

import pandas as pd

from maestra.config import load_dotenv
from maestra.profiling import description_context, profile_dataframe
from maestra.validation import adversarial_validation
from maestra.validation_strategist import propose_fold_strategy, validate_fold_strategy

_HIGH_MISSING = 0.5      # fraction of missing values that is a real modelling risk
_HIGH_CARD_TEXT = 0.9    # unique fraction above which an object column is likely free text / noise


@dataclass
class AuditReport:
    csv: str
    n_rows: int
    n_cols: int
    target: str
    fold_strategy: dict
    fold_log: list[str]
    leakage_warnings: list[dict]
    id_like: list[str] = field(default_factory=list)
    constant: list[str] = field(default_factory=list)
    high_missing: list[tuple] = field(default_factory=list)      # (column, missing_frac)
    high_card_text: list[str] = field(default_factory=list)
    adversarial_auc: float | None = None


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
        adversarial_auc=adv, **flags,
    )


def _adversarial_verdict(auc: float) -> str:
    if auc < 0.6:
        return "no meaningful train/test shift — a random split is representative"
    if auc < 0.8:
        return "MODERATE shift — some columns differ between train and test; check them and consider adversarial-weighted validation"
    return "STRONG shift — train and test are easily told apart; your validation will over-estimate test performance"


def render_report(r: AuditReport) -> str:
    """Render the audit as Markdown."""
    out = [f"# Data-risk audit: `{r.csv}`", "",
           f"{r.n_rows:,} rows x {r.n_cols} columns  ·  target: `{r.target}`", ""]

    out += ["## 1. How to validate this data"]
    strat = r.fold_strategy["strategy"]
    if strat == "group":
        out.append(f"**Group folds by `{r.fold_strategy['group_column']}`.** A random split would leak "
                   "an entity across train and validation — the CV would over-state real accuracy.")
    elif strat == "time":
        out.append(f"**Time-ordered folds by `{r.fold_strategy['time_column']}`.** Validate on the "
                   "future, never train on it.")
    else:
        out.append("**Random folds are appropriate** — no grouping or temporal structure detected.")
    if r.fold_strategy.get("rationale"):
        out.append(f"> {r.fold_strategy['rationale']}")

    out += ["", "## 2. Leakage risks (LLM-flagged)"]
    if r.leakage_warnings:
        out += [f"- **`{w.get('column')}`** — {w.get('reason')}" for w in r.leakage_warnings]
    else:
        out.append("_None flagged._")

    out += ["", "## 3. Structural flags (deterministic)"]
    any_flag = False
    if r.id_like:
        any_flag = True
        out.append(f"- **ID-like columns** (drop from features): {', '.join(f'`{c}`' for c in r.id_like)}")
    if r.constant:
        any_flag = True
        out.append(f"- **Constant columns** (no signal): {', '.join(f'`{c}`' for c in r.constant)}")
    if r.high_missing:
        any_flag = True
        cols = ', '.join(f"`{c}` ({f:.0%})" for c, f in r.high_missing)
        out.append(f"- **High missingness** (>50%): {cols}")
    if r.high_card_text:
        any_flag = True
        out.append(f"- **High-cardinality text** (likely free text / noise): "
                   f"{', '.join(f'`{c}`' for c in r.high_card_text)}")
    if not any_flag:
        out.append("_None._")

    out += ["", "## 4. Train/test distribution shift"]
    if r.adversarial_auc is None:
        out.append("_Not checked (no `--test` set provided)._")
    else:
        out.append(f"Adversarial AUC **{r.adversarial_auc:.3f}** — {_adversarial_verdict(r.adversarial_auc)}.")

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="maestra-audit",
                                description="Data-risk report (validation strategy, leakage, "
                                            "structural traps) — run before building a model.")
    p.add_argument("--csv", required=True, help="Training CSV to audit.")
    p.add_argument("--target", required=True, help="Target column.")
    p.add_argument("--test", help="Optional test CSV, for a train/test distribution-shift check.")
    p.add_argument("--description", help="Optional dataset-description file (column semantics).")
    p.add_argument("--model", default="gpt-4o", help="LiteLLM model for the Validation Strategist.")
    p.add_argument("--time-limit", type=int, default=30, help="Budget for the adversarial classifier.")
    p.add_argument("--out", help="Write the Markdown report here (default: stdout).")
    args = p.parse_args(argv)
    load_dotenv()

    train_df = pd.read_csv(args.csv)
    test_df = pd.read_csv(args.test) if args.test else None
    description = open(args.description).read() if args.description else None

    report = audit(train_df, args.target, model=args.model, test_df=test_df,
                   description=description, time_limit=args.time_limit, csv=args.csv)
    text = render_report(report)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"Audit written to {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
