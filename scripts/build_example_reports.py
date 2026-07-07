"""Generate the three example HTML reports for the P1 10-minute path (docs/examples/reports/):

  * bike-sharing  — a run dossier on the K1 temporal setup (fold-advisor + target framing on),
  * house-prices  — a run dossier on the M6/M11 rich-semantics task (cleaning + framing),
  * grunfeld      — a pre-modelling AUDIT showing group leakage caught before any model is built.

The two run dossiers need real LLM + AutoGluon execution; Helena runs this with API keys. Offline,
the module still exposes the specs and a ``--dry-run`` that renders a SYNTHETIC dossier (no LLM, no
AutoGluon) to prove the output path — that part is what the test exercises.

    ./.venv/bin/python scripts/build_example_reports.py                # real runs (needs keys)
    ./.venv/bin/python scripts/build_example_reports.py --dry-run      # offline, synthetic
"""
from __future__ import annotations

import argparse
import os

_OUT_DIR = "docs/examples/reports"

# name, kind, csv, target, and the pipeline flags that make the example illustrative.
EXAMPLES = [
    {"name": "bike-sharing", "kind": "dossier", "csv": "data/kaggle_bike/train.csv",
     "target": "count", "cv": 3, "fold_advisor": True, "target_framing": True,
     "drop": ["casual", "registered"],
     "blurb": "Temporal demand forecasting: fold-advisor + log1p target framing (K1)."},
    {"name": "house-prices", "kind": "dossier", "csv": "data/house-prices/train.csv",
     "target": "SalePrice", "cv": 3, "fold_advisor": True, "target_framing": True,
     "blurb": "Rich-semantics regression: cleaning judgment + framing (M6/M11)."},
    {"name": "grunfeld", "kind": "audit", "csv": "data/grunfeld.csv", "target": "inv",
     "blurb": "Pre-modelling audit: group leakage (firm) caught before a model is built (M1)."},
]


def _synthetic_dossier_html() -> str:
    """A deterministic dossier from a fixed fake result — no LLM, no AutoGluon (for --dry-run
    and the offline test). Proves the render+write path without a real run."""
    import pandas as pd

    from maestra.dossier import render_dossier
    from maestra.engine import TrainingResult
    from maestra.pipeline import PipelineResult
    from maestra.validation import CVResult

    result = PipelineResult(
        n_cols_before=12, n_cols_after=9, plan={"columns_to_drop": []},
        training=TrainingResult("regression", "root_mean_squared_error",
                                pd.DataFrame({"model": ["m"]}), {}),
        cv=CVResult("root_mean_squared_error", "regression", [46.0, 44.0, 45.0], 45.0, 1.0, 3,
                    False, greater_is_better=False),
        fold_strategy={"strategy": "time", "time_column": "datetime", "group_column": None,
                       "period_column": None, "rationale": "the task forecasts future demand"},
        hybrid=[{"name": "age_of_house", "idea": "example rejected feature", "source": "profile",
                 "cv_delta": 0.0, "kept": False, "reason": "no_improvement"}],
        target_framing={"transform": "log1p", "accepted": True, "cv_delta": 3.0, "log": []},
        cv_budget={"limit": None, "trials_spent": 1}, adversarial_auc=0.53,
    )
    return render_dossier(result, verdict_sentence="Example dossier (synthetic, offline dry-run).")


def build_example(spec: dict, *, model: str, out_dir: str, time_limit: int) -> str:
    """Run the real pipeline/audit for one example and write its HTML. Returns the path.
    Needs LLM + AutoGluon; raises if the data file is missing."""
    import pandas as pd

    from maestra.config import load_dotenv
    load_dotenv()
    path = os.path.join(out_dir, f"{spec['name']}.html")
    df = pd.read_csv(spec["csv"])
    for col in spec.get("drop", []):
        df = df.drop(columns=[col], errors="ignore")

    if spec["kind"] == "audit":
        from maestra.audit import audit, write_audit_html
        report = audit(df, spec["target"], model=model, time_limit=time_limit, csv=spec["csv"])
        write_audit_html(report, path)
    else:
        from maestra.dossier import dossier_narrative, write_dossier
        from maestra.pipeline import run_pipeline
        result = run_pipeline(df, spec["target"], model=model, test_size=0.2,
                              time_limit=time_limit, seed=42,
                              model_dir=f"AutogluonModels/example_{spec['name']}",
                              cv_folds=spec.get("cv"), fold_advisor=spec.get("fold_advisor", False),
                              target_framing=spec.get("target_framing", False))
        write_dossier(result, path, **dossier_narrative(model, result))
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true",
                   help="Write a synthetic offline dossier per example (no LLM/AutoGluon).")
    p.add_argument("--model", default="gpt-4o")
    p.add_argument("--time-limit", type=int, default=120)
    p.add_argument("--out-dir", default=_OUT_DIR)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    for spec in EXAMPLES:
        path = os.path.join(args.out_dir, f"{spec['name']}.html")
        if args.dry_run:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(_synthetic_dossier_html())
            print(f"[dry-run] {path}")
        elif not os.path.exists(spec["csv"]):
            print(f"SKIP {spec['name']}: data missing ({spec['csv']})")
        else:
            print(f"building {spec['name']} ({spec['kind']}) — {spec['blurb']}")
            print(f"  wrote {build_example(spec, model=args.model, out_dir=args.out_dir, time_limit=args.time_limit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
