"""Schritt 1: CSV + Zielspalte -> AutoGluon TabularPredictor -> Metriken auf Holdout.

Kein LLM. Die Engine (AutoGluon) macht Split-freies Training, Modellwahl und Metriken;
dieses Script reicht nur Daten rein und gibt die Ergebnisse aus.
"""
import argparse

from autogluon.tabular import TabularDataset, TabularPredictor


def main() -> None:
    p = argparse.ArgumentParser(description="Train AutoGluon on a CSV and report holdout metrics.")
    p.add_argument("--csv", required=True, help="Pfad zum Trainings-CSV")
    p.add_argument("--target", required=True, help="Name der Zielspalte")
    p.add_argument("--test-size", type=float, default=0.2, help="Anteil Holdout (default 0.2)")
    p.add_argument("--time-limit", type=int, default=120, help="Trainings-Budget in Sekunden (default 120)")
    p.add_argument("--seed", type=int, default=42, help="Random Seed fuer den Split (default 42)")
    p.add_argument("--model-dir", default="AutogluonModels", help="Ablageordner fuer das Modell")
    args = p.parse_args()

    df = TabularDataset(args.csv)
    if args.target not in df.columns:
        raise SystemExit(f"Zielspalte '{args.target}' nicht im CSV. Vorhandene Spalten: {list(df.columns)}")

    test = df.sample(frac=args.test_size, random_state=args.seed)
    train = df.drop(index=test.index)
    print(f"Zeilen gesamt={len(df)}  train={len(train)}  holdout={len(test)}")

    predictor = TabularPredictor(label=args.target, path=args.model_dir).fit(
        train, time_limit=args.time_limit
    )

    print(f"\nProblemtyp (von AutoGluon inferiert): {predictor.problem_type}")
    print(f"Eval-Metrik: {predictor.eval_metric.name}")

    print("\n=== Leaderboard auf Holdout ===")
    print(predictor.leaderboard(test))

    print("\n=== Metriken bestes Modell auf Holdout ===")
    perf = predictor.evaluate(test)
    for k, v in perf.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
