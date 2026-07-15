"""Step 7 runner: business angle — cost-based threshold, segmentation, playbook.

    ./venv/bin/python scripts/06_business.py

Illustrative unit costs (a real project would source these from finance):
    COST_FP = $50   wasted retention offer on a customer who would have stayed
    COST_FN = $500  a missed churner (lost customer / margin)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _msg in ("divide by zero encountered in matmul",
             "overflow encountered in matmul",
             "invalid value encountered in matmul"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)

import mlflow  # noqa: E402
import numpy as np  # noqa: E402

from src import business, data_loader, models, preprocess, tracking  # noqa: E402

SELECTED = "logreg_balanced"
COST_FP = 50.0
COST_FN = 500.0


def main() -> None:
    df = preprocess.prepare(data_loader.load_raw())
    numeric, categorical = preprocess.feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.split(df)

    spec = models.get_model_specs()[SELECTED]
    pipe = models.make_pipeline(spec["estimator"], numeric, categorical,
                                scaler=spec["scaler"], encoder=spec["encoder"])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    # --- 1. cost-based threshold ---
    table, best = business.cost_curve(y_test, proba, COST_FP, COST_FN)
    cost_default = int(table.loc[table["threshold"] == 0.5, "total_cost"].iloc[0])
    cost_best = int(best["total_cost"])
    print("=" * 66 + "\n1. COST-BASED THRESHOLD\n" + "=" * 66)
    print(f"default 0.50 -> cost ${cost_default:,}  (fn misses cost ${COST_FN:.0f} each)")
    print(f"optimal {best['threshold']:.2f} -> cost ${cost_best:,} "
          f"(tp={int(best['tp'])}, fp={int(best['fp'])}, fn={int(best['fn'])})")
    print(f"saving vs default: ${cost_default - cost_best:,}")

    # --- 2. risk x value segmentation ---
    value = X_test["MonthlyCharges"]
    vthr = float(value.median())
    seg = business.segment(value, proba, risk_threshold=best["threshold"],
                           value_threshold=vthr)
    counts = seg["segment"].value_counts()
    priority = seg[seg["segment"].str.startswith("priority")]
    rev_at_risk = priority["monthly_value"].sum()
    print("\n" + "=" * 66 + "\n2. RISK x VALUE SEGMENTS\n" + "=" * 66)
    print(counts.to_string())
    print(f"\nPriority segment: {len(priority)} customers, "
          f"${rev_at_risk:,.0f}/month revenue at risk "
          f"(value cutoff = median ${vthr:.0f})")

    # --- 3. retention playbook (drivers -> actions, from Step 6 SHAP) ---
    print("\n" + "=" * 66 + "\n3. RETENTION PLAYBOOK (SHAP driver -> action)\n" + "=" * 66)
    playbook = [
        ("Month-to-month contract", "Offer incentive to move to 1/2-year term"),
        ("Low tenure (new customer)", "Early-life onboarding & check-in"),
        ("Fibre optic + high charges", "Loyalty pricing / service-quality review"),
        ("Electronic-check payment", "Nudge to autopay (autopay churns less)"),
        ("Few add-on services", "Bundle add-ons to raise switching cost"),
    ]
    for driver, action in playbook:
        print(f"  {driver:28s} -> {action}")

    # --- figures + MLflow ---
    figs = [
        business.plot_cost_curve(table, best),
        business.plot_segments(seg, best["threshold"], vthr),
    ]
    tracking.setup_mlflow()
    with mlflow.start_run(run_name=f"business_{SELECTED}"):
        mlflow.set_tag("stage", "business")
        mlflow.log_params({"cost_fp": COST_FP, "cost_fn": COST_FN})
        mlflow.log_metric("optimal_threshold", float(best["threshold"]))
        mlflow.log_metric("cost_saving_vs_default", cost_default - cost_best)
        mlflow.log_metric("priority_customers", len(priority))
        mlflow.log_metric("monthly_revenue_at_risk", float(rev_at_risk))
        for path in figs:
            mlflow.log_artifact(str(path), artifact_path="business")

    print("\nFigures:")
    for path in figs:
        print(" ", path)


if __name__ == "__main__":
    main()
