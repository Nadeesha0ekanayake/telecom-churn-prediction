"""Step 4-5 runner: train a zoo of models, compare, pick the best.

    ./venv/bin/python scripts/04_model_zoo.py

Trains logreg_balanced + RandomForest + XGBoost + LightGBM (each with matched
preprocessing), cross-validates, evaluates on the held-out test set, logs every
run to the same MLflow experiment, and reports a ranked comparison.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

for _msg in ("divide by zero encountered in matmul",
             "overflow encountered in matmul",
             "invalid value encountered in matmul"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer",
                        category=UserWarning)
# Benign: ColumnTransformer passes a NumPy array (no names) to LightGBM at predict.
warnings.filterwarnings("ignore", message="X does not have valid feature names",
                        category=UserWarning)

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from src import data_loader, evaluate, models, preprocess, tracking  # noqa: E402


def main() -> None:
    df = preprocess.prepare(data_loader.load_raw())
    numeric, categorical = preprocess.feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.split(df)

    tracking.setup_mlflow()
    results: dict = {}

    for name, spec in models.get_model_specs().items():
        pipe = models.make_pipeline(
            spec["estimator"], numeric, categorical,
            scaler=spec["scaler"], encoder=spec["encoder"],
        )
        cv_mean, cv_std = evaluate.cross_val_auc(pipe, X_train, y_train)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate.classification_metrics(y_test, y_pred, y_proba)

        results[name] = {
            "pipeline": pipe, "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
            "test": test_metrics, "y_pred": y_pred, "y_proba": y_proba,
        }
        print(f"  trained {name:16s} cv_auc={cv_mean:.3f} test_auc={test_metrics['roc_auc']:.3f}")

        with mlflow.start_run(run_name=name):
            mlflow.set_tag("stage", "model_zoo")
            mlflow.log_params({
                "model": name, "scaler": spec["scaler"], "encoder": spec["encoder"],
                **spec["estimator"].get_params(),
            })
            mlflow.log_metric("cv_roc_auc_mean", cv_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_std)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_train, y_pred),
                input_example=X_train.head(3),
            )

    # --- ranked comparison (by CV AUC — our model-selection metric) ---
    table = pd.DataFrame({
        name: {
            "cv_roc_auc": round(r["cv_auc_mean"], 3),
            "cv_std": round(r["cv_auc_std"], 3),
            **{k: round(v, 3) for k, v in r["test"].items()},
        }
        for name, r in results.items()
    }).T.sort_values("cv_roc_auc", ascending=False)

    print("\n" + "=" * 74 + "\nMODEL ZOO — ranked by CV ROC-AUC\n" + "=" * 74)
    print(table.to_string())

    best = table.index[0]
    print(f"\nBest by CV ROC-AUC: {best} "
          f"(cv={table.loc[best, 'cv_roc_auc']}, test={table.loc[best, 'roc_auc']})")

    # --- figures ---
    print("\nFigures:")
    print(" ", evaluate.plot_roc(results, y_test, fname="07_zoo_roc.png"))
    print(" ", evaluate.plot_metric_bars(results))
    print(" ", evaluate.plot_feature_importance(results[best]["pipeline"]))


if __name__ == "__main__":
    main()
