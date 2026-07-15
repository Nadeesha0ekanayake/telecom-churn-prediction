"""Step 6 runner: SHAP interpretation of the selected model.

    ./venv/bin/python scripts/05_shap.py

Refits the selected model (logreg_balanced), computes SHAP values on the test set,
and saves global (beeswarm, bar) + local (waterfall) explanations. Logs the figures
to an MLflow run tagged stage=interpret.
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

from src import data_loader, interpret, models, preprocess, tracking  # noqa: E402

SELECTED = "logreg_balanced"


def main() -> None:
    df = preprocess.prepare(data_loader.load_raw())
    numeric, categorical = preprocess.feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.split(df)

    spec = models.get_model_specs()[SELECTED]
    pipe = models.make_pipeline(spec["estimator"], numeric, categorical,
                                scaler=spec["scaler"], encoder=spec["encoder"])
    pipe.fit(X_train, y_train)

    expl = interpret.explain_linear(pipe, X_train, X_test)

    # Pick a confidently-correct churner and non-churner for the local views
    proba = pipe.predict_proba(X_test)[:, 1]
    y = y_test.to_numpy()
    churner_i = int(np.argmax(np.where(y == 1, proba, -1)))       # highest-risk true churner
    nonchurner_i = int(np.argmin(np.where(y == 0, proba, 2)))     # lowest-risk true non-churner
    print(f"local examples -> churner idx {churner_i} (p={proba[churner_i]:.2f}), "
          f"non-churner idx {nonchurner_i} (p={proba[nonchurner_i]:.2f})")

    figs = {
        "beeswarm": interpret.plot_beeswarm(expl),
        "bar": interpret.plot_bar(expl),
        "waterfall_churner": interpret.plot_waterfall(
            expl, churner_i, "12_shap_waterfall_churner.png"),
        "waterfall_nonchurner": interpret.plot_waterfall(
            expl, nonchurner_i, "13_shap_waterfall_nonchurner.png"),
    }

    # Global ranking (mean |SHAP|) printed for the notes
    order = np.argsort(np.abs(expl.values).mean(0))[::-1][:10]
    print("\nTop 10 churn drivers (mean |SHAP|):")
    for r in order:
        print(f"  {expl.feature_names[r]:35s} {np.abs(expl.values).mean(0)[r]:.3f}")

    tracking.setup_mlflow()
    with mlflow.start_run(run_name=f"shap_{SELECTED}"):
        mlflow.set_tag("stage", "interpret")
        for path in figs.values():
            mlflow.log_artifact(str(path), artifact_path="shap")

    print("\nFigures:")
    for path in figs.values():
        print(" ", path)


if __name__ == "__main__":
    main()
