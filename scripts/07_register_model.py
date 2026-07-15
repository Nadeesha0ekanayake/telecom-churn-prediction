"""Step S1a: register the selected model to the MLflow Model Registry.

    ./venv/bin/python scripts/07_register_model.py

Fits logreg_balanced, logs it, registers it as 'telecom-churn', and points the
'champion' alias at the new version. Downstream inference references
models:/telecom-churn@champion (alias, not a hard-coded version).
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

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from src import data_loader, inference, models, preprocess, tracking  # noqa: E402

SELECTED = "logreg_balanced"


def main() -> None:
    df = preprocess.prepare(data_loader.load_raw())
    numeric, categorical = preprocess.feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.split(df)

    spec = models.get_model_specs()[SELECTED]
    pipe = models.make_pipeline(spec["estimator"], numeric, categorical,
                                scaler=spec["scaler"], encoder=spec["encoder"])
    pipe.fit(X_train, y_train)

    tracking.setup_mlflow()
    with mlflow.start_run(run_name=f"register_{SELECTED}"):
        mlflow.set_tag("stage", "register")
        info = mlflow.sklearn.log_model(
            pipe, name="model",
            signature=infer_signature(X_train, pipe.predict(X_train)),
            input_example=X_train.head(3),
            registered_model_name=inference.REGISTERED_MODEL,
        )

    client = mlflow.MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:  # fallback: newest version
        version = max(int(v.version) for v in
                      client.search_model_versions(f"name='{inference.REGISTERED_MODEL}'"))
    client.set_registered_model_alias(inference.REGISTERED_MODEL,
                                      inference.CHAMPION_ALIAS, version)

    print(f"Registered '{inference.REGISTERED_MODEL}' version {version}")
    print(f"Alias '{inference.CHAMPION_ALIAS}' -> version {version}")
    print(f"Reference it as: {inference.model_uri()}")


if __name__ == "__main__":
    main()
