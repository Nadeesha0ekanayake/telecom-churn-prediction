"""Step S1b: batch inference with the registered model.

    ./venv/bin/python scripts/08_batch_inference.py

Loads models:/telecom-churn@champion, scores a sample of raw customers
(probability + decision + risk band), saves predictions, and writes a JSON
payload that the REST-serving demo reuses.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

warnings.filterwarnings("ignore", message="X does not have valid feature names",
                        category=UserWarning)

from src import config, data_loader, inference  # noqa: E402


def main() -> None:
    raw = data_loader.load_raw()
    # A spread of customers (mix of likely churn / not) for a readable demo
    sample = raw.sample(8, random_state=config.RANDOM_SEED).reset_index(drop=True)

    model = inference.load_sklearn()
    scored = inference.score(model, sample)
    out = sample[["customerID", "tenure", "Contract",
                  "InternetService", "MonthlyCharges"]].join(scored)

    print("=" * 78 + "\nBATCH INFERENCE (models:/telecom-churn@champion)\n" + "=" * 78)
    print(out.to_string(index=False))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = config.PROCESSED_DIR / "predictions.csv"
    out.to_csv(pred_path, index=False)
    print(f"\nsaved {pred_path.relative_to(config.PROJECT_ROOT)}")

    # REST payload (dataframe_split): the model's post-prepare feature columns
    X = inference.feature_frame(sample).head(3)
    payload = {"dataframe_split": {"columns": X.columns.tolist(),
                                   "data": X.astype(object).values.tolist()}}
    payload_path = config.PROCESSED_DIR / "sample_payload.json"
    payload_path.write_text(json.dumps(payload))
    print(f"saved {payload_path.relative_to(config.PROJECT_ROOT)} (for REST demo)")


if __name__ == "__main__":
    main()
