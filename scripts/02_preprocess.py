"""Step 2 runner: cleaning, feature engineering, split, preprocessing pipeline.

    ./venv/bin/python scripts/02_preprocess.py

Demonstrates the transforms, prints before/after checks, and saves the cleaned
frame to data/processed/ (gitignored — regenerable from raw).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data_loader, preprocess  # noqa: E402


def section(title: str) -> None:
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main() -> None:
    raw = data_loader.load_raw()

    section("1. CLEAN (deterministic fixes)")
    clean = preprocess.clean(raw)
    print("TotalCharges dtype :", clean["TotalCharges"].dtype,
          "| nulls:", clean["TotalCharges"].isna().sum())
    print("Churn values       :", sorted(clean[config.TARGET].unique()))
    print("SeniorCitizen values:", sorted(clean["SeniorCitizen"].unique()))
    print("'No * service' left :",
          int((clean == "No internet service").sum().sum()
              + (clean == "No phone service").sum().sum()))

    section("2. FEATURE ENGINEERING (row-wise, leakage-safe)")
    feat = preprocess.add_features(clean)
    print(feat[["tenure", "tenure_group", "num_addon_services",
                "InternetService", "has_internet"]].head(6).to_string(index=False))

    section("3. FEATURE COLUMNS")
    numeric, categorical = preprocess.feature_columns(feat)
    print(f"numeric ({len(numeric)}): {numeric}")
    print(f"categorical ({len(categorical)}): {categorical}")

    section("4. STRATIFIED SPLIT")
    X_train, X_test, y_train, y_test = preprocess.split(feat)
    print(f"train: {X_train.shape[0]:,} rows | churn rate {y_train.mean():.4f}")
    print(f"test : {X_test.shape[0]:,} rows | churn rate {y_test.mean():.4f}")

    section("5. PREPROCESSOR — option comparison (fit on TRAIN only)")
    for enc in ("onehot", "ordinal"):
        pre = preprocess.build_preprocessor(numeric, categorical,
                                            scaler="standard", encoder=enc)
        Xt = pre.fit_transform(X_train)
        print(f"encoder={enc:8s} -> matrix {Xt.shape} "
              f"({Xt.shape[1]} model features)")

    section("6. SAVE CLEANED DATA")
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = config.PROCESSED_DIR / "telco_clean.csv"
    feat.to_csv(out, index=False)
    print(f"saved {out.relative_to(config.PROJECT_ROOT)} "
          f"({feat.shape[0]:,} rows x {feat.shape[1]} cols)")


if __name__ == "__main__":
    main()
