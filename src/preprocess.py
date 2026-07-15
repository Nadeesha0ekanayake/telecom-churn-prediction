"""Cleaning, feature engineering, splitting, and the preprocessing pipeline.

Design (why it's split this way):
- `clean()` applies only **deterministic, rule-based** fixes (no statistics learned
  from data) → safe to run once, upfront, no leakage risk.
- `add_features()` adds **row-wise** engineered features (each row computed from itself)
  → also leakage-safe.
- `build_preprocessor()` returns an **unfitted** ColumnTransformer. It scales/encodes
  using statistics that MUST be learned on training data only, so it is fit *inside*
  the model pipeline during cross-validation (Step 3) — never on the full dataset here.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from src import config

# Service add-ons: counted into a "how invested is this customer" feature
ADDON_SERVICES = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns that read as "No internet service" / "No phone service" (a 3rd level
# that means "No", but also encodes a dependency)
_NO_SERVICE_TOKENS = ["No internet service", "No phone service"]


def clean(df: pd.DataFrame, collapse_no_service: bool = True) -> pd.DataFrame:
    """Apply deterministic cleaning (Step 1 decisions 1–3).

    - TotalCharges: text -> float; the 11 tenure-0 blanks -> 0 (accrued nothing yet)
    - Churn: 'Yes'/'No' -> 1/0 (model target)
    - SeniorCitizen: 0/1 -> 'No'/'Yes' (treat as categorical, not a scaled number)
    - optionally collapse 'No internet/phone service' -> 'No'
    """
    df = df.copy()

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[
        df["tenure"] == 0, "TotalCharges"
    ].fillna(0.0)

    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    if collapse_no_service:
        df = df.replace(_NO_SERVICE_TOKENS, "No")

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add row-wise engineered features (leakage-safe).

    - num_addon_services: count of the 6 add-ons a customer subscribes to.
      Intuition: more services = higher switching cost = stickier.
    - tenure_group: bucketed tenure. Intuition: gives linear models a non-linear
      view of tenure (the churn effect is front-loaded in the first year).
    - has_internet: whether they have any internet product. Intuition: internet
      customers behave differently (add-ons, fibre pricing) from phone-only.
    """
    df = df.copy()
    df["num_addon_services"] = (df[ADDON_SERVICES] == "Yes").sum(axis=1)
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12", "12-24", "24-48", "48-72"],
    ).astype(str)
    df["has_internet"] = (df["InternetService"] != "No").map({True: "Yes", False: "No"})
    return df


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols), excluding id and target."""
    drop = {config.ID_COL, config.TARGET}
    numeric = [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical = [c for c in df.columns if c not in drop and c not in numeric]
    return numeric, categorical


def split(df: pd.DataFrame, test_size: float = 0.2):
    """Stratified train/test split (preserves the 26.5% churn rate in both)."""
    X = df.drop(columns=[config.ID_COL, config.TARGET], errors="ignore")
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.RANDOM_SEED
    )


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    scaler: str = "standard",
    encoder: str = "onehot",
) -> ColumnTransformer:
    """Build an UNFITTED ColumnTransformer (fit inside the model pipeline).

    scaler:  'standard' | 'minmax' | 'none'
    encoder: 'onehot'   | 'ordinal'
    """
    num_transformer = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "none": "passthrough",
    }[scaler]

    cat_transformer = {
        "onehot": OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        "ordinal": OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        ),
    }[encoder]

    return ColumnTransformer(
        [
            ("num", num_transformer, numeric),
            ("cat", cat_transformer, categorical),
        ],
        remainder="drop",
    )


def prepare(df: pd.DataFrame, engineer: bool = True) -> pd.DataFrame:
    """Convenience: clean (+ optional feature engineering) in one call."""
    df = clean(df)
    if engineer:
        df = add_features(df)
    return df
