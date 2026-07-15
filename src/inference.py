"""Inference helpers: load the registered model and score customers (Step S1).

Note on input schema: the registered Pipeline handles scaling + encoding, but the
deterministic `clean()` / `add_features()` steps run *before* it. So the model expects
**post-prepare** columns (the engineered feature set), not the raw CSV schema. `score()`
runs `prepare()` first. (Productionisation improvement: fold `prepare` into the pipeline
as FunctionTransformer steps so the model accepts fully raw input.)
"""
from __future__ import annotations

import mlflow
import numpy as np
import pandas as pd

from src import config, preprocess, tracking

REGISTERED_MODEL = "telecom-churn"
CHAMPION_ALIAS = "champion"
DECISION_THRESHOLD = 0.20  # cost-optimal threshold from Step 7


def model_uri(alias: str = CHAMPION_ALIAS) -> str:
    return f"models:/{REGISTERED_MODEL}@{alias}"


def load_sklearn(alias: str = CHAMPION_ALIAS):
    """Load the registered Pipeline (sklearn flavor → keeps predict_proba)."""
    tracking.setup_mlflow()
    return mlflow.sklearn.load_model(model_uri(alias))


def feature_frame(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Raw customer rows → the model's expected (post-prepare) feature columns."""
    feat = preprocess.prepare(raw_df)
    return feat.drop(columns=[config.ID_COL, config.TARGET], errors="ignore")


def score(pipeline, raw_df: pd.DataFrame,
          threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Score raw customers → probability, decision, and risk band."""
    X = feature_frame(raw_df)
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    band = np.where(proba >= 0.6, "high", np.where(proba >= threshold, "medium", "low"))
    return pd.DataFrame({
        "churn_proba": proba.round(3),
        "churn_decision": pred,
        "risk_band": band,
    })
