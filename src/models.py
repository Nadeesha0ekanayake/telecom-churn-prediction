"""Model definitions and the preprocessor+estimator pipeline factory.

Keeping every model behind a single `make_pipeline()` means preprocessing is always
bundled with the estimator into one object — so cross-validation fits the scaler/encoder
on each training fold only (leakage-safe), and the saved model preprocesses raw input
itself at inference time.
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src import config, preprocess

# Class imbalance ratio (neg/pos) for XGBoost's scale_pos_weight ≈ 5174/1869
POS_WEIGHT = 2.77


def make_pipeline(
    estimator,
    numeric: list[str],
    categorical: list[str],
    scaler: str = "standard",
    encoder: str = "onehot",
) -> Pipeline:
    """Compose an unfitted preprocessor with an estimator into one Pipeline."""
    pre = preprocess.build_preprocessor(numeric, categorical, scaler, encoder)
    return Pipeline([("pre", pre), ("clf", estimator)])


def get_baseline_models() -> dict:
    """Step 3 baselines.

    - dummy_most_frequent: predicts the majority class — the score floor any real
      model must beat (its ROC-AUC is 0.5 by construction).
    - logreg: plain logistic regression — an interpretable, honest linear baseline.
    - logreg_balanced: same but class_weight='balanced' — re-weights the 26.5%
      minority so the model stops ignoring churners. Lets us *see* the trade-off.
    """
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "logreg_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
    }


def get_model_specs() -> dict:
    """Step 4–5 model zoo: estimator + its matched preprocessing.

    Each spec pairs a model with the preprocessing it wants:
    - linear (logreg) → scale numerics + one-hot (needs scaling, no false order)
    - trees (RF/XGB/LGBM) → no scaling (splits are scale-invariant), one-hot cats
    Imbalance is handled per-model: class_weight for logreg/RF/LGBM,
    scale_pos_weight for XGBoost.
    """
    seed = config.RANDOM_SEED
    return {
        "logreg_balanced": {
            "estimator": LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=seed
            ),
            "scaler": "standard", "encoder": "onehot",
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced",
                random_state=seed, n_jobs=-1
            ),
            "scaler": "none", "encoder": "onehot",
        },
        "xgboost": {
            "estimator": XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                scale_pos_weight=POS_WEIGHT, random_state=seed, n_jobs=-1
            ),
            "scaler": "none", "encoder": "onehot",
        },
        "lightgbm": {
            "estimator": LGBMClassifier(
                n_estimators=300, learning_rate=0.05, num_leaves=31,
                class_weight="balanced", random_state=seed, n_jobs=-1, verbose=-1
            ),
            "scaler": "none", "encoder": "onehot",
        },
    }
