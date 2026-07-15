# 04 · Model Zoo (Steps 4–5, Path α)

> Learning notes. Code: `src/models.py` (`get_model_specs`, `make_pipeline`),
> `src/evaluate.py` (plots); runner: `scripts/04_model_zoo.py`.
> Run: `./venv/bin/python scripts/04_model_zoo.py`

## What we ran

Four models, each with **preprocessing matched to the model** and **imbalance handled
per-model**, all logged to the same MLflow experiment (tag `stage=model_zoo`):

| Model | Scaler | Encoder | Imbalance handling |
|---|---|---|---|
| logreg_balanced | standard | one-hot | `class_weight='balanced'` |
| random_forest | none | one-hot | `class_weight='balanced'` |
| xgboost | none | one-hot | `scale_pos_weight≈2.77` |
| lightgbm | none | one-hot | `class_weight='balanced'` |

Trees skip scaling (their splits are scale-invariant). XGBoost/LightGBM use ~300 shallow
trees with a low learning rate — sensible, untuned defaults (tuning is the next option).

## Results — ranked by CV ROC-AUC (our selection metric)

| model | cv_roc_auc | cv_std | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|---|
| **logreg_balanced** | **0.846** | 0.011 | 0.732 | 0.497 | 0.791 | 0.611 | 0.842 |
| xgboost | 0.843 | 0.011 | 0.749 | 0.519 | 0.767 | 0.619 | 0.839 |
| lightgbm | 0.836 | 0.009 | 0.762 | 0.538 | 0.743 | 0.624 | 0.832 |
| random_forest | 0.824 | 0.011 | 0.780 | 0.614 | 0.460 | 0.526 | 0.823 |

## The big lessons

1. **Gradient boosting did *not* automatically win.** The **logistic-regression baseline
   is the top model** by CV AUC (0.846), edging out XGBoost (0.843). This surprises people
   who assume XGBoost/LightGBM always dominate tabular problems. Why here?
   - The dataset is **modest (~7k rows) and mostly categorical**, with fairly **linear,
     additive** signal (contract type, tenure, charges). That's logistic regression's
     home turf; boosting's edge (complex interactions) has little to exploit.

2. **The top three are a statistical tie.** logreg 0.846, xgboost 0.843, lightgbm 0.836 —
   the gaps are within ~1 CV std (±0.01). Treat them as **equivalent on ranking**; pick on
   *other* grounds (simplicity, interpretability, inference cost) → favours logreg.

3. **The "best" model depends on the metric** (see `08_zoo_metrics.png`):
   - **ROC-AUC / recall** → logreg_balanced
   - **F1** → lightgbm (0.624) — best precision/recall balance
   - **precision / accuracy** → random_forest (0.61 / 0.78) — but its **recall is only 0.46**
     (misses over half the churners). High precision, low recall: it only flags the most
     obvious churners.

4. **Feature importance sanity-checks the EDA** (`09_feature_importance.png`). Winner's top
   drivers: **tenure**, **fibre-optic internet**, **contract type** (two-year ↓, month-to-
   month ↑), **monthly charges**. Exactly the signals Step 1 surfaced — the model is
   learning real structure, not noise.

## Model selected: `logreg_balanced`

By our primary metric (CV ROC-AUC) it's the best, and it's also the **simplest and most
interpretable** — the right default winner. XGBoost is effectively tied and could overtake
it with hyperparameter tuning.

## Options not yet exercised (available for later)

- **Hyperparameter tuning** (`optuna` / `RandomizedSearchCV`) — could lift XGBoost/LightGBM
  past logreg; untuned here on purpose to keep the comparison honest.
- **SMOTE** instead of class weights (needs an `imblearn` pipeline to stay leakage-safe).
- **Ordinal encoding** for trees (more compact than one-hot).

## Carried into Step 6
- Interpret the selected model with **SHAP** — global drivers + individual explanations,
  going beyond the coarse `|coef|` importance shown here.
- (SHAP also lets us contrast the linear winner with a tree model's view of the data.)
