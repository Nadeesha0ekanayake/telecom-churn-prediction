# 09 · Path β on Databricks — Steps 1–3

> Learning notes. Notebook: `notebooks/pathb_telecom_churn.py` (also published to the
> Databricks workspace at `/Users/nadeeshaekanayake@…/telecom-churn-pathb`).
> Run interactively in the Databricks UI on an ML-runtime cluster.

## Goal
Re-run the local Path α flow (load → clean/features → baseline) on Databricks with
**managed MLflow**, and confirm the results match. A port-and-compare exercise.

## The only real code change
```python
# Path α (local):    mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Path β (Databricks):
mlflow.set_tracking_uri("databricks")
```
Everything else — cleaning, feature engineering, the split (same seed 42), the
preprocessing, the three baseline models — is byte-for-byte the same logic as `src/`.

## Result: identical to 3 decimals ✅

| model | cv_roc_auc | accuracy | precision | recall | f1 | roc_auc |
|---|---|---|---|---|---|---|
| dummy_most_frequent | 0.500 | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| logreg | 0.846 | 0.799 | 0.653 | 0.519 | 0.578 | 0.842 |
| logreg_balanced | 0.846 | 0.732 | 0.497 | 0.791 | 0.611 | 0.842 |

Every value matches Path α (Step 3) exactly. **Same data + same code + same seed → same
result**, regardless of environment. That's the payoff of keeping DS logic separate from
where it runs.

## What actually changed (the environment, not the results)
- **MLflow**: runs now live in the **workspace Experiments UI** (`telecom-churn-pathb`) —
  no local `mlflow ui`, no SQLite file, no setup. The managed tracking server just works.
- **Environment**: the cluster already had pandas / scikit-learn / mlflow — **no venv, no
  `brew install libomp`**. What Path α set up by hand, Databricks provides.
- **Data**: loaded straight from the public URL on the cluster (no upload).

## How it was delivered (mechanics worth remembering)
- Notebook published via the **Workspace import API** (`/api/2.0/workspace/import`,
  SOURCE format, base64 content) using the **workspace token** from `~/.databrickscfg`
  (the CLI marks that token invalid, but curl against the REST API works).
- Stored in the repo in **Databricks source format** (`# Databricks notebook source` +
  `# COMMAND ----------` cells) so it renders on GitHub *and* re-imports cleanly.
- **Sync caveat:** edits made *inside* Databricks don't flow back to git automatically —
  re-export via the Workspace API and commit to keep the repo copy current.

## Next on Path β
- Steps 4–5 (XGBoost / LightGBM) + SHAP in a second notebook — needs an **ML runtime**
  (those libs pre-installed) or a `%pip install`.
- **Databricks Model Serving** in place of the local `mlflow models serve` (the S2 step),
  including the Unity Catalog model registry.
