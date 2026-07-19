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

---

# Part 2 · Steps 4–5 + SHAP + Serving

> Notebook: `notebooks/pathb_telecom_churn_2_zoo_shap_serving.py`
> (workspace: `.../telecom-churn-pathb-2`). Standard-runtime cluster → the first cell
> `%pip install`s xgboost/lightgbm/shap (an ML runtime would already have them).

## Zoo — same winner, one instructive difference

| model | Path α CV AUC | Path β CV AUC | match |
|---|---|---|---|
| **logreg_balanced** (winner) | 0.846 | 0.846 | ✅ exact |
| xgboost | 0.843 | 0.844 | ⚠️ +0.001 |
| lightgbm | 0.836 | 0.836 | ✅ exact |
| random_forest | 0.824 | 0.824 | ✅ exact |

Same winner, same ranking. **Lesson in the XGBoost drift:** sklearn, LightGBM and
RandomForest reproduced *exactly*, but XGBoost moved by 0.001 — because we `%pip`-installed
the **latest** XGBoost on Databricks vs local **2.1.4**, and XGBoost's parallel histogram
tree-building sums gradients in a version/threading-dependent order, so it isn't
bit-identical across environments. Well within CV noise (±0.01); changes nothing. The
takeaway: **most of the pipeline is perfectly reproducible; a highly-parallel GBM is the
one place you see platform drift.**

## SHAP
Rendered inline on Databricks via `shap.summary_plot(..., show=False)` + `display(plt.gcf())`.
Same drivers as Path α (tenure, MonthlyCharges, fibre, contract).

## Serving on Databricks

- **Batch inference — done.** Loaded the best model from MLflow and scored the test set
  with `mlflow.pyfunc.load_model(...).predict(...)`; predictions match the SHAP story
  (month-to-month + fibre + low tenure → churn).
- **Runtime gotcha:** `mlflow.pyfunc.spark_udf` (the *scale-out* path) hits a version-parse
  bug on **serverless / Photon 18.x** — `Invalid version: '18.x-photon-scala2'`. In-process
  `pyfunc.predict` works on any runtime; `spark_udf` is the swap-in for a classic
  ML-runtime cluster with millions of rows.
- **Real-time endpoint — left as reference (decision).** The notebook has the full
  register-to-UC + create-endpoint code, gated behind `CREATE_ENDPOINT = False`. We chose
  **not** to deploy a live endpoint: the real-time pattern is already proven locally
  (Path α S1), and a serving endpoint provisions **paid compute on the company workspace**
  — not warranted for a personal learning project. The code stands as the "how" for when
  it's needed.

## Auth lesson worth remembering
`SHOW CATALOGS` run **interactively** listed 10 UC catalogs (`hip_dev_catalog`,
`lakehouse_development`, `main`, …), but the **PAT token** earlier saw **zero**. Your
interactive identity has broader Unity Catalog permissions than the API token — a common
source of "works in the notebook, fails via API" confusion.

## Path β — COMPLETE ✅
Steps 1–6 reproduced on Databricks with managed MLflow (results match Path α); batch
serving demonstrated; real-time endpoint documented as opt-in reference. The whole port
came down to `mlflow.set_tracking_uri("databricks")` + a `%pip install` — proof that
keeping the DS logic in `src/` makes the environment swappable.
