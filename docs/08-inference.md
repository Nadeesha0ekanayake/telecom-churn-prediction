# 08 · MLflow Inference (Step S1, Path α)

> Learning notes. Code: `src/inference.py`; runners: `scripts/07_register_model.py`,
> `scripts/08_batch_inference.py`. Closes the loop: raw data → a callable churn service.

## 1. Register the model — `scripts/07_register_model.py`

We log the selected pipeline **and register** it in one call
(`registered_model_name="telecom-churn"`), then point an **alias** at the version:

```
Registered 'telecom-churn' version 1
Alias 'champion' -> version 1
Reference:  models:/telecom-churn@champion
```

**Why an alias, not a hard-coded version?** Downstream code references
`models:/telecom-churn@champion`. When we train a better model later, we register v2 and
move the `champion` alias to it — **every batch job and endpoint picks up the new model
with no code change**. (Aliases replaced the old Staging/Production "stages" in MLflow 3.x.)

## 2. Input-schema subtlety (important)

The registered `Pipeline` contains **scaling + encoding**, but the deterministic
`clean()` / `add_features()` steps run *before* it. So the model's signature is the
**post-prepare** feature set, not the raw CSV. `inference.score()` runs `prepare()` first,
then the model.

> **Productionisation improvement:** fold `prepare` into the pipeline as
> `FunctionTransformer` steps, so the deployed model accepts *fully raw* input and there's
> one artifact with no pre-step to remember. Left as a noted next step.

## 3. Batch inference — `scripts/08_batch_inference.py`

Load with the **sklearn flavor** (keeps `predict_proba`), score raw customers → probability
+ decision (Step 7 threshold 0.20) + risk band:

```python
model = mlflow.sklearn.load_model("models:/telecom-churn@champion")
```

Sample output (8 customers):

| customer | tenure | contract | internet | monthly | proba | decision | band |
|---|---|---|---|---|---|---|---|
| 1024-GUALD | 1 | Month-to-month | DSL | 24.8 | 0.832 | 1 | high |
| 6910-HADCM | 1 | Month-to-month | Fibre | 76.4 | 0.828 | 1 | high |
| 6818-WOBHJ | 68 | Month-to-month | Fibre | 89.6 | 0.544 | 1 | medium |
| 3620-EHIMZ | 52 | Two year | No | 19.4 | 0.020 | 0 | low |
| 4737-AQCPU | 72 | Two year | DSL | 72.1 | 0.015 | 0 | low |

New month-to-month customers score high; long-tenure two-year contracts near zero — the
model behaves exactly as the SHAP story predicted.

## Which data did we score? (evaluation vs. demo — important)

A model split into train/test raises a fair question: *which base did we run inference on?*

| Split | Rows | Used for |
|---|---|---|
| **train** (80%) | 5,634 | fitting the model + 5-fold cross-**validation** (folds = validation) |
| **test** (20%) | 1,409 | held-out **evaluation**: all metrics, ROC/confusion, cost-threshold (0.20), segmentation, SHAP |

- The model is **fit on train only** — including in registration (`pipe.fit(X_train, ...)`).
  Every *performance number* in this project was computed on the **held-out test set**, so
  the evaluation is clean (no leakage).
- **The inference demo is different.** `scripts/08_batch_inference.py` samples 8 rows from
  the **entire raw base** (`raw.sample(8)`), so some scored rows were **in the training
  set**. That is intentional and fine **for demonstrating the serving *mechanics*** (does
  the registered model load, take raw rows, and return predictions?) — but it is **not a
  performance measurement**. Scoring training rows would look optimistically good.
- **In production**, inference runs on genuinely **new, unseen** customers — the equivalent
  of scoring the test split here. To make the demo mirror that, sample the demo rows from
  the test split instead of the full base (a couple of lines; deliberately left as-is here
  since the demo only needs to prove the plumbing).

**Summary:** evaluation = held-out test (honest); inference demo = random sample of the
full base (fine to show the pipeline works, overlaps training data by design).

## 4. Real-time REST serving — `mlflow models serve`

```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"
export PATH="$PWD/venv/bin:$PATH"          # gotcha below
mlflow models serve -m "models:/telecom-churn@champion" -p 5001 --env-manager local
```

POST a customer and get a live prediction:
```bash
curl -s -X POST http://127.0.0.1:5001/invocations \
  -H 'Content-Type: application/json' -d @data/processed/sample_payload.json
# -> {"predictions": [1, 0, 0]}
```

**Three things learned setting this up:**
1. **Serving needs `uvicorn` + `fastapi`** — MLflow 3.x's scoring server runs on them; they
   aren't in the base `mlflow` install. Added to `requirements.txt`. (Symptom: server exits
   with `return code 127` = `uvicorn` not found.)
2. **`--env-manager local`** serves in *our* venv instead of building a fresh conda/virtualenv
   per model (faster, and avoids rebuild errors while iterating).
3. **`PATH` gotcha:** the server launches `uvicorn` in a subprocess via `PATH`, so `venv/bin`
   must be on `PATH` — running `./venv/bin/mlflow` alone isn't enough.
4. **Payload format:** `{"dataframe_split": {"columns": [...], "data": [[...]]}}`, with the
   model's post-prepare feature columns.

**Caveat — REST returns labels, not probabilities.** The `pyfunc` endpoint returns
`predict()` output = class labels at the model's internal **0.5** cutoff (`[1,0,0]`), while
batch applied our cost-optimal **0.20** threshold to probabilities. To serve probabilities
or a custom threshold, log a **custom `pyfunc` wrapper** whose `predict()` returns
`predict_proba` (or applies 0.20). Noted for later.

## Path α — COMPLETE ✅
Raw data → cleaning → baseline → model zoo → SHAP → business → **registered, batch-scored,
and REST-served model**. All logged in MLflow, all in `src/` (env-agnostic).

## Next: Path β (Databricks)
Run the *same* `src/` logic from Databricks notebooks; the only real changes are
`mlflow.set_tracking_uri("databricks")` and Databricks Model Serving in place of the local
REST server. A port-and-compare exercise.
