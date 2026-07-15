# 02 · Cleaning & Feature Engineering (Step 2, Path α)

> Learning notes. Code: `src/preprocess.py`; runner: `scripts/02_preprocess.py`.
> Run: `./venv/bin/python scripts/02_preprocess.py`

## The one principle that shapes everything here: **no data leakage**

Leakage = letting information from the test set (or the future) sneak into training,
giving falsely optimistic scores that collapse in production. To avoid it we split the
work by *what kind of transform it is*:

| Kind | Examples | When applied | Why safe |
|---|---|---|---|
| **Deterministic / rule-based** | text→number, tenure-0 → \$0, relabelling | once, upfront (`clean`) | no statistic learned from data |
| **Row-wise features** | count of add-ons, tenure bucket | once, upfront (`add_features`) | each row computed from itself only |
| **Learned transforms** | scaling (mean/std), one-hot categories | **fit on TRAIN only, inside CV** (Step 3) | stats depend on the data → must not see test |

That's why `build_preprocessor()` returns an **unfitted** `ColumnTransformer`. We do *not*
scale/encode the whole dataset here — Step 3's pipeline fits it on each training fold.

## 1. Cleaning — `clean()` (Step 1 decisions 1–3)

- **`TotalCharges`: text → float**, and the **11 tenure-0 blanks → \$0** (verified in Step 1
  these are un-billed new customers). Result: dtype `float64`, **0 nulls**.
- **`Churn`: 'Yes'/'No' → 1/0** — the numeric target models need.
- **`SeniorCitizen`: 0/1 → 'No'/'Yes'** — it's a category, not a quantity; this stops us
  from accidentally *scaling* it as if 1 were "more" than 0 in a continuous sense.
- **Collapse "No internet/phone service" → "No"** (toggle `collapse_no_service`). These
  meant "No" anyway; the dependency they encoded is captured better by `has_internet`.
  After this, **0** such tokens remain.

## 2. Feature engineering — `add_features()` (each with its intuition)

| Feature | Definition | Intuition |
|---|---|---|
| `num_addon_services` | count of the 6 add-ons subscribed (0–6) | more services = higher switching cost = **stickier** customer |
| `tenure_group` | tenure bucketed: 0-12 / 12-24 / 24-48 / 48-72 | gives **linear** models a non-linear view — churn risk is front-loaded in year 1 |
| `has_internet` | InternetService ≠ "No" | internet customers (fibre pricing, add-ons) churn differently from phone-only |

All three are row-wise → safe to compute upfront. (Target/mean encoding would be
powerful but **leaky** unless done inside CV — deferred as a later option.)

Result: **4 numeric** + **18 categorical** features (24-column cleaned frame incl. id+target).

## 3. Stratified split — `split()`

- 80/20 train/test, **stratified on churn** and seeded (`RANDOM_SEED=42`).
- Check: train churn rate **0.2654**, test churn rate **0.2654** — identical to the full
  26.5%. Stratification worked; both sets are representative.
- 5,634 train / 1,409 test rows.

## 4. The preprocessor & its options — `build_preprocessor(scaler=, encoder=)`

A `ColumnTransformer` = "apply transform A to these columns, transform B to those".

**Options exposed (to compare later, per the learning plan):**
- `scaler`: `standard` (z-score) · `minmax` (0–1) · `none` (trees don't need scaling)
- `encoder`: `onehot` (one column per level; good for linear models) ·
  `ordinal` (single integer column per feature; compact, natural for trees)

**Concrete effect of the encoder choice** (fit on train only):
| encoder | resulting model features |
|---|---|
| one-hot | **46** |
| ordinal | **22** |

One-hot expands each category into indicator columns (more features, no false ordering);
ordinal keeps one column but imposes an integer order (fine for trees, risky for linear).

## Carried into Step 3 (modelling)

- Compose `build_preprocessor(...)` **+ estimator** into a single sklearn `Pipeline`, so
  preprocessing is fit per-fold inside cross-validation (leakage-safe by construction).
- Match preprocessing to model: linear → `standard` + `onehot`; trees → `none` +
  `onehot`/`ordinal`.
- Handle the 26.5% imbalance via `class_weight` / SMOTE / threshold tuning.
- Headline metrics: AUC-ROC, precision, recall, F1.
