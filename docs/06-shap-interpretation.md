# 06 · SHAP Interpretation (Step 6, Path α)

> Learning notes. Code: `src/interpret.py`; runner: `scripts/05_shap.py`.
> Run: `./venv/bin/python scripts/05_shap.py`. Model interpreted: `logreg_balanced`.

## What SHAP is (and why it beats plain coefficients)

SHAP assigns every feature a **signed contribution** to a single prediction, in the
model's output space (**log-odds** for logistic regression). For each customer:

```
predicted log-odds  =  base value E[f(x)]  +  Σ (SHAP contributions)
```

It's **exact and additive** — not a post-hoc approximation — and, unlike a single global
coefficient, it shows the contribution **per customer** and its **direction**. Because our
model is a `Pipeline`, we push data through the preprocessor first, then explain the
estimator on the transformed matrix (so SHAP values carry readable feature names).

We use `shap.LinearExplainer` — exact and instant for linear models.

## Global view

### Top churn drivers (mean |SHAP|)
| rank | feature | mean \|SHAP\| |
|---|---|---|
| 1 | tenure | 0.934 |
| 2 | MonthlyCharges | 0.445 |
| 3 | InternetService = Fiber optic | 0.434 |
| 4 | Contract = Month-to-month | 0.355 |
| 5 | Contract = Two year | 0.278 |
| 6–10 | InternetService/has_internet, TotalCharges | — |

### Direction — `10_shap_beeswarm.png` (each dot = one customer; red = high value)
- **tenure**: high tenure (red) → **negative** SHAP (pushes *away* from churn); low tenure
  (blue) → **positive** (toward churn). New customers churn — as expected.
- **MonthlyCharges**: high (red) → positive (toward churn). Bigger bills, more churn.
- **Fibre optic = yes** (red) → positive. Fibre customers churn more (price/competition).
- **Month-to-month = yes** (red) → positive; **Two-year = yes** (red) → negative. Contract
  length is a strong protective/risk signal.

The bar chart `11_shap_bar.png` is the same ranking as a simple magnitude bar.

## Local view — explaining one customer

### High-risk churner — `12_shap_waterfall_churner.png`
- Model output **f(x) = 2.995 log-odds → p ≈ 0.95**, from a base of E[f(x)] = −0.64.
- What drove it up: **very low tenure (+1.37)**, **fibre optic (+0.43)**,
  **month-to-month (+0.34)**, **electronic-check payment (+0.17)**.
- This is the archetypal high-risk profile: a **brand-new fibre customer on a flexible
  month-to-month contract paying by electronic check**.
- Nuance worth noticing: this customer's high `MonthlyCharges` actually contributed
  **−0.51** (toward staying) — a reminder that a **local** explanation can differ from the
  **global** average, and that correlated features (charges/fibre/total) share credit.

### Confident non-churner — `13_shap_waterfall_nonchurner.png`
- Predicted **p ≈ 0.01**. The mirror image: long tenure and a long contract push strongly
  toward "won't churn".

## Why this matters
- **Trust / debugging:** the model's drivers match domain reality (Step 1 EDA), so it's
  learning signal, not artefacts.
- **Actionability (Step 7):** SHAP turns "this customer will churn" into "*because* they're
  new, on month-to-month, on fibre" — which points directly at retention levers
  (contract incentives, onboarding, fibre pricing).

## MLflow
The four SHAP figures are logged as artifacts under an MLflow run tagged
`stage=interpret` (open the run in the UI to view them alongside the model).

## Carried into Step 7
- Convert per-customer SHAP + churn probability into **segments** (high-value, high-risk).
- Choose an operating **threshold** by business cost (missed churner vs wasted offer).
- Translate drivers into a concrete **retention playbook / Next Best Action**.
