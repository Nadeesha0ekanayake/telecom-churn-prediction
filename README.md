# Telecom Churn Prediction

A hands-on **learning project** for predicting customer churn on the public
[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset (7,043 customers, 21 features).

The aim isn't just a good model — it's to try **multiple options at each stage**
(models, preprocessing, environments, serving) and understand how they compare.

See **[ROADMAP.md](ROADMAP.md)** for the full plan and progress.

## Layout

```
telecom-churn-prediction/
├── data/
│   ├── raw/         # source dataset (Telco-Customer-Churn.csv)
│   └── processed/   # cleaned / feature-engineered outputs (gitignored)
├── notebooks/       # exploratory + Databricks runners
├── src/             # environment-agnostic data-science logic
├── models/          # serialized models (binaries gitignored)
├── reports/figures/ # generated plots
└── scripts/         # CLI entry points
```

## Status

Path α (VS Code local + local MLflow) — **in progress**. Setup complete; Step 1 next.
