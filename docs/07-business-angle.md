# 07 · Business Angle (Step 7, Path α)

> Learning notes. Code: `src/business.py`; runner: `scripts/06_business.py`.
> Run: `./venv/bin/python scripts/06_business.py`. Model: `logreg_balanced`.

The model gives *probabilities*; the business needs *decisions* and *priorities*. This
step converts the churn model + SHAP drivers into money and actions.

## 1. The threshold is a business decision, not 0.5

A classifier's default 0.5 cutoff assumes a false positive and a false negative cost the
same. They don't. We set **illustrative** unit costs (a real project sources these from
finance):

- **COST_FP = \$50** — a wasted retention offer on a customer who'd have stayed.
- **COST_FN = \$500** — a missed churner (lost customer / margin). 10× worse.

Sweeping the threshold and totalling cost on the test set (`14_threshold_cost.png`):

| threshold | total cost | tp | fp | fn |
|---|---|---|---|---|
| 0.50 (default) | **\$53,950** | — | — | — |
| **0.20 (optimal)** | **\$35,500** | 359 | 560 | 15 |

- **Choosing 0.20 saves \$18,450** on 1,409 test customers vs the naïve 0.5.
- Why so low? Since missing a churner costs 10× a wasted offer, it pays to **cast a wide
  net** — catch 359 of 374 churners (only 15 missed) and accept 560 false alarms.
- The curve rises steeply to the right: being conservative (high threshold) is expensive
  here. **The optimal threshold moves with the cost ratio** — if offers were expensive or
  churn cheap, it would shift up. That's the lesson: tune the threshold to the economics.

## 2. Who to contact first — risk × value

Not every at-risk customer is worth equal effort. We cross **churn risk** (proba ≥ 0.20)
with **customer value** (monthly charges ≥ median \$70) → four segments
(`15_risk_value_segments.png`):

| segment | customers | action |
|---|---|---|
| **priority** (high-risk, high-value) | **548** | contact first — biggest \$ at stake |
| high-risk, low-value | 371 | cheaper automated retention |
| monitor (low-risk, high-value) | 157 | protect, watch for drift |
| low priority | 333 | no action |

- **Priority segment = 548 customers ≈ \$48,994/month revenue at risk.** That's the
  concrete number a retention team can be handed and measured against.

## 3. Retention playbook — SHAP drivers → actions

Step 6 told us *why* customers churn; each driver implies a lever (this is the seed of a
**Next Best Action** system):

| SHAP driver | Retention action |
|---|---|
| Month-to-month contract | Incentive to move to a 1/2-year term |
| Low tenure (new customer) | Early-life onboarding & check-in |
| Fibre optic + high charges | Loyalty pricing / service-quality review |
| Electronic-check payment | Nudge to autopay (autopay customers churn less) |
| Few add-on services | Bundle add-ons to raise switching cost |

## MLflow
Logged under a run tagged `stage=business`: the cost params, optimal threshold, cost
saving, priority count, and monthly revenue at risk — plus both figures.

## Caveats (honesty matters)
- Unit costs are **illustrative** — the exact optimal threshold depends on real finance
  numbers, and offer success isn't 100% (a fuller model would weight TP by save-rate).
- Value = current monthly charges is a proxy for **CLV**; a lifetime-value estimate would
  refine the priority ranking.

## Path α — complete
Steps 1–7 done end-to-end (data → clean → baseline → model zoo → SHAP → business).
Remaining in Path α: **S1 · MLflow inference** (register the model, batch + local REST).
Then **Path β**: port the same `src/` logic onto Databricks.
