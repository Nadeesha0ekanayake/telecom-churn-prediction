"""Turn model probabilities into business decisions (Step 7).

Two ideas:
- A model outputs a *probability*; the business needs a *decision*. The 0.5 cutoff is
  arbitrary — the right threshold depends on the relative cost of a missed churner vs a
  wasted retention offer. `cost_curve` finds the cost-minimising threshold.
- Not every at-risk customer is worth the same. `segment` crosses churn risk with
  customer value so retention effort goes to high-value-at-risk customers first.
"""
from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config  # noqa: E402


def cost_curve(y_true, y_proba, cost_fp: float, cost_fn: float,
               thresholds=None) -> tuple[pd.DataFrame, pd.Series]:
    """Expected cost across decision thresholds. Returns (table, best_row).

    cost_fp = cost of a wasted retention offer (predict churn, customer would stay)
    cost_fn = cost of a missed churner   (predict stay, customer actually leaves)
    """
    if thresholds is None:
        thresholds = np.round(np.linspace(0.05, 0.95, 19), 3)
    y = np.asarray(y_true)
    rows = []
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tp = int(((pred == 1) & (y == 1)).sum())
        rows.append({"threshold": t, "tp": tp, "fp": fp, "fn": fn,
                     "total_cost": fp * cost_fp + fn * cost_fn})
    table = pd.DataFrame(rows)
    return table, table.loc[table["total_cost"].idxmin()]


def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_cost_curve(table: pd.DataFrame, best: pd.Series,
                    fname: str = "14_threshold_cost.png"):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(table["threshold"], table["total_cost"], marker="o")
    ax.axvline(best["threshold"], color="green", ls="--",
               label=f"optimal = {best['threshold']:.2f}")
    ax.axvline(0.5, color="grey", ls=":", label="default = 0.50")
    ax.set(xlabel="decision threshold", ylabel="total expected cost ($)",
           title="Cost vs decision threshold (test set)")
    ax.legend()
    return _save(fig, fname)


def segment(value: pd.Series, proba: np.ndarray, risk_threshold: float,
            value_threshold: float | None = None) -> pd.DataFrame:
    """Cross churn risk with customer value into a per-customer segment table."""
    if value_threshold is None:
        value_threshold = float(np.median(value))
    df = pd.DataFrame({
        "monthly_value": np.asarray(value),
        "churn_proba": proba,
        "high_risk": proba >= risk_threshold,
        "high_value": np.asarray(value) >= value_threshold,
    })
    df["segment"] = np.select(
        [df["high_risk"] & df["high_value"],
         df["high_risk"] & ~df["high_value"],
         ~df["high_risk"] & df["high_value"]],
        ["priority (high-risk, high-value)", "high-risk, low-value",
         "monitor (low-risk, high-value)"],
        default="low priority",
    )
    return df


def plot_segments(seg: pd.DataFrame, risk_threshold: float,
                  value_threshold: float, fname: str = "15_risk_value_segments.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    priority = seg["segment"].str.startswith("priority")
    ax.scatter(seg.loc[~priority, "monthly_value"], seg.loc[~priority, "churn_proba"],
               s=10, alpha=0.3, color="grey", label="other")
    ax.scatter(seg.loc[priority, "monthly_value"], seg.loc[priority, "churn_proba"],
               s=14, alpha=0.6, color="crimson", label="priority")
    ax.axhline(risk_threshold, color="green", ls="--")
    ax.axvline(value_threshold, color="blue", ls="--")
    ax.set(xlabel="monthly charges ($) = customer value",
           ylabel="predicted churn probability",
           title="Risk × value segmentation (test set)")
    ax.legend()
    return _save(fig, fname)
