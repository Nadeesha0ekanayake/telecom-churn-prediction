"""Metrics, cross-validation, and evaluation plots for classification.

Metrics chosen for an imbalanced target (Step 1): we report accuracy for context but
judge on precision / recall / F1 / ROC-AUC. ROC-AUC is threshold-independent and uses
predicted probabilities, so it's our primary model-selection metric.
"""
from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

from src import config  # noqa: E402


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    """Standard classification metrics for the positive (churn=1) class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def cross_val_auc(pipeline, X, y, n_splits: int = 5) -> tuple[float, float]:
    """Stratified k-fold ROC-AUC on the training set → (mean, std).

    Because the whole pipeline (preprocessor + model) is passed in, each fold refits
    preprocessing on its own training portion — no leakage across folds.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=config.RANDOM_SEED)
    scores = cross_val_score(pipeline, X, y, scoring="roc_auc", cv=cv)
    return scores.mean(), scores.std()


def _save(fig, name: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc(results: dict, y_test, fname: str = "05_baseline_roc.png"):
    """Overlay ROC curves for every model with a predicted probability."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, r in results.items():
        if r.get("y_proba") is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['test']['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="chance (0.5)")
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Baseline ROC curves")
    ax.legend(loc="lower right")
    return _save(fig, fname)


def plot_confusions(results: dict, y_test, fname: str = "06_baseline_confusion.png"):
    """Side-by-side confusion matrices for all models."""
    names = list(results)
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        cm = confusion_matrix(y_test, results[name]["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        ax.set_title(name, fontsize=10)
    return _save(fig, fname)


def plot_metric_bars(
    results: dict,
    metrics=("roc_auc", "f1", "recall", "precision"),
    fname: str = "08_zoo_metrics.png",
):
    """Grouped bar chart of test metrics across all models."""
    names = list(results)
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(names)), 5))
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, [results[n]["test"][m] for n in names], width, label=m)
    ax.set_xticks(x + width * (len(metrics) - 1) / 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set(ylim=(0, 1), ylabel="test score", title="Model zoo — test metrics")
    ax.legend(ncol=len(metrics), loc="lower right")
    return _save(fig, fname)


def plot_feature_importance(pipeline, top_n: int = 15,
                            fname: str = "09_feature_importance.png"):
    """Native feature importance (trees) or |coef| (linear) for a fitted pipeline."""
    pre = pipeline.named_steps["pre"]
    clf = pipeline.named_steps["clf"]
    names = pre.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    else:
        importance = np.abs(clf.coef_).ravel()
    order = np.argsort(importance)[::-1][:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order], [importance[i] for i in order])
    ax.set_title(f"Top {top_n} features — {type(clf).__name__}")
    fig.tight_layout()
    return _save(fig, fname)
