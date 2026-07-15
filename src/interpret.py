"""SHAP interpretation for the selected pipeline (Step 6).

SHAP assigns each feature a signed contribution to a single prediction, in the model's
output space (log-odds for logistic regression). Summed over a customer, the
contributions + a base value reconstruct that customer's predicted log-odds — so the
explanation is *exact and additive*, not a post-hoc guess.

Our model is a Pipeline (preprocessor + estimator). SHAP works on the estimator, so we
first push data through the preprocessor to get the numeric matrix + readable feature
names, then explain the estimator on that matrix.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import shap  # noqa: E402

from src import config  # noqa: E402


def transform(pipeline, X):
    """Return (transformed_matrix, feature_names) from the pipeline's preprocessor."""
    pre = pipeline.named_steps["pre"]
    return pre.transform(X), list(pre.get_feature_names_out())


def explain_linear(pipeline, X_train, X_test):
    """Exact SHAP values for a linear estimator, as a shap.Explanation over X_test."""
    clf = pipeline.named_steps["clf"]
    X_train_t, names = transform(pipeline, X_train)
    X_test_t, _ = transform(pipeline, X_test)

    explainer = shap.LinearExplainer(clf, X_train_t)
    values = explainer.shap_values(X_test_t)
    base = np.repeat(explainer.expected_value, len(values))
    return shap.Explanation(values=values, base_values=base,
                            data=X_test_t, feature_names=names)


def _save_current(fname: str):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / fname
    plt.gcf().savefig(path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return path


def plot_beeswarm(expl, fname: str = "10_shap_beeswarm.png", max_display: int = 15):
    """Global view: each dot is a customer; colour = feature value, x = SHAP impact."""
    shap.plots.beeswarm(expl, max_display=max_display, show=False)
    return _save_current(fname)


def plot_bar(expl, fname: str = "11_shap_bar.png", max_display: int = 15):
    """Global importance: mean |SHAP| per feature."""
    shap.plots.bar(expl, max_display=max_display, show=False)
    return _save_current(fname)


def plot_waterfall(expl, i: int, fname: str, max_display: int = 12):
    """Local view: how one customer's features push its prediction up/down."""
    shap.plots.waterfall(expl[i], max_display=max_display, show=False)
    return _save_current(fname)
