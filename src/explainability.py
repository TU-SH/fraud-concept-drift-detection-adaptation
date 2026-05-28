"""
explainability.py
-----------------
SHAP-based feature importance analysis.
Compares feature contributions BEFORE and AFTER drift to show
which features changed most — this is the 'explainable drift' angle.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_shap_values(model, X: np.ndarray, sample_size: int = 500):
    """
    Compute SHAP values for a model.
    Uses TreeExplainer (fast for XGBoost/RF).
    Samples to keep runtime manageable.
    """
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # fraud class

    return shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, title: str, save_path: str):
    """Bar chart of mean absolute SHAP values per feature."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=FEATURE_COLS).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(feature_importance)))
    feature_importance.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_ylabel("Feature", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [SHAP] Saved → {save_path}")
    return feature_importance


def plot_drift_feature_shift(shap_before, shap_after, save_path: str):
    """
    Side-by-side bar chart comparing feature importance before vs after drift.
    Highlights which features changed the most.
    """
    imp_before = pd.Series(np.abs(shap_before).mean(axis=0), index=FEATURE_COLS)
    imp_after = pd.Series(np.abs(shap_after).mean(axis=0), index=FEATURE_COLS)

    shift = (imp_after - imp_before).abs().sort_values(ascending=False)
    top_features = shift.head(15).index.tolist()

    df_plot = pd.DataFrame({
        "Before Drift": imp_before[top_features].values,
        "After Drift": imp_after[top_features].values,
    }, index=top_features)

    fig, ax = plt.subplots(figsize=(10, 7))
    x = np.arange(len(top_features))
    width = 0.35
    bars1 = ax.bar(x - width / 2, df_plot["Before Drift"], width, label="Before Drift", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, df_plot["After Drift"], width, label="After Drift", color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right")
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title("Feature Importance Shift: Before vs After Concept Drift\n(Top 15 most-shifted features)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [SHAP] Drift shift chart saved → {save_path}")

    # Return top shifted features as a DataFrame
    result = pd.DataFrame({
        "feature": top_features,
        "importance_before": imp_before[top_features].values,
        "importance_after": imp_after[top_features].values,
        "abs_shift": shift[top_features].values
    }).sort_values("abs_shift", ascending=False)
    return result


def run_full_shap_analysis(model_before, model_after, X_pre, X_post, scaler):
    """
    Full SHAP analysis pipeline comparing two model states.
    Saves plots and returns summary DataFrames.
    """
    print("\n[SHAP] Running explainability analysis...")

    X_pre_sc = scaler.transform(X_pre[:500])
    X_post_sc = scaler.transform(X_post[:500])

    print("  Computing SHAP values (pre-drift model)...")
    shap_before, _ = compute_shap_values(model_before, X_pre_sc)

    print("  Computing SHAP values (post-drift model)...")
    shap_after, _ = compute_shap_values(model_after, X_post_sc)

    plot_shap_summary(
        shap_before, X_pre_sc,
        "Feature Importance — Pre-Drift Model",
        f"{REPORTS_DIR}/shap_pre_drift.png"
    )
    plot_shap_summary(
        shap_after, X_post_sc,
        "Feature Importance — Post-Drift Retrained Model",
        f"{REPORTS_DIR}/shap_post_drift.png"
    )
    shift_df = plot_drift_feature_shift(
        shap_before, shap_after,
        f"{REPORTS_DIR}/shap_drift_shift.png"
    )
    print("\n  Top features that shifted most due to drift:")
    print(shift_df.head(10).to_string(index=False))

    return shift_df


if __name__ == "__main__":
    import joblib
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_data, get_period_splits

    df = load_data("data/creditcard_with_drift.csv")
    splits = get_period_splits(df)

    model = joblib.load("models/xgb_baseline.pkl")
    scaler = joblib.load("models/scaler_baseline.pkl")

    X_pre = splits["pre_drift"][[f"V{i}" for i in range(1, 29)] + ["Amount"]].values[:500]
    shap_vals, _ = compute_shap_values(model, scaler.transform(X_pre))
    plot_shap_summary(shap_vals, X_pre, "Baseline SHAP", f"{REPORTS_DIR}/shap_baseline.png")
    print("Done.")
