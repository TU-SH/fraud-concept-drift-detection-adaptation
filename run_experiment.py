"""
run_experiment.py
-----------------
END-TO-END PIPELINE:
  1. Load data and train baseline model (pre-drift only)
  2. Stream all data through the drift monitor chunk-by-chunk
  3. On drift detection → trigger adaptive retraining
  4. Track performance metrics at every chunk
  5. Compare static model vs adaptive model
  6. Generate SHAP explainability plots
  7. Save all results to reports/

Run: python run_experiment.py
"""

import sys
import os
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from src.data_preprocessing import (
    load_data, get_period_splits, prepare_train_test,
    get_streaming_chunks, FEATURE_COLS, TARGET_COL
)
from src.baseline_model import train_xgboost, train_random_forest, evaluate_model, save_model
from src.drift_detector import DriftMonitor
from src.adaptive_retraining import AdaptiveRetrainer
from src.explainability import run_full_shap_analysis

REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
CHUNK_SIZE = 400


def compute_chunk_metrics(model, scaler, X_chunk, y_chunk) -> dict:
    """Compute metrics for one data chunk."""
    if y_chunk.sum() == 0:
        return {"roc_auc": 0.5, "avg_precision": 0.0, "f1": 0.0, "error_rate": 0.0}
    X_sc = scaler.transform(X_chunk)
    y_prob = model.predict_proba(X_sc)[:, 1]
    y_pred = model.predict(X_sc)
    return {
        "roc_auc": roc_auc_score(y_chunk, y_prob),
        "avg_precision": average_precision_score(y_chunk, y_prob),
        "f1": f1_score(y_chunk, y_pred, zero_division=0),
        "error_rate": (y_pred != y_chunk).mean()
    }


def main():
    print("=" * 65)
    print("  FRAUD DETECTION WITH CONCEPT DRIFT DETECTION & ADAPTATION")
    print("=" * 65)

    # ── 1. Load data
    print("\n[Step 1] Loading dataset...")
    df = load_data("data/creditcard_with_drift.csv")
    print("\nPeriod splits:")
    splits = get_period_splits(df)

    # ── 2. Train baseline on pre-drift data
    print("\n[Step 2] Training baseline model (pre-drift data only)...")
    pre = splits["pre_drift"]
    X_train, X_test, y_train, y_test, scaler_base = prepare_train_test(pre)
    xgb_baseline = train_xgboost(X_train, y_train)
    evaluate_model(xgb_baseline, X_test, y_test, "XGBoost Baseline (Pre-Drift Test Set)")
    save_model(xgb_baseline, "xgb_baseline")
    save_model(scaler_base, "scaler_baseline")

    # ── 3. Set up drift monitor + adaptive retrainer
    print("\n[Step 3] Initialising drift monitor and adaptive retrainer...")
    monitor = DriftMonitor(use_river=True)
    retrainer = AdaptiveRetrainer(window_size=3000, min_samples_to_retrain=600)

    # ── 4. Stream ALL data (all three periods) through the pipeline
    print("\n[Step 4] Streaming data through pipeline...")
    print(f"  Chunk size: {CHUNK_SIZE} samples")
    print(f"  Total samples: {len(df):,}")

    # Track metrics per chunk for both static and adaptive models
    static_metrics = []
    adaptive_metrics = []
    drift_chunks = []
    retrain_chunks = []
    chunk_periods = []

    # Static model uses baseline throughout
    static_model = xgb_baseline
    static_scaler = scaler_base

    for chunk_idx, X_chunk, y_chunk in get_streaming_chunks(df, chunk_size=CHUNK_SIZE):
        period = df["Period"].iloc[chunk_idx * CHUNK_SIZE]
        chunk_periods.append(period)

        # ── Static model predictions (no adaptation)
        sm = compute_chunk_metrics(static_model, static_scaler, X_chunk, y_chunk)
        static_metrics.append(sm)

        # ── Adaptive model predictions
        am = compute_chunk_metrics(retrainer.model or static_model,
                                   retrainer.scaler or static_scaler,
                                   X_chunk, y_chunk)
        adaptive_metrics.append(am)

        # ── Feed predictions to drift monitor
        X_sc = (retrainer.scaler or static_scaler).transform(X_chunk)
        y_pred = (retrainer.model or static_model).predict(X_sc)
        drifts = monitor.update_chunk(y_chunk, y_pred)

        if drifts > 0:
            drift_chunks.append(chunk_idx)
            print(f"\n[DRIFT] Chunk {chunk_idx} | Period: {period} | Triggering retrain...")
            log = retrainer.retrain(X_chunk, y_chunk, trigger="detector_fired", chunk_index=chunk_idx)
            if log:
                retrain_chunks.append(chunk_idx)
                monitor.reset()

        # Add to retrainer buffer regardless
        retrainer.add_samples(X_chunk, y_chunk)

        if chunk_idx % 10 == 0:
            print(f"  Chunk {chunk_idx:3d}/{len(df)//CHUNK_SIZE} | "
                  f"Period: {period:12s} | "
                  f"Static F1: {sm['f1']:.3f} | "
                  f"Adaptive F1: {am['f1']:.3f}")

    print("\n[Step 4] Streaming complete.")

    # ── 5. Results summary
    print("\n[Step 5] Generating results...")
    static_df = pd.DataFrame(static_metrics)
    adaptive_df = pd.DataFrame(adaptive_metrics)

    print("\n── Static Model (no adaptation) ──")
    print(f"  Mean ROC-AUC      : {static_df['roc_auc'].mean():.4f}")
    print(f"  Mean Avg Precision: {static_df['avg_precision'].mean():.4f}")
    print(f"  Mean F1 (fraud)   : {static_df['f1'].mean():.4f}")

    print("\n── Adaptive Model (with drift detection + retraining) ──")
    print(f"  Mean ROC-AUC      : {adaptive_df['roc_auc'].mean():.4f}")
    print(f"  Mean Avg Precision: {adaptive_df['avg_precision'].mean():.4f}")
    print(f"  Mean F1 (fraud)   : {adaptive_df['f1'].mean():.4f}")

    print(f"\n  Total drift events detected : {len(monitor.drift_events)}")
    print(f"  Total retraining events     : {retrainer.retrain_count}")

    # ── 6. Plots
    print("\n[Step 6] Generating plots...")
    _plot_performance_comparison(static_df, adaptive_df, drift_chunks, retrain_chunks, chunk_periods)
    _plot_drift_events(monitor)
    _plot_retrain_improvement(retrainer)

    # ── 7. SHAP analysis
    print("\n[Step 7] Running SHAP explainability...")
    try:
        X_pre_raw = splits["pre_drift"][FEATURE_COLS].values
        X_post_raw = splits["post_drift"][FEATURE_COLS].values
        final_model = retrainer.model or xgb_baseline
        final_scaler = retrainer.scaler or scaler_base
        run_full_shap_analysis(xgb_baseline, final_model, X_pre_raw, X_post_raw, final_scaler)
    except Exception as e:
        print(f"  [SHAP] Skipped due to error: {e}")

    # ── 8. Save metrics
    static_df.to_csv(f"{REPORTS_DIR}/static_model_metrics.csv", index=False)
    adaptive_df.to_csv(f"{REPORTS_DIR}/adaptive_model_metrics.csv", index=False)
    monitor.get_drift_summary().to_csv(f"{REPORTS_DIR}/drift_events.csv", index=False)

    print("\n✅ Experiment complete. All results saved to reports/")
    print(f"   - reports/static_model_metrics.csv")
    print(f"   - reports/adaptive_model_metrics.csv")
    print(f"   - reports/drift_events.csv")
    print(f"   - reports/retrain_log.csv")
    print(f"   - reports/performance_comparison.png")
    print(f"   - reports/drift_events_plot.png")
    print(f"   - reports/shap_drift_shift.png")


def _plot_performance_comparison(static_df, adaptive_df, drift_chunks, retrain_chunks, chunk_periods):
    """Plot F1 score over time for static vs adaptive model with drift markers."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    metrics = [("f1", "F1 Score (Fraud Class)"),
               ("roc_auc", "ROC-AUC Score"),
               ("avg_precision", "Average Precision")]

    # Map period to color band
    period_colors = {"pre_drift": "#E3F2FD", "post_drift": "#FFEBEE", "recovery": "#E8F5E9"}

    for ax, (metric, title) in zip(axes, metrics):
        chunks = range(len(static_df))

        # Background color bands by period
        prev_period = None
        band_start = 0
        for i, p in enumerate(chunk_periods + [None]):
            if p != prev_period and prev_period is not None:
                ax.axvspan(band_start, i, alpha=0.3,
                           color=period_colors.get(prev_period, "white"), label=f"_{prev_period}")
                band_start = i
            prev_period = p

        ax.plot(chunks, static_df[metric], color="#EF5350", linewidth=1.5,
                label="Static Model", alpha=0.85)
        ax.plot(chunks, adaptive_df[metric], color="#42A5F5", linewidth=1.5,
                label="Adaptive Model", alpha=0.85)

        # Rolling average
        window = 8
        static_roll = static_df[metric].rolling(window, min_periods=1).mean()
        adaptive_roll = adaptive_df[metric].rolling(window, min_periods=1).mean()
        ax.plot(chunks, static_roll, color="#B71C1C", linewidth=2.5, linestyle="--", alpha=0.7)
        ax.plot(chunks, adaptive_roll, color="#0D47A1", linewidth=2.5, linestyle="--", alpha=0.7)

        # Drift and retrain markers
        for dc in drift_chunks:
            if dc < len(static_df):
                ax.axvline(x=dc, color="orange", linewidth=1.5, linestyle=":", alpha=0.8)
        for rc in retrain_chunks:
            if rc < len(static_df):
                ax.axvline(x=rc, color="green", linewidth=2.0, linestyle="-.", alpha=0.9)

        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if ax == axes[0]:
            ax.legend(loc="upper right", fontsize=9)

    # Custom legend
    patches = [
        mpatches.Patch(color="#EF5350", label="Static Model"),
        mpatches.Patch(color="#42A5F5", label="Adaptive Model"),
        mpatches.Patch(color="orange", label="Drift Detected"),
        mpatches.Patch(color="green", label="Model Retrained"),
        mpatches.Patch(color="#BBDEFB", alpha=0.5, label="Pre-Drift Period"),
        mpatches.Patch(color="#FFCDD2", alpha=0.5, label="Post-Drift Period"),
        mpatches.Patch(color="#C8E6C9", alpha=0.5, label="Recovery Period"),
    ]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.01))

    axes[-1].set_xlabel("Chunk Index (time →)", fontsize=11)
    fig.suptitle("Static vs Adaptive Fraud Detector: Performance Under Concept Drift",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/performance_comparison.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {path}")


def _plot_drift_events(monitor: DriftMonitor):
    """Plot rolling error rate with drift event markers."""
    rolling = monitor.rolling_error_rate(window=150)
    drift_df = monitor.get_drift_summary()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(rolling, color="#546E7A", linewidth=1.5, label="Rolling Error Rate (w=150)")
    ax.fill_between(range(len(rolling)), rolling, alpha=0.15, color="#546E7A")

    colors = {"ADWIN": "#FF5722", "DDM": "#9C27B0", "PageHinkley": "#2196F3"}
    labeled = set()
    for _, row in drift_df.iterrows():
        if row["drift_type"] == "drift":
            c = colors.get(row["detector"], "red")
            lbl = row["detector"] if row["detector"] not in labeled else None
            ax.axvline(x=row["sample_index"], color=c, linewidth=1.5,
                       linestyle="--", alpha=0.8, label=lbl)
            labeled.add(row["detector"])

    ax.set_xlabel("Sample Index", fontsize=11)
    ax.set_ylabel("Rolling Error Rate", fontsize=11)
    ax.set_title("Concept Drift Detection Events (ADWIN | DDM | PageHinkley)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = f"{REPORTS_DIR}/drift_events_plot.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {path}")


def _plot_retrain_improvement(retrainer: AdaptiveRetrainer):
    """Bar chart: ROC-AUC before vs after each retraining event."""
    log = retrainer.get_retrain_log()
    if log.empty:
        print("  [Plot] No retrain events to plot.")
        return

    fig, ax = plt.subplots(figsize=(max(6, len(log) * 1.5), 5))
    x = np.arange(len(log))
    width = 0.35
    ax.bar(x - width / 2, log["before_roc_auc"], width, label="Before Retrain",
           color="#EF9A9A", edgecolor="black", alpha=0.9)
    ax.bar(x + width / 2, log["after_roc_auc"], width, label="After Retrain",
           color="#80CBC4", edgecolor="black", alpha=0.9)

    for i, row in log.iterrows():
        delta = row["after_roc_auc"] - row["before_roc_auc"]
        sign = "+" if delta >= 0 else ""
        ax.text(i, max(row["before_roc_auc"], row["after_roc_auc"]) + 0.01,
                f"{sign}{delta:.3f}", ha="center", fontsize=9, fontweight="bold",
                color="green" if delta >= 0 else "red")

    ax.set_xticks(x)
    ax.set_xticklabels([f"Retrain #{r}" for r in log["retrain_id"]], rotation=15)
    ax.set_ylabel("ROC-AUC Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("ROC-AUC Before vs After Each Drift-Triggered Retraining",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = f"{REPORTS_DIR}/retrain_improvement.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [Plot] Saved → {path}")


if __name__ == "__main__":
    main()
