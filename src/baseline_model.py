"""
baseline_model.py
-----------------
Train baseline fraud detection models (XGBoost + Random Forest).
Evaluates on pre-drift data only — this is the 'naive static' model
that will degrade when concept drift hits.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# ── Paths
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """Compute and print full evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(f"  ROC-AUC Score      : {roc_auc:.4f}")
    print(f"  Avg Precision (PR) : {ap:.4f}")
    print(f"  Confusion Matrix   :\n{cm}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return {
        "roc_auc": roc_auc,
        "avg_precision": ap,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist()
    }


def train_xgboost(X_train, y_train) -> XGBClassifier:
    """Train an XGBoost classifier with class-imbalance handling."""
    scale_pos = int((y_train == 0).sum() / (y_train == 1).sum())
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Train a Random Forest with SMOTE oversampling."""
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_res, y_res)
    return model


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[save_model] Saved → {path}")


def load_model(name: str):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    return joblib.load(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_preprocessing import load_data, get_period_splits, prepare_train_test

    df = load_data("data/creditcard_with_drift.csv")
    splits = get_period_splits(df)

    # Train ONLY on pre-drift data (this is the static baseline)
    pre = splits["pre_drift"]
    X_train, X_test, y_train, y_test, scaler = prepare_train_test(pre)

    print("\n[1] Training XGBoost baseline...")
    xgb_model = train_xgboost(X_train, y_train)
    xgb_metrics = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Pre-Drift)")
    save_model(xgb_model, "xgb_baseline")
    save_model(scaler, "scaler_baseline")

    print("\n[2] Training Random Forest baseline...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "RandomForest (Pre-Drift)")
    save_model(rf_model, "rf_baseline")

    print("\n✅ Baseline models trained and saved.")
