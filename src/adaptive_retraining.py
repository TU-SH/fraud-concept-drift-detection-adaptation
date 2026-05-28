"""
adaptive_retraining.py
----------------------
When a drift detector fires, this module:
  1. Collects recent data (sliding window)
  2. Retrains the XGBoost model on the new distribution
  3. Evaluates improvement vs. the stale model
  4. Logs the retrain event with before/after metrics

This is the core differentiator: a self-healing fraud detector.
"""

import numpy as np
import pandas as pd
import joblib
import os
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"
MODEL_DIR = "models"
REPORTS_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


class AdaptiveRetrainer:
    """
    Maintains a sliding window of recent samples and retrains on drift.
    Tracks performance before/after each retraining event.
    """

    def __init__(
        self,
        window_size: int = 3000,
        min_samples_to_retrain: int = 500,
        retrain_log_path: str = "reports/retrain_log.csv"
    ):
        self.window_size = window_size
        self.min_samples = min_samples_to_retrain
        self.retrain_log_path = retrain_log_path

        self.X_buffer = []
        self.y_buffer = []
        self.retrain_count = 0
        self.retrain_log = []

        # Load baseline model
        self.model = None
        self.scaler = None
        self._load_baseline()

    def _load_baseline(self):
        """Load the pre-trained baseline model."""
        xgb_path = os.path.join(MODEL_DIR, "xgb_baseline.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler_baseline.pkl")
        if os.path.exists(xgb_path):
            self.model = joblib.load(xgb_path)
            self.scaler = joblib.load(scaler_path)
            print("[AdaptiveRetrainer] Loaded baseline XGBoost model.")
        else:
            print("[AdaptiveRetrainer] No baseline model found. Will train on first batch.")

    def add_samples(self, X_chunk: np.ndarray, y_chunk: np.ndarray):
        """Add new data to the sliding window buffer."""
        for x, y in zip(X_chunk, y_chunk):
            self.X_buffer.append(x)
            self.y_buffer.append(y)
        # Keep only last `window_size` samples
        if len(self.X_buffer) > self.window_size:
            excess = len(self.X_buffer) - self.window_size
            self.X_buffer = self.X_buffer[excess:]
            self.y_buffer = self.y_buffer[excess:]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using current model (scaled internally)."""
        if self.model is None:
            return np.zeros(len(X), dtype=int)
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return fraud probability scores."""
        if self.model is None:
            return np.zeros(len(X))
        X_scaled = self.scaler.transform(X) if self.scaler else X
        return self.model.predict_proba(X_scaled)[:, 1]

    def retrain(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        trigger: str = "drift_detected",
        chunk_index: int = 0
    ) -> dict:
        """
        Retrain model on buffered data + new samples.
        Returns dict of before/after metrics.
        """
        # Add new samples to buffer
        self.add_samples(X_new, y_new)

        if len(self.y_buffer) < self.min_samples:
            print(f"  [Retrainer] Not enough samples ({len(self.y_buffer)}). Skipping retrain.")
            return {}

        X_buf = np.array(self.X_buffer)
        y_buf = np.array(self.y_buffer)

        # Check class balance — need at least a few fraud samples
        if y_buf.sum() < 10:
            print("  [Retrainer] Too few fraud samples in buffer. Skipping retrain.")
            return {}

        # ── Before metrics (stale model on new data)
        before_metrics = {}
        if self.model is not None:
            y_prob_before = self.predict_proba(X_new)
            y_pred_before = self.predict(X_new)
            before_metrics = {
                "roc_auc": roc_auc_score(y_new, y_prob_before) if y_new.sum() > 0 else 0,
                "avg_precision": average_precision_score(y_new, y_prob_before) if y_new.sum() > 0 else 0,
                "f1": f1_score(y_new, y_pred_before, zero_division=0)
            }

        # ── Refit scaler on buffer data
        new_scaler = StandardScaler()
        X_buf_scaled = new_scaler.fit_transform(X_buf)

        # ── Retrain XGBoost
        t0 = time.time()
        scale_pos = max(1, int((y_buf == 0).sum() / max(1, (y_buf == 1).sum())))
        new_model = XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1
        )
        new_model.fit(X_buf_scaled, y_buf)
        elapsed = time.time() - t0

        # ── After metrics (new model on same new data)
        self.model = new_model
        self.scaler = new_scaler
        y_prob_after = self.predict_proba(X_new)
        y_pred_after = self.predict(X_new)
        after_metrics = {
            "roc_auc": roc_auc_score(y_new, y_prob_after) if y_new.sum() > 0 else 0,
            "avg_precision": average_precision_score(y_new, y_prob_after) if y_new.sum() > 0 else 0,
            "f1": f1_score(y_new, y_pred_after, zero_division=0)
        }

        self.retrain_count += 1
        log_entry = {
            "retrain_id": self.retrain_count,
            "timestamp": datetime.now().isoformat(),
            "chunk_index": chunk_index,
            "trigger": trigger,
            "buffer_size": len(y_buf),
            "fraud_rate_in_buffer": round(y_buf.mean(), 4),
            "retrain_time_sec": round(elapsed, 2),
            "before_roc_auc": round(before_metrics.get("roc_auc", 0), 4),
            "after_roc_auc": round(after_metrics["roc_auc"], 4),
            "before_avg_precision": round(before_metrics.get("avg_precision", 0), 4),
            "after_avg_precision": round(after_metrics["avg_precision"], 4),
            "before_f1": round(before_metrics.get("f1", 0), 4),
            "after_f1": round(after_metrics["f1"], 4),
        }
        self.retrain_log.append(log_entry)
        self._save_log()

        # Save updated model
        joblib.dump(self.model, os.path.join(MODEL_DIR, f"xgb_retrained_v{self.retrain_count}.pkl"))
        joblib.dump(self.scaler, os.path.join(MODEL_DIR, f"scaler_v{self.retrain_count}.pkl"))

        print(f"\n  ✅ RETRAIN #{self.retrain_count} complete | "
              f"ROC-AUC: {before_metrics.get('roc_auc', 0):.3f} → {after_metrics['roc_auc']:.3f} | "
              f"Time: {elapsed:.1f}s")

        return log_entry

    def _save_log(self):
        pd.DataFrame(self.retrain_log).to_csv(self.retrain_log_path, index=False)

    def get_retrain_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.retrain_log)


if __name__ == "__main__":
    # Quick test with random data
    retrainer = AdaptiveRetrainer()
    np.random.seed(42)
    X_fake = np.random.randn(1000, 29)
    y_fake = np.random.choice([0, 1], size=1000, p=[0.97, 0.03])
    retrainer.add_samples(X_fake, y_fake)
    result = retrainer.retrain(X_fake[:200], y_fake[:200], trigger="test")
    print(result)
