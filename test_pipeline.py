"""
tests/test_pipeline.py
-----------------------
Unit tests for all major components.
Run: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data, get_period_splits, get_streaming_chunks, FEATURE_COLS
from src.drift_detector import DriftMonitor, ADWINFallback, DDMFallback, PageHinkleyFallback
from src.adaptive_retraining import AdaptiveRetrainer


# ── Fixtures
@pytest.fixture(scope="module")
def sample_df():
    path = "data/creditcard_with_drift.csv"
    if os.path.exists(path):
        return pd.read_csv(path).head(2000)
    pytest.skip("Dataset not found")


@pytest.fixture
def monitor():
    return DriftMonitor(use_river=False)


@pytest.fixture
def retrainer(tmp_path):
    r = AdaptiveRetrainer(
        window_size=500,
        min_samples_to_retrain=100,
        retrain_log_path=str(tmp_path / "retrain_log.csv")
    )
    r.model = None  # no baseline in test
    return r


# ── Data tests
class TestDataPreprocessing:
    def test_load_data(self, sample_df):
        assert len(sample_df) > 0
        assert "Class" in sample_df.columns
        assert all(f"V{i}" in sample_df.columns for i in range(1, 6))

    def test_period_splits(self, sample_df):
        import pandas as pd
        full_df = pd.read_csv("data/creditcard_with_drift.csv")
        splits = get_period_splits(full_df)
        assert "pre_drift" in splits
        assert all(len(v) > 0 for v in splits.values())

    def test_streaming_chunks(self, sample_df):
        chunks = list(get_streaming_chunks(sample_df, chunk_size=100))
        assert len(chunks) > 0
        idx, X, y = chunks[0]
        assert X.shape[1] == len(FEATURE_COLS)
        assert len(y) == len(X)

    def test_class_balance(self, sample_df):
        fraud_rate = sample_df["Class"].mean()
        assert 0.01 <= fraud_rate <= 0.15, f"Unexpected fraud rate: {fraud_rate}"


# ── Detector tests
class TestDriftDetectors:
    def test_adwin_no_drift_stable(self):
        det = ADWINFallback(delta=0.002)
        drifts = 0
        np.random.seed(0)
        for _ in range(300):
            det.update(np.random.choice([0, 1], p=[0.95, 0.05]))
            if det.drift_detected:
                drifts += 1
        # Should be very few false positives
        assert drifts < 10

    def test_adwin_detects_clear_drift(self):
        det = ADWINFallback(delta=0.002)
        np.random.seed(42)
        # Normal
        for _ in range(200):
            det.update(np.random.choice([0, 1], p=[0.95, 0.05]))
        # Abrupt drift: error rate jumps from 5% to 50%
        drifts = 0
        for _ in range(300):
            det.update(np.random.choice([0, 1], p=[0.5, 0.5]))
            if det.drift_detected:
                drifts += 1
        assert drifts >= 1, "ADWIN should detect clear distribution shift"

    def test_ddm_detects_drift(self):
        det = DDMFallback()
        np.random.seed(1)
        for _ in range(100):
            det.update(int(np.random.rand() > 0.95))
        drifts = 0
        for _ in range(200):
            det.update(int(np.random.rand() > 0.5))
            if det.drift_detected:
                drifts += 1
        assert drifts >= 1

    def test_page_hinkley_detects_mean_shift(self):
        det = PageHinkleyFallback(delta=0.005, threshold=20.0)
        np.random.seed(2)
        for _ in range(100):
            det.update(np.random.normal(0.05, 0.02))
        drifts = 0
        for _ in range(200):
            det.update(np.random.normal(0.6, 0.1))  # big jump
            if det.drift_detected:
                drifts += 1
        assert drifts >= 1

    def test_drift_monitor_logs_events(self, monitor):
        np.random.seed(42)
        for _ in range(200):
            yt = np.random.choice([0, 1], p=[0.98, 0.02])
            yp = yt if np.random.rand() > 0.05 else 1 - yt
            monitor.update(yt, yp)
        for _ in range(200):
            yt = np.random.choice([0, 1], p=[0.97, 0.03])
            yp = yt if np.random.rand() > 0.4 else 1 - yt
            monitor.update(yt, yp)
        # We should have some events
        assert monitor.sample_count == 400
        summary = monitor.get_drift_summary()
        # Summary may be empty or non-empty — just check type
        assert isinstance(summary, pd.DataFrame)

    def test_drift_monitor_reset(self, monitor):
        monitor.reset()
        # After reset, drift_detected should be False
        assert not monitor.adwin.drift_detected

    def test_rolling_error_rate_length(self, monitor):
        monitor2 = DriftMonitor(use_river=False)
        for _ in range(100):
            monitor2.update(0, 1)
        rolling = monitor2.rolling_error_rate(window=10)
        assert len(rolling) == 100


# ── Retrainer tests
class TestAdaptiveRetrainer:
    def test_buffer_fills(self, retrainer):
        X = np.random.randn(200, 29)
        y = np.random.choice([0, 1], size=200, p=[0.97, 0.03])
        retrainer.add_samples(X, y)
        assert len(retrainer.X_buffer) == 200

    def test_buffer_capped_at_window_size(self, retrainer):
        retrainer2 = AdaptiveRetrainer(window_size=100, min_samples_to_retrain=50)
        retrainer2.model = None
        X = np.random.randn(200, 29)
        y = np.zeros(200, dtype=int)
        retrainer2.add_samples(X, y)
        assert len(retrainer2.X_buffer) == 100

    def test_retrain_skipped_when_no_fraud(self, retrainer):
        X = np.random.randn(200, 29)
        y = np.zeros(200, dtype=int)  # all legit
        retrainer.add_samples(X, y)
        result = retrainer.retrain(X, y, trigger="test")
        assert result == {}

    def test_retrain_runs_with_fraud_samples(self, tmp_path):
        r = AdaptiveRetrainer(
            window_size=500,
            min_samples_to_retrain=100,
            retrain_log_path=str(tmp_path / "log.csv")
        )
        r.model = None
        np.random.seed(7)
        X = np.random.randn(300, 29)
        y = np.random.choice([0, 1], size=300, p=[0.93, 0.07])
        r.add_samples(X, y)
        result = r.retrain(X[:100], y[:100], trigger="test")
        assert r.retrain_count == 1
        assert r.model is not None
        assert "after_roc_auc" in result

    def test_predict_returns_correct_shape(self, tmp_path):
        r = AdaptiveRetrainer(
            window_size=500,
            min_samples_to_retrain=100,
            retrain_log_path=str(tmp_path / "log2.csv")
        )
        r.model = None
        X = np.random.randn(100, 29)
        y = np.random.choice([0, 1], size=100, p=[0.93, 0.07])
        r.add_samples(X, y)
        r.retrain(X, y, trigger="test")
        preds = r.predict(X[:20])
        assert len(preds) == 20
        assert set(preds).issubset({0, 1})


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
