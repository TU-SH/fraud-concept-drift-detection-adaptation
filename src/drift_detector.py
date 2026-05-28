"""
drift_detector.py
-----------------
Implements three concept drift detectors:
  1. ADWIN   – Adaptive Windowing (distribution-based, fast)
  2. DDM     – Drift Detection Method (error-rate based)
  3. Page-Hinkley – Sequential drift test (mean-shift detection)

All detectors are wrapped in a unified DriftMonitor class that can run
in batch or streaming mode and logs all drift events.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
import warnings
warnings.filterwarnings("ignore")

try:
    from river.drift import ADWIN, DDM, PageHinkley
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False
    print("[Warning] river not installed. Using fallback implementations.")


# ────────────────────────────────────────────────
#  Fallback implementations (no river needed)
# ────────────────────────────────────────────────

class ADWINFallback:
    """Simplified ADWIN: splits window when variance changes significantly."""
    def __init__(self, delta: float = 0.002):
        self.delta = delta
        self.window = []
        self.drift_detected = False
        self.warning_detected = False

    def update(self, value: float):
        self.window.append(value)
        self.drift_detected = False
        if len(self.window) > 50:
            half = len(self.window) // 2
            m1 = np.mean(self.window[:half])
            m2 = np.mean(self.window[half:])
            n1, n2 = half, len(self.window) - half
            variance = np.var(self.window) + 1e-9
            epsilon_cut = np.sqrt((1 / (2 * min(n1, n2))) * np.log(4 * len(self.window) / self.delta))
            if abs(m2 - m1) >= epsilon_cut:
                self.drift_detected = True
                self.window = self.window[half:]  # reset to new window


class DDMFallback:
    """DDM: monitors error rate mean + std. Drift when p + 3s > p_min + 3s_min."""
    def __init__(self, min_samples: int = 30, warning_level: float = 2.0, drift_level: float = 3.0):
        self.min_samples = min_samples
        self.warning_level = warning_level
        self.drift_level = drift_level
        self._reset()
        self.drift_detected = False
        self.warning_detected = False

    def _reset(self):
        self.n = 1
        self.p = 1.0
        self.s = 0.0
        self.p_min = float("inf")
        self.s_min = float("inf")

    def update(self, error: int):  # error = 1 if wrong, 0 if correct
        self.drift_detected = False
        self.warning_detected = False
        self.n += 1
        self.p += (error - self.p) / self.n
        self.s = np.sqrt(self.p * (1 - self.p) / self.n)

        if self.n >= self.min_samples:
            if self.p + self.s < self.p_min + self.s_min:
                self.p_min = self.p
                self.s_min = self.s

            if self.p + self.s > self.p_min + self.drift_level * self.s_min:
                self.drift_detected = True
                self._reset()
            elif self.p + self.s > self.p_min + self.warning_level * self.s_min:
                self.warning_detected = True


class PageHinkleyFallback:
    """Page-Hinkley: detects persistent mean increase in a stream."""
    def __init__(self, delta: float = 0.005, threshold: float = 50.0, alpha: float = 0.9999):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.x_mean = 0.0
        self.sum = 0.0
        self.n = 0
        self.drift_detected = False
        self.warning_detected = False

    def update(self, value: float):
        self.drift_detected = False
        self.n += 1
        self.x_mean = self.alpha * self.x_mean + (1 - self.alpha) * value
        self.sum = max(0, self.sum + value - self.x_mean - self.delta)
        if self.sum > self.threshold:
            self.drift_detected = True
            self.sum = 0.0


# ────────────────────────────────────────────────
#  Drift Event Dataclass
# ────────────────────────────────────────────────

@dataclass
class DriftEvent:
    detector: str
    sample_index: int
    chunk_index: int
    drift_type: str  # 'drift' or 'warning'
    error_rate_at_detection: float


# ────────────────────────────────────────────────
#  Unified DriftMonitor
# ────────────────────────────────────────────────

class DriftMonitor:
    """
    Wraps ADWIN, DDM, and Page-Hinkley detectors.
    Feed predictions + true labels sample-by-sample or in chunks.
    Logs all drift events with exact sample indices.
    """

    def __init__(self, use_river: bool = True):
        self.use_river = use_river and RIVER_AVAILABLE
        self._init_detectors()
        self.drift_events: List[DriftEvent] = []
        self.error_history: List[float] = []
        self.sample_count = 0
        self.chunk_count = 0

    def _init_detectors(self):
        if self.use_river:
            self.adwin = ADWIN(delta=0.002)
            self.ddm = DDM(warm_start=30, warning_level=2.0, drift_level=3.0)
            self.ph = PageHinkley(delta=0.005, threshold=50.0, alpha=0.9999)
        else:
            self.adwin = ADWINFallback(delta=0.002)
            self.ddm = DDMFallback()
            self.ph = PageHinkleyFallback()

    def update(self, y_true: int, y_pred: int):
        """Process a single sample. Returns True if any detector fires drift."""
        error = int(y_true != y_pred)
        self.error_history.append(error)
        self.sample_count += 1
        drift_fired = False

        detectors = {
            "ADWIN": (self.adwin, error),
            "DDM": (self.ddm, error),
            "PageHinkley": (self.ph, float(error)),
        }

        for name, (det, val) in detectors.items():
            det.update(val)
            if det.drift_detected:
                event = DriftEvent(
                    detector=name,
                    sample_index=self.sample_count,
                    chunk_index=self.chunk_count,
                    drift_type="drift",
                    error_rate_at_detection=np.mean(self.error_history[-100:])
                )
                self.drift_events.append(event)
                drift_fired = True
                print(f"  🚨 DRIFT DETECTED [{name}] at sample {self.sample_count} "
                      f"(chunk {self.chunk_count}) | err_rate={event.error_rate_at_detection:.3f}")

            if hasattr(det, "warning_detected") and det.warning_detected:
                event = DriftEvent(
                    detector=name,
                    sample_index=self.sample_count,
                    chunk_index=self.chunk_count,
                    drift_type="warning",
                    error_rate_at_detection=np.mean(self.error_history[-100:])
                )
                self.drift_events.append(event)
                print(f"  ⚠️  WARNING [{name}] at sample {self.sample_count}")

        return drift_fired

    def update_chunk(self, y_true_arr, y_pred_arr):
        """Process a full chunk of predictions."""
        self.chunk_count += 1
        drifts = 0
        for yt, yp in zip(y_true_arr, y_pred_arr):
            if self.update(int(yt), int(yp)):
                drifts += 1
        return drifts

    def reset(self):
        """Reinitialise detectors after drift-triggered retraining."""
        self._init_detectors()
        print("  🔄 Detectors reset after retraining.")

    def get_drift_summary(self) -> pd.DataFrame:
        if not self.drift_events:
            return pd.DataFrame()
        return pd.DataFrame([{
            "detector": e.detector,
            "sample_index": e.sample_index,
            "chunk_index": e.chunk_index,
            "drift_type": e.drift_type,
            "error_rate": e.error_rate_at_detection
        } for e in self.drift_events])

    def rolling_error_rate(self, window: int = 200) -> List[float]:
        """Compute rolling error rate for visualisation."""
        errors = np.array(self.error_history, dtype=float)
        if len(errors) < window:
            return list(errors)
        return [np.mean(errors[max(0, i - window):i]) for i in range(1, len(errors) + 1)]


if __name__ == "__main__":
    # Quick smoke test
    monitor = DriftMonitor(use_river=False)
    np.random.seed(42)

    # Normal period — low error
    for _ in range(500):
        yt = np.random.choice([0, 1], p=[0.98, 0.02])
        yp = yt if np.random.rand() > 0.05 else 1 - yt
        monitor.update(yt, yp)

    # Drift period — high error
    for _ in range(500):
        yt = np.random.choice([0, 1], p=[0.97, 0.03])
        yp = yt if np.random.rand() > 0.35 else 1 - yt
        monitor.update(yt, yp)

    print("\nDrift Summary:")
    print(monitor.get_drift_summary())
