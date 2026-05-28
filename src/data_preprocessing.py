"""
data_preprocessing.py
---------------------
Loads, cleans, and prepares the fraud dataset for drift detection experiments.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"


def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    print(f"[load_data] Loaded {len(df):,} rows | Fraud rate: {df[TARGET_COL].mean():.2%}")
    return df


def scale_features(df: pd.DataFrame, scaler=None, fit: bool = True):
    """
    Scale Amount and V-features. Returns (scaled_df, fitted_scaler).
    Pass fit=False + existing scaler to transform without refitting.
    """
    df = df.copy()
    if fit:
        scaler = StandardScaler()
        df[FEATURE_COLS] = scaler.fit_transform(df[FEATURE_COLS])
    else:
        df[FEATURE_COLS] = scaler.transform(df[FEATURE_COLS])
    return df, scaler


def get_period_splits(df: pd.DataFrame):
    """
    Split dataset into three temporal periods matching concept drift phases.
    Returns dict: {'pre_drift': df1, 'post_drift': df2, 'recovery': df3}
    """
    splits = {}
    for period in ["pre_drift", "post_drift", "recovery"]:
        splits[period] = df[df["Period"] == period].reset_index(drop=True)
        n = len(splits[period])
        fraud = splits[period][TARGET_COL].mean()
        print(f"  [{period}] {n:,} rows | fraud rate: {fraud:.2%}")
    return splits


def prepare_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Standard train/test split with scaling."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_streaming_chunks(df: pd.DataFrame, chunk_size: int = 500):
    """
    Generator: yields (chunk_index, X_chunk, y_chunk) for streaming simulation.
    Used by drift detectors that process data sequentially.
    """
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values
    n = len(df)
    for i in range(0, n, chunk_size):
        yield i // chunk_size, X[i:i + chunk_size], y[i:i + chunk_size]


if __name__ == "__main__":
    df = load_data("data/creditcard_with_drift.csv")
    print("\nPeriod splits:")
    splits = get_period_splits(df)
    print("\nSample rows:")
    print(df[FEATURE_COLS[:5] + [TARGET_COL, "Period"]].head())
