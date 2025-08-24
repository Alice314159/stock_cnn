
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd
from .feature_engineering import engineer_features
from .labeling import make_binary_labels

def _enforce_float32(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if df[c].dtype.kind in "fi":
            df[c] = df[c].astype(np.float32, copy=False)
    return df

class DataProcessor:
    """Load CSV, engineer features, label, and split into train/val/test (time order)."""
    def __init__(self, csv_path: str, window_size: int, stride: int,
                 prediction_horizon: int, binary_threshold: float):
        self.csv_path = csv_path
        self.window_size = window_size
        self.stride = stride
        self.prediction_horizon = prediction_horizon
        self.binary_threshold = binary_threshold

    def load_and_prepare(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)
        df = engineer_features(df)
        df = make_binary_labels(df, horizon=self.prediction_horizon, threshold=self.binary_threshold)
        df = df.dropna().reset_index(drop=True)
        return _enforce_float32(df)

    def split(self, df: pd.DataFrame, train_ratio: float, val_ratio: float):
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = df.iloc[:n_train]
        val = df.iloc[n_train:n_train+n_val]
        test = df.iloc[n_train+n_val:]
        return train, val, test
