"""Sliding-window creation for direct multi-horizon forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(df: pd.DataFrame, input_length: int, horizon: int, target_col: str = "y") -> tuple[np.ndarray, np.ndarray]:
    """Return X windows with shape [n, input_length] and y with shape [n, horizon]."""
    values = df[target_col].to_numpy(dtype=float)
    n = len(values) - input_length - horizon + 1
    if n <= 0:
        raise ValueError(
            f"Not enough rows ({len(values)}) for input_length={input_length}, horizon={horizon}."
        )
    x = np.empty((n, input_length), dtype=float)
    y = np.empty((n, horizon), dtype=float)
    for i in range(n):
        x[i] = values[i : i + input_length]
        y[i] = values[i + input_length : i + input_length + horizon]
    return x, y


def make_forecast_origins(df: pd.DataFrame, input_length: int, horizon: int) -> pd.Series:
    """Return timestamps corresponding to the first predicted step of each window."""
    n = len(df) - input_length - horizon + 1
    if n <= 0:
        raise ValueError("Not enough rows for forecast origins.")
    return df["ds"].iloc[input_length : input_length + n].reset_index(drop=True)


def append_context(train: pd.DataFrame, part: pd.DataFrame, input_length: int) -> pd.DataFrame:
    """Prepend the last context rows from train-like data to a later split."""
    context = train.tail(input_length)
    return pd.concat([context, part], ignore_index=True)

