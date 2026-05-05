"""Direct multi-output XGBoost forecasting wrapper."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor


def calendar_features(ds: pd.Series) -> np.ndarray:
    """Create cyclic calendar features from timestamps."""
    ts = pd.to_datetime(ds)
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    month = ts.dt.month.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7),
            np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ]
    )


def make_xgb_features(x: np.ndarray, origins: pd.Series | None = None) -> np.ndarray:
    """Lag/rolling/calendar features from input windows."""
    lag_1 = x[:, -1]
    lag_24 = x[:, -24]
    lag_168 = x[:, -168]
    roll_24 = x[:, -24:].mean(axis=1)
    roll_168 = x[:, -168:].mean(axis=1)
    base = np.column_stack([lag_1, lag_24, lag_168, roll_24, roll_168])
    if origins is None:
        return base
    return np.column_stack([base, calendar_features(origins)])


class XGBoostDirectForecaster:
    """Train one multi-output regressor for all horizon steps."""

    def __init__(self, **params):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("XGBoostDirectForecaster requires xgboost.") from exc

        defaults = {
            "objective": "reg:squarederror",
            "max_depth": 3,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": 42,
            "n_jobs": 4,
        }
        defaults.update(params)
        self.model = MultiOutputRegressor(XGBRegressor(**defaults))

    def fit(self, x: np.ndarray, y: np.ndarray, origins: pd.Series | None = None) -> "XGBoostDirectForecaster":
        self.model.fit(make_xgb_features(x, origins), y)
        return self

    def predict(self, x: np.ndarray, origins: pd.Series | None = None) -> np.ndarray:
        return np.asarray(self.model.predict(make_xgb_features(x, origins)), dtype=float)

