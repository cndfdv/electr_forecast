"""Classical and simple baseline models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SeasonalNaiveForecaster:
    """Repeat the latest observed seasonal pattern."""

    season_length: int = 24

    def fit(self, y: np.ndarray) -> "SeasonalNaiveForecaster":
        y = np.asarray(y, dtype=float)
        if len(y) < self.season_length:
            raise ValueError("Training series shorter than season_length.")
        self.history_ = y.copy()
        return self

    def predict_from_context(self, context: np.ndarray, horizon: int) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        if len(context) < self.season_length:
            raise ValueError("Context shorter than season_length.")
        pattern = context[-self.season_length :]
        reps = int(np.ceil(horizon / self.season_length))
        return np.tile(pattern, reps)[:horizon]

    def predict_windows(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return np.vstack([self.predict_from_context(row, horizon) for row in x])


class AutoARIMAForecaster:
    """StatsForecast AutoARIMA wrapper."""

    def __init__(self, season_length: int = 24, seasonal: bool = True, max_p: int = 3, max_q: int = 3):
        self.season_length = season_length
        self.seasonal = seasonal
        self.max_p = max_p
        self.max_q = max_q

    def fit_predict(self, y: np.ndarray, horizon: int) -> np.ndarray:
        try:
            from statsforecast.models import AutoARIMA
        except ImportError as exc:
            raise ImportError("AutoARIMA requires statsforecast. Install the project environment.") from exc

        model = AutoARIMA(
            season_length=self.season_length,
            seasonal=self.seasonal,
            max_p=self.max_p,
            max_q=self.max_q,
        )
        fitted = model.fit(np.asarray(y, dtype=float))
        forecast = fitted.predict(h=horizon)
        if isinstance(forecast, dict):
            return np.asarray(forecast["mean"], dtype=float)
        return np.asarray(forecast, dtype=float)

    def predict_windows(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return np.vstack([self.fit_predict(row, horizon) for row in x])

