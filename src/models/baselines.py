"""Classical and simple baseline models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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


class FixedSARIMAForecaster:
    """SARIMA with fixed orders. Fitted once on train; per-window forecasts apply
    the pre-fit parameters to a new context via ``forward`` (no re-fit per window).

    Falls back to per-window fits if the installed statsforecast lacks ``forward``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self._fitted = None
        self._supports_forward = False

    def _make_model(self):
        from statsforecast.models import ARIMA

        return ARIMA(order=self.order, seasonal_order=self.seasonal_order)

    def fit(self, y_train: np.ndarray) -> "FixedSARIMAForecaster":
        try:
            self._fitted = self._make_model().fit(np.asarray(y_train, dtype=float))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit FixedSARIMA(order={self.order}, "
                f"seasonal_order={self.seasonal_order}) on train: {exc!r}"
            ) from exc
        self._supports_forward = hasattr(self._fitted, "forward")
        return self

    def _coerce(self, forecast: Any) -> np.ndarray:
        if isinstance(forecast, dict):
            return np.asarray(forecast["mean"], dtype=float)
        return np.asarray(forecast, dtype=float)

    def predict_from_context(self, context: np.ndarray, horizon: int) -> np.ndarray:
        context = np.asarray(context, dtype=float)
        if self._fitted is None:
            raise RuntimeError("FixedSARIMAForecaster.fit must be called before predict.")
        if self._supports_forward:
            forecast = self._fitted.forward(y=context, h=horizon)
        else:
            refit = self._make_model().fit(context)
            forecast = refit.predict(h=horizon)
        return self._coerce(forecast)

    def predict_windows(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return np.vstack([self.predict_from_context(row, horizon) for row in x])
