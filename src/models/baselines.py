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


DEFAULT_SARIMA_GRID: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] = [
    ((1, 1, 1), (1, 1, 1, 24)),
    ((2, 1, 1), (1, 1, 1, 24)),
    ((1, 1, 2), (1, 1, 1, 24)),
    ((2, 1, 2), (1, 1, 1, 24)),
    ((2, 1, 2), (2, 1, 1, 24)),
    ((2, 1, 2), (1, 1, 2, 24)),
]


def _model_aic(fitted_model: Any) -> float:
    """Best-effort AICc/AIC extraction across statsforecast versions."""
    for attr in ("aicc", "aic"):
        if hasattr(fitted_model, attr):
            value = getattr(fitted_model, attr)
            if callable(value):
                try:
                    value = value()
                except Exception:
                    continue
            if value is not None:
                return float(value)
    inner = getattr(fitted_model, "model_", None) or getattr(fitted_model, "model", None)
    if isinstance(inner, dict):
        for key in ("aicc", "aic"):
            if key in inner and inner[key] is not None:
                return float(inner[key])
    return float("inf")


class FixedSARIMAForecaster:
    """SARIMA with a small grid of candidate orders. Each candidate is fitted once
    on train; the lowest-AICc fit is kept and reused on every test window via
    ``forward`` (no re-fit per window).

    Falls back to per-window fits if the installed statsforecast lacks ``forward``.
    """

    def __init__(
        self,
        order: tuple[int, int, int] | None = None,
        seasonal_order: tuple[int, int, int, int] | None = None,
        grid: list[tuple[tuple[int, int, int], tuple[int, int, int, int]]] | None = None,
    ) -> None:
        if grid is not None:
            self.grid = list(grid)
        elif order is not None and seasonal_order is not None:
            self.grid = [(order, seasonal_order)]
        else:
            self.grid = list(DEFAULT_SARIMA_GRID)
        self._fitted = None
        self._supports_forward = False
        self.selected_order: tuple[int, int, int] | None = None
        self.selected_seasonal_order: tuple[int, int, int, int] | None = None
        self.selected_aic: float | None = None
        self.tried: list[dict[str, Any]] = []

    @staticmethod
    def _make_model(order: tuple[int, int, int], seasonal_order: tuple[int, int, int, int]):
        from statsforecast.models import ARIMA

        return ARIMA(
            order=order,
            seasonal_order=seasonal_order[:3],
            season_length=seasonal_order[3],
        )

    def fit(self, y_train: np.ndarray) -> "FixedSARIMAForecaster":
        y = np.asarray(y_train, dtype=float)
        best = None
        best_aic = float("inf")
        best_order: tuple[int, int, int] | None = None
        best_seasonal: tuple[int, int, int, int] | None = None
        for order, seasonal_order in self.grid:
            try:
                fitted = self._make_model(order, seasonal_order).fit(y)
            except Exception as exc:
                self.tried.append({"order": order, "seasonal_order": seasonal_order, "aic": None, "error": repr(exc)})
                continue
            aic = _model_aic(fitted)
            self.tried.append({"order": order, "seasonal_order": seasonal_order, "aic": aic, "error": None})
            if aic < best_aic:
                best, best_aic, best_order, best_seasonal = fitted, aic, order, seasonal_order
        if best is None:
            tried_repr = "; ".join(f"{t['order']}{t['seasonal_order']}: {t['error']}" for t in self.tried)
            raise RuntimeError(f"All SARIMA candidates failed to fit. Tried: {tried_repr}")
        self._fitted = best
        self._supports_forward = hasattr(self._fitted, "forward")
        self.selected_order = best_order
        self.selected_seasonal_order = best_seasonal
        self.selected_aic = best_aic
        print(
            f"[FixedSARIMA] selected order={best_order} seasonal_order={best_seasonal} "
            f"AICc/AIC={best_aic:.2f} (from {len(self.grid)} candidates, "
            f"{sum(1 for t in self.tried if t['error'] is None)} converged)",
            flush=True,
        )
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
            refit = self._make_model(self.selected_order, self.selected_seasonal_order).fit(context)
            forecast = refit.predict(h=horizon)
        return self._coerce(forecast)

    def predict_windows(self, x: np.ndarray, horizon: int) -> np.ndarray:
        return np.vstack([self.predict_from_context(row, horizon) for row in x])
