"""Forecast accuracy metrics."""

from __future__ import annotations

import numpy as np


EPS = 1e-8


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(err * err))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), EPS)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = max(float(np.sum(np.abs(y_true))), EPS)
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
        "wMAPE": wmape(y_true, y_pred),
        "MSE": mse(y_true, y_pred),
    }


def per_window_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Return one absolute-error value per forecast window."""
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), axis=1)

