"""NeuralForecast wrappers.

These wrappers intentionally keep imports lazy so the repository can run
baseline smoke tests without heavy deep-learning dependencies installed.
"""

from __future__ import annotations

import pandas as pd


SUPPORTED = {"dlinear", "patchtst", "itransformer", "lstm", "timesnet"}


def to_neuralforecast_df(df: pd.DataFrame, unique_id: str = "series") -> pd.DataFrame:
    out = df[["ds", "y"]].copy()
    out.insert(0, "unique_id", unique_id)
    return out


def build_neuralforecast_model(name: str, horizon: int, input_size: int, learning_rate: float = 1e-3, **kwargs):
    """Build a NeuralForecast model by name."""
    name = name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported NeuralForecast model '{name}'.")

    try:
        from neuralforecast.models import DLinear, LSTM, PatchTST, TimesNet, iTransformer
    except ImportError as exc:
        raise ImportError("NeuralForecast models require neuralforecast.") from exc

    common = {"h": horizon, "input_size": input_size, "learning_rate": learning_rate}
    common.update(kwargs)
    if name == "dlinear":
        return DLinear(**common)
    if name == "patchtst":
        return PatchTST(**common)
    if name == "itransformer":
        return iTransformer(**common)
    if name == "lstm":
        return LSTM(**common)
    if name == "timesnet":
        return TimesNet(**common)
    raise AssertionError(name)


def fit_predict_neuralforecast(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    name: str,
    horizon: int,
    input_size: int,
    learning_rate: float = 1e-3,
    **kwargs,
) -> pd.DataFrame:
    """Fit a NeuralForecast model and return its raw prediction dataframe."""
    try:
        from neuralforecast import NeuralForecast
    except ImportError as exc:
        raise ImportError("NeuralForecast models require neuralforecast.") from exc

    model = build_neuralforecast_model(name, horizon, input_size, learning_rate, **kwargs)
    nf = NeuralForecast(models=[model], freq="H")
    nf.fit(df=to_neuralforecast_df(train_df))
    return nf.predict(df=to_neuralforecast_df(test_df))

