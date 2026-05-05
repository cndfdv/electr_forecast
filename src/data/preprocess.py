"""Dataset loading, validation, splitting, and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class TimeSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


@dataclass(frozen=True)
class ScaledSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def make_synthetic_series(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Create a deterministic hourly synthetic load series for smoke tests."""
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2020-01-01", periods=n, freq="h")
    hour = ds.hour.to_numpy()
    dow = ds.dayofweek.to_numpy()
    daily = 10.0 * np.sin(2 * np.pi * hour / 24)
    weekly = 5.0 * np.cos(2 * np.pi * dow / 7)
    trend = np.linspace(0, 4, n)
    noise = rng.normal(0, 1.5, n)
    y = 100 + daily + weekly + trend + noise
    return pd.DataFrame({"ds": ds, "y": y})


def load_ecl_aggregate(path: Path) -> pd.DataFrame:
    """Load ECL and return hourly aggregate load with columns ds,y."""
    if not path.exists():
        gz_path = path.with_suffix(path.suffix + ".gz")
        if gz_path.exists():
            path = gz_path
        else:
            raise FileNotFoundError(path)

    df = pd.read_csv(path, sep=None, engine="python", header=None, compression="infer")
    first_col = df.iloc[:, 0]
    has_timestamp_col = False
    maybe_dates: pd.Series | None = None
    # Only attempt timestamp parsing when the first column is non-numeric. Numeric
    # values get interpreted as nanoseconds from the Unix epoch by pd.to_datetime,
    # which silently misclassifies the LSTNet ECL format (no timestamp column,
    # numeric client readings) as if it had timestamps anchored at 1970.
    if first_col.dtype == object or pd.api.types.is_string_dtype(first_col):
        maybe_dates = pd.to_datetime(first_col, errors="coerce")
        if maybe_dates.notna().mean() > 0.95:
            sample_year = maybe_dates.dropna().iloc[0].year
            has_timestamp_col = sample_year >= 1990
    if has_timestamp_col and maybe_dates is not None:
        dates = maybe_dates
        values = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    else:
        # The laiguokun processed file has no timestamp column: rows are hourly
        # observations and columns are clients. Use a synthetic hourly index for
        # calendar features while keeping the temporal order unchanged.
        dates = pd.date_range("2012-01-01", periods=len(df), freq="h")
        values = df.apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame({"ds": dates, "y": values.sum(axis=1, skipna=False)})
    return out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)


def load_etth1(path: Path, target_col: str = "HUFL") -> pd.DataFrame:
    """Load ETTh1 and return columns ds,y for the selected target."""
    if target_col == "OT":
        raise ValueError("ETTh1 OT is oil temperature, not load. Use a broader paper framing first.")
    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError("ETTh1 must contain a 'date' column.")
    if target_col not in df.columns:
        raise ValueError(f"Target '{target_col}' not found in ETTh1 columns: {list(df.columns)}")
    out = pd.DataFrame({"ds": pd.to_datetime(df["date"]), "y": pd.to_numeric(df[target_col])})
    return out.dropna(subset=["ds", "y"]).sort_values("ds").reset_index(drop=True)


def validate_hourly_series(df: pd.DataFrame, freq: str = "h") -> dict[str, int | float | bool]:
    """Validate core assumptions for an hourly univariate forecasting dataset."""
    required = {"ds", "y"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ordered = df.sort_values("ds")
    duplicate_timestamps = int(ordered["ds"].duplicated().sum())
    missing_y = int(ordered["y"].isna().sum())
    zero_y = int((ordered["y"] == 0).sum())
    near_zero_y = int((ordered["y"].abs() < 1e-8).sum())

    expected = pd.date_range(ordered["ds"].iloc[0], ordered["ds"].iloc[-1], freq=freq)
    regular_grid = len(expected) == len(ordered) and ordered["ds"].reset_index(drop=True).equals(
        pd.Series(expected, name="ds")
    )
    return {
        "rows": int(len(ordered)),
        "duplicate_timestamps": duplicate_timestamps,
        "missing_y": missing_y,
        "zero_y": zero_y,
        "near_zero_y": near_zero_y,
        "regular_hourly_grid": bool(regular_grid),
        "start": str(ordered["ds"].iloc[0]),
        "end": str(ordered["ds"].iloc[-1]),
    }


def split_by_fraction(df: pd.DataFrame, train: float = 0.70, val: float = 0.10) -> TimeSplit:
    """Split a time-ordered dataframe by fractions."""
    if train <= 0 or val <= 0 or train + val >= 1:
        raise ValueError("Invalid split fractions.")
    n = len(df)
    train_end = int(n * train)
    val_end = int(n * (train + val))
    return TimeSplit(
        train=df.iloc[:train_end].reset_index(drop=True),
        val=df.iloc[train_end:val_end].reset_index(drop=True),
        test=df.iloc[val_end:].reset_index(drop=True),
    )


def split_etth1_standard(df: pd.DataFrame) -> TimeSplit:
    """Use the common 12/4/4 month ETTh1 split by row counts."""
    train_end = 12 * 30 * 24
    val_end = train_end + 4 * 30 * 24
    if len(df) < val_end:
        raise ValueError("ETTh1 dataframe is too short for the standard split.")
    return TimeSplit(
        train=df.iloc[:train_end].reset_index(drop=True),
        val=df.iloc[train_end:val_end].reset_index(drop=True),
        test=df.iloc[val_end:].reset_index(drop=True),
    )


def scale_split(split: TimeSplit, target_col: str = "y") -> ScaledSplit:
    """Fit StandardScaler on train only and transform all splits."""
    scaler = StandardScaler()
    scaler.fit(split.train[[target_col]])

    def transform(part: pd.DataFrame) -> pd.DataFrame:
        out = part.copy()
        out[target_col] = scaler.transform(out[[target_col]]).reshape(-1)
        return out

    return ScaledSplit(transform(split.train), transform(split.val), transform(split.test), scaler)


def inverse_transform(values: Iterable[float], scaler: StandardScaler) -> np.ndarray:
    """Inverse-transform a one-dimensional target array."""
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)
