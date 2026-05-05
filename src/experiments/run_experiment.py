"""Run one forecasting experiment and append metrics to CSV."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.preprocess import (
    inverse_transform,
    load_ecl_aggregate,
    load_etth1,
    make_synthetic_series,
    scale_split,
    split_by_fraction,
    split_etth1_standard,
    validate_hourly_series,
)
from src.data.windows import append_context, make_forecast_origins, make_windows
from src.evaluation.metrics import compute_metrics
from src.evaluation.metrics import per_window_absolute_error
from src.evaluation.timing import get_peak_gpu_mb, measure_time
from src.models.baselines import AutoARIMAForecaster, FixedSARIMAForecaster, SeasonalNaiveForecaster


RESULT_COLUMNS = [
    "model",
    "dataset",
    "target",
    "horizon",
    "seed",
    "MAE",
    "RMSE",
    "sMAPE",
    "wMAPE",
    "MSE",
    "train_time_s",
    "inference_ms",
    "n_params",
    "peak_gpu_mb",
]


def load_dataset(name: str, raw_path: Path | None, smoke: bool) -> tuple[pd.DataFrame, str]:
    if smoke or name == "synthetic":
        return make_synthetic_series(seed=42), "synthetic_load"
    if name == "ecl":
        if raw_path is None:
            raw_path = Path("data/raw/electricity.txt")
        return load_ecl_aggregate(raw_path), "aggregate_load"
    if name == "etth1":
        if raw_path is None:
            raw_path = Path("data/raw/ETTh1.csv")
        return load_etth1(raw_path, target_col="HUFL"), "HUFL"
    raise ValueError(f"Unknown dataset '{name}'.")


def build_prediction(
    model_name: str,
    train_df: pd.DataFrame,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    test_origins: pd.Series,
    horizon: int,
    seed: int,
    latency_repeats: int = 20,
) -> tuple[np.ndarray, float, float, int, float]:
    """Fit/predict a model and return predictions plus efficiency metrics."""
    model_name = model_name.lower()
    n_params = 0
    get_peak_gpu_mb(reset=True)

    with measure_time() as train_timer:
        if model_name == "seasonal_naive":
            model = SeasonalNaiveForecaster(season_length=24).fit(train_df["y"].to_numpy())
        elif model_name == "auto_arima":
            model = AutoARIMAForecaster(season_length=24, seasonal=True, max_p=3, max_q=3)
        elif model_name == "sarima":
            model = FixedSARIMAForecaster()  # uses DEFAULT_SARIMA_GRID, AICc-selected
            model.fit(train_df["y"].to_numpy())
        elif model_name == "xgboost":
            from src.models.xgboost_model import XGBoostDirectForecaster

            model = XGBoostDirectForecaster(random_state=seed)
            train_origins = make_forecast_origins(train_df, x_train.shape[1], horizon)
            model.fit(x_train, y_train, train_origins)
        elif model_name in {"chronos_bolt_small", "chronos"}:
            raise ValueError("Use src.experiments.run_foundation_experiment for Chronos-Bolt.")
        elif model_name == "timesfm":
            raise ValueError("Use src.experiments.run_foundation_experiment for TimesFM.")
        else:
            raise ValueError(
                f"Model '{model_name}' is not supported by this direct-window runner yet. "
                "Use NeuralForecast-specific pipeline for neural models."
            )

    train_time_s = train_timer.elapsed_s
    if model_name == "auto_arima":
        with measure_time() as rolling_timer:
            y_pred = model.predict_windows(x_test, horizon)
        # AutoARIMA refits for every origin. Report the full rolling fit+forecast
        # pass as train/evaluation cost, and single-origin refit+forecast latency.
        train_time_s = rolling_timer.elapsed_s
        repeats = max(3, min(latency_repeats, 5))
        model.predict_windows(x_test[:1], horizon)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict_windows(x_test[:1], horizon)
        inference_ms = (time.perf_counter() - start) / repeats * 1000.0
    elif model_name == "xgboost":
        y_pred = model.predict(x_test, test_origins)
        n_params = count_xgboost_structure(model)
        repeats = max(latency_repeats, 1)
        model.predict(x_test[:1], test_origins.iloc[:1])
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict(x_test[:1], test_origins.iloc[:1])
        inference_ms = (time.perf_counter() - start) / repeats * 1000.0
    else:
        y_pred = model.predict_windows(x_test, horizon)
        repeats = max(latency_repeats, 1)
        model.predict_windows(x_test[:1], horizon)
        start = time.perf_counter()
        for _ in range(repeats):
            model.predict_windows(x_test[:1], horizon)
        inference_ms = (time.perf_counter() - start) / repeats * 1000.0
    peak_gpu_mb = get_peak_gpu_mb(reset=True)
    return y_pred, train_time_s, inference_ms, n_params, peak_gpu_mb


def count_xgboost_structure(model: Any) -> int:
    """Approximate XGBoost model size by total leaf count across horizon models."""
    total = 0
    try:
        estimators = model.model.estimators_
    except Exception:
        return 0
    for estimator in estimators:
        try:
            frame = estimator.get_booster().trees_to_dataframe()
            total += int((frame["Feature"] == "Leaf").sum())
        except Exception:
            try:
                total += len(estimator.get_booster().get_dump())
            except Exception:
                pass
    return total


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_COLUMNS})


def save_window_errors(
    output_dir: Path,
    model: str,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    origins: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Persist per-window losses for statistical tests."""
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = per_window_absolute_error(y_true, y_pred)
    out = pd.DataFrame(
        {
            "origin": origins.iloc[: len(errors)].astype(str).to_numpy(),
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            "model": model,
            "abs_error": errors,
        }
    )
    path = output_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv"
    out.to_csv(path, index=False)


def save_prediction_sample(
    output_dir: Path,
    model: str,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    origins: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Persist one full forecast trajectory for plotting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(origins.iloc[0])
    ds = pd.date_range(origin, periods=horizon, freq="h")
    out = pd.DataFrame(
        {
            "ds": ds,
            "step": np.arange(1, horizon + 1),
            "origin": origin,
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            "model": model,
            "y_true": y_true[0],
            "y_pred": y_pred[0],
        }
    )
    out.to_csv(output_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv", index=False)


def run(args: argparse.Namespace) -> dict[str, Any]:
    np.random.seed(args.seed)
    df, target = load_dataset(args.dataset, args.raw_path, args.smoke)
    report = validate_hourly_series(df)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/data_validation_{args.dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.dataset == "etth1" and not args.smoke:
        split = split_etth1_standard(df)
    else:
        split = split_by_fraction(df)
    scaled = scale_split(split)

    train_for_windows = scaled.train
    test_with_context = append_context(pd.concat([scaled.train, scaled.val], ignore_index=True), scaled.test, args.input_length)

    x_train, y_train = make_windows(train_for_windows, args.input_length, args.horizon)
    x_test, y_test = make_windows(test_with_context, args.input_length, args.horizon)
    test_origins = make_forecast_origins(test_with_context, args.input_length, args.horizon)

    y_pred, train_time_s, inference_ms, n_params, peak_gpu_mb = build_prediction(
        args.model,
        train_for_windows,
        x_train,
        y_train,
        x_test,
        test_origins,
        args.horizon,
        args.seed,
        args.latency_repeats,
    )
    y_test_original = inverse_transform(y_test.reshape(-1), scaled.scaler).reshape(y_test.shape)
    y_pred_original = inverse_transform(y_pred.reshape(-1), scaled.scaler).reshape(y_pred.shape)
    metrics = compute_metrics(y_test_original, y_pred_original)
    save_window_errors(
        Path("results/window_errors"),
        args.model,
        args.dataset,
        target,
        args.horizon,
        args.seed,
        test_origins,
        y_test_original,
        y_pred_original,
    )
    save_prediction_sample(
        Path("results/prediction_samples"),
        args.model,
        args.dataset,
        target,
        args.horizon,
        args.seed,
        test_origins,
        y_test_original,
        y_pred_original,
    )
    row = {
        "model": args.model,
        "dataset": args.dataset,
        "target": target,
        "horizon": args.horizon,
        "seed": args.seed,
        **metrics,
        "train_time_s": train_time_s,
        "inference_ms": inference_ms,
        "n_params": n_params,
        "peak_gpu_mb": peak_gpu_mb,
    }
    append_result(args.output, row)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="synthetic", choices=["synthetic", "ecl", "etth1"])
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--model", default="seasonal_naive")
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    row = run(parse_args())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
