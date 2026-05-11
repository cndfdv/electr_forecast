"""Hyperparameter-tune XGBoost per (dataset, horizon) and evaluate on test.

For each (dataset, horizon) this module runs a small grid search on the
validation split (max_depth in {4, 6}, n_estimators in {500, 1000},
learning_rate in {0.05, 0.1}), picks the configuration with the lowest
validation MAE, retrains on train+val, and writes per-seed test metrics
(plus per-window errors and prediction samples) so that downstream
statistical tests and figures can use seed-pooled results.

This is the canonical XGBoost entry point used by ``run_all`` and is
the version that produced ``results/main_results_final.csv``.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.preprocess import (
    inverse_transform,
    load_ecl_aggregate,
    load_etth1,
    scale_split,
    split_by_fraction,
    split_etth1_standard,
)
from src.data.windows import append_context, make_forecast_origins, make_windows
from src.evaluation.metrics import compute_metrics
from src.experiments.run_experiment import (
    append_result,
    count_xgboost_structure,
    save_prediction_sample,
    save_window_errors,
)
from src.models.xgboost_model import XGBoostDirectForecaster

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "main_results_final.csv"
DEFAULT_TUNING_LOG = ROOT / "results" / "xgboost_tuning.csv"

DATASETS = {
    "ecl": ("data/raw/electricity.txt.gz", "aggregate_load"),
    "etth1": ("data/raw/ETTh1.csv", "HUFL"),
}
HORIZONS = [24, 96, 168]
SEEDS = [42, 43, 44, 45, 46]

GRID = [
    {"max_depth": md, "n_estimators": ne, "learning_rate": lr}
    for md in (4, 6)
    for ne in (500, 1000)
    for lr in (0.05, 0.1)
]


def load_dataset(name: str) -> pd.DataFrame:
    raw, _ = DATASETS[name]
    if name == "ecl":
        return load_ecl_aggregate(Path(raw))
    if name == "etth1":
        return load_etth1(Path(raw), target_col="HUFL")
    raise ValueError(name)


def fit_predict_xgb(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_origins: pd.Series,
    x_eval: np.ndarray,
    eval_origins: pd.Series,
    seed: int,
    params: dict,
) -> tuple[np.ndarray, "XGBoostDirectForecaster", float, float, int]:
    model = XGBoostDirectForecaster(random_state=seed, **params)
    start_fit = time.perf_counter()
    model.fit(x_train, y_train, train_origins)
    fit_time = time.perf_counter() - start_fit
    start_inf = time.perf_counter()
    model.predict(x_eval[:1], eval_origins.iloc[:1])
    repeats = 5
    for _ in range(repeats):
        model.predict(x_eval[:1], eval_origins.iloc[:1])
    inf_ms = (time.perf_counter() - start_inf) / repeats * 1000.0
    y_pred = model.predict(x_eval, eval_origins)
    n_params = count_xgboost_structure(model)
    return y_pred, model, fit_time, inf_ms, n_params


def make_splits(name: str, horizon: int, input_length: int):
    df = load_dataset(name)
    if name == "etth1":
        split = split_etth1_standard(df)
    else:
        split = split_by_fraction(df)
    scaled = scale_split(split)

    train_df = scaled.train
    val_df_with_context = append_context(scaled.train, scaled.val, input_length)
    test_df_with_context = append_context(
        pd.concat([scaled.train, scaled.val], ignore_index=True),
        scaled.test,
        input_length,
    )
    train_full_df = pd.concat([scaled.train, scaled.val], ignore_index=True)

    x_train, y_train = make_windows(train_df, input_length, horizon)
    x_val, y_val = make_windows(val_df_with_context, input_length, horizon)
    x_test, y_test = make_windows(test_df_with_context, input_length, horizon)
    x_train_full, y_train_full = make_windows(train_full_df, input_length, horizon)

    train_origins = make_forecast_origins(train_df, input_length, horizon)
    val_origins = make_forecast_origins(val_df_with_context, input_length, horizon)
    test_origins = make_forecast_origins(test_df_with_context, input_length, horizon)
    train_full_origins = make_forecast_origins(train_full_df, input_length, horizon)
    return {
        "scaler": scaled.scaler,
        "x_train": x_train,
        "y_train": y_train,
        "train_origins": train_origins,
        "x_val": x_val,
        "y_val": y_val,
        "val_origins": val_origins,
        "x_test": x_test,
        "y_test": y_test,
        "test_origins": test_origins,
        "x_train_full": x_train_full,
        "y_train_full": y_train_full,
        "train_full_origins": train_full_origins,
    }


def append_tuning_log(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "dataset", "target", "horizon", "max_depth", "n_estimators",
        "learning_rate", "val_MAE", "fit_time_s",
    ]
    exists = path.exists()
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def tune_one(name: str, horizon: int, input_length: int, tuning_log: Path) -> dict:
    splits = make_splits(name, horizon, input_length)
    target = DATASETS[name][1]
    scaler = splits["scaler"]
    print(f"[tune] {name}/H{horizon}: train_windows={len(splits['x_train'])}, "
          f"val_windows={len(splits['x_val'])}", flush=True)

    grid_results = []
    best = None
    for params in GRID:
        y_pred_scaled, _, fit_time, _, _ = fit_predict_xgb(
            x_train=splits["x_train"],
            y_train=splits["y_train"],
            train_origins=splits["train_origins"],
            x_eval=splits["x_val"],
            eval_origins=splits["val_origins"],
            seed=42,
            params=params,
        )
        y_val_orig = inverse_transform(splits["y_val"].reshape(-1), scaler).reshape(splits["y_val"].shape)
        y_pred_orig = inverse_transform(y_pred_scaled.reshape(-1), scaler).reshape(y_pred_scaled.shape)
        val_mae = float(np.mean(np.abs(y_val_orig - y_pred_orig)))
        record = {
            "dataset": name,
            "target": target,
            "horizon": horizon,
            "max_depth": params["max_depth"],
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "val_MAE": val_mae,
            "fit_time_s": fit_time,
        }
        grid_results.append(record)
        print(f"[tune] {name}/H{horizon} {params} val_MAE={val_mae:.2f} "
              f"fit_time={fit_time:.1f}s", flush=True)
        if best is None or val_mae < best["val_MAE"]:
            best = record

    append_tuning_log(grid_results, tuning_log)
    print(f"[tune] {name}/H{horizon} BEST: max_depth={best['max_depth']} "
          f"n_estimators={best['n_estimators']} lr={best['learning_rate']} "
          f"val_MAE={best['val_MAE']:.2f}", flush=True)
    return {"splits": splits, "best": best, "target": target}


def evaluate_seeds(
    name: str,
    horizon: int,
    splits: dict,
    best: dict,
    target: str,
    seeds: list[int],
    output: Path,
) -> None:
    scaler = splits["scaler"]
    params = {
        "max_depth": best["max_depth"],
        "n_estimators": best["n_estimators"],
        "learning_rate": best["learning_rate"],
    }
    for seed in seeds:
        y_pred_scaled, _, fit_time, inf_ms, n_params = fit_predict_xgb(
            x_train=splits["x_train_full"],
            y_train=splits["y_train_full"],
            train_origins=splits["train_full_origins"],
            x_eval=splits["x_test"],
            eval_origins=splits["test_origins"],
            seed=seed,
            params=params,
        )
        y_test_orig = inverse_transform(splits["y_test"].reshape(-1), scaler).reshape(splits["y_test"].shape)
        y_pred_orig = inverse_transform(y_pred_scaled.reshape(-1), scaler).reshape(y_pred_scaled.shape)
        metrics = compute_metrics(y_test_orig, y_pred_orig)
        row = {
            "model": "xgboost",
            "dataset": name,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            **metrics,
            "train_time_s": fit_time,
            "inference_ms": inf_ms,
            "n_params": n_params,
            "peak_gpu_mb": 0.0,
        }
        append_result(output, row)
        save_window_errors(
            ROOT / "results" / "window_errors",
            "xgboost", name, target, horizon, seed,
            splits["test_origins"], y_test_orig, y_pred_orig,
        )
        save_prediction_sample(
            ROOT / "results" / "prediction_samples",
            "xgboost", name, target, horizon, seed,
            splits["test_origins"], y_test_orig, y_pred_orig,
        )
        print(f"[eval] {name}/H{horizon} seed{seed} MAE={metrics['MAE']:.2f} "
              f"sMAPE={metrics['sMAPE']:.2f} fit_time={fit_time:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--tuning-log", type=Path, default=DEFAULT_TUNING_LOG)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.tuning_log.parent.mkdir(parents=True, exist_ok=True)
    overall_start = time.perf_counter()

    for name in args.datasets:
        for horizon in args.horizons:
            block_start = time.perf_counter()
            tune_out = tune_one(name, horizon, args.input_length, args.tuning_log)
            evaluate_seeds(
                name, horizon,
                tune_out["splits"], tune_out["best"], tune_out["target"],
                args.seeds, args.output,
            )
            elapsed = time.perf_counter() - block_start
            print(f"[done] {name}/H{horizon} in {elapsed/60:.1f} min", flush=True)

    total = time.perf_counter() - overall_start
    print(f"All XGBoost tuning + eval done in {total/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
