"""Run one NeuralForecast experiment with rolling test cross-validation."""

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
from src.evaluation.metrics import compute_metrics
from src.evaluation.timing import get_peak_gpu_mb
from src.experiments.run_experiment import RESULT_COLUMNS, append_result


def load_dataset(name: str, raw_path: Path | None, smoke: bool) -> tuple[pd.DataFrame, str]:
    if smoke or name == "synthetic":
        return make_synthetic_series(n=1200, seed=42), "synthetic_load"
    if name == "ecl":
        return load_ecl_aggregate(raw_path or Path("data/raw/electricity.txt")), "aggregate_load"
    if name == "etth1":
        return load_etth1(raw_path or Path("data/raw/ETTh1.csv"), target_col="HUFL"), "HUFL"
    raise ValueError(name)


def to_nf_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["ds", "y"]].copy()
    out.insert(0, "unique_id", "series")
    return out


def build_model(
    name: str,
    horizon: int,
    input_length: int,
    seed: int,
    max_steps: int,
    accelerator: str,
    devices: int,
    learning_rate: float,
    early_stop_patience_steps: int,
):
    from neuralforecast.models import DLinear, PatchTST, iTransformer

    common = {
        "h": horizon,
        "input_size": input_length,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "batch_size": 32,
        "valid_batch_size": 32,
        "windows_batch_size": 1024,
        "inference_windows_batch_size": 1024,
        "random_seed": seed,
        "scaler_type": "identity",
        "accelerator": accelerator,
        "devices": devices,
        "early_stop_patience_steps": early_stop_patience_steps,
        "val_check_steps": 50,
        "enable_checkpointing": False,
        "logger": False,
    }
    if name == "dlinear":
        return DLinear(alias="dlinear", **common)
    if name == "patchtst":
        return PatchTST(alias="patchtst", hidden_size=64, n_heads=4, encoder_layers=2, **common)
    if name == "itransformer":
        return iTransformer(alias="itransformer", n_series=1, hidden_size=128, n_heads=4, e_layers=2, d_ff=256, **common)
    raise ValueError(name)


def save_window_errors(
    output_dir: Path,
    model: str,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    cv: pd.DataFrame,
    pred_col: str,
    scaler,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cutoff, part in cv.groupby("cutoff"):
        y_true = inverse_transform(part["y"].to_numpy(), scaler)
        y_pred = inverse_transform(part[pred_col].to_numpy(), scaler)
        rows.append(
            {
                "origin": str(cutoff),
                "dataset": dataset,
                "target": target,
                "horizon": horizon,
                "seed": seed,
                "model": model,
                "abs_error": float(np.mean(np.abs(y_true - y_pred))),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv", index=False)


def save_prediction_sample(
    output_dir: Path,
    model: str,
    dataset: str,
    target: str,
    horizon: int,
    seed: int,
    cv: pd.DataFrame,
    pred_col: str,
    scaler,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cutoff = cv["cutoff"].iloc[0]
    part = cv[cv["cutoff"] == cutoff].head(horizon)
    y_true = inverse_transform(part["y"].to_numpy(), scaler)
    y_pred = inverse_transform(part[pred_col].to_numpy(), scaler)
    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(part["ds"]),
            "step": np.arange(1, len(part) + 1),
            "origin": str(part["ds"].iloc[0]),
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            "model": model,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    out.to_csv(output_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv", index=False)


def count_params(model: Any) -> int:
    try:
        return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    except Exception:
        return 0


def measure_single_window_inference(nf: Any, history: pd.DataFrame, repeats: int = 20) -> float:
    """Measure latency for one forecast window after the model is trained."""
    # One warm-up call keeps CUDA/kernel initialization out of the reported timing.
    nf.predict(df=to_nf_df(history), verbose=False)
    start = time.perf_counter()
    for _ in range(repeats):
        nf.predict(df=to_nf_df(history), verbose=False)
    return (time.perf_counter() - start) / max(repeats, 1) * 1000.0


def parse_learning_rates(values: list[str]) -> list[float]:
    rates: list[float] = []
    for item in values:
        for part in item.split(","):
            part = part.strip()
            if part:
                rates.append(float(part))
    if not rates:
        raise ValueError("At least one learning rate is required.")
    return rates


def run_cross_validation(
    args: argparse.Namespace,
    full: pd.DataFrame,
    test_size: int,
    val_size: int,
    learning_rate: float,
    accelerator: str,
    devices: int,
):
    from neuralforecast import NeuralForecast

    model = build_model(
        args.model,
        args.horizon,
        args.input_length,
        args.seed,
        args.max_steps,
        accelerator,
        devices,
        learning_rate,
        args.early_stop_patience_steps,
    )
    nf = NeuralForecast(models=[model], freq="h")
    start = time.perf_counter()
    cv = nf.cross_validation(
        df=to_nf_df(full),
        n_windows=None,
        test_size=test_size,
        val_size=val_size,
        step_size=args.step_size,
        verbose=False,
    )
    elapsed = time.perf_counter() - start
    return model, nf, cv, elapsed


def tune_learning_rate(
    args: argparse.Namespace,
    scaled,
    accelerator: str,
    devices: int,
    learning_rates: list[float],
) -> tuple[float, float, list[dict[str, float]]]:
    """Select learning rate on the chronological validation split."""
    if len(learning_rates) == 1:
        return learning_rates[0], 0.0, []

    tune_full = pd.concat([scaled.train, scaled.val], ignore_index=True)
    val_size = len(scaled.val)
    inner_val_size = min(max(args.horizon, 24), max(len(scaled.train) // 10, 1))
    if len(tune_full) <= val_size + inner_val_size + args.input_length + args.horizon:
        inner_val_size = max(args.horizon, 1)

    records: list[dict[str, float]] = []
    tune_elapsed = 0.0
    best_lr = learning_rates[0]
    best_mae = float("inf")
    for lr in learning_rates:
        _, _, cv, elapsed = run_cross_validation(
            args,
            tune_full,
            test_size=val_size,
            val_size=inner_val_size,
            learning_rate=lr,
            accelerator=accelerator,
            devices=devices,
        )
        tune_elapsed += elapsed
        y_true = inverse_transform(cv["y"].to_numpy(), scaled.scaler)
        y_pred = inverse_transform(cv[args.model].to_numpy(), scaled.scaler)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        records.append({"learning_rate": lr, "val_MAE": mae, "elapsed_s": elapsed})
        if mae < best_mae:
            best_mae = mae
            best_lr = lr
    return best_lr, tune_elapsed, records


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    torch.set_float32_matmul_precision("medium")
    np.random.seed(args.seed)

    df, target = load_dataset(args.dataset, args.raw_path, args.smoke)
    report = validate_hourly_series(df)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/data_validation_{args.dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    split = split_etth1_standard(df) if args.dataset == "etth1" and not args.smoke else split_by_fraction(df)
    scaled = scale_split(split)
    full = pd.concat([scaled.train, scaled.val, scaled.test], ignore_index=True)
    val_size = len(scaled.val)
    test_size = len(scaled.test)
    if test_size <= args.horizon:
        raise ValueError("test split shorter than horizon")

    accelerator = "gpu" if torch.cuda.is_available() and not args.cpu else "cpu"
    devices = 1
    learning_rates = parse_learning_rates(args.learning_rates)

    get_peak_gpu_mb(reset=True)
    best_lr, tune_elapsed, tune_records = tune_learning_rate(args, scaled, accelerator, devices, learning_rates)
    model, nf, cv, final_elapsed = run_cross_validation(
        args,
        full,
        test_size=test_size,
        val_size=val_size,
        learning_rate=best_lr,
        accelerator=accelerator,
        devices=devices,
    )
    history_for_latency = pd.concat([scaled.train, scaled.val], ignore_index=True)
    inference_ms = measure_single_window_inference(nf, history_for_latency, repeats=args.latency_repeats)
    peak_gpu_mb = get_peak_gpu_mb(reset=True)

    pred_col = args.model
    y_true = inverse_transform(cv["y"].to_numpy(), scaled.scaler)
    y_pred = inverse_transform(cv[pred_col].to_numpy(), scaled.scaler)
    metrics = compute_metrics(y_true, y_pred)
    row = {
        "model": args.model,
        "dataset": args.dataset,
        "target": target,
        "horizon": args.horizon,
        "seed": args.seed,
        **metrics,
        "train_time_s": tune_elapsed + final_elapsed,
        "inference_ms": inference_ms,
        "n_params": count_params(model),
        "peak_gpu_mb": peak_gpu_mb,
    }
    append_result(args.output, row)
    save_window_errors(Path("results/window_errors"), args.model, args.dataset, target, args.horizon, args.seed, cv, pred_col, scaled.scaler)
    save_prediction_sample(Path("results/prediction_samples"), args.model, args.dataset, target, args.horizon, args.seed, cv, pred_col, scaled.scaler)
    if tune_records:
        Path("results/tuning").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(tune_records).assign(
            model=args.model,
            dataset=args.dataset,
            target=target,
            horizon=args.horizon,
            seed=args.seed,
            selected_learning_rate=best_lr,
        ).to_csv(
            Path("results/tuning") / f"{args.dataset}_{target}_h{args.horizon}_{args.model}_seed{args.seed}.csv",
            index=False,
        )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ecl", "etth1"], required=True)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--model", choices=["dlinear", "patchtst", "itransformer"], required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--step-size", type=int, default=1)
    parser.add_argument("--latency-repeats", type=int, default=20)
    parser.add_argument("--learning-rates", nargs="+", default=["1e-3", "5e-4"])
    parser.add_argument("--early-stop-patience-steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    row = run(parse_args())
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
