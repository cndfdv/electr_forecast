"""Run zero-shot foundation-model experiments on rolling windows."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocess import (
    load_ecl_aggregate,
    load_etth1,
    make_synthetic_series,
    split_by_fraction,
    split_etth1_standard,
    validate_hourly_series,
)
from src.data.windows import append_context, make_forecast_origins, make_windows
from src.evaluation.metrics import compute_metrics, per_window_absolute_error
from src.evaluation.timing import get_peak_gpu_mb
from src.experiments.run_experiment import append_result


def load_dataset(name: str, raw_path: Path | None, smoke: bool) -> tuple[pd.DataFrame, str]:
    if smoke or name == "synthetic":
        return make_synthetic_series(n=1200, seed=42), "synthetic_load"
    if name == "ecl":
        return load_ecl_aggregate(raw_path or Path("data/raw/electricity.txt")), "aggregate_load"
    if name == "etth1":
        return load_etth1(raw_path or Path("data/raw/ETTh1.csv"), target_col="HUFL"), "HUFL"
    raise ValueError(name)


def iter_batches(x: np.ndarray, batch_size: int):
    for start in range(0, len(x), batch_size):
        yield start, x[start : start + batch_size]


def load_chronos(checkpoint: str):
    import torch
    from chronos import ChronosBoltPipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "cuda" if torch.cuda.is_available() else "cpu"
    return ChronosBoltPipeline.from_pretrained(checkpoint, device_map=device_map, torch_dtype=dtype)


def forecast_chronos(pipeline, x: np.ndarray, horizon: int, batch_size: int) -> np.ndarray:
    import torch

    preds = []
    for _, batch in iter_batches(x, batch_size):
        inputs = torch.tensor(batch, dtype=torch.float32)
        out = pipeline.predict(inputs, prediction_length=horizon)
        arr = out.detach().cpu().float().numpy()
        if arr.ndim == 3:
            arr = np.median(arr, axis=1)
        preds.append(arr[:, :horizon])
    return np.vstack(preds)


def count_chronos_params(pipeline) -> int:
    try:
        return int(sum(p.numel() for p in pipeline.model.parameters()))
    except Exception:
        return 0


def load_timesfm(checkpoint: str, context_len: int, horizon: int, batch_size: int):
    import timesfm

    if hasattr(timesfm, "TimesFM_2p5_200M_torch"):
        model = timesfm.TimesFM_2p5_200M_torch._from_pretrained(
            model_id=checkpoint,
            revision=None,
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
            torch_compile=False,
        )
        model.compile(
            timesfm.ForecastConfig(
                max_context=context_len,
                max_horizon=horizon,
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
            )
        )
        return model, "timesfm_2p5", 200_000_000

    model = timesfm.TimesFm(
        hparams=timesfm.TimesFmHparams(
            backend="gpu",
            per_core_batch_size=batch_size,
            horizon_len=horizon,
            context_len=context_len,
        ),
        checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=checkpoint),
    )
    return model, "timesfm_legacy", 0


def forecast_timesfm(model, api: str, x: np.ndarray, horizon: int, batch_size: int) -> np.ndarray:
    preds = []
    if api == "timesfm_2p5":
        for _, batch in iter_batches(x, batch_size):
            point_forecast, _ = model.forecast(horizon=horizon, inputs=[row for row in batch])
            preds.append(np.asarray(point_forecast, dtype=float)[:, :horizon])
    else:
        for _, batch in iter_batches(x, batch_size):
            forecasts, _ = model.forecast(list(batch), freq=[0] * len(batch), normalize=True)
            preds.append(np.asarray(forecasts, dtype=float)[:, :horizon])
    return np.vstack(preds)


def measure_single_window_latency(fn, x: np.ndarray, repeats: int) -> float:
    fn(x[:1])
    start = time.perf_counter()
    for _ in range(max(repeats, 1)):
        fn(x[:1])
    return (time.perf_counter() - start) / max(repeats, 1) * 1000.0


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
    output_dir.mkdir(parents=True, exist_ok=True)
    errors = per_window_absolute_error(y_true, y_pred)
    pd.DataFrame(
        {
            "origin": origins.iloc[: len(errors)].astype(str).to_numpy(),
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            "model": model,
            "abs_error": errors,
        }
    ).to_csv(output_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv", index=False)


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
    output_dir.mkdir(parents=True, exist_ok=True)
    origin = pd.Timestamp(origins.iloc[0])
    out = pd.DataFrame(
        {
            "ds": pd.date_range(origin, periods=horizon, freq="h"),
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


def run(args: argparse.Namespace) -> dict:
    np.random.seed(args.seed)
    df, target = load_dataset(args.dataset, args.raw_path, args.smoke)
    report = validate_hourly_series(df)
    Path("results").mkdir(exist_ok=True)
    Path(f"results/data_validation_{args.dataset}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    split = split_etth1_standard(df) if args.dataset == "etth1" and not args.smoke else split_by_fraction(df)
    history = pd.concat([split.train, split.val], ignore_index=True)
    test_with_context = append_context(history, split.test, args.input_length)
    x_test, y_test = make_windows(test_with_context, args.input_length, args.horizon)
    origins = make_forecast_origins(test_with_context, args.input_length, args.horizon)
    if args.max_windows:
        x_test = x_test[: args.max_windows]
        y_test = y_test[: args.max_windows]
        origins = origins.iloc[: args.max_windows]

    get_peak_gpu_mb(reset=True)
    start = time.perf_counter()
    if args.model == "chronos_bolt_small":
        pipeline = load_chronos(args.checkpoint)
        y_pred = forecast_chronos(pipeline, x_test, args.horizon, args.batch_size)
        inference_ms = measure_single_window_latency(
            lambda x: forecast_chronos(pipeline, x, args.horizon, 1),
            x_test,
            args.latency_repeats,
        )
        n_params = count_chronos_params(pipeline)
    elif args.model == "timesfm":
        model, api, n_params = load_timesfm(args.checkpoint, x_test.shape[1], args.horizon, args.batch_size)
        y_pred = forecast_timesfm(model, api, x_test, args.horizon, args.batch_size)
        inference_ms = measure_single_window_latency(
            lambda x: forecast_timesfm(model, api, x, args.horizon, 1),
            x_test,
            args.latency_repeats,
        )
    else:
        raise ValueError(args.model)
    elapsed = time.perf_counter() - start
    peak_gpu_mb = get_peak_gpu_mb(reset=True)

    metrics = compute_metrics(y_test, y_pred)
    row = {
        "model": args.model,
        "dataset": args.dataset,
        "target": target,
        "horizon": args.horizon,
        "seed": args.seed,
        **metrics,
        "train_time_s": 0.0,
        "inference_ms": inference_ms,
        "n_params": n_params,
        "peak_gpu_mb": peak_gpu_mb,
    }
    append_result(args.output, row)
    save_window_errors(Path("results/window_errors"), args.model, args.dataset, target, args.horizon, args.seed, origins, y_test, y_pred)
    save_prediction_sample(Path("results/prediction_samples"), args.model, args.dataset, target, args.horizon, args.seed, origins, y_test, y_pred)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["synthetic", "ecl", "etth1"], required=True)
    parser.add_argument("--model", choices=["chronos_bolt_small", "timesfm"], required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--checkpoint", default="amazon/chronos-bolt-small")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--latency-repeats", type=int, default=10)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2))


if __name__ == "__main__":
    main()
