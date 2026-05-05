"""Run a conservative batch of configured experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ecl", "etth1"])
    parser.add_argument("--models", nargs="+", default=["seasonal_naive", "xgboost"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 96, 168])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--output", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    deterministic = {"seasonal_naive", "auto_arima", "chronos", "chronos_bolt_small", "timesfm"}
    neuralforecast_models = {"dlinear", "patchtst", "itransformer"}
    foundation_models = {"chronos_bolt_small", "timesfm"}

    for dataset in args.datasets:
        for model in args.models:
            model_seeds = [args.seeds[0]] if model in deterministic else args.seeds
            for horizon in args.horizons:
                for seed in model_seeds:
                    if model in neuralforecast_models:
                        module = "src.experiments.run_neuralforecast_experiment"
                    elif model in foundation_models:
                        module = "src.experiments.run_foundation_experiment"
                    else:
                        module = "src.experiments.run_experiment"
                    cmd = [
                        sys.executable,
                        "-m",
                        module,
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                        "--horizon",
                        str(horizon),
                        "--input-length",
                        str(args.input_length),
                        "--seed",
                        str(seed),
                        "--output",
                        str(args.output),
                    ]
                    if args.smoke:
                        cmd.append("--smoke")
                    if model == "timesfm":
                        cmd.extend(["--checkpoint", "google/timesfm-2.5-200m-pytorch"])
                    print(" ".join(cmd), flush=True)
                    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
