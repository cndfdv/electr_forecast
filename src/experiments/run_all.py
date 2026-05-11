"""Run the full benchmark end-to-end.

Sequentially launches every (model, dataset, horizon, seed) combination
that contributes to ``results/main_results_final.csv``:

* SeasonalNaive, SARIMA: deterministic, one seed.
* XGBoost: validation-tuned grid search per (dataset, horizon), then
  five-seed evaluation on the test split.
* DLinear, PatchTST, iTransformer: trained with --max-steps 5000 and
  validation-driven early stopping, evaluated across five seeds.
* Chronos-Bolt-Small, TimesFM 2.5: zero-shot, one seed each.

The script delegates per-config execution to
``src.experiments.run_experiment``,
``src.experiments.run_neuralforecast_experiment``,
``src.experiments.run_foundation_experiment``, and
``src.experiments.tune_xgboost`` so that this entry point only
orchestrates and does not duplicate model logic.

Default invocation reproduces the paper:
    python -m src.experiments.run_all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent

ALL_MODELS = [
    "seasonal_naive",
    "sarima",
    "xgboost",
    "dlinear",
    "patchtst",
    "itransformer",
    "chronos_bolt_small",
    "timesfm",
]

DETERMINISTIC = {"seasonal_naive", "sarima", "chronos_bolt_small", "timesfm"}
NEURALFORECAST = {"dlinear", "patchtst", "itransformer"}
FOUNDATION = {"chronos_bolt_small", "timesfm"}

RAW_PATHS = {
    "ecl": "data/raw/electricity.txt.gz",
    "etth1": "data/raw/ETTh1.csv",
}


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ecl", "etth1"])
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 96, 168])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--input-length", type=int, default=336)
    parser.add_argument("--max-steps", type=int, default=5000,
                        help="NeuralForecast training budget (early stopping enabled).")
    parser.add_argument("--output", type=Path,
                        default=Path("results/main_results_final.csv"))
    parser.add_argument("--smoke", action="store_true",
                        help="Use the synthetic series and short runs for a sanity check.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        if model == "xgboost":
            cmd = [
                sys.executable, "-m", "src.experiments.tune_xgboost",
                "--datasets", *args.datasets,
                "--horizons", *(str(h) for h in args.horizons),
                "--seeds", *(str(s) for s in args.seeds),
                "--input-length", str(args.input_length),
                "--output", str(args.output),
            ]
            run(cmd)
            continue

        seeds = [args.seeds[0]] if model in DETERMINISTIC else args.seeds
        for dataset in args.datasets:
            for horizon in args.horizons:
                for seed in seeds:
                    if model in NEURALFORECAST:
                        module = "src.experiments.run_neuralforecast_experiment"
                    elif model in FOUNDATION:
                        module = "src.experiments.run_foundation_experiment"
                    else:
                        module = "src.experiments.run_experiment"
                    cmd = [
                        sys.executable, "-m", module,
                        "--dataset", dataset,
                        "--raw-path", RAW_PATHS[dataset],
                        "--model", model,
                        "--horizon", str(horizon),
                        "--input-length", str(args.input_length),
                        "--seed", str(seed),
                        "--output", str(args.output),
                    ]
                    if model in NEURALFORECAST:
                        cmd.extend(["--max-steps", str(args.max_steps)])
                    if model == "timesfm":
                        cmd.extend(["--checkpoint", "google/timesfm-2.5-200m-pytorch"])
                    if args.smoke:
                        cmd.append("--smoke")
                    run(cmd)


if __name__ == "__main__":
    main()
