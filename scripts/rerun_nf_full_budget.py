"""Re-run all NeuralForecast experiments with a full training budget.

The existing results in ``results/main_results.csv`` were produced with
``--max-steps 500``, which leaves PatchTST / iTransformer severely
under-trained. This script reruns all NF configurations with
``--max-steps 5000`` so that early stopping (val_check_steps=50,
patience=10) becomes the binding constraint instead of the step cap.

Outputs land in ``results/main_results_v2.csv`` so the previous run
remains intact for comparison. Every (model, dataset, horizon, seed)
combination is wrapped in try/except so that a single failure cannot
abort the overnight batch.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "main_results_v2.csv"
DEFAULT_LOG = ROOT / "results" / "rerun_log.txt"

MODELS = ["dlinear", "patchtst", "itransformer"]
DATASETS = [
    ("ecl", "data/raw/electricity.txt.gz"),
    ("etth1", "data/raw/ETTh1.csv"),
]
HORIZONS = [24, 96, 168]
SEEDS = [42, 43, 44, 45, 46]


def run_config(
    *,
    model: str,
    dataset: str,
    raw_path: str,
    horizon: int,
    seed: int,
    max_steps: int,
    output: Path,
    env: str,
    log_path: Path,
) -> tuple[bool, float, str]:
    cmd = [
        "mamba", "run", "-n", env,
        "python", "-m", "src.experiments.run_neuralforecast_experiment",
        "--dataset", dataset,
        "--raw-path", raw_path,
        "--model", model,
        "--horizon", str(horizon),
        "--input-length", "336",
        "--seed", str(seed),
        "--max-steps", str(max_steps),
        "--learning-rates", "1e-3", "5e-4",
        "--early-stop-patience-steps", "10",
        "--output", str(output),
    ]
    start = time.perf_counter()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=60 * 60,  # 1h per config hard cap
        )
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - start
        return False, elapsed, "TIMEOUT after 1h"
    elapsed = time.perf_counter() - start
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-15:])
        return False, elapsed, f"exit {result.returncode}\n{tail}"
    return True, elapsed, ""


def log(line: str, log_path: Path) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{stamp}] {line}"
    print(msg, flush=True)
    with log_path.open("a") as fh:
        fh.write(msg + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--env", default="electr-forecast-foundation")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--datasets", nargs="+", default=[d[0] for d in DATASETS])
    parser.add_argument("--horizons", nargs="+", type=int, default=HORIZONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    raw_paths = {name: path for name, path in DATASETS}

    configs = []
    for dataset in args.datasets:
        for model in args.models:
            for horizon in args.horizons:
                for seed in args.seeds:
                    configs.append((model, dataset, horizon, seed))

    log(f"Starting rerun: {len(configs)} configurations, max_steps={args.max_steps}, "
        f"output={args.output}", args.log)

    overall_start = time.perf_counter()
    successes = 0
    failures = 0
    for i, (model, dataset, horizon, seed) in enumerate(configs, 1):
        prefix = f"[{i}/{len(configs)}] {model}/{dataset}/H{horizon}/seed{seed}"
        log(f"{prefix} START", args.log)
        ok, elapsed, err = run_config(
            model=model,
            dataset=dataset,
            raw_path=raw_paths[dataset],
            horizon=horizon,
            seed=seed,
            max_steps=args.max_steps,
            output=args.output,
            env=args.env,
            log_path=args.log,
        )
        if ok:
            successes += 1
            log(f"{prefix} OK in {elapsed:.1f}s", args.log)
        else:
            failures += 1
            log(f"{prefix} FAIL in {elapsed:.1f}s: {err}", args.log)

    total = time.perf_counter() - overall_start
    log(f"DONE: {successes} ok, {failures} failed, total {total/60:.1f} min", args.log)


if __name__ == "__main__":
    main()
