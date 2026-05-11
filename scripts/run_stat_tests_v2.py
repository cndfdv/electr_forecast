"""Diebold-Mariano with seed-averaged per-window errors.

The original run_stat_tests.py picked the top-2 models by MAE at one
seed (42) and ran DM on their window-level absolute errors at that seed.
For stochastic models (XGBoost, DLinear, PatchTST, iTransformer) this
is statistically weaker than it could be: each seed gives one MAE,
several seeds give a more stable estimate.

This script averages absolute errors across all available seeds per
(model, dataset, target, horizon) before computing DM. Deterministic
models (SeasonalNaive, SARIMA, Chronos-Bolt, TimesFM) only have one
seed and contribute that single trajectory.

Inputs : ``results/main_results_final.csv``
         ``results/window_errors/*.csv``
Outputs: ``results/stat_tests_final.csv``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import diebold_mariano

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = ROOT / "results" / "main_results_final.csv"
DEFAULT_ERRORS = ROOT / "results" / "window_errors"
DEFAULT_OUTPUT = ROOT / "results" / "stat_tests_final.csv"


COLUMNS = [
    "dataset", "target", "horizon", "metric",
    "model_a", "model_b",
    "n_seeds_a", "n_seeds_b",
    "n_windows",
    "test_name", "statistic", "p_value", "significant",
]


def load_pooled_errors(
    errors_dir: Path,
    dataset: str,
    target: str,
    horizon: int,
    model: str,
    seeds: list[int],
) -> tuple[np.ndarray | None, list[str] | None, int]:
    """Return seed-averaged absolute errors per origin (aligned by origin)."""
    parts = []
    used = 0
    for seed in seeds:
        path = errors_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)[["origin", "abs_error"]]
        df = df.rename(columns={"abs_error": f"err_{seed}"})
        parts.append(df)
        used += 1
    if not parts:
        return None, None, 0
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on="origin", how="inner")
    err_cols = [c for c in merged.columns if c.startswith("err_")]
    averaged = merged[err_cols].mean(axis=1).to_numpy()
    origins = merged["origin"].tolist()
    return averaged, origins, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--errors-dir", type=Path, default=DEFAULT_ERRORS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    results = pd.read_csv(args.results)
    rows = []
    for (dataset, target, horizon), part in results.groupby(["dataset", "target", "horizon"]):
        agg = part.groupby("model").agg(mean_MAE=("MAE", "mean")).reset_index()
        agg = agg.sort_values("mean_MAE")
        if len(agg) < 2:
            continue
        model_a, model_b = agg.iloc[0]["model"], agg.iloc[1]["model"]
        seeds_a = sorted(part[part.model == model_a]["seed"].unique().tolist())
        seeds_b = sorted(part[part.model == model_b]["seed"].unique().tolist())
        err_a, origins_a, n_a = load_pooled_errors(args.errors_dir, dataset, target, horizon, model_a, seeds_a)
        err_b, origins_b, n_b = load_pooled_errors(args.errors_dir, dataset, target, horizon, model_b, seeds_b)
        row = {
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "metric": "MAE",
            "model_a": model_a,
            "model_b": model_b,
            "n_seeds_a": n_a,
            "n_seeds_b": n_b,
        }
        if err_a is None or err_b is None:
            row.update({
                "n_windows": 0,
                "test_name": "missing_per_window_errors",
                "statistic": "", "p_value": "", "significant": "",
            })
        else:
            # Align both pooled vectors by origin (inner-join on shared origins).
            df_a = pd.DataFrame({"origin": origins_a, "ea": err_a})
            df_b = pd.DataFrame({"origin": origins_b, "eb": err_b})
            joined = df_a.merge(df_b, on="origin", how="inner")
            test = diebold_mariano(joined["ea"].to_numpy(), joined["eb"].to_numpy())
            row.update({"n_windows": len(joined), **test})
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
