"""Diebold-Mariano tests on seed-averaged per-window errors.

For each (dataset, target, horizon) this picks the top-2 models by
mean MAE and runs the Diebold-Mariano test on their per-window
absolute-error vectors. For stochastic models (XGBoost, DLinear,
PatchTST, iTransformer) the per-window errors are first averaged
across all available seeds before the test, which is statistically
more conservative than evaluating a single seed.

Default outputs:
  ``results/stat_tests_final.csv``

Inputs:
  ``--results``    main_results CSV (default results/main_results_final.csv)
  ``--errors-dir`` directory with per-(model,dataset,target,horizon,seed) CSVs
                  produced during the main experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.statistical_tests import diebold_mariano


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
        df = pd.read_csv(path)[["origin", "abs_error"]].rename(
            columns={"abs_error": f"err_{seed}"}
        )
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


def write_tests(results_path: Path, output_path: Path, errors_dir: Path) -> None:
    results = pd.read_csv(results_path)
    rows = []
    for (dataset, target, horizon), part in results.groupby(["dataset", "target", "horizon"]):
        agg = part.groupby("model").agg(mean_MAE=("MAE", "mean")).reset_index()
        agg = agg.sort_values("mean_MAE")
        if len(agg) < 2:
            continue
        model_a, model_b = agg.iloc[0]["model"], agg.iloc[1]["model"]
        seeds_a = sorted(part[part.model == model_a]["seed"].unique().tolist())
        seeds_b = sorted(part[part.model == model_b]["seed"].unique().tolist())
        err_a, origins_a, n_a = load_pooled_errors(
            errors_dir, dataset, target, horizon, model_a, seeds_a
        )
        err_b, origins_b, n_b = load_pooled_errors(
            errors_dir, dataset, target, horizon, model_b, seeds_b
        )
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
            df_a = pd.DataFrame({"origin": origins_a, "ea": err_a})
            df_b = pd.DataFrame({"origin": origins_b, "eb": err_b})
            joined = df_a.merge(df_b, on="origin", how="inner")
            test = diebold_mariano(joined["ea"].to_numpy(), joined["eb"].to_numpy())
            row.update({"n_windows": len(joined), **test})
        rows.append(row)
    out = pd.DataFrame(rows, columns=COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/main_results_final.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/stat_tests_final.csv"))
    parser.add_argument("--errors-dir", type=Path, default=Path("results/window_errors"))
    args = parser.parse_args()
    write_tests(args.results, args.output, args.errors_dir)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
