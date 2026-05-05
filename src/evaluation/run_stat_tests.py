"""Build pairwise statistical-test CSV from per-window error files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.evaluation.statistical_tests import diebold_mariano


COLUMNS = [
    "dataset",
    "target",
    "horizon",
    "seed",
    "metric",
    "model_a",
    "model_b",
    "test_name",
    "statistic",
    "p_value",
    "significant",
]


def load_error_file(errors_dir: Path, dataset: str, target: str, horizon: int, model: str, seed: int) -> pd.DataFrame | None:
    path = errors_dir / f"{dataset}_{target}_h{horizon}_{model}_seed{seed}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def write_tests(results_path: Path, output_path: Path, errors_dir: Path, seed: int) -> None:
    results = pd.read_csv(results_path)
    results = results[results["seed"] == seed].copy()
    rows = []
    for (dataset, target, horizon, seed), part in results.groupby(["dataset", "target", "horizon", "seed"]):
        models = list(part.sort_values("MAE")["model"].drop_duplicates())
        if len(models) < 2:
            continue
        model_a, model_b = models[0], models[1]
        err_a = load_error_file(errors_dir, dataset, target, horizon, model_a, seed)
        err_b = load_error_file(errors_dir, dataset, target, horizon, model_b, seed)
        row = {
            "dataset": dataset,
            "target": target,
            "horizon": horizon,
            "seed": seed,
            "metric": "MAE",
            "model_a": model_a,
            "model_b": model_b,
        }
        if err_a is None or err_b is None:
            row.update({"test_name": "missing_per_window_errors", "statistic": "", "p_value": "", "significant": ""})
        else:
            merged = err_a[["origin", "abs_error"]].merge(
                err_b[["origin", "abs_error"]],
                on="origin",
                suffixes=("_a", "_b"),
            )
            test = diebold_mariano(merged["abs_error_a"].to_numpy(), merged["abs_error_b"].to_numpy())
            row.update(test)
        rows.append(row)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/stat_tests.csv"))
    parser.add_argument("--errors-dir", type=Path, default=Path("results/window_errors"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    write_tests(args.results, args.output, args.errors_dir, args.seed)


if __name__ == "__main__":
    main()
