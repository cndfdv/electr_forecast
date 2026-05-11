"""Merge baseline results (foundation + classical) with the v2 rerun.

The v1 file ``results/main_results.csv`` was produced with
``--max-steps 500`` for NeuralForecast models and with untuned XGBoost
defaults. The v2 file ``results/main_results_v2.csv`` contains the
re-tuned NF (5000 steps) and grid-searched XGBoost results.

This script writes ``results/main_results_final.csv`` keeping the
foundation and classical baselines from v1 (they were already at their
maximum-quality configuration) and replacing the NF and XGBoost rows
with the v2 numbers.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
V1 = ROOT / "results" / "main_results.csv"
V2 = ROOT / "results" / "main_results_v2.csv"
OUT = ROOT / "results" / "main_results_final.csv"

KEEP_FROM_V1 = {"seasonal_naive", "sarima", "chronos_bolt_small", "timesfm"}
REPLACE_FROM_V2 = {"dlinear", "patchtst", "itransformer", "xgboost"}


def main() -> None:
    v1 = pd.read_csv(V1)
    v2 = pd.read_csv(V2)
    keep = v1[v1.model.isin(KEEP_FROM_V1)].copy()
    new = v2[v2.model.isin(REPLACE_FROM_V2)].copy()
    final = pd.concat([keep, new], ignore_index=True)
    final = final.sort_values(["dataset", "horizon", "model", "seed"]).reset_index(drop=True)
    final.to_csv(OUT, index=False)
    print(f"Saved {OUT} ({len(final)} rows)")
    print("Counts per model:")
    print(final.model.value_counts())


if __name__ == "__main__":
    main()
