"""Generate LaTeX tables from experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def fmt_mean_std(mean: float, std: float | None, digits: int = 2) -> str:
    if pd.isna(std) or std == 0:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def tex_escape(value: object) -> str:
    return str(value).replace("_", "\\_")


def accuracy_table(df: pd.DataFrame, dataset: str, output: Path) -> None:
    part = df[df["dataset"] == dataset].copy()
    agg = (
        part.groupby(["model", "horizon"])
        .agg(
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            sMAPE_mean=("sMAPE", "mean"),
            sMAPE_std=("sMAPE", "std"),
            wMAPE_mean=("wMAPE", "mean"),
            wMAPE_std=("wMAPE", "std"),
        )
        .reset_index()
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{Accuracy on {dataset.upper()} test split.}}",
        f"\\label{{tab:{dataset}_accuracy}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & H & MAE & RMSE & sMAPE (\\%) & wMAPE (\\%) \\\\",
        "\\midrule",
    ]
    for _, row in agg.sort_values(["horizon", "MAE_mean"]).iterrows():
        lines.append(
            f"{tex_escape(row['model'])} & {int(row['horizon'])} & "
            f"{fmt_mean_std(row['MAE_mean'], row['MAE_std'])} & "
            f"{fmt_mean_std(row['RMSE_mean'], row['RMSE_std'])} & "
            f"{fmt_mean_std(row['sMAPE_mean'], row['sMAPE_std'])} & "
            f"{fmt_mean_std(row['wMAPE_mean'], row['wMAPE_std'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def efficiency_table(df: pd.DataFrame, output: Path) -> None:
    agg = (
        df.groupby(["model", "dataset", "horizon"])
        .agg(
            train_s=("train_time_s", "mean"),
            infer_ms=("inference_ms", "mean"),
            n_params=("n_params", "mean"),
            gpu_mb=("peak_gpu_mb", "mean"),
        )
        .reset_index()
    )
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Computational efficiency.}",
        "\\label{tab:efficiency}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Model & Data/H & Train s & Infer ms & Params & GPU MB \\\\",
        "\\midrule",
    ]
    for _, row in agg.sort_values(["dataset", "horizon", "model"]).iterrows():
        lines.append(
            f"{tex_escape(row['model'])} & {tex_escape(row['dataset'])}/H{int(row['horizon'])} & "
            f"{row['train_s']:.3f} & {row['infer_ms']:.4f} & {row['n_params']:.0f} & {row['gpu_mb']:.1f} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/main_results_final.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/tables"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.results)
    accuracy_table(df, "ecl", args.output_dir / "table_ecl_accuracy.tex")
    accuracy_table(df, "etth1", args.output_dir / "table_etth1_accuracy.tex")
    efficiency_table(df, args.output_dir / "table_efficiency.tex")


if __name__ == "__main__":
    main()
