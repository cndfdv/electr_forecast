"""Generate polished figures from experiment CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_LABELS = {
    "seasonal_naive": "SeasonalNaive",
    "xgboost": "XGBoost",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "chronos_bolt_small": "Chronos-Bolt",
    "timesfm": "TimesFM 2.5",
}

PALETTE = {
    "SeasonalNaive": "#7a7a7a",
    "XGBoost": "#1b9e77",
    "DLinear": "#d95f02",
    "PatchTST": "#7570b3",
    "iTransformer": "#e7298a",
    "Chronos-Bolt": "#66a61e",
    "TimesFM 2.5": "#e6ab02",
}


def label_model(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    df["model_label"] = df["model"].map(label_model)
    return df


def save_methodology(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.2))
    ax.axis("off")
    labels = [
        ("Public data", "ECL aggregate load\nETTh1/HUFL"),
        ("Leakage-safe setup", "Chronological split\nTrain-only scaling"),
        ("Model families", "Baseline, ML, DL\nFoundation models"),
        ("Evaluation", "MAE/RMSE/sMAPE\nLatency, memory, DM tests"),
        ("Deliverables", "CSV, figures\nPDF and Word draft"),
    ]
    x = np.linspace(0.08, 0.92, len(labels))
    for i, (xi, (title, body)) in enumerate(zip(x, labels)):
        ax.text(
            xi,
            0.62,
            title,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#17324d",
        )
        ax.text(
            xi,
            0.42,
            body,
            ha="center",
            va="center",
            fontsize=10,
            color="#263238",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#eef4f7", "edgecolor": "#b6c5cc"},
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.09, 0.42),
                xytext=(xi + 0.09, 0.42),
                arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#607d8b"},
            )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "horizon", "model", "model_label"], as_index=False)
        .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), sMAPE=("sMAPE", "mean"), inference_ms=("inference_ms", "mean"))
    )


def save_normalized_heatmap(df: pd.DataFrame, path: Path) -> None:
    agg = aggregate(df)
    agg["rel_mae"] = agg.groupby(["dataset", "horizon"])["MAE"].transform(lambda s: s / s.min())
    order = (
        agg.groupby("model_label")["rel_mae"]
        .mean()
        .sort_values()
        .index.tolist()
    )
    pivot = agg.pivot_table(index="model_label", columns=["dataset", "horizon"], values="rel_mae").reindex(order)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=1.0, vmax=min(1.35, np.nanmax(pivot.to_numpy())))
    ax.set_yticks(range(len(pivot.index)), pivot.index, fontsize=10)
    ax.set_xticks(range(len(pivot.columns)), [f"{d.upper()}\nH={h}" for d, h in pivot.columns], fontsize=10)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}x", ha="center", va="center", fontsize=9, color="#102027")
    ax.set_title("Relative MAE by Dataset and Horizon (Best Model = 1.00x)", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Relative MAE", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_accuracy_lines(df: pd.DataFrame, dataset: str, path: Path) -> None:
    part = aggregate(df)
    part = part[part["dataset"] == dataset].copy()
    order = part.groupby("model_label")["MAE"].mean().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for model in order:
        sub = part[part["model_label"] == model].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["MAE"],
            marker="o",
            linewidth=2.2,
            markersize=6,
            label=model,
            color=PALETTE.get(model),
        )
    ax.set_title(f"MAE Across Forecast Horizons: {dataset.upper()}", fontsize=14, pad=12)
    ax.set_xlabel("Forecast horizon, hours")
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_xticks([24, 96, 168])
    ax.legend(ncols=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_pareto(df: pd.DataFrame, path: Path, horizon: int = 96) -> None:
    part = aggregate(df)
    part = part[part["horizon"] == horizon].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.7), sharex=False)
    for ax, dataset in zip(axes, ["ecl", "etth1"]):
        sub = part[part["dataset"] == dataset].copy()
        x = np.log10(sub["inference_ms"].clip(lower=1e-4))
        y = sub["MAE"]
        for _, row in sub.iterrows():
            label = row["model_label"]
            ax.scatter(
                np.log10(max(row["inference_ms"], 1e-4)),
                row["MAE"],
                s=75,
                color=PALETTE.get(label, "#333333"),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.annotate(label, (np.log10(max(row["inference_ms"], 1e-4)), row["MAE"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
        ax.set_title(dataset.upper(), fontsize=12)
        ax.set_xlabel("log10 inference latency (ms/window)")
        ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Accuracy vs. Inference Latency at H={horizon}", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_family_bars(df: pd.DataFrame, path: Path) -> None:
    agg = aggregate(df)
    families = {
        "SeasonalNaive": "Baseline",
        "XGBoost": "ML",
        "DLinear": "DL",
        "PatchTST": "DL",
        "iTransformer": "DL",
        "Chronos-Bolt": "Foundation",
        "TimesFM 2.5": "Foundation",
    }
    agg["family"] = agg["model_label"].map(families)
    winners = agg.loc[agg.groupby(["dataset", "horizon"])["MAE"].idxmin()].copy()
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    labels = [f"{r.dataset.upper()}\nH={int(r.horizon)}" for r in winners.itertuples()]
    colors = [{"ML": "#1b9e77", "DL": "#d95f02", "Foundation": "#66a61e", "Baseline": "#7a7a7a"}[f] for f in winners["family"]]
    ax.bar(labels, winners["MAE"], color=colors)
    for i, row in enumerate(winners.itertuples()):
        ax.text(i, row.MAE, row.model_label, ha="center", va="bottom", fontsize=9, rotation=0)
    ax.set_title("Best Model per Dataset and Horizon", fontsize=14, pad=12)
    ax.set_ylabel("MAE")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_sample_forecast(path: Path, samples_dir: Path = Path("results/prediction_samples")) -> None:
    files = []
    preferred = ["xgboost", "chronos_bolt_small", "timesfm", "dlinear", "seasonal_naive"]
    for model in preferred:
        candidate = samples_dir / f"ecl_aggregate_load_h168_{model}_seed42.csv"
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError("Prediction samples are missing; rerun experiments before creating fig2.")

    frames = [pd.read_csv(path) for path in files]
    truth = frames[0]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.plot(pd.to_datetime(truth["ds"]), truth["y_true"], color="#111111", linewidth=2.5, label="Ground truth")
    for frame in frames:
        model = frame["model"].iloc[0]
        label = label_model(model)
        ax.plot(
            pd.to_datetime(frame["ds"]),
            frame["y_pred"],
            linewidth=1.8,
            alpha=0.9,
            label=label,
            color=PALETTE.get(label),
        )
    ax.set_title("Sample 168-hour Forecasts on ECL Test Split", fontsize=14, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Aggregate load")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(ncols=2, fontsize=9, frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results)
    save_methodology(args.output_dir / "fig1_methodology.pdf")
    save_sample_forecast(args.output_dir / "fig2_forecasts.pdf")
    save_normalized_heatmap(df, args.output_dir / "fig3_heatmap.pdf")
    save_pareto(df, args.output_dir / "fig4_pareto.pdf")
    save_family_bars(df, args.output_dir / "fig5_horizon.pdf")
    save_accuracy_lines(df, "ecl", args.output_dir / "fig6_ecl_accuracy.pdf")
    save_accuracy_lines(df, "etth1", args.output_dir / "fig7_etth1_accuracy.pdf")


if __name__ == "__main__":
    main()
