"""Generate publication figures from experiment CSV files using Plotly."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


MODEL_LABELS = {
    "seasonal_naive": "SeasonalNaive",
    "sarima": "SARIMA",
    "xgboost": "XGBoost",
    "dlinear": "DLinear",
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "chronos_bolt_small": "Chronos-Bolt",
    "timesfm": "TimesFM 2.5",
}

PALETTE = {
    "SeasonalNaive": "#6B7280",
    "SARIMA": "#0F766E",
    "XGBoost": "#2563EB",
    "DLinear": "#F97316",
    "PatchTST": "#7C3AED",
    "iTransformer": "#DB2777",
    "Chronos-Bolt": "#16A34A",
    "TimesFM 2.5": "#D97706",
}

FAMILY_COLORS = {
    "Baseline": "#6B7280",
    "Classical": "#0F766E",
    "ML": "#2563EB",
    "DL": "#7C3AED",
    "Foundation": "#16A34A",
}

FIG_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Arial, Helvetica, sans-serif", "size": 16, "color": "#111827"},
    "title": {"font": {"size": 22, "color": "#111827"}, "x": 0.02, "xanchor": "left"},
    "paper_bgcolor": "white",
    "plot_bgcolor": "white",
    "margin": {"l": 72, "r": 28, "t": 74, "b": 64},
}


def label_model(name: str) -> str:
    return MODEL_LABELS.get(name, name)


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    df["model_label"] = df["model"].map(label_model)
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["dataset", "horizon", "model", "model_label"], as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            sMAPE=("sMAPE", "mean"),
            inference_ms=("inference_ms", "mean"),
        )
    )


def apply_axes_style(fig: go.Figure) -> None:
    fig.update_layout(**FIG_LAYOUT)
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="outside", tickfont={"size": 13})
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E5E7EB",
        gridwidth=1,
        zeroline=False,
        ticks="outside",
        tickfont={"size": 13},
    )


def write_figure(fig: go.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(path, scale=2)


def save_methodology(path: Path) -> None:
    fig = go.Figure()
    steps = [
        ("Data", "ECL aggregate load; ETTh1/HUFL"),
        ("Protocol", "Chronological split; train-only scaling"),
        ("Models", "Seasonal/ARIMA, XGBoost, DL, foundation"),
        ("Metrics", "MAE, RMSE, sMAPE, wMAPE; latency"),
        ("Outputs", "Tables, statistical tests, figures, DOCX"),
    ]
    ys = np.linspace(0.88, 0.16, len(steps))
    for i, (y, (title, body)) in enumerate(zip(ys, steps)):
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="paper",
            x0=0.085,
            x1=0.915,
            y0=y - 0.058,
            y1=y + 0.058,
            line={"color": "#CBD5E1", "width": 1.25},
            fillcolor="#F9FAFB",
            layer="below",
        )
        fig.add_shape(
            type="circle",
            xref="paper",
            yref="paper",
            x0=0.115,
            x1=0.175,
            y0=y - 0.030,
            y1=y + 0.030,
            line={"color": "#1D4ED8", "width": 1},
            fillcolor="#2563EB",
            layer="below",
        )
        fig.add_annotation(
            x=0.145,
            y=y,
            xref="paper",
            yref="paper",
            text=str(i + 1),
            showarrow=False,
            font={"size": 15, "color": "white"},
            align="center",
        )
        fig.add_annotation(
            x=0.235,
            y=y + 0.018,
            xref="paper",
            yref="paper",
            text=f"<b>{title}</b>",
            showarrow=False,
            font={"size": 14, "color": "#0F172A"},
            align="left",
            xanchor="left",
        )
        fig.add_annotation(
            x=0.235,
            y=y - 0.020,
            xref="paper",
            yref="paper",
            text=body,
            showarrow=False,
            font={"size": 12.5, "color": "#334155"},
            align="left",
            xanchor="left",
        )
        if i < len(steps) - 1:
            fig.add_shape(
                type="line",
                xref="paper",
                yref="paper",
                x0=0.145,
                x1=0.145,
                y0=y - 0.063,
                y1=ys[i + 1] + 0.063,
                line={"color": "#94A3B8", "width": 1.4},
            )
            fig.add_annotation(
                x=0.145,
                y=ys[i + 1] + 0.063,
                xref="paper",
                yref="paper",
                text="",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.0,
                arrowwidth=1.4,
                arrowcolor="#94A3B8",
                ax=0,
                ay=-20,
            )
    fig.update_layout(**FIG_LAYOUT)
    fig.update_layout(
        width=760,
        height=760,
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
    )
    write_figure(fig, path)


def save_normalized_heatmap(df: pd.DataFrame, path: Path) -> None:
    agg = aggregate(df)
    agg["rel_mae"] = agg.groupby(["dataset", "horizon"])["MAE"].transform(lambda s: s / s.min())
    order = agg.groupby("model_label")["rel_mae"].mean().sort_values().index.tolist()
    pivot = agg.pivot_table(index="model_label", columns=["dataset", "horizon"], values="rel_mae").reindex(order)
    columns = [f"{d.upper()}<br>H={h}" for d, h in pivot.columns]
    values = pivot.to_numpy()
    fig = go.Figure(
        data=go.Heatmap(
            z=values,
            x=columns,
            y=pivot.index.tolist(),
            colorscale=[
                [0.0, "#ECFDF5"],
                [0.45, "#A7F3D0"],
                [0.72, "#38BDF8"],
                [1.0, "#1E3A8A"],
            ],
            zmin=1.0,
            zmax=min(1.35, np.nanmax(values)),
            colorbar={"title": "Relative<br>MAE", "thickness": 14},
            text=np.vectorize(lambda x: f"{x:.2f}x")(values),
            texttemplate="%{text}",
            textfont={"size": 13, "color": "#0F172A"},
            hovertemplate="%{y}<br>%{x}<br>relative MAE=%{z:.3f}x<extra></extra>",
        )
    )
    fig.update_layout(width=1100, height=620, title="Relative MAE by Dataset and Horizon")
    apply_axes_style(fig)
    fig.update_xaxes(side="top")
    write_figure(fig, path)


def save_accuracy_lines(df: pd.DataFrame, dataset: str, path: Path) -> None:
    part = aggregate(df)
    part = part[part["dataset"] == dataset].copy()
    order = part.groupby("model_label")["MAE"].mean().sort_values().index.tolist()
    fig = go.Figure()
    for model in order:
        sub = part[part["model_label"] == model].sort_values("horizon")
        fig.add_trace(
            go.Scatter(
                x=sub["horizon"],
                y=sub["MAE"],
                mode="lines+markers",
                name=model,
                line={"width": 3, "color": PALETTE.get(model)},
                marker={"size": 9, "line": {"color": "white", "width": 1}},
                hovertemplate=f"{model}<br>H=%{{x}}<br>MAE=%{{y:,.3f}}<extra></extra>",
            )
        )
    fig.update_layout(
        width=1050,
        height=620,
        title=f"MAE Across Forecast Horizons: {dataset.upper()}",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.24, "xanchor": "center", "x": 0.5},
    )
    apply_axes_style(fig)
    fig.update_xaxes(title="Forecast horizon, hours", tickmode="array", tickvals=[24, 96, 168])
    fig.update_yaxes(title="MAE")
    write_figure(fig, path)


def save_pareto(df: pd.DataFrame, path: Path, horizon: int = 96) -> None:
    part = aggregate(df)
    part = part[part["horizon"] == horizon].copy()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("ECL", "ETTh1/HUFL"), horizontal_spacing=0.12)
    for col, dataset in enumerate(["ecl", "etth1"], start=1):
        sub = part[part["dataset"] == dataset].copy()
        sub["latency_log"] = np.log10(sub["inference_ms"].clip(lower=1e-4))
        for _, row in sub.iterrows():
            label = row["model_label"]
            fig.add_trace(
                go.Scatter(
                    x=[row["latency_log"]],
                    y=[row["MAE"]],
                    mode="markers",
                    marker={
                        "size": 14,
                        "color": PALETTE.get(label, "#111827"),
                        "line": {"color": "white", "width": 1.5},
                    },
                    name=label,
                    legendgroup=label,
                    showlegend=col == 1,
                    hovertemplate=f"{label}<br>latency=%{{customdata:.3f}} ms<br>MAE=%{{y:,.3f}}<extra></extra>",
                    customdata=[row["inference_ms"]],
                ),
                row=1,
                col=col,
            )
    fig.update_layout(
        width=1200,
        height=600,
        title=f"Accuracy vs. Single-Window Inference Latency at H={horizon}",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.24, "xanchor": "center", "x": 0.5},
    )
    apply_axes_style(fig)
    fig.update_xaxes(title="log10 latency (ms/window)")
    fig.update_yaxes(title="MAE")
    write_figure(fig, path)


def save_family_bars(df: pd.DataFrame, path: Path) -> None:
    agg = aggregate(df)
    families = {
        "SeasonalNaive": "Baseline",
        "SARIMA": "Classical",
        "XGBoost": "ML",
        "DLinear": "DL",
        "PatchTST": "DL",
        "iTransformer": "DL",
        "Chronos-Bolt": "Foundation",
        "TimesFM 2.5": "Foundation",
    }
    agg["family"] = agg["model_label"].map(families)
    winners = agg.loc[agg.groupby(["dataset", "horizon"])["MAE"].idxmin()].copy()
    winners = winners.sort_values(["dataset", "horizon"])
    labels = [f"{row.dataset.upper()}<br>H={int(row.horizon)}" for row in winners.itertuples()]
    fig = go.Figure(
        go.Bar(
            x=labels,
            y=winners["MAE"],
            marker={"color": [FAMILY_COLORS[f] for f in winners["family"]]},
            text=winners["model_label"],
            textposition="outside",
            textfont={"size": 13},
            hovertemplate="%{text}<br>%{x}<br>MAE=%{y:,.3f}<extra></extra>",
        )
    )
    fig.update_layout(width=1050, height=560, title="Best Model per Dataset and Horizon", showlegend=False)
    apply_axes_style(fig)
    fig.update_xaxes(title="")
    fig.update_yaxes(title="MAE")
    write_figure(fig, path)


def save_sample_forecast(path: Path, samples_dir: Path = Path("results/prediction_samples")) -> None:
    files = []
    preferred = ["xgboost", "chronos_bolt_small", "timesfm", "sarima"]
    for model in preferred:
        candidate = samples_dir / f"ecl_aggregate_load_h168_{model}_seed42.csv"
        if candidate.exists():
            files.append(candidate)
    if not files:
        raise FileNotFoundError("Prediction samples are missing; rerun experiments before creating fig2.")

    frames = [pd.read_csv(file) for file in files]
    truth = frames[0]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=truth["step"],
            y=truth["y_true"],
            mode="lines",
            name="Ground truth",
            line={"color": "#111827", "width": 3.2},
            hovertemplate="Ground truth<br>step=%{x}<br>%{y:,.0f}<extra></extra>",
        )
    )
    for frame in frames:
        model = frame["model"].iloc[0]
        label = label_model(model)
        fig.add_trace(
            go.Scatter(
                x=frame["step"],
                y=frame["y_pred"],
                mode="lines",
                name=label,
                line={"color": PALETTE.get(label), "width": 2.7},
                hovertemplate=f"{label}<br>step=%{{x}}<br>%{{y:,.0f}}<extra></extra>",
            )
        )
    fig.update_layout(
        width=1200,
        height=620,
        title="Sample 168-Hour ECL Forecast Window",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.24, "xanchor": "center", "x": 0.5},
    )
    apply_axes_style(fig)
    fig.update_xaxes(title="Forecast step, hour")
    fig.update_yaxes(title="Aggregate load")
    write_figure(fig, path)


def save_all(df: pd.DataFrame, output_dir: Path, suffix: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_methodology(output_dir / f"fig1_methodology.{suffix}")
    save_sample_forecast(output_dir / f"fig2_forecasts.{suffix}")
    save_normalized_heatmap(df, output_dir / f"fig3_heatmap.{suffix}")
    save_pareto(df, output_dir / f"fig4_pareto.{suffix}")
    save_family_bars(df, output_dir / f"fig5_horizon.{suffix}")
    save_accuracy_lines(df, "ecl", output_dir / f"fig6_ecl_accuracy.{suffix}")
    save_accuracy_lines(df, "etth1", output_dir / f"fig7_etth1_accuracy.{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/main_results.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--format", choices=["pdf", "png", "svg"], default="pdf")
    parser.add_argument("--png-output-dir", type=Path, default=None)
    args = parser.parse_args()

    df = load_results(args.results)
    save_all(df, args.output_dir, args.format)
    if args.png_output_dir is not None:
        save_all(df, args.png_output_dir, "png")


if __name__ == "__main__":
    main()
