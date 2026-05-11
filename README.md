# Short-Term Electricity Load Forecasting Benchmark

This repository contains a reproducible benchmark for short-term electricity
load forecasting that compares classical statistical models, gradient-boosted
trees, neural forecasting models, and time-series foundation models under a
single chronological evaluation protocol.

The primary dataset is ECL aggregate load. ETTh1/HUFL is used as a secondary
industrial time-series benchmark. The ETTh1 `OT` column is not used as an
electricity-load target because it represents transformer oil temperature.

## Environments

Two conda/mamba environments are provided:

- `environment.yml` (`electr-forecast`): light environment with pandas,
  scikit-learn, XGBoost, StatsForecast, plotly, python-docx. Used for
  classical baselines, XGBoost tuning, figures, statistical tests.
- `environment_foundation.yml` (`electr-forecast-foundation`): heavier
  environment with PyTorch + CUDA, NeuralForecast, Chronos, and TimesFM.
  Used for DLinear, PatchTST, iTransformer, Chronos-Bolt-Small, TimesFM 2.5.

```bash
mamba env create -f environment.yml
mamba env create -f environment_foundation.yml
```

## Reproducing the paper results

`results/main_results_final.csv` and `results/stat_tests_final.csv` are
produced by the commands below. The two environments are split because
PyTorch and the foundation-model dependencies are isolated from the light
environment.

```bash
# Classical baselines + XGBoost grid-tune (CPU, no PyTorch needed):
mamba run -n electr-forecast python -m src.experiments.run_all \
    --models seasonal_naive sarima xgboost \
    --output results/main_results_final.csv

# Neural forecasting + foundation models (GPU recommended):
mamba run -n electr-forecast-foundation python -m src.experiments.run_all \
    --models dlinear patchtst itransformer chronos_bolt_small timesfm \
    --max-steps 5000 \
    --output results/main_results_final.csv

# Seed-pooled Diebold-Mariano tests on per-window errors:
mamba run -n electr-forecast python -m src.evaluation.run_stat_tests

# Regenerate figures from the merged final CSV:
mamba run -n electr-forecast python -m src.figures.make_figures \
    --results results/main_results_final.csv \
    --output-dir figures --format pdf \
    --png-output-dir paper/docx_figures
```

Defaults of `run_all`:

- Datasets: `ecl etth1`
- Horizons: `24 96 168`
- Seeds: `42 43 44 45 46`
- Input window: `336` hours
- `--max-steps 5000` with validation-driven early stopping (patience 10
  over 50-step checks) for DLinear / PatchTST / iTransformer
- XGBoost is run through `src.experiments.tune_xgboost`: a grid search
  over `max_depth ∈ {4, 6}`, `n_estimators ∈ {500, 1000}`,
  `learning_rate ∈ {0.05, 0.1}` selected on the validation split, then
  five-seed evaluation on the test split
- Deterministic models (SeasonalNaive, SARIMA, Chronos-Bolt, TimesFM) are
  evaluated once per dataset/horizon

Running with `--smoke` substitutes a synthetic series for the real
datasets and is useful for a quick end-to-end sanity check.

## Repository layout

```text
configs/                Experiment and model configs (reference only)
data/                   Raw and processed datasets (raw files git-ignored)
src/data/               Loading, validation, splits, scaling, windowing
src/models/             Baselines, XGBoost, NeuralForecast model helpers
src/evaluation/         Metrics, timing, Diebold-Mariano tests
src/experiments/        Per-model CLIs and the run_all orchestrator
src/figures/            Figure generation
results/                main_results_final.csv, stat_tests_final.csv
figures/                Publication figures (PDF) and PNG copies for DOCX
paper/                  Manuscript and embedded figure assets
```

## Reproducibility rules

- No test-set hyperparameter tuning. Validation splits are used for all
  hyperparameter selection (XGBoost grid, neural learning rate, SARIMA AICc).
- Preprocessing (StandardScaler) is fitted only on the training split.
  Neural-forecast models have their internal scaler disabled to avoid
  double-scaling pre-standardised inputs.
- Public-dataset pretraining contamination for foundation models is treated
  as a limitation in the paper.
- ETTh1 `OT` is not reported as an electricity-load target.
- The `iTransformer` model is run with `n_series=1` (univariate) so that
  all eight models share the same protocol; this is acknowledged in
  Limitations.
