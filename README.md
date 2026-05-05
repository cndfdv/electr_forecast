# Electricity Load Forecasting Benchmark

Reproducible benchmark for short-term electricity load forecasting with classical, ML, deep learning, and time-series foundation models.

Primary dataset:
- ECL aggregate load.

Secondary benchmark:
- ETTh1 with an explicitly selected load-related target such as `HUFL`, or `OT` only under the broader transformer condition / oil-temperature forecasting framing.

## Quick Start

```bash
mamba env create -f environment.yml
mamba activate electr-forecast
python -m src.experiments.run_experiment --dataset synthetic --model seasonal_naive --horizon 24 --seed 42 --smoke
```

The default environment is intentionally light and supports data processing,
SeasonalNaive, AutoARIMA/StatsForecast, XGBoost, metrics, and figures. GPU and
foundation-model dependencies are isolated in `environment_foundation.yml`
because PyTorch/CUDA downloads are large and can be brittle.

Run the currently lightweight real-data baseline:

```bash
python -m src.experiments.run_all --datasets ecl etth1 --models seasonal_naive --horizons 24 96 168 --seeds 42
python -m src.figures.make_figures --results results/main_results.csv --output-dir figures
python -m src.evaluation.run_stat_tests --results results/main_results.csv --output results/stat_tests.csv
```

After installing the heavy dependencies, extend `--models` to `xgboost`, `auto_arima`, `chronos`, `timesfm`, and the NeuralForecast models. The neural wrappers are present, but full neural training should be run deliberately because it is GPU- and time-intensive.

Full experiments are configured through `configs/*.yaml`. Heavy model families are optional at import time, but require the relevant packages from `environment.yml`.

## Repository Layout

```text
configs/        Experiment and model configs
data/           Raw and processed datasets
src/data/       Downloading, validation, preprocessing, windowing
src/models/     Baselines, XGBoost, NeuralForecast model helpers
src/evaluation/ Metrics, timing, statistical tests
src/experiments Experiment CLIs
src/figures/    Figure generation
results/        CSV outputs
figures/        PDF figures
paper/          IEEE paper sources
```

## Reproducibility Rules

- No test-set hyperparameter tuning.
- Preprocessing is fitted only on the permitted training split.
- Public-dataset pretraining contamination for foundation models is treated as a limitation.
- ETTh1 `OT` is not reported as electricity load.
- Missing numerical results must remain `NA`/TODO, never fabricated.
