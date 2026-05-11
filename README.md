# Short-Term Electricity Load Forecasting Benchmark

This repository contains a reproducible benchmark for short-term electricity
load forecasting with classical statistical models, gradient-boosted trees,
neural forecasting models, and time-series foundation models.

The primary dataset is ECL aggregate load. ETTh1/HUFL is used as a secondary
industrial time-series benchmark. The ETTh1 `OT` column is not used as an
electricity-load target because it represents transformer oil temperature.

## Quick Start

```bash
mamba env create -f environment.yml
mamba activate electr-forecast
python -m src.experiments.run_experiment --dataset synthetic --model seasonal_naive --horizon 24 --seed 42 --smoke
```

The default environment supports data processing, classical baselines, XGBoost,
metrics, statistical tests, and figure generation. GPU and foundation-model
dependencies are isolated in `environment_foundation.yml` because PyTorch/CUDA
and pretrained forecasting packages are substantially heavier.

Run a small real-data baseline:

```bash
python -m src.experiments.run_all --datasets ecl etth1 --models seasonal_naive --horizons 24 96 168 --seeds 42
python -m src.figures.make_figures --results results/main_results.csv --output-dir figures
python -m src.evaluation.run_stat_tests --results results/main_results.csv --output results/stat_tests.csv
```

After installing the foundation environment, extend `--models` to `xgboost`,
`auto_arima`, `chronos_bolt_small`, `timesfm`, `dlinear`, `patchtst`, and
`itransformer`. Full neural and foundation-model runs are GPU-intensive and
should be launched deliberately.

The final reported result tables are stored in:

- `results/main_results_final.csv`
- `results/stat_tests_final.csv`

The final manuscript is `paper/main_scopus.docx`.

## Repository Layout

```text
configs/        Experiment and model configs
data/           Raw and processed datasets
src/data/       Downloading, validation, preprocessing, windowing
src/models/     Baselines, XGBoost, NeuralForecast model helpers
src/evaluation/ Metrics, timing, statistical tests
src/experiments Experiment CLIs
src/figures/    Figure generation
results/        Final CSV result tables
figures/        Publication figures
paper/          Final manuscript and embedded figure assets
```

## Reproducibility Rules

- No test-set hyperparameter tuning.
- Preprocessing is fitted only on the permitted training split.
- Public-dataset pretraining contamination for foundation models is treated as a limitation.
- ETTh1 `OT` is not reported as electricity load.
- Missing numerical results must remain explicitly marked, never fabricated.
