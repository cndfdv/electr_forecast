::: IEEEkeywords
short-term load forecasting, time series, deep learning, foundation
models, XGBoost, smart grid
:::

# Introduction

Short-term load forecasting (STLF) supports dispatch planning,
balancing, tariff-aware operation, maintenance planning, and
anomaly-aware industrial energy management. In modern smart-grid and
Industry 4.0 settings, forecasting models are expected not only to be
accurate, but also to be reproducible, computationally affordable, and
robust across operating regimes.

The methodological landscape is broad. Classical methods such as ARIMA
and seasonal naive baselines remain common references; machine-learning
models such as gradient boosting are attractive because they incorporate
lag and calendar features with modest engineering cost; and recent
deep-learning and foundation models promise broader transfer across
time-series domains. This creates a practical question: when do
expensive architectures or zero-shot foundation models actually improve
over strong inexpensive baselines?

This paper reports a reproducible execution of the benchmark on two
public datasets. The contribution is pragmatic: (i) an end-to-end
repository with data validation, leakage-aware scaling, metrics, timing,
figures, and paper generation; (ii) real results for SeasonalNaive,
XGBoost, DLinear, PatchTST, iTransformer, Chronos-Bolt-Small, and
TimesFM 2.5; (iii) Diebold--Mariano tests computed from per-window
errors; and (iv) an accuracy--latency analysis that separates practical
deployment cost from raw accuracy.

# Related Work

Classical time-series forecasting methods, including ARIMA-family
models, remain important because they establish interpretable and
inexpensive references. Machine-learning approaches based on lagged
targets, calendar variables, and gradient-boosted trees are widely used
in load forecasting because they are fast to train and robust under
tabular feature engineering [@chen2016xgboost]. Deep-learning models
such as LSTM [@hochreiter1997lstm], attention-based architectures
[@vaswani2017attention], Informer [@zhou2021informer], DLinear
[@zeng2023dlinear], PatchTST [@nie2023patchtst], iTransformer
[@liu2024itransformer], and TimesNet [@wu2023timesnet] have broadened
the long-horizon forecasting toolbox.

Time-series foundation models such as TimesFM [@das2024timesfm] and
Chronos [@ansari2024chronos] motivate a new evaluation axis:
no-task-specific-training inference versus supervised training on the
target dataset. A fair comparison must state that public benchmarks may
have appeared in pretraining corpora, so zero-shot performance on ECL or
ETT should not be interpreted as guaranteed dataset-unseen
generalization.

# Methodology

Given a univariate context window $\mathbf{x}_{t-L+1:t}$, the task is to
predict $\hat{\mathbf{y}}_{t+1:t+H}$ for horizons $H \in \{24,96,168\}$.
The executed experiments use a fixed context length of $L=336$ hours,
which includes daily and weekly seasonal information and permits use of
lag-168 features.

The SeasonalNaive baseline repeats the most recent daily seasonal
pattern. XGBoost is trained as a direct multi-output regressor using
lag-1, lag-24, lag-168, rolling means over 24 and 168 hours, and cyclic
calendar features. DLinear, PatchTST, and iTransformer are trained
through NeuralForecast in the same target-only setting, with a short
validation-only learning-rate sweep over $\{10^{-3},5\cdot10^{-4}\}$ and
early stopping. Chronos-Bolt-Small and TimesFM 2.5 are evaluated in
zero-shot mode without task-specific training. All preprocessing for
trained models is fitted only on the training split, and metrics are
computed after inverse-transforming predictions to the original target
scale. Foundation models are evaluated directly on the original target
scale.

![Experimental
pipeline.](../figures/fig1_methodology.pdf){#fig:methodology
width="\\linewidth"}

# Experimental Setup

The primary dataset is ECL, processed as an aggregate hourly electricity
load across 321 clients. The secondary dataset is ETTh1 with HUFL as the
target. ETTh1 OT is oil temperature and is not used as electricity load.
ECL is split chronologically into 70% train, 10% validation, and 20%
test. ETTh1 uses the standard 12-month, 4-month, 4-month chronological
split.

Accuracy is measured with MAE, RMSE, sMAPE, and wMAPE. Efficiency is
measured with training time, single-window inference latency, trainable
parameters, and peak GPU memory. Diebold--Mariano tests are computed on
per-window absolute errors for the top two models by MAE at seed 42 for
each dataset and horizon. Because per-window averaging and flattened
point-wise MAE weight errors differently, the statistical-test ranking
can differ slightly from the table ranking when models are very close.

# Results

![Sample 168-hour ECL forecasts from representative
models.](../figures/fig2_forecasts.pdf){#fig:sample_forecasts
width="\\linewidth"}

On ECL, the strongest group consists of XGBoost and the two foundation
models. TimesFM obtains the lowest point-wise MAE at H=24, while XGBoost
is best at H=96 and H=168. Chronos-Bolt is within approximately 0.3% of
the best MAE at all three horizons. DLinear, PatchTST, and iTransformer
improve substantially over SeasonalNaive, but they do not match the tree
model or foundation models under the current training budget. The
efficiency table shows the practical trade-off under single-window
latency measurement: XGBoost remains faster than the neural and TimesFM
models, Chronos-Bolt is competitive, and TimesFM is the slowest despite
strong accuracy.

On ETTh1/HUFL, Chronos-Bolt is best at H=24, PatchTST is the strongest
trained model at H=24, and DLinear is best at H=96 and H=168. TimesFM is
competitive but not the best on this benchmark. The results support the
importance of including both linear, transformer, and foundation
baselines: the best model class changes with dataset and horizon.

The statistical tests confirm that the top comparisons are not only
small numerical differences. On ECL, the H=96 and H=168 results favor
XGBoost over Chronos-Bolt, while H=24 is nearly tied between TimesFM and
XGBoost depending on whether one uses flattened point-wise MAE or
per-window error aggregation. On ETTh1/HUFL, the tests support
Chronos-Bolt at H=24 and DLinear at the two longer horizons. This
strengthens the central observation that no single family dominates
across datasets and horizons.

![MAE heatmap across executed datasets and
horizons.](../figures/fig3_heatmap.pdf){#fig:heatmap
width="\\linewidth"}

![Accuracy--latency view at
H=96.](../figures/fig4_pareto.pdf){#fig:pareto width="\\linewidth"}

![MAE by horizon for the executed model
set.](../figures/fig5_horizon.pdf){#fig:horizon width="\\linewidth"}

# Discussion

The results show that model choice is dataset-dependent. On aggregate
ECL load, engineered lag features with XGBoost remain an extremely
strong and inexpensive option. Foundation models are attractive when
avoiding task-specific training is valuable: Chronos-Bolt is close to
XGBoost on ECL and best at the shortest ETTh1/HUFL horizon. DLinear is
the strongest trained neural model in this benchmark and dominates the
longer ETTh1/HUFL horizons.

From a deployment perspective, the models occupy different operating
regimes. XGBoost is the most attractive choice when a conventional
supervised training pipeline is acceptable: it is accurate on ECL, has
low single-window latency, and does not require GPU inference.
Chronos-Bolt is useful when task-specific training is undesirable, for
example in cold-start industrial monitoring, rapid prototyping, or sites
where historical data cannot be pooled for model training. TimesFM shows
strong point accuracy at the shortest ECL horizon, but its latency is
substantially higher in this implementation. DLinear is a good neural
reference model because it is simple, compact, and strong on the longer
ETTh1/HUFL horizons.

The transformer results should be read carefully. PatchTST and
iTransformer are credible architectures, but this benchmark uses a
compact target-only configuration and a fixed training budget. Their
underperformance here does not imply that transformers are generally
unsuitable for load forecasting; rather, it shows that they do not
automatically beat simpler methods under constrained, reproducible
settings. A production study should include a broader hyperparameter
search, multivariate exogenous inputs, and possibly dataset-specific
tuning.

The main limitations are public-benchmark pretraining-contamination risk
for foundation models, limited hyperparameter search, target-only
modeling, and the absence of private industrial deployment data. A
rolling AutoARIMA baseline was attempted, but full per-origin refitting
was prohibitively expensive in this protocol; the partial run is
retained in the repository logs but excluded from the main comparison
tables. Chronos-Bolt also emits a warning that prediction lengths above
64 are outside its recommended range; therefore, H=96 and H=168 Chronos
results should be interpreted as practical stress tests rather than
optimal use of that model.

The foundation-model results should also not be overinterpreted as
guaranteed dataset-unseen zero-shot generalization. ECL and ETT are
public benchmarks and may overlap with pretraining or evaluation corpora
used during model development. The correct claim is therefore narrower:
Chronos-Bolt and TimesFM were run without task-specific training in this
repository. This remains practically relevant, but it is not the same as
evaluating on a private unseen industrial dataset.

# Conclusion

We built and executed a reproducible STLF benchmark pipeline on ECL and
ETTh1/HUFL. XGBoost is the most practical choice on ECL when training
data are available, TimesFM slightly leads H=24 point-wise MAE, and
Chronos-Bolt provides competitive zero-shot accuracy. On ETTh1/HUFL,
Chronos-Bolt wins H=24, PatchTST is the best trained model at H=24, and
DLinear wins H=96 and H=168. These findings support a pragmatic
recommendation: compare against strong simple baselines first, use
DLinear and PatchTST as neural references, and treat foundation models
as valuable cold-start tools rather than universally dominant
replacements for supervised forecasting.
