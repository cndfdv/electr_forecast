# ТЗ для агента: Comparative Study of Deep Learning and Foundation Models for Short-Term Electricity Load Forecasting

## 0. Контекст, deadline и цель

**Критический pre-check до реализации:** сегодня 2026-05-04, конференция заявлена на май 2026. Перед запуском экспериментов обязательно проверить актуальный submission deadline, camera-ready deadline и статус приема работ на сайте конференции/воркшопа. Если submission deadline уже прошел, не начинать full pipeline без решения пользователя: либо искать другой venue, либо готовить статью как arXiv/preprint/backup submission.

**Финальный продукт:** научная статья 6 страниц в IEEE conference template (двухколоночный формат, английский) + полный воспроизводимый репозиторий.

**Конференция:** IEEE, воркшоп «Intelligent Methods and Digital Technologies of Industrial Transformation», май 2026, индексация Scopus.

**Кейс:** сравнить классические, ML, deep learning и time-series foundation models на задаче краткосрочного прогнозирования электрической нагрузки. Основной load forecasting датасет должен быть ECL. ETTh1 использовать только как secondary industrial benchmark: либо target = `HUFL` как load-related series, либо честно переформулировать вторую задачу как transformer oil temperature forecasting при target = `OT`.

**Среда:** Python 3.11, PyTorch 2.x, RTX 4060 Ti 16 GB, WSL2.

---

## 1. Заголовок и аннотация

**Основной title:**
> A Comparative Study of Deep Learning and Foundation Models for Short-Term Electricity Load Forecasting in Industrial Power Systems

**Если ETTh1 target = OT, использовать более честный title:**
> A Comparative Study of Deep Learning and Foundation Models for Short-Term Forecasting in Industrial Power Systems

**Abstract (<=220 слов), структура:**
1. Контекст индустрии: smart grid, predictive operations, short-term load forecasting.
2. Проблема: landscape методов фрагментирован; foundation models активно развиваются, но их практический trade-off против supervised методов в power-system forecasting не всегда ясен.
3. Что сделано: систематическое сравнение 8-10 методов на ECL как основном load dataset и ETTh1 как secondary industrial benchmark, горизонты 24/96/168 ч, несколько сидов для stochastic trained моделей.
4. Главный результат заполнять только после экспериментов: какой класс выигрывает на каком горизонте, цена по latency/training cost, что дают no-task-specific-training foundation models.
5. Practical contribution: рекомендации по выбору метода под бюджет, данные и горизонт.
6. Ключевые слова: short-term load forecasting, time series, deep learning, foundation models, Industry 4.0, smart grid.

---

## 2. Датасеты

### Основной — ECL (Electricity Consuming Load)
- Источник: `https://github.com/laiguokun/multivariate-time-series-data`
- 321 клиент, 26,304 hourly точки после ресемплинга.
- Primary target: агрегированная нагрузка, сумма по 321 клиенту.
- Optional robustness target: один представительный клиент, например `MT_320`, только если хватает времени.
- Split: 70/10/20 по времени, без перемешивания.
- Использовать как главный датасет для всех load forecasting выводов.

### Secondary — ETTh1 (Electricity Transformer Temperature)
- Источник: `https://github.com/zhouhaoyi/ETDataset`
- 17,420 hourly точек, 7 признаков: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`.
- Для load forecasting title: target = `HUFL` или другая явно load-related series, выбранная до экспериментов и зафиксированная в статье.
- Если target = `OT`, не называть это electricity load forecasting. Формулировать как transformer oil temperature forecasting / transformer condition forecasting.
- Стандартный split: 12 мес train / 4 мес val / 4 мес test.
- Использовать как secondary industrial benchmark, а не как главный load dataset.

**Препроцессинг:**
- Перед обучением проверить пропуски, дубликаты timestamp, регулярность hourly grid и нули в target.
- Imputation не делать автоматически. Если пропуски есть, зафиксировать стратегию явно: linear interpolation для коротких gaps или исключение проблемных фрагментов.
- `StandardScaler` fit только на train при финальной оценке. Для финального retrain на train+val разрешен отдельный scaler fit на train+val, но это должно быть явно отражено в протоколе.
- Скользящее окно: базово input length `L=336` часов, output `H in {24, 96, 168}`. Это нужно, чтобы XGBoost мог честно использовать `lag-168`.
- Для foundation models дополнительно разрешен контекст до их native/recommended context length, но основной fair-comparison режим фиксирует общий `L=336`. Если используется разный context length, это выносится в ablation/limitations.

---

## 3. Модели

Основной набор — 8 моделей. Расширенный набор — до 10, если остается время.

| # | Модель | Класс | Библиотека | Тип | Примечание |
|---|---|---|---|---|---|
| 1 | SeasonalNaive | baseline | `statsforecast` | deterministic | обязательный sanity baseline, season length 24/168 |
| 2 | AutoARIMA | статистический | `statsforecast` | trained per-series | univariate target-only |
| 3 | XGBoost | ML + lag features | `xgboost` | trained | target lags + calendar features |
| 4 | DLinear | linear baseline | `neuralforecast` | trained | обязательный |
| 5 | PatchTST | transformer | `neuralforecast` | trained | обязательный |
| 6 | iTransformer | transformer | `neuralforecast` | trained | обязательный |
| 7 | Chronos-Bolt-Small | foundation model | `chronos-forecasting` / AutoGluon | zero-shot inference | no task-specific training |
| 8 | TimesFM | foundation model | `timesfm` | zero-shot inference | проверить актуальную доступную версию |
| 9 | LSTM | RNN | `neuralforecast` | trained | optional |
| 10 | TimesNet или Prophet | DL/decomposition | `neuralforecast` / `prophet` | trained | optional, только если хватает времени |

**Если время поджимает:** оставить 8 обязательных моделей. Prophet и TimesNet не обязательны.

**Важно про fair comparison:**
- AutoARIMA, Prophet и SeasonalNaive являются univariate target-only моделями.
- XGBoost использует target lags и календарные признаки.
- NeuralForecast модели могут использовать multivariate/exogenous features только если это одинаково и явно описано. Если часть моделей exogenous не поддерживает, основной эксперимент лучше сделать target-only, а multivariate режим вынести отдельно.
- Foundation models не fine-tune. В тексте писать “zero-shot / no task-specific training”, но не утверждать, что модель точно не видела ECL/ETT в pretraining.

**Feature engineering для XGBoost:**
- Lag-фичи: `lag-1`, `lag-24`, `lag-168`.
- Календарные: hour-of-day, day-of-week, month как sin/cos или one-hot.
- Скользящие средние: windows 24 и 168.

**Hyperparameters:**
- Нейронные: компактные конфигурации `neuralforecast` + короткий validation-only sweep по `learning_rate in {1e-3, 5e-4}` и `early_stop_patience_steps=10`.
- XGBoost: фиксированный direct multi-output baseline; отдельный validation sweep является future/extension item, без использования test.
- AutoARIMA: начать с ограничений `max_p<=3`, `max_q<=3`. Для ECL aggregate проверить `seasonal=True, m=24`; если слишком медленно, оставить как простой baseline и явно написать ограничение.
- TimesFM: перед запуском проверить актуальную версию `timesfm` и доступные checkpoint names. Если TimesFM 2.5 недоступен в установленном пакете, fallback: latest stable TimesFM или исключение модели с записью причины.

---

## 4. Протокол эксперимента

Основной протокол:

```
for dataset in {ECL, ETTh1_secondary}:
    for horizon H in {24, 96, 168}:
        split data by time into train/val/test
        fit preprocessing on train only
        fit fixed training configuration on train/val protocol
        for model M:
            for seed in model_seeds(M):
                train M on train with selected hyperparameters
                predict test windows
                save metrics, timing, memory
        aggregate mean +- std where seeds are meaningful
```

Seed protocol:
- Stochastic trained models: seeds `{42, 43, 44, 45, 46}` on ECL if time allows; minimum 3 seeds `{42, 43, 44}`.
- ETTh1 secondary: minimum 3 seeds for stochastic trained models.
- Deterministic models: SeasonalNaive, AutoARIMA, Prophet and zero-shot foundation models run once; report std as NA, not fake zero unless deterministic repeat is explicitly verified.

Foundation models:
- Chronos and TimesFM: zero-shot inference only, no fine-tuning.
- Limitations must state that public datasets such as ECL/ETT may have appeared in pretraining corpora, so zero-shot does not imply guaranteed dataset-unseen evaluation.
- If model uses probabilistic samples, fix sampling settings and seed; report point forecast as median/mean consistently.

Final training policy:
- Conservative version: train on train only, tune on val, evaluate on test. This is simplest and avoids ambiguity.
- Optional version: after choosing hyperparameters, retrain on train+val and evaluate on test. If used, scaler and preprocessing must be fit on train+val for that final run and clearly documented.

**Важно:**
- Не использовать test для hyperparameter tuning.
- Не делать rolling-origin tuning на test.
- Для latency считать warm-up отдельно и усреднять минимум по 100 forward passes where feasible.
- Для GPU memory использовать `torch.cuda.reset_peak_memory_stats()` и `torch.cuda.max_memory_allocated()`.

---

## 5. Метрики и статистическая проверка

**Основные метрики точности:**
- MAE
- RMSE
- sMAPE (%)
- wMAPE (%) для load dataset

**Дополнительно:**
- MSE можно сохранять в CSV, но не выносить в основные таблицы вместе с RMSE, чтобы не дублировать информацию.
- MAPE использовать только если в target нет нулей и near-zero значений; иначе не использовать.

**Эффективность:**
- Training time, seconds per model x horizon.
- Inference time, ms per forecast window, average after warm-up.
- Number of trainable parameters.
- Peak GPU memory, MB.

**Статистическая значимость:**
- Для top-2/top-3 моделей по MAE на каждом горизонте добавить paired Diebold-Mariano test или paired t-test по test-window errors.
- В статье не делать сильных claims о превосходстве без статистической проверки или явной оговорки.

Все сохранить в `results/main_results.csv` со схемой:

```csv
model,dataset,target,horizon,seed,MAE,RMSE,sMAPE,wMAPE,MSE,train_time_s,inference_ms,n_params,peak_gpu_mb
```

Статистические тесты сохранить в `results/stat_tests.csv`:

```csv
dataset,target,horizon,metric,model_a,model_b,test_name,statistic,p_value,significant
```

---

## 6. Структура репозитория

Создать воспроизводимый репозиторий:

```text
.
├── README.md
├── TODO.md
├── environment.yml
├── requirements.txt
├── configs/
│   ├── ecl.yaml
│   ├── etth1.yaml
│   └── models.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── download.py
│   │   ├── preprocess.py
│   │   └── windows.py
│   ├── models/
│   │   ├── baselines.py
│   │   ├── xgboost_model.py
│   │   ├── neuralforecast_models.py
│   │   └── neuralforecast_models.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── timing.py
│   │   └── statistical_tests.py
│   ├── experiments/
│   │   ├── run_experiment.py
│   │   └── run_all.py
│   └── figures/
│       └── make_figures.py
├── results/
│   ├── main_results.csv
│   └── stat_tests.csv
├── figures/
└── paper/
    ├── main.tex
    ├── references.bib
    └── ieeeconf.bst / IEEEtran.bst
```

Среда:
- Создать `environment.yml` для mamba.
- Зафиксировать версии ключевых библиотек.
- Если `timesfm`/`chronos` конфликтуют с основным окружением, разрешено отдельное окружение `environment_foundation.yml`, но результаты должны сохраняться в той же CSV-схеме.

Код:
- Комментарии и docstrings на английском.
- CLI должен позволять запускать один датасет/модель/горизонт отдельно.
- Все random seeds фиксировать централизованно.

---

## 7. Структура статьи IEEE

### I. Introduction
- Параграф 1: важность STLF в smart grid и Industry 4.0, кейсы диспетчеризации, балансирования и predictive operations.
- Параграф 2: эволюция методов от ARIMA и gradient boosting к DLinear, transformers и time-series foundation models.
- Параграф 3: gap — практический trade-off между supervised моделями и foundation models для industrial/power-system forecasting остается недостаточно ясным, особенно по accuracy vs computational cost.
- Параграф 4: contributions:
  1. Воспроизводимое сравнение 8-10 методов через несколько архитектурных семейств на ECL и ETTh1.
  2. Практическая оценка no-task-specific-training foundation models против trained baselines.
  3. Анализ accuracy vs training/inference cost и Pareto front.
  4. Practical guidelines для выбора модели под бюджет, доступность данных и горизонт.
- Последний параграф: roadmap по секциям.

### II. Related Work
Подсекции:
- A. Classical methods: ARIMA, exponential smoothing.
- B. Machine learning approaches: gradient boosting, lag/calendar features.
- C. Deep learning era: LSTM, TCN, Informer, Autoformer, FEDformer.
- D. Linear baselines: DLinear, NLinear, RLinear.
- E. Patch-based and inverted transformers: PatchTST, iTransformer.
- F. Time-series foundation models: TimesFM, Chronos, Moirai, Lag-Llama, MOMENT.
- G. Data contamination and benchmark reuse in foundation-model evaluation.

### III. Methodology
- A. Problem formulation: given context window `x_{t-L+1:t}`, predict `y_{t+1:t+H}`.
- B. Dataset targets: ECL aggregate load as primary target; ETTh1 target policy explicitly stated.
- C. Compared methods: concise paragraph per model family.
- D. Fairness boundaries: univariate vs exogenous-capable models, context length, no task-specific training for foundation models.
- E. Methodology pipeline figure.

### IV. Experimental Setup
- A. Datasets: ECL and ETTh1, sizes, targets, splits, missing-value checks.
- B. Evaluation protocol: input/output lengths, horizons, seeds, deterministic-model handling.
- C. Metrics and statistical tests.
- D. Implementation: Python 3.11, PyTorch 2.x, NeuralForecast, StatsForecast, XGBoost, RTX 4060 Ti 16 GB, exact library versions.

### V. Results
- A. Main accuracy results on ECL — Table I.
- B. Secondary ETTh1 results — Table II.
- C. Per-horizon analysis — Figure 5.
- D. Heatmap of MAE/sMAPE — Figure 3.
- E. Computational efficiency — Table III.
- F. Foundation models: what they provide without task-specific training, with contamination caveat.
- G. Pareto front — Figure 4.
- H. Sample forecasts — Figure 2.
- I. Statistical significance summary.

### VI. Discussion
- When simple baselines/DLinear are sufficient.
- When PatchTST/iTransformer justify extra training cost.
- When foundation models are attractive: cold-start, no training pipeline, limited engineering time.
- Practical recommendations: decision table.
- Limitations:
  - possible pretraining contamination for public datasets ECL/ETT;
  - only public benchmarks, not private industrial deployment data;
  - limited hyperparameter search;
  - limited exogenous-variable setting;
  - no uncertainty quantification unless Chronos samples are analyzed separately.

### VII. Conclusion and Future Work
- Summary: 8-10 methods, 2 datasets, key measured trade-offs.
- Future: private multi-site load data, online learning, probabilistic forecasting, fine-tuning/adaptation of foundation models.

### References (~25-35 ссылок)
Минимально обязательный bibtex: только реальные записи с DOI/arXiv ID. Если DOI не найден, использовать arXiv/official URL и не выдумывать.

- Box & Jenkins, Time Series Analysis.
- Hochreiter & Schmidhuber 1997, LSTM.
- Vaswani et al. 2017, Attention.
- Zhou et al. 2021, Informer / ETT dataset.
- Wu et al. 2021, Autoformer.
- Zhou et al. 2022, FEDformer.
- Zeng et al. 2023, DLinear.
- Nie et al. 2023, PatchTST.
- Liu et al. 2024, iTransformer.
- Wu et al. 2023, TimesNet.
- Das et al., TimesFM.
- Ansari et al., Chronos.
- Goswami et al., MOMENT.
- Woo et al., Moirai.
- Taylor & Letham 2018, Prophet.
- Chen & Guestrin 2016, XGBoost.
- TimesFM paper/work on benchmark contamination or evaluation caveats, if applicable.
- 5-7 industrial STLF papers from Google Scholar / IEEE Xplore by query `short-term load forecasting smart grid`.

---

## 8. Фигуры

1. **fig1_methodology.pdf** — pipeline: data -> preprocessing -> model families -> metrics/statistical tests.
2. **fig2_forecasts.pdf** — line plot на 7 днях test, 4-5 лучших моделей + ground truth.
3. **fig3_heatmap.pdf** — heatmap MAE/sMAPE: rows = models, columns = horizons x datasets.
4. **fig4_pareto.pdf** — scatter: x = log(inference_ms), y = MAE на H=96, Pareto front line.
5. **fig5_horizon.pdf** — grouped bar chart: horizon vs MAE, top-6 models only.

Все figures: PDF vector, minimum 9 pt font, без лишних рамок, ColorBrewer/viridis-compatible palette, readable in IEEE two-column layout.

---

## 9. Что агент НЕ должен делать

- НЕ начинать full pipeline до проверки conference deadline.
- НЕ выдумывать численные результаты. Если что-то не запустилось, оставить TODO/NA и записать причину.
- НЕ использовать test для hyperparameter tuning.
- НЕ допускать data leakage: scaler/preprocessing fit только на разрешенной training portion.
- НЕ выдавать ETTh1 `OT` за electricity load.
- НЕ утверждать, что zero-shot foundation models точно не видели public datasets в pretraining.
- НЕ забывать seed-усреднение для stochastic trained models.
- НЕ ставить fake citations. Все ссылки должны быть реальными с DOI/arXiv/official URL.
- НЕ делать claims типа “outperforms all baselines” без результатов и statistical test.
- НЕ использовать MAPE при нулях или near-zero target.

---

## 10. Известные риски и mitigation

| Риск | Mitigation |
|---|---|
| Submission deadline уже прошел | Сразу остановить full implementation и предложить другой venue/preprint strategy |
| ETTh1 `OT` не является load | Использовать ECL как main dataset; ETTh1 target=`HUFL` или сменить framing на temperature forecasting |
| Public-data contamination у foundation models | Явно указать limitation; писать no-task-specific-training вместо guaranteed unseen zero-shot |
| `timesfm` версия/чекпоинт недоступны | Проверить package/checkpoint до экспериментов; fallback на latest stable TimesFM или исключение с объяснением |
| `chronos-forecasting` или `timesfm` конфликтуют с окружением | Разнести foundation inference в отдельное mamba env |
| AutoARIMA виснет | Ограничить search space; для ECL проверить seasonal=True, m=24; если медленно, оставить bounded baseline |
| NeuralForecast/pandas конфликты | Зафиксировать версии в `environment.yml` и `requirements.txt` |
| Foundation models не помещаются в 16 GB | batch_size=1, fp16/bfloat16, CPU offload если поддерживается |
| ECL single-seed вывод шаткий | Минимум 3 seeds для stochastic trained models |
| LaTeX-сборка не идет | Использовать Overleaf fallback, но сохранить исходники в `paper/` |

---

## 11. Чеклист перед отправкой статьи

- [ ] Submission deadline/camera-ready deadline проверены на официальном сайте.
- [ ] Статья ровно около 6 страниц в IEEE conference template или в лимите конкретного call for papers.
- [ ] Название соответствует target: ECL load или ETTh1 temperature/load-related target.
- [ ] Все таблицы заполнены реальными числами, std/NA указаны корректно.
- [ ] Для top comparisons добавлены statistical tests.
- [ ] Все 5 фигур в PDF, читаются в двухколоночном IEEE layout.
- [ ] Все цитаты в bibtex, DOI/arXiv/URL корректны.
- [ ] Abstract в пределах 220 слов.
- [ ] Keywords: 5-7 штук.
- [ ] Author info, affiliation, email.
- [ ] Acknowledgments, если применимо.
- [ ] Ссылка на репозиторий с кодом и данными в footnote/conclusion.
- [ ] Spellcheck английского.
- [ ] PDF проходит IEEE PDF eXpress, если требуется.

---

## 12. Что нужно от пользователя ДО запуска агента

1. Подтвердить, что deadline еще актуален, или дать новый venue.
2. Подтвердить title policy: ECL load forecasting как main или broader industrial forecasting.
3. Указать author info: имя, аффилиация, email.
4. Подготовить IEEE conference template файлы (`IEEEtran.cls`, `IEEEtran.bst`) или разрешить агенту скачать официальные шаблоны.
5. Подтвердить, что комментарии/docstrings в коде пишутся на английском.

---

## 13. Финальная инструкция агенту

> Ты — senior ML engineer. Задача: реализовать это ТЗ end-to-end в одном воспроизводимом репозитории. Сначала проверь deadline и package availability. Затем работай итеративно: data validation + metrics, baselines, trained models, foundation inference, statistical tests, figures, paper. Не выдумывай цифры и citations. Не используй test для tuning. Не выдавай ETTh1 OT за load. Если что-то не запускается — фиксируй причину, упрощай протокол и продолжай после явной записи limitation. Финальный deliverable: рабочий repo + `paper/main.pdf` в IEEE template с реальными результатами.
