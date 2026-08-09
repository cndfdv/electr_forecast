# Текст доклада на 5 минут — RU и EN

Структура — 9 слайдов под шаблон ICIE (Oral presentation).
Темп: ~140 слов/мин ⇒ ~700 слов на язык.

Тайминг (всего ~5 мин):
- Слайд 1 (Cover) — 20 с
- Слайд 2 (Мотивация) — 45 с
- Слайд 3 (Цель и задачи) — 30 с
- Слайд 4 (Протокол и данные) — 45 с
- Слайд 5 (Результаты на ECL) — 60 с
- Слайд 6 (Результаты на ETTh1) — 50 с
- Слайд 7 (Абляция по бюджету обучения) — 45 с
- Слайд 8 (Точность и задержка) — 30 с
- Слайд 9 (Выводы) — 35 с

---

# RU — Русская версия (~5 минут)

**(Слайд 1 — Cover)**

Добрый день, уважаемые коллеги. Меня зовут Арсений Князев, Российский университет дружбы народов. Тема доклада — сравнительное исследование классических, машинно-обучаемых, глубоких и фундаментальных моделей для краткосрочного прогнозирования потребления электроэнергии. Соавтор работы — Никита Гречаников.

---

**(Слайд 2 — Мотивация)**

Актуальность работы определяется двумя обстоятельствами. С одной стороны, краткосрочный прогноз потребления электроэнергии — базовая задача планирования работы любой энергосистемы: от его точности напрямую зависят выбор состава генерирующего оборудования, расчёт резерва мощности и управление спросом. Ошибка прогноза на проценты выливается в реальные затраты на резервирование и в риски надёжности. С другой стороны, за последний год появилось новое поколение так называемых фундаментальных моделей для временных рядов — Chronos, TimesFM, Lag-Llama. Они предобучены на огромных корпусах разнородных рядов и обещают качественный прогноз без какого-либо дообучения на целевом ряде, что особенно привлекательно для отрасли — это потенциальная экономия на разработке и поддержке локальных моделей.

Однако в литературе почти нет прямых сравнений таких моделей со специально обученными нейросетевыми и классическими методами по единому протоколу без утечек данных. Без честного сравнения непонятно, можно ли заменить локально обученную модель на готовую фундаментальную. Поэтому возникает естественный вопрос: насколько прогноз без дообучения конкурентоспособен против локально обученной модели в задаче краткосрочного прогноза нагрузки?

---

**(Слайд 3 — Цель и задачи)**

Цель работы — количественно оценить, как фундаментальные модели в режиме без дообучения сравниваются со специально обученными нейросетевыми и классическими методами по единому хронологическому протоколу без утечек данных.

Задачи: построить единый конвейер обработки с контекстом 336 часов и горизонтами 24, 96 и 168 часов; сравнить восемь моделей из трёх семейств на двух наборах данных по пяти случайным инициализациям; провести абляционное исследование по бюджету обучения; и проанализировать компромисс между точностью прогноза и задержкой инференса.

---

**(Слайд 4 — Протокол и данные)**

Используем два набора данных. Первый — ECL, агрегированная нагрузка электросети, типичная задача для энергетика. Второй — ETTh1, сигнал датчика «high useful load» трансформаторной подстанции; этот ряд широко используется в литературе по моделированию временных рядов и по характеру ближе к данным, на которых обучались фундаментальные модели.

Разбиение строго хронологическое: 70 процентов — обучение, по 15 — валидация и тест. Стандартизация считается только по обучающей части, чтобы исключить утечку. Длина входного окна — 336 часов, одинаковая для всех нейронных и фундаментальных моделей. Горизонты — сутки, четверо суток и неделя. Пять случайных инициализаций на каждую конфигурацию.

---

**(Слайд 5 — Результаты на ECL)**

Главный результат на ECL. На горизонте 24 часа лучшая модель — iTransformer, средняя абсолютная ошибка 26 083. Далее DLinear — 27 038, PatchTST — 31 430 и XGBoost — 31 738. Классические методы: SARIMA — 263 914, сезонное наивное предсказание — 338 936. Фундаментальные модели TimesFM и Chronos-Bolt дают около 259 тысяч — рядом с SARIMA.

То есть обученные нейросети и фундаментальные модели без дообучения образуют два чётко разделённых кластера: разрыв почти на порядок. iTransformer остаётся лучшим на ECL на всех трёх горизонтах; на длинных DLinear — стабильно второй.

---

**(Слайд 6 — Результаты на ETTh1)**

Картина на ETTh1 принципиально другая. На горизонте 24 часа лучший результат показывает Chronos-Bolt — средняя абсолютная ошибка три ноль ноль. DLinear — 3.07, PatchTST — 3.10, TimesFM — 3.16. То есть фундаментальная модель не просто конкурентна, а лидер.

Это явное подтверждение: прогноз без дообучения работает там, где целевой ряд по характеру близок к данным предобучения, и проигрывает там, где он далёк. На горизонтах 96 и 168 часов на ETTh1 вперёд выходит DLinear, фундаментальные модели держатся в пределах нескольких процентов.

---

**(Слайд 7 — Абляция по бюджету обучения)**

Возможное возражение — что нейросети просто недообучены. Чтобы это закрыть, мы провели отдельное абляционное исследование на ECL при горизонте 24 часа. PatchTST и iTransformer обучались по 500, 1000, 2000 и 5000 шагов с ранней остановкой по валидации. PatchTST остаётся в диапазоне от 31.4 до 33.7 тысячи MAE, iTransformer — от 26.1 до 29.8 тысячи. Разброс между бюджетами — меньше четырёх процентов, ранняя остановка срабатывает уже до тысячи шагов. Средняя ошибка фундаментальных моделей на той же задаче остаётся около 260 тысяч независимо от бюджета обучения нейросетей.

Значит разрыв на ECL — не следствие недостаточного обучения, а собственное ограничение режима без дообучения при существенном сдвиге распределения.

---

**(Слайд 8 — Точность и задержка)**

Кратко о задержке инференса. Нейросети доминируют по точности, классические методы — по скорости; фундаментальные модели на ECL оказались в неудачной «середине»: они в 13.6 раза медленнее SARIMA и в 4.2 раза тяжелее DLinear по числу параметров, при этом сильно проигрывают по точности. На практике это диктует разумную схему многоуровневого развёртывания: SARIMA там, где критична задержка; нейросети — для основной нагрузки; фундаментальные модели — для холодного старта на новых точках.

---

**(Слайд 9 — Выводы)**

Подытожим. Во-первых, по единому протоколу без утечек данных обученные глубокие модели — iTransformer, DLinear, PatchTST — доминируют на ECL, а XGBoost остаётся сильнейшим методом без глубокого обучения. Во-вторых, фундаментальные модели без дообучения отстают на ECL примерно на порядок, но конкурентны и даже лучшие на ETTh1 при горизонте 24 часа. В-третьих, этот разрыв — не следствие недостаточного бюджета обучения, а собственное ограничение режима без дообучения. В-четвёртых, исходный код, конфигурации и все 144 строки результатов опубликованы для воспроизведения.

Практический вывод для отрасли: фундаментальные модели на сегодня нельзя брать как готовую замену для конкретной энергосистемы — нужна либо адаптация, либо честное сравнение с локально обученной моделью. Но как стартовая точка для редких или коротких рядов они уже работают.

Спасибо за внимание, готов ответить на вопросы.

---

## Запас на вопросы — короткие ответы

- **Почему iTransformer с одной серией?** Чтобы протокол был одинаковым для всех моделей: одна целевая серия, один тип входа.
- **Почему именно ECL и ETTh1?** ECL — классический энергетический бенчмарк, ETTh1 — стандарт в литературе по моделированию временных рядов; их разные распределения дают честный контраст.
- **Контаминация фундаментальных моделей?** Оба набора данных публичные, возможно входили в данные предобучения. Это указано в ограничениях работы — но даже при таком оптимистичном для них сценарии они проигрывают на ECL.
- **Почему DLinear так силён?** Линейные модели хорошо ловят сильную суточно-недельную сезонность, которая доминирует на ECL.
- **Оборудование?** Видеокарта RTX 4060 Ti, CUDA 13, PyTorch 2.11; полный прогон — за ночь.
- **Почему SARIMA с фиксированными параметрами, а не автоподбор?** AutoARIMA проводит поиск по сетке на каждом тестовом окне — на ECL это десятки часов на одну пару «набор–горизонт». Мы фиксировали (1,1,1)(1,1,1,24), один раз подобрали на обучении, затем применяли «вперёд» с уже подобранными параметрами.

---

# EN — English version (~5 minutes)

**(Slide 1 — Cover)**

Good afternoon. My name is Arsenii Kniazev, RUDN University. The title of our work is "A Comparative Study of Machine Learning, Deep Learning and Foundation Models for Short-Term Electricity Load Forecasting." Co-author — Nikita Grechanikov.

---

**(Slide 2 — Motivation)**

Short-term load forecasting underpins unit commitment, reserve sizing and demand-response — a core task in any power system. Over the last year a new generation of foundation models for time series has appeared — Chronos, TimesFM, Lag-Llama. They are pre-trained on huge corpora and promise high-quality zero-shot forecasts without any fine-tuning on the target series. Yet the literature offers very few like-for-like comparisons of foundation models against supervised DL and classical baselines under a single leakage-safe protocol. So the natural question is: how competitive is zero-shot against a locally trained model on STLF?

---

**(Slide 3 — Aim and Tasks)**

Our aim is to quantify how zero-shot foundation models compare against purpose-trained ML/DL and classical baselines under one chronological, leakage-safe protocol.

The tasks: build a unified pipeline with a 336-hour context and horizons 24, 96, 168; evaluate eight models from three families on two datasets across five seeds; run a training-budget ablation; and analyze the accuracy–latency trade-off.

---

**(Slide 4 — Protocol and Data)**

Two datasets. ECL — aggregated grid load, a typical industrial energy task. ETTh1, the HUFL column — also widely used in the time-series literature and, importantly, closer in distribution to the data the foundation models were pre-trained on.

The split is strictly chronological, 70 / 15 / 15. StandardScaler is fitted on train only — no leakage. The input window is 336 hours, identical across all neural and foundation models. Horizons are one day, four days, and one week. Five random seeds per configuration.

---

**(Slide 5 — ECL results)**

The headline result on ECL. At the 24-hour horizon the best model is iTransformer at MAE 26 083, followed by DLinear at 27 038, PatchTST at 31 430 and XGBoost at 31 738. SARIMA is at 263 914, SeasonalNaive at 338 936. Foundation models TimesFM and Chronos-Bolt are around 259 thousand, next to SARIMA. So supervised DL and zero-shot foundation fall into two clearly separated clusters, almost an order of magnitude apart. iTransformer stays best on ECL at every horizon; DLinear is the most stable runner-up at longer ones.

---

**(Slide 6 — ETTh1 flip)**

The picture flips on ETTh1. At the 24-hour horizon Chronos-Bolt delivers the best MAE — three point zero zero. DLinear 3.07, PatchTST 3.10, TimesFM 3.16. The foundation model is not just competitive — it leads. A clear illustration: zero-shot works where the target distribution is close to the pretraining mixture, and loses where it is far. At horizons 96 and 168 on ETTh1 DLinear takes the lead, with the foundation models within a few percent.

---

**(Slide 7 — Training-budget ablation)**

The natural counter-argument is that we simply under-trained the supervised models. To close that off we ran a separate ablation on ECL / horizon 24. We re-trained PatchTST and iTransformer at 500, 1000, 2000 and 5000 optimizer steps with validation-driven early stopping. PatchTST stayed in the 31.4 – 33.7 k MAE band, iTransformer in 26.1 – 29.8 k; between-budget spread below four percent, early stopping triggered before 1000 steps. Foundation-model MAE on the same task stayed near 260 k regardless of the supervised budget. So the gap on ECL is an intrinsic zero-shot limitation under domain shift, not an under-training artefact.

---

**(Slide 8 — Accuracy and latency)**

A short note on latency. Deep models dominate accuracy, classical baselines dominate latency, and the foundation models on ECL sit in an unfortunate middle ground — 13.6× slower than SARIMA and 4.2× heavier than DLinear in parameters, while losing on accuracy. Practically this suggests a tiered deployment: SARIMA where latency is critical; supervised DL for the main workload; foundation models for cold start.

---

**(Slide 9 — Conclusion)**

To conclude. First, under a single leakage-safe protocol, supervised deep models — iTransformer, DLinear, PatchTST — dominate ECL, and XGBoost remains the strongest non-deep baseline. Second, zero-shot foundation models lag by roughly an order of magnitude on ECL but are competitive — and best at the 24-hour horizon — on ETTh1. Third, this gap is not a training-budget artefact, it is an intrinsic zero-shot limitation. Fourth, all code, configurations and 144 result rows are publicly released for full reproduction.

The practical takeaway for industry: today's foundation models are not a drop-in for a specific power grid — they need either adaptation or a fair comparison against a locally trained model. But as a warm start for short or rare series, they already work.

Thank you for your attention. I am happy to take questions.

---

## Reserve answers for Q&A (EN)

- **Why iTransformer with n_series=1?** To keep the protocol identical across all models — one target series, one input layout.
- **Why ECL and ETTh1?** ECL is the canonical energy benchmark; ETTh1 is a standard in the TS-modelling literature. Their distinct distributions give a fair contrast.
- **Foundation-model contamination?** Both datasets are public and may overlap with pretraining. We list this in Limitations — but even in this optimistic-for-them setting they lose on ECL.
- **Why is DLinear so strong?** Linear models capture the strong daily and weekly seasonality that dominates ECL.
- **Hardware?** RTX 4060 Ti, CUDA 13, PyTorch 2.11; the full sweep ran overnight.
- **Why SARIMA with fixed parameters and not AutoARIMA?** AutoARIMA runs a grid search per test window (5000+ on ECL) — tens of hours per (dataset, horizon). We fix (1,1,1)(1,1,1,24), fit once on train and forward with the fitted parameters.
