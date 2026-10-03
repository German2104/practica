# Подтверждающая синтетическая серия

Статус: Реализовано

Источник чисел: [машинный отчёт](../../data/reports/confirmation-v2.json).

Seed: 2026; бутстрэп: 10 повторов; полное время 2928.72 с.
Выбор использует валидацию и критерии качества; тест оценивается после фиксации модели, без переизбрания.

| Сценарий / окно | Выбрано | RMSE вал. | RMSE тест | Устойчивость | Итог |
| --- | --- | ---: | ---: | ---: | --- |
| Кредитный | GAM с парами | 5.53264 | 5.20698 | 0.957254 | explanation_validated |
| Линейный | Ridge | 3.14114e-05 | 3.04232e-05 | 1 | explanation_validated |
| Пороговый | GAM с парами | 2.17963 | 2.08967 | 0.988935 | explanation_validated |
| Насыщение | GAM | 0.00551867 | 0.00566234 | 0.999981 | explanation_validated |
| U-образный | Ridge² | 3.23282e-05 | 3.1443e-05 | 1 | explanation_validated |
| Взаимодействие | Ridge² | 1.74905 | 1.85333 | 0.99738 | explanation_validated |
| Смешанный | EBM | 2.685 | 3.02402 | 0.971247 | explanation_validated |
| Смена режима | GAM с парами | 59.3198 | 59.4753 | 0.988298 | attribution_checked |
| Корреляция | GAM | 2.19178 | 2.27029 | 0.976327 | explanation_validated |
| Нет сигнала | EBM | 0 | 0 | — | attribution_checked |

## Полное сравнение семейств

Ошибки и остатки в следующих таблицах относятся к валидации, не всему портфелю. Выбранный внутри семьи профиль и все попытки находятся в JSON.

### Кредитный

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 73.6626 | 61.6267 | -15970.6 | — | — | rejected |
| Ridge | 28.3491 | 18.2755 | -4375.72 | 0.992208 | 1 | rejected |
| GAM | 22.625 | 15.3496 | -3156.19 | 0.957402 | 1 | rejected |
| XGBoost | 10.2241 | 5.54373 | -3330.35 | 0.975871 | 0.98 | rejected |
| EBM | 8.61839 | 4.65668 | -2243.34 | 0.930078 | 0.833333 | rejected |
| Ridge² | 13.7711 | 7.86621 | -2927.57 | 0.992466 | 1 | rejected |
| GAM с парами | 5.53264 | 3.02958 | -926.028 | 0.957254 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- GAM: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- XGBoost: Unexplained remainder exceeds the configured materiality limit.
- EBM: Signs of material effects are not sufficiently consistent.
- Ridge²: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.

Полный портфель: Δ = -79983.2; сумма эффектов = -75986.7; остаток = -3996.53.
Файлы: `data/experiments/confirmation-v2/credit/`.

### Линейный

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 36.3916 | 30.3913 | -5228.71 | — | — | rejected |
| Ridge | 3.14114e-05 | 2.67599e-05 | -0.00405817 | 1 | 1 | accepted |
| GAM | 0.00250629 | 0.00108912 | 0.0178293 | 0.999995 | 1 | accepted |
| XGBoost | 2.62871 | 1.75311 | -534.388 | 0.982901 | 1 | accepted |
| EBM | 1.79868 | 0.811635 | -320.556 | 0.994423 | 1 | accepted |
| Ridge² | 3.17356e-05 | 2.69785e-05 | -0.00428873 | 1 | 1 | accepted |
| GAM с парами | 2.37867 | 0.595192 | 551.834 | 0.997887 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.

Полный портфель: Δ = -26237.7; сумма эффектов = -26237.7; остаток = -0.00953097.
Файлы: `data/experiments/confirmation-v2/linear/`.

### Пороговый

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 40.4783 | 32.7646 | -9845.31 | — | — | rejected |
| Ridge | 11.9367 | 8.84278 | -1837.17 | 0.992835 | 1 | rejected |
| GAM | 2.31128 | 1.14719 | -554.23 | 0.996021 | 1 | accepted |
| XGBoost | 4.79736 | 2.92539 | -1754.71 | 0.982254 | 1 | rejected |
| EBM | 3.32139 | 1.5291 | -1049.09 | 0.994542 | 1 | accepted |
| Ridge² | 4.03399 | 3.4874 | -1036.41 | 0.996401 | 1 | accepted |
| GAM с парами | 2.17963 | 0.976617 | 67.6457 | 0.988935 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- XGBoost: Unexplained remainder exceeds the configured materiality limit.

Полный портфель: Δ = -49460.1; сумма эффектов = -49348.5; остаток = -111.599.
Файлы: `data/experiments/confirmation-v2/threshold/`.

### Насыщение

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 48.0026 | 40.0651 | -4807.36 | — | — | rejected |
| Ridge | 5.63632 | 4.70097 | -352.409 | 0.996596 | 1 | accepted |
| GAM | 0.00551867 | 0.00313685 | -0.510196 | 0.999981 | 1 | accepted |
| XGBoost | 2.74215 | 1.87307 | -569.734 | 0.981768 | 1 | accepted |
| EBM | 1.8209 | 0.84639 | -304.329 | 0.99343 | 1 | accepted |
| Ridge² | 1.7899 | 1.43135 | 154.088 | 0.997749 | 1 | accepted |
| GAM с парами | 2.36535 | 0.593234 | 548.42 | 0.997735 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.

Полный портфель: Δ = -24086.9; сумма эффектов = -24084.9; остаток = -2.00687.
Файлы: `data/experiments/confirmation-v2/saturation/`.

### U-образный

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 37.1513 | 30.9563 | -4761.88 | — | — | rejected |
| Ridge | 6.00713 | 4.72024 | 514.748 | 0.995172 | 1 | rejected |
| GAM | 0.12246 | 0.0237424 | 22.7231 | 0.999989 | 1 | accepted |
| XGBoost | 3.20102 | 2.16898 | -157.305 | 0.979973 | 1 | accepted |
| EBM | 2.2373 | 1.11761 | -15.2052 | 0.993584 | 1 | accepted |
| Ridge² | 3.23282e-05 | 2.73783e-05 | -0.000848457 | 1 | 1 | accepted |
| GAM с парами | 2.4568 | 0.662034 | 620.925 | 0.997681 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.

Полный портфель: Δ = -23485.5; сумма эффектов = -23485.5; остаток = -0.00371823.
Файлы: `data/experiments/confirmation-v2/u_shaped/`.

### Взаимодействие

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 38.165 | 31.531 | -6845.73 | — | — | rejected |
| Ridge | 4.391 | 3.04821 | -255.187 | 0.997259 | 1 | accepted |
| GAM | 3.84372 | 2.70905 | -113.257 | 0.984037 | 1 | accepted |
| XGBoost | 3.11029 | 2.10328 | -875.168 | 0.979029 | 1 | accepted |
| EBM | 2.08818 | 1.1185 | -356.62 | 0.957442 | 0.9 | accepted |
| Ridge² | 1.74905 | 1.26809 | 77.6651 | 0.99738 | 1 | accepted |
| GAM с парами | 3.66576 | 1.17115 | 750.027 | 0.989085 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.

Полный портфель: Δ = -34640.6; сумма эффектов = -35210.3; остаток = 569.763.
Файлы: `data/experiments/confirmation-v2/interaction/`.

### Смешанный

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 52.2644 | 42.3247 | -10574.2 | — | — | rejected |
| Ridge | 10.9524 | 8.13601 | -1928.44 | 0.994761 | 1 | rejected |
| GAM | 4.58533 | 2.95065 | -645.325 | 0.989399 | 1 | accepted |
| XGBoost | 4.2307 | 2.81657 | -1580.33 | 0.977987 | 0.88 | rejected |
| EBM | 2.685 | 1.46254 | -685.55 | 0.971247 | 0.92 | accepted |
| Ridge² | 4.59428 | 3.75777 | -804.652 | 0.995546 | 0.98 | accepted |
| GAM с парами | 3.10111 | 1.23834 | 389.909 | 0.988633 | 0.98 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- XGBoost: Signs of material effects are not sufficiently consistent.

Полный портфель: Δ = -52959.8; сумма эффектов = -49606; остаток = -3353.79.
Файлы: `data/experiments/confirmation-v2/mixed/`.

### Смена режима

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 87.0315 | 71.9916 | -70574.2 | — | — | rejected |
| Ridge | 63.3038 | 62.3952 | -61928.4 | 0.994761 | 1 | rejected |
| GAM | 60.8181 | 60.6484 | -60645.3 | 0.989399 | 1 | rejected |
| XGBoost | 61.628 | 61.5009 | -61580.3 | 0.977987 | 0.88 | rejected |
| EBM | 60.7438 | 60.6884 | -60685.5 | 0.971247 | 0.92 | rejected |
| Ridge² | 60.9769 | 60.8089 | -60804.7 | 0.995546 | 0.98 | rejected |
| GAM с парами | 59.3198 | 59.1705 | -59166.1 | 0.988298 | 0.98 | rejected |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- GAM: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- XGBoost: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; Signs of material effects are not sufficiently consistent.
- EBM: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- Ridge²: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- GAM с парами: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.

Полный портфель: Δ = -352960; сумма эффектов = -56072.3; остаток = -296888.
Файлы: `data/experiments/confirmation-v2/regime_shift/`.

### Корреляция

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 57.5507 | 46.6704 | -11719.8 | — | — | rejected |
| Ridge | 13.3585 | 10.0479 | -2445.44 | 0.934158 | 1 | rejected |
| GAM | 2.19178 | 1.06066 | -551.466 | 0.976327 | 1 | accepted |
| XGBoost | 5.4747 | 3.00433 | -1768.88 | 0.965039 | 0.775 | rejected |
| EBM | 4.29449 | 1.70205 | -1096.03 | 0.993055 | 1 | accepted |
| Ridge² | 5.40809 | 4.59919 | -1288.38 | 0.939014 | 0.857143 | rejected |
| GAM с парами | 2.21123 | 0.893471 | 297.343 | 0.993199 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- XGBoost: Unexplained remainder exceeds the configured materiality limit.; Signs of material effects are not sufficiently consistent.
- Ridge²: Signs of material effects are not sufficiently consistent.

Полный портфель: Δ = -58534.5; сумма эффектов = -55896.1; остаток = -2638.48.
Файлы: `data/experiments/confirmation-v2/correlated/`.

### Нет сигнала

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 0 | 0 | 0 | — | — | incomplete |
| Ridge | 0 | 0 | 0 | — | — | incomplete |
| GAM | 0 | 0 | 0 | — | — | incomplete |
| XGBoost | 0 | 0 | 0 | — | — | incomplete |
| EBM | 0 | 0 | 0 | — | — | incomplete |
| Ridge² | 0 | 0 | 0 | — | — | incomplete |
| GAM с парами | 0 | 0 | 0 | — | — | incomplete |

Причины отказа или неполноты:

- Среднее: A complete bootstrap with enough repeats is required.
- Ridge: A complete bootstrap with enough repeats is required.
- GAM: A complete bootstrap with enough repeats is required.
- XGBoost: A complete bootstrap with enough repeats is required.
- EBM: A complete bootstrap with enough repeats is required.
- Ridge²: A complete bootstrap with enough repeats is required.
- GAM с парами: A complete bootstrap with enough repeats is required.

Полный портфель: Δ = 0; сумма эффектов = 0; остаток = 0.
Файлы: `data/experiments/confirmation-v2/no_signal/`.

## Ограничения

Показатели устойчивости относятся к фиксированной валидации, квантили предварительные. Вклады зависят от модели и не доказывают причинность. Бюджеты семейств различаются; полное время измерено при одновременном выполнении двух серий.
Новые seed и объём отличаются от исторического эксперимента. Улучшение относительно Ridge корректно сравнивать внутри текущего общего разбиения; прямое сравнение со старым тестом не является парным.

Определения и пороги — в [метриках](metrics.md), воспроизведение — в [инструкции](../development/README.md).
