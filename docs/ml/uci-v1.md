# Три временных окна UCI

Статус: Реализовано

Источник чисел: [машинный отчёт](../../data/reports/uci-v1.json).

Seed: 2026; бутстрэп: 10 повторов; полное время 152.637 с.
Выбор использует валидацию и критерии качества; тест оценивается после фиксации модели, без переизбрания.

| Сценарий / окно | Выбрано | RMSE вал. | RMSE тест | Устойчивость | Итог |
| --- | --- | ---: | ---: | ---: | --- |
| 2005-04-05 | Ridge² | 0.000117086 | 0.000116586 | 0.987873 | explanation_validated |
| 2005-06-07 | Ridge² | 0.000159936 | 0.000154904 | 0.988315 | explanation_validated |
| 2005-08-09 | Ridge² | 0.000164532 | 0.000158684 | 0.987085 | explanation_validated |

## Полное сравнение семейств

Ошибки и остатки в следующих таблицах относятся к валидации, не всему портфелю. Выбранный внутри семьи профиль и все попытки находятся в JSON.

### 2005-04-05

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 1843.44 | 585.853 | 58912 | — | — | rejected |
| Ridge | 1543.67 | 491.777 | 65357.5 | 0.940951 | 1 | rejected |
| GAM | 1549.33 | 473.685 | 4484.89 | 0.897856 | 1 | rejected |
| XGBoost | 484.296 | 22.5748 | -13669 | 0.899066 | 1 | rejected |
| EBM | 149.138 | 56.7781 | -24906.1 | 0.757063 | 0.8 | rejected |
| Ridge² | 0.000117086 | 5.90523e-05 | 0.010592 | 0.987873 | 1 | accepted |
| GAM с парами | 0.188349 | 0.00653713 | -12.2488 | 0.987179 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- GAM: Prediction error exceeds the configured normalized RMSE limit.
- XGBoost: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- EBM: Unexplained remainder exceeds the configured materiality limit.; Attribution is below the configured stability threshold.; Signs of material effects are not sufficiently consistent.

Полный портфель: Δ = 50759.1; сумма эффектов = 50759.1; остаток = 0.0544575.
Файлы: `data/experiments/uci-v1/2005-04-05/`.

### 2005-06-07

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 2172.05 | 720.743 | -438843 | — | — | rejected |
| Ridge | 1786.55 | 635.259 | -9817.91 | 0.954152 | 1 | rejected |
| GAM | 1812.2 | 639.845 | -20851.7 | 0.881969 | 1 | rejected |
| XGBoost | 466.521 | 23.575 | 41763.1 | — | — | rejected |
| EBM | 276.577 | 41.7245 | 13893.8 | 0.881575 | 0.966667 | accepted |
| Ridge² | 0.000159936 | 8.34552e-05 | -0.0553808 | 0.988315 | 1 | accepted |
| GAM с парами | 416.966 | 5.61171 | 31724.5 | 0.978514 | 1 | rejected |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.
- GAM: Prediction error exceeds the configured normalized RMSE limit.
- XGBoost: Prediction error exceeds the configured normalized RMSE limit.; Attribution failed: ValueError: Local attribution fails additivity; max error=0.1425018310546875; A complete bootstrap with enough repeats is required.
- GAM с парами: Prediction error exceeds the configured normalized RMSE limit.

Полный портфель: Δ = -1.91762e+06; сумма эффектов = -1.91762e+06; остаток = -0.272427.
Файлы: `data/experiments/uci-v1/2005-06-07/`.

### 2005-08-09

| Семейство | RMSE | MAE | Остаток | L1 | Знаки | Статус |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Среднее | 2347.35 | 877.983 | 46703.8 | — | — | rejected |
| Ridge | 1845.5 | 780.905 | 50465.2 | 0.941887 | 1 | rejected |
| GAM | 1894.45 | 736.2 | 57844.1 | 0.469055 | 0.8 | rejected |
| XGBoost | 136.887 | 21.4174 | 503.001 | — | — | incomplete |
| EBM | 338.405 | 92.5196 | 55822.8 | 0.907711 | 0.966667 | rejected |
| Ridge² | 0.000164532 | 9.12709e-05 | -0.0129676 | 0.987085 | 1 | accepted |
| GAM с парами | 1.87973 | 0.239771 | -1165.37 | 0.964512 | 1 | accepted |

Причины отказа или неполноты:

- Среднее: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; A complete bootstrap with enough repeats is required.
- Ridge: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.
- GAM: Prediction error exceeds the configured normalized RMSE limit.; Unexplained remainder exceeds the configured materiality limit.; Attribution is below the configured stability threshold.; Signs of material effects are not sufficiently consistent.
- XGBoost: A complete bootstrap with enough repeats is required.
- EBM: Unexplained remainder exceeds the configured materiality limit.

Полный портфель: Δ = 19309.2; сумма эффектов = 19309.3; остаток = -0.100064.
Файлы: `data/experiments/uci-v1/2005-08-09/`.

## Ограничения

Показатели устойчивости относятся к фиксированной валидации, квантили предварительные. Вклады зависят от модели и не доказывают причинность. Бюджеты семейств различаются; полное время измерено при одновременном выполнении двух серий.
UCI содержит реальные счета и сценарную формулу model_npv; это не банковский NPV или модель прогнозирования дефолта.

Определения и пороги — в [метриках](metrics.md), воспроизведение — в [инструкции](../development/README.md).
