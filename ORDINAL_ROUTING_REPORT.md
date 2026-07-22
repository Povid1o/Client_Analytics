# Cumulative ordinal routing

## Постановка

Доход разбит на восемь упорядоченных диапазонов:

`20–50k`, `50–75k`, `75–100k`, `100–150k`, `150–250k`,
`250–400k`, `400–700k`, `700k+`.

Вместо одного независимого multiclass-решения обучаются семь бинарных
границ `P(target >= threshold)`. Вероятности приводятся к монотонному CDF и
преобразуются в распределение по диапазонам. Для каждого диапазона обучен
локальный LightGBM-регрессор на самом диапазоне и двух соседних. Итоговый
прогноз — confidence-gated мягкая смесь специалистов и базового CatBoost.

Все модели outer-validation обучаются только на complementary train rows.
Feature engineering не использует target. Параметры проверены на двух
разбиениях: stratified random outer fold и June temporal holdout.

## Результаты

| Вариант | Random WMAE | Delta | Time WMAE | Delta |
|---|---:|---:|---:|---:|
| Base CatBoost | 62 792 | — | 60 874 | — |
| Multiclass ordinal experts | 60 484 | −2 308 | 58 581 | −2 293 |
| Cumulative ordinal, best | 59 918 | −2 875 | 58 015 | −2 859 |
| Cumulative-only, fixed common config | 59 994 | −2 799 | 58 031 | −2 843 |
| Final minimax mix | 59 830 | −2 962 | 58 023 | −2 851 |

Final minimax mix использует коэффициенты:

```text
prediction = base
           + 0.75 * ordinal_correction
           + 0.25 * tail_correction
           + 0.25 * multi_anchor_source_correction
```

LightGBM base diversity не вошёл в финальную формулу: после ordinal routing
его дополнительный сигнал не был устойчивым на обоих разбиениях.

## Диагностика классификатора

На random outer fold:

| Router | Weighted exact accuracy | Within one band | Weighted class MAE |
|---|---:|---:|---:|
| Weighted multiclass | 59.45% | 82.83% | 0.800 |
| Cumulative ordinal | 60.33% | 84.01% | 0.752 |

На temporal holdout cumulative router показал 59.94% exact accuracy,
84.46% within-one и class MAE 0.741.

Если использовать истинный диапазон и только спроецировать base внутрь
него, потенциальное улучшение составляет 28.6–31.2k WMAE. Следовательно,
основное ограничение по-прежнему находится в распознавании диапазона, а не
в качестве специалистов.

## Воспроизведение

Эксперимент:

```bash
python3 ordinal_routing_experiment.py --cumulative --engineered
python3 ordinal_routing_experiment.py --cumulative --engineered \
  --base-predictions outputs/partial/base_time_predictions.csv \
  --output outputs/partial/ordinal_cumulative_time_results.csv
```

Full-train и submission:

```bash
python3 train_ordinal_router.py
```

Выходы:

- `outputs/submission_ordinal_router.csv`;
- `outputs/partial/submission_ordinal_router_components.csv`;
- `outputs/partial/ordinal_cumulative_results.csv`;
- `outputs/partial/ordinal_cumulative_time_results.csv`.

## Вывод

Гипотеза подтверждена: ordinal classification даёт стабильный независимый
сигнал и сильнее прежнего tail-routing. Однако дополнительный выигрыш
относительно текущей power-target базы составляет около 2.9k, а не 10k.
От исходного baseline 70 999 до random-fold оценки финального решения
снижение превышает 11k, но считать это ещё одним улучшением на 10k поверх
уже улучшенной модели нельзя.
