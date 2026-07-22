# Tail, distributional и local-neighbour эксперименты

## Протокол

- Random outer fold: 80% train → 20% validation.
- Temporal outer holdout: январь–май → июнь.
- Base-прогнозы validation никогда не являются in-sample.
- Коэффициенты финального blend выбраны на random outer fold и без
  переоптимизации применены к temporal holdout.
- Метрика и веса одинаковы во всех сравнениях.

## Tail classifier + specialists

| Порог | Weighted AUC | Soft-gate Δ | Oracle hard-gate Δ |
|---:|---:|---:|---:|
| 150k | 0,946 | −1 270 | −9 949 |
| 300k | 0,939 | −645 | −10 213 |
| 500k | 0,942 | −280 | −9 156 |

Oracle подтверждает нужный масштаб потенциального улучшения. Однако реальные
вероятности недостаточно точно отделяют false positives от настоящего tail.
Hierarchical gate 150k/300k/500k дал −1 435 на random fold и −495 на temporal
при фиксированных параметрах.

## CatBoost leaf-space neighbours

- Локальные KNN-estimators: WMAE 78–88k.
- Лучший blend на random fold: −122 при весе neighbours 0,05.
- На temporal holdout оптимальный вес neighbours: 0.

Гипотеза отвергнута: совпадение листьев не делает индивидуальные target
достаточно однородными.

## Distributional ensemble

- Oracle-выбор среди base/q10/q25/q50/q75/q90: около 26k WMAE.
- Реальный q75 tail gate: −992 random, но +183 temporal с фиксированными
  random-параметрами.
- Стабильный q50 blend: −439 random и −287 temporal.
- После добавления tail и LightGBM q50 получает оптимальный вес 0.

Условное распределение содержит большой oracle-потенциал, но доступные
признаки пока не позволяют выбрать правильный квантиль конкретного клиента.

## Source-specific experts

| Expert | Random Δ | Temporal Δ |
|---|---:|---:|
| `salary_6to12m_avg` present | −325 | −261 |
| `anchors >= 3` | −544 | −499 |
| sparse anchors | −153 | −95 |
| `first_salary_income` present | −439 | неприменим |

`first_salary_income` полностью отсутствует начиная с июня и во всём test,
поэтому соответствующий random-CV выигрыш не используется.

## LightGBM diversity

- CatBoost + LightGBM base blend: −321 random, −857 temporal.
- LightGBM tail gate: −884 random, −822 temporal.
- В общем blend отдельный LightGBM tail correction оказался избыточным, но
  base correction остался полезным.

## Выбранный ансамбль

```text
prediction = base
           + CatBoost hierarchical tail correction
           + 0.5 * (LightGBM prediction - base)
           + 0.5 * multi-anchor expert correction
```

| Split | Base | Advanced | Δ WMAE |
|---|---:|---:|---:|
| Random outer fold | 62 792 | 60 795 | **−1 998** |
| Temporal holdout | 60 874 | 59 509 | **−1 364** |

Это устойчивое улучшение, но оно пока существенно меньше целевых −10 000.
Основной оставшийся bottleneck — качество индивидуального tail/quantile gate,
а не качество самих специалистов.

## Артефакты

- `train_advanced_tail.py` — обучение всех выбранных компонентов на полном
  train.
- `outputs/submission_advanced_tail.csv` — новый кандидат-сабмит.
- `outputs/partial/submission_advanced_tail_components.csv` — component-level
  диагностика прогноза.
- `tail_mixture_experiment.py`, `leaf_neighbors_experiment.py`,
  `distributional_ensemble_experiment.py`, `source_experts_experiment.py`,
  `lightgbm_diversity_experiment.py` — воспроизводимые абляции.

Запуск финального кандидата:

```bash
python3 train_advanced_tail.py
```
