# Nested-OOF downstream-regret boundaries

## Постановка

Для outer-train построена nested 3-fold OOF-матрица из восьми band experts.
На каждом inner fold специалисты обучались только на complementary rows.
Для каждой cumulative boundary вычислялся regret перехода на неправильную
сторону:

```text
regret = max(0,
    abs(target - wrong_side_expert)
    - abs(target - true_band_expert)
)
```

Boundary classifier обучался с весом `w * scaled_regret_multiplier`.
Проверены strength 0.5/1.0 и смеси regret posterior с исходным posterior.

## Результаты

Лучший random-кандидат — 50% baseline + 50% regret posterior:

| Вариант | Random WMAE | Temporal WMAE |
|---|---:|---:|
| Baseline при той же routing policy | 59 924 | 58 145 |
| 50% regret blend | 59 844 | 58 143 |
| Изменение | **−80** | **−2** |

Более стабильная общая смесь 25% regret при рабочей routing policy
`T=0.5, soft, confidence gamma=1, strength=1` изменила ordinal-only delta:

- random: `−2 799 → −2 847`;
- temporal: `−2 843 → −2 846`.

После совместной minimax-оптимизации с tail/source лучший regret-вариант
уступил текущему production ensemble примерно 5 WMAE на худшем holdout.

## Диагностика

Положительный regret наблюдается у 93–99.7% строк в зависимости от boundary.
То есть неправильный специалист почти всегда хуже правильного, но сам regret
плохо разделяет сложных клиентов: он в основном повторно кодирует расстояние
до порога и масштаб target. Это объясняет минимальный дополнительный сигнал.

## Решение

Regret-sensitive posterior не добавляется в production. Текущий submission
`outputs/submission_ordinal_router.csv` оставлен без изменений.

