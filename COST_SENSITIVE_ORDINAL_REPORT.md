# Cost-sensitive cumulative boundaries

## Гипотеза

Семь cumulative LightGBM classifiers переобучены с дополнительной стоимостью
ошибки относительно каждого income threshold. Множитель веса симметричен на
логарифмической шкале и растёт с расстоянием target от границы:

```text
boundary_weight = w * (1 + strength * clipped_log_distance)
```

Проверены `strength=0.5` и `strength=1.5`, а также смеси нового posterior с
исходным. Базовый CatBoost и восемь band experts не переобучались, поэтому
эксперимент изолирует влияние boundary loss.

## Результаты

При одинаковом routing (`T=0.5`, soft, confidence gamma=1, strength=1):

| Posterior | Random WMAE | Temporal WMAE |
|---|---:|---:|
| Baseline cumulative | 59 994 | 58 031 |
| Mild cost-sensitive | 59 825 | 58 084 |
| Изменение | **−169** | **+53** |

Сильная схема `strength=1.5` ухудшила random outer-fold и была отвергнута до
temporal запуска.

Лучшая minimax-смесь (`25% mild cost-sensitive posterior`) в полном
tail/source ансамбле изменила результат примерно на `−41 WMAE` random и
`+10 WMAE` temporal. Интерполяция от 0% до 100% cost-sensitive сигнала
монотонно ухудшает temporal holdout, поэтому безопасного положительного
коэффициента нет.

## Вывод

Distance-based cost-sensitive weighting не переносится между random и
temporal валидациями и не добавляется в production. Рабочий submission
`outputs/submission_ordinal_router.csv` оставлен без изменений.

Следующий вариант cost-sensitive подхода, если к нему возвращаться, должен
использовать не геометрическое расстояние до порога, а OOF downstream regret:
фактическую разницу ошибки соседних band experts. Для этого нужны nested OOF
предсказания специалистов на outer-train, иначе веса будут in-sample и дадут
утечку.

