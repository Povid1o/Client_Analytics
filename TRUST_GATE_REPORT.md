# Trust gate experiment

## Цель

Попытка получить дополнительный WMAE-сигнал без новых salary regressors:
предсказывать, когда cumulative ordinal correction следует ослабить или
полностью отключить.

Meta-train состоит только из strict OOF random-fold строк до июня. June
temporal holdout не используется ни для обучения gate, ни для выбора
abstention policy.

## Проверенные targets

1. `harm`: ordinal correction увеличивает абсолютную ошибку относительно
   non-ordinal прогноза.
2. `far`: ошибка hard ordinal class составляет два диапазона или больше.
3. `alpha`: оптимальная непрерывная доля ordinal correction на отрезке
   `[0, 1]`.

Для `harm` веса равны `w * abs(utility)`. Для `far` используются исходные
WMAE-веса. Для `alpha` — `w * abs(ordinal step)`. Проверены компактный набор
posterior/anchor diagnostics и полный набор числовых признаков.

## Результаты

| Gate | Meta-OOF | Temporal | Решение |
|---|---:|---:|---|
| Harm compact | +1 | +5 | reject |
| Harm full numeric | −3 | −1 | reject |
| Far compact | 0 | 0 | reject |
| Far full numeric | −1 | −9 | reject |
| Continuous alpha compact | 0 | 0 | reject |
| Continuous alpha full numeric | 0 | 0 | reject |

`far` classifier имеет хороший weighted AUC: 0.808 compact и 0.816 full.
Однако policy, выбранная непосредственно по WMAE, почти всегда предпочитает
оставить текущую коррекцию без изменений. Высокая вероятность дальнего
промаха класса не эквивалентна вредной численной коррекции: специалист может
двигать прогноз в правильную сторону даже при неверном hard class.

## Вывод

Trust gate не добавляется в production. Текущий лучший submission и его
метрики остаются без изменений. Следующий более обоснованный путь — менять
loss и веса самих cumulative boundaries, особенно для дорогих переходов
между несоседними диапазонами, а не пытаться исправлять готовый posterior
после обучения.

