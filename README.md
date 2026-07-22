# Client_Analytics — прогнозирование дохода клиента (WMAE)

Регрессия дохода клиента (`target`), метрика — **WMAE** (Weighted Mean
Absolute Error). Используются только предоставленные `train.csv`/`test.csv` —
никаких внешних данных (запрещено регламентом конкурса).

## Структура

```
data/raw/            # сюда положить train.csv, test.csv (см. ниже)
data/processed/      # кэш очищенных данных (не используется в baseline, зарезервировано)
notebooks/
  00_eda_experiment_log.ipynb # хронологический лог EDA-гипотез (см. EDA_experiment_log.md)
  01_eda.ipynb              # EDA: dtype-аудит, target, пропуски, регион, train/test shift
  02_baseline_pipeline.ipynb # препроцессинг, региональный OOF-признак, CatBoost, сабмит
src/                 # переиспользуемый код, импортируется из ноутбуков
tests/                # unit-тесты для src/metrics.py и src/region_encoding.py
outputs/              # только итоговые submission полных production-систем
  partial/            # validation/test-предикты, компоненты, аудиты и графики
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Данные

Положите `train.csv` и `test.csv` в `data/raw/` (эти файлы в `.gitignore`,
в репозиторий не попадают). Файлы читаются как
`pd.read_csv(path, sep=';', decimal=',')`.

- `train.csv`: 222 входных признака + `id`, `target`, `w` (вес наблюдения,
  используется как `sample_weight` при обучении и при расчёте локального
  CV WMAE — организаторы, по условиям задачи, применяют свой вес на
  приватном тесте).
- `test.csv`: те же 222 входных признака + `id`, без `target`/`w`.

`sample_submission.csv` в материалах конкурса обнаружен не был. Формат
сабмита выбран как допущение: **`id;predict`** (два столбца, разделитель
`;`, запятая как десятичный разделитель — тот же формат, что и во входных
`train.csv`/`test.csv`) — **это нужно подтвердить у организаторов** перед
финальной отправкой.

## Порядок запуска

1. `notebooks/00_eda_experiment_log.ipynb` — Restart & Run All. Хронологическое
   воспроизведение всех EDA-гипотез и проверок в том порядке, в котором они
   изначально прогонялись (см. `EDA_experiment_log.md`) — каждый найденный
   результат печатается рядом с референсным значением из лога для
   трассируемости. Не заменяет `01_eda.ipynb`, а дополняет его как
   хронологический слой.
2. `notebooks/01_eda.ipynb` — Restart & Run All. Финальный тематический EDA,
   сохраняет `outputs/partial/feature_stats.csv` и графики в
   `outputs/partial/figures/`.
3. `notebooks/02_baseline_pipeline.ipynb` — Restart & Run All. Строит
   региональный OOF-признак, обучает CatBoost, выводит CV WMAE и sanity
   floor, сохраняет `outputs/submission_baseline.csv` и
   `outputs/partial/feature_importance.csv`.

Все три ноутбука воспроизводимы (seed=42 везде, см. `src/config.py`) и
рассчитаны на выполнение целиком без ручных правок.

## Валидация

CV использует `StratifiedKFold` (`src/validation.py`), стратифицированный по
децилям target (`pd.qcut(target, 10)`), а не обычный случайный `KFold`: без
стратификации редкие дорогие клиенты верхнего дециля (у которых `w` почти на
порядок выше среднего — см. ноутбук 00, D1) могут по случайности
сконцентрироваться в одном фолде, что раздувает fold-to-fold разброс WMAE и
не отражает реальную неопределённость модели. Те же стратифицированные
фолды используются и для регионального OOF-энкодинга (`src/region_encoding.py`),
чтобы модель и признак оценивались консистентно.

Дополнительно к per-fold WMAE (`mean ± std` по 5 фолдам) считается **pooled
OOF WMAE** с bootstrap 95% CI (`src/metrics.py::bootstrap_wmae_ci`, 2000
ресемплов на всём пуле OOF-предсказаний). Наивный `std` по 5 fold-level
числам — оценка на выборке размера 5, она может заметно завышать
неопределённость; bootstrap CI использует весь объём train. Количественное
сравнение (наивный `KFold` → `StratifiedKFold` → bootstrap) приведено в
`02_baseline_pipeline.ipynb`, раздел 5.

## Тесты

```bash
pytest tests/ -v
```

Покрывают `src/metrics.py::wmae`/`weighted_median` (ручные примеры) и
leakage-safety `src/region_encoding.py` (OOF-энкодинг не использует
собственный target/w строки).

## Где искать результаты

- `outputs/submission_baseline.csv` — финальный сабмит.
- `outputs/partial/feature_importance.csv` — топ-20 признаков финальной модели.
- `outputs/partial/feature_stats.csv` — сводка по всем признакам (dtype, семейство,
  доля пропусков train/test, |Spearman ρ| с target).
- `outputs/partial/figures/` — графики из EDA.

## Ограничения этой итерации (см. ТЗ, раздел 6)

Никаких ансамблей/стекинга, сложного FE сверх регионального признака,
агрессивного тюнинга гиперпараметров или внешних данных. Это baseline и
точка отсчёта для дальнейших итераций.

## Улучшенная модель

В `train_improved.py` добавлена вторая, независимая от baseline итерация:

- ансамбль двух weighted-RMSE CatBoost на `target ** 0.25` и `sqrt(target)`;
- смешивание прогноза с `salary_6to12m_avg` только там, где этот признак
  доступен (никакие target-derived признаки на test не используются);
- ограничение прогноза известным диапазоном target;
- региональный target encoding исключён: прежний OOF-признак безопасен для
  отдельной строки, но при внешнем CV часть его значений обучающей выборки
  могла зависеть от target в validation-фолде.

Честный 5-fold OOF для лучшей одиночной модели составил **63 698 WMAE**, а
после зарплатной калибровки — **62 342 WMAE** против **70 999** у baseline
(улучшение на 12,2%). На временном holdout новый ансамбль также улучшил
результат: примерно 62,3 тыс. до зарплатной калибровки против 69,2 тыс. у
baseline.

Быстро обучить финальные модели и создать сабмит:

```bash
python3 train_improved.py
```

Повторить полный 5-fold CV перед финальным обучением (заметно дольше):

```bash
python3 train_improved.py --cv
```

Результат сохраняется в `outputs/submission_improved.csv`.

## Tail/diversity итерация

Продолжение экспериментов с tail classifiers, conditional quantiles,
leaf-space neighbours, source experts и LightGBM diversity описано в
`ADVANCED_TAIL_EXPERIMENT_REPORT.md`. Выбранный ансамбль улучшил WMAE на
1 998 на random outer fold и на 1 364 на temporal holdout.

```bash
python3 train_advanced_tail.py
```

Новый кандидат сохраняется в `outputs/submission_advanced_tail.csv`.

## Hierarchical ordinal router

Надстройка классифицирует клиента локальным бинарным деревом из семи узлов и
восстанавливает распределение по восьми диапазонам. Затем confidence-gated
routing мягко смешивает локальных регрессоров-специалистов. Иерархия заменила
семь прежних cumulative boundaries, не увеличивая число моделей.

Полный ансамбль получил **59 712 WMAE** на random outer fold и **57 772 WMAE**
на June temporal holdout — на 118 и 251 лучше прежнего production. Полная
диагностика классификации и сравнение с кластеризацией приведены в
`HIERARCHICAL_ROUTING_REPORT.md`.

```bash
python3 train_ordinal_router.py
```

Submission сохраняется в `outputs/submission_hierarchical_router.csv`, а
вероятности диапазонов и все коррекции — в
`outputs/partial/submission_hierarchical_router_components.csv`.
