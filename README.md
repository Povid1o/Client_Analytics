# Client_Analytics — прогнозирование дохода клиента (WMAE)

Регрессия дохода клиента (`target`), метрика — **WMAE** (Weighted Mean
Absolute Error). Используются только предоставленные `train.csv`/`test.csv` —
никаких внешних данных (запрещено регламентом конкурса).

## Структура

```
data/raw/            # сюда положить train.csv, test.csv (см. ниже)
data/processed/      # кэш очищенных данных (не используется в baseline, зарезервировано)
notebooks/
  01_eda.ipynb              # EDA: dtype-аудит, target, пропуски, регион, train/test shift
  02_baseline_pipeline.ipynb # препроцессинг, региональный OOF-признак, CatBoost, сабмит
src/                 # переиспользуемый код, импортируется из ноутбуков
tests/                # unit-тесты для src/metrics.py и src/region_encoding.py
outputs/              # результаты запуска ноутбуков (submission, importance, figures)
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
сабмита выбран как допущение: **`id,target`** (два столбца, разделитель
`,`, точка как десятичный разделитель) — **это нужно подтвердить у
организаторов** перед финальной отправкой.

## Порядок запуска

1. `notebooks/01_eda.ipynb` — Restart & Run All. Строит EDA, сохраняет
   `outputs/feature_stats.csv` и графики в `outputs/figures/`.
2. `notebooks/02_baseline_pipeline.ipynb` — Restart & Run All. Строит
   региональный OOF-признак, обучает CatBoost, выводит CV WMAE и sanity
   floor, сохраняет `outputs/submission_baseline.csv` и
   `outputs/feature_importance.csv`.

Оба ноутбука воспроизводимы (seed=42 везде, см. `src/config.py`) и
рассчитаны на выполнение целиком без ручных правок.

## Тесты

```bash
pytest tests/ -v
```

Покрывают `src/metrics.py::wmae`/`weighted_median` (ручные примеры) и
leakage-safety `src/region_encoding.py` (OOF-энкодинг не использует
собственный target/w строки).

## Где искать результаты

- `outputs/submission_baseline.csv` — финальный сабмит.
- `outputs/feature_importance.csv` — топ-20 признаков финальной модели.
- `outputs/feature_stats.csv` — сводка по всем признакам (dtype, семейство,
  доля пропусков train/test, |Spearman ρ| с target).
- `outputs/figures/` — графики из EDA.

## Ограничения этой итерации (см. ТЗ, раздел 6)

Никаких ансамблей/стекинга, сложного FE сверх регионального признака,
агрессивного тюнинга гиперпараметров или внешних данных. Это baseline и
точка отсчёта для дальнейших итераций.
