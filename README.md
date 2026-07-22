# Client Analytics — прогнозирование дохода по WMAE

В проекте зафиксированы ровно две production-системы: абсолютный лидер и
компактный лидер. Исследовательские выводы, включая feature ablation,
confusion matrix, bias по группам и анализ публичного решения, собраны в одном
ноутбуке [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb).

## Результаты

| Система | Моделей | Random WMAE | Temporal WMAE | Назначение |
|---|---:|---:|---:|---|
| Full champion | 26 | **59 291** | **57 740** | максимальный validated score |
| Compact champion | 16 | 59 501 | 57 827 | проще развивать и обслуживать |

Random — strict outer validation. Temporal — зафиксированный holdout:
январь–май обучают модель, июнь проверяет перенос во времени.

## Структура

```text
train_full_champion.py       # полный 26-model ансамбль
train_compact_champion.py    # компактный 16-model ансамбль
src/                         # общие production-компоненты
notebooks/EDA.ipynb          # весь EDA и журнал исследований
data/raw/                    # train.csv и test.csv
outputs/                     # только два production submission
outputs/partial/             # компоненты, OOF, аудиты и старые эксперименты
tests/                       # тесты переиспользуемых компонентов
```

## Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Файлы `data/raw/train.csv` и `data/raw/test.csv` читаются с параметрами
`sep=";", decimal=","`. Train содержит `target` и `w`; test — только входные
признаки и `id`.

## Запуск

Полный чемпион:

```bash
python3 train_full_champion.py
```

Результаты:

- `outputs/submission_full_champion.csv`;
- `outputs/partial/submission_full_champion_components.csv`.

Компактный чемпион:

```bash
python3 train_compact_champion.py
```

Результаты:

- `outputs/submission_compact_champion.csv`;
- `outputs/partial/submission_compact_champion_components.csv`.

Параметры итераций доступны через `python3 <entry-point> --help`. Значения по
умолчанию являются validated-конфигурацией; изменение параметров превращает
запуск в новый эксперимент.

## Архитектура полного чемпиона

1. Base CatBoost на `target ** 0.25` с последующим blend зарплатного anchor.
2. Три tail classifiers и три tail experts для 150k/300k/500k.
3. Multi-anchor expert для клиентов с тремя и более источниками дохода.
4. Семь локальных узлов hierarchical LightGBM router.
5. Восемь LightGBM band experts и мягкий confidence-gated routing.
6. Три unweighted log-quantile LightGBM, lognormal-аппроксимация и точная
   WMAE-optimal weighted median.
7. Финал: `0.80 × hierarchical_prediction + 0.20 × distribution_prediction`.

Для top-class G0/G4 используется temperature `0.3`, для остальных — `0.5`.

## Архитектура compact champion

Base, tail, source и семь узлов router совпадают с полным backbone. Восемь
локальных band experts заменены одной band-conditioned LightGBM: строка
реплицируется для своего и соседних диапазонов, а requested band передаётся
явным контекстом. Distribution head отсутствует. Итого 16 моделей.

## EDA

```bash
jupyter notebook notebooks/EDA.ipynb
```

Ноутбук рассчитан на `Restart & Run All` и включает:

- схему, dtype-аудит, target, пропуски и train/test drift;
- точное восстановление функции веса WMAE;
- anchors, scale ratios, flows, trends, region normalization, expense shares,
  log/rank, missing flags и recency;
- random/temporal feature ablation;
- confusion matrix, bias, MAE и вклад каждой группы;
- clustering, boundary repair, cost/regret/trust gates и attractor experiments;
- анализ distribution stacking публичного решения;
- обоснование двух текущих production-архитектур.

## Проверки

```bash
pytest -q
python3 train_full_champion.py --help
python3 train_compact_champion.py --help
```

Submission имеет два столбца `id;predict`, разделитель `;`, десятичный знак
`,`; перед отправкой формат следует сверить с актуальными правилами конкурса.
