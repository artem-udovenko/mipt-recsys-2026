# mipt-recsys-2026
## Удовенко Артём, Б05-322

Домашнее задание 1 по рекомендательным системам.



Датасет: Goodreads Book Graph (UCSD), жанровое подмножество Poetry - https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

Используются файлы goodreads_interactions_poetry.json.gz, goodreads_books_poetry.json.gz, goodreads_book_authors.json.gz; они скачиваются автоматически в data/ при запуске ноутбука.

## Структура
- experiments.ipynb - данные, EDA, протокол оценки, подбор гиперпараметров, сравнение моделей, выводы
- models/ - реализации с нуля (NumPy / SciPy / PyTorch):
  - knn.py - item-based kNN (косинус с shrinkage)
  - als.py - iALS
  - ease.py - EASE
  - fm.py - факторизационная машина с метапризнаками, BPR
  - timesvdpp.py - timeSVD++
  - data.py, metrics.py - загрузка и разбиение данных, метрики

## Запуск
\[Продублировано в самом ноутбуке\]
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```

