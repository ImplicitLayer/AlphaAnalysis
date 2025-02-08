<div align="center">
  <img src="https://github.com/user-attachments/assets/c9426b1f-9ce1-4a44-9547-7c77a77ec1f6">
</div>
</br>

**AlphaAnalysis** — это библиотека для анализа финансовых показателей, которая включает классические методы анализа данных и машинное обучение. Проект предназначен для использования в финансовых рынках и может быть полезен для анализа, прогнозирования и оптимизации торговых стратегий, а также для управления портфелем.

Основной функционал доступен бесплатно, но для доступа к полному функционалу, включая продвинутые модели машинного обучения и алгоритмическую торговлю, необходимо приобрести **premium** версию.

## Описание модулей

### 1. data
Модуль для загрузки, очистки и предобработки данных. Сюда входят функции для извлечения данных из различных источников, очистки и преобразования данных, а также создания новых признаков для дальнейшего анализа.

### 2. visualization
Модуль для визуализации финансовых данных, включая построение графиков временных рядов, корреляций и технических индикаторов, таких как скользящие средние и полосы Боллинджера.

### 3. models
Этот модуль включает различные модели для анализа финансовых данных, как классические (например, ARIMA, GARCH, VAR), так и современные модели машинного обучения (Random Forest, XGBoost, CatBoost), а также глубокие нейронные сети (LSTM, GRU, Transformer) и алгоритмы машинного обучения с подкреплением для торговли.

### 4. trading
Модуль для алгоритмической торговли. Включает функциональность для бэктестинга торговых стратегий, симуляции торговли в реальном времени и управления рисками.

### 5. portfolio
Модуль для управления инвестиционным портфелем. Включает в себя инструменты для оптимизации портфеля (MPT, Black-Litterman), оценки рисков (VaR, CVaR) и кластеризации активов.

### 6. signal_generation
Модуль для генерации торговых сигналов. Включает в себя анализ новостей и соцсетей (sentiment analysis), создание технических сигналов и фундаментальный анализ.

## Установка

Для установки библиотеки используйте pip:

```bash
pip install alphaanalysis
```

## Premium версия

Для доступа к полному функционалу необходимо приобрести **premium** версию.

|                Категория        |          Модуль          |  Open Source  |  Premium   |
| ------------------------------- | ------------------------ | ------------- | ---------- |
| **Загрузка и обработка данных** |     `data_loader.py`     |        ☑️     |     -      |
|                                 |   `data_cleaning.py`     |        ☑️     |     -      |
|                                 | `feature_engineering.py` |       -       |     ☑️     |
|      **Визуализация**           |     `plots.py`           |        ☑️     |     -      |
|                                 |   `indicators.py`        |        ☑️     |     -      |
|          **Модели**             | `classical_models.py`    |        ☑️     |     -      |
|                                 |       `ml_models.py`     |        ☑️     |     -      |
|                                 |    `deep_learning.py`    |       -       |     ☑️     |
|                                 |       `auto_ml.py`       |       -       |     ☑️     |
|          **Трейдинг**           |    `backtesting.py`      |       -       |     ☑️     |
|                                 |  `risk_management.py`    |       -       |     ☑️     |
|    **Управление портфелем**     |    `optimization.py`     |       -       |     ☑️     |
|                                 |   `risk_analysis.py`     |        ☑️     |     -      |
|                                 |      `clustering.py`     |        ☑️     |     -      |
|    **Генерация сигналов**	      | `technical_signals.py`   |        ☑️     |     -      |
|                                 | `fundamental_signals.py` |       -       |     ☑️     |
|                                 | `sentiment_analysis.py`  |       -       |     ☑️     |

Для получения доступа к premium функционалу, свяжитесь с нами по [электронной почте](burenok023@gmail.com).

## Примеры использования
Пример использования для прогнозирования временных рядов:

```python
from AlphaAnalysis.models import classical_models
from AlphaAnalysis.data import data_loader

# Загрузка данных
data = data_loader.load_from_csv('historical_stock_data.csv')

# Применение модели ARIMA
classical_models = classical_models.ClassicalModels()

forecast, summary = classical_models.arima_model(data, 'price', order=(5, 1, 0))
print("ARIMA Model Summary:")
print(summary)
```

Пример кластеризации активов:

```python
from AlphaAnalysis.portfolio.clustering import PortfolioClustering
import numpy as np
import pandas as pd

np.random.seed(42)
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'XOM', 'META']
num_assets = len(assets)
num_days = 252

# Симуляция доходностей
simulated_returns = pd.DataFrame(np.random.randn(num_days, num_assets) / 100, columns=assets)

clustering = PortfolioClustering(simulated_returns)

# K-Means
kmeans_result = clustering.kmeans_clustering(num_clusters=3)
print("K-Means Clustering:")
print(kmeans_result)
```

## Планы на будущее

Мы активно развиваем AlphaAnalysis и планируем добавить:

* **Генеративные модели (GANs)** для симуляции рыночных данных
* **Кастомизируемые торговые стратегии** через пользовательские скрипты
* **Обучение с подкреплением**
* Более сложные **методы моделирования**
  
Если у вас есть предложения, откройте issue в репозитории!

## Контакты

C нами можно связаться с помощью:

📩 Email: burenok023@gmail.com

🔗 Telegram: @artemburenok




