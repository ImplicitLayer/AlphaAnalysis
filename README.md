<div align="center">
  <img src="https://github.com/user-attachments/assets/c9426b1f-9ce1-4a44-9547-7c77a77ec1f6">
</div>
</br>

**AlphaAnalysis** — is a library for analyzing financial indicators, which includes classical data analysis methods and machine learning. The project is intended for use in financial markets and can be useful for analysis, forecasting and optimization of trading strategies, as well as for portfolio management.

Basic functionality is available for free, but to access the full functionality, including advanced machine learning models and algorithmic trading, you must purchase the **premium** version.

## Module Description

### 1. data
Module for data loading, cleaning and preprocessing. This includes functions for extracting data from various sources, cleaning and transforming data, and creating new attributes for further analysis.

### 2. visualization
A module for visualizing financial data, including plotting time series, correlations and technical indicators such as moving averages and Bollinger Bands.

### 3. models
This module includes various models for analyzing financial data, both classical (e.g. ARIMA, GARCH, VAR) and modern machine learning models (Random Forest, XGBoost, CatBoost), as well as deep neural networks (LSTM, GRU, Transformer) for trading.

### 4. trading
Module for algorithmic trading. Includes functionality for backtesting trading strategies and risk management.

### 5. portfolio
Module for investment portfolio management. Includes tools for portfolio optimization (MPT, Black-Litterman), risk assessment (VaR, CVaR) and asset clustering.

### 6. signal_generation
Module for generating trading signals. Includes analysis of news and social networks (sentiment analysis), creation of technical signals and fundamental analysis.

## Installation

Use pip to install the library:

```bash
pip install alphaanalysis
```

## Premium version

To access the full functionality you need to purchase the **premium** version.

|                Category         |          Module          |  Open Source  |  Premium   |
| ------------------------------- | ------------------------ | ------------- | ---------- |
| **Data loading and processing** |     `data_loader.py`     |        ☑️     |     -      |
|                                 |   `data_cleaning.py`     |        ☑️     |     -      |
|                                 | `feature_engineering.py` |       -       |     ☑️     |
|      **Visualisation**          |     `plots.py`           |        ☑️     |     -      |
|                                 |   `indicators.py`        |        ☑️     |     -      |
|          **Models**             | `classical_models.py`    |        ☑️     |     -      |
|                                 |       `ml_models.py`     |        ☑️     |     -      |
|                                 |    `deep_learning.py`    |       -       |     ☑️     |
|                                 |       `auto_ml.py`       |       -       |     ☑️     |
|          **Trading**            |    `backtesting.py`      |       -       |     ☑️     |
|                                 |  `risk_management.py`    |       -       |     ☑️     |
|    **Portfolio management**     |    `optimization.py`     |       -       |     ☑️     |
|                                 |   `risk_analysis.py`     |        ☑️     |     -      |
|                                 |      `clustering.py`     |        ☑️     |     -      |
|    **Signal generation**	      | `technical_signals.py`   |        ☑️     |     -      |
|                                 | `fundamental_signals.py` |       -       |     ☑️     |
|                                 | `sentiment_analysis.py`  |       -       |     ☑️     |

To access premium functionality, contact us.

📩 Email: burenok023@gmail.com

## Examples of use

Example of use for time series forecasting:

```python
from AlphaAnalysis.models import classical_models
from AlphaAnalysis.data import data_loader

# load data
data = data_loader.load_from_csv('historical_stock_data.csv')

# using ARIMA model
classical_models = classical_models.ClassicalModels()

forecast, summary = classical_models.arima_model(data, 'price', order=(5, 1, 0))
print("ARIMA Model Summary:")
print(summary)
```

The example of asset clustering:

```python
from AlphaAnalysis.portfolio.clustering import PortfolioClustering
import numpy as np
import pandas as pd

np.random.seed(42)
assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NFLX', 'NVDA', 'JPM', 'XOM', 'META']
num_assets = len(assets)
num_days = 252

# profit simulation
simulated_returns = pd.DataFrame(np.random.randn(num_days, num_assets) / 100, columns=assets)

clustering = PortfolioClustering(simulated_returns)

# K-Means
kmeans_result = clustering.kmeans_clustering(num_clusters=3)
print("K-Means Clustering:")
print(kmeans_result)
```

## Future plans

We are actively developing AlphaAnalysis and plan to add:

* **Generative models (GANs)** to simulate market data
* **Customisable trading strategies** via custom scripts
* **Reinforcement learning**
* More complex **modelling methods**
    
If you have suggestions, open an issue in the repository!

## Contacts

We can be contacted by:

📩 Email: burenok023@gmail.com

🔗 Telegram: @artemburenok




