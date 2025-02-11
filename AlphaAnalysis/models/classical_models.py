import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
import matplotlib.pyplot as plt


class ClassicalModels:
    @staticmethod
    def arima_model(df, column, order=(5, 1, 0)):
        """
        ARIMA model for time series forecasting.
        :param df: Input DataFrame.
        :param column: Column to apply ARIMA on.
        :param order: The (p, d, q) order of the ARIMA model.
        :return: Forecasted values and model summary.
        """
        # Fit ARIMA model
        model = ARIMA(df[column], order=order)
        model_fit = model.fit()

        # Forecasting the next 10 periods
        forecast = model_fit.forecast(steps=10)

        # Plot the result
        plt.figure(figsize=(10, 6))
        plt.plot(df[column], label="Actual")
        plt.plot(pd.date_range(df.index[-1], periods=11, freq='D')[1:], forecast, label="Forecast", color='red')
        plt.legend(loc='best')
        plt.title(f'ARIMA Model Forecast (p={order[0]}, d={order[1]}, q={order[2]})')
        plt.show()

        return forecast, model_fit.summary()

    @staticmethod
    def garch_model(df, column, p=1, q=1):
        """
        GARCH model for volatility forecasting.
        :param df: Input DataFrame.
        :param column: Column to apply GARCH on (usually returns).
        :param p: The order of the GARCH model (lag for conditional variance).
        :param q: The order of the ARCH model (lag for returns).
        :return: Forecasted volatility and model summary.
        """
        # Calculate returns as percentage change
        returns = df[column].pct_change().dropna()

        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=p, q=q)
        model_fit = model.fit()

        # Forecasting the next 10 periods
        forecast = model_fit.forecast(horizon=10)

        # Extract forecasted volatility (standard deviation)
        forecast_volatility = np.sqrt(forecast.variance.values[-1, :])

        # Create date range for forecasted periods
        forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=10, freq='D')

        # Plot the result
        plt.figure(figsize=(10, 6))
        plt.plot(returns.index, returns, label="Returns")
        plt.plot(forecast_dates, forecast_volatility, label="Forecasted Volatility", color='red')
        plt.legend(loc='best')
        plt.title(f'GARCH Model Forecast (p={p}, q={q})')
        plt.show()

        return forecast_volatility, model_fit.summary()

    @staticmethod
    def var_model(df, columns, lags=5):
        """
        Vector Autoregression (VAR) model for multivariate time series forecasting.
        :param df: Input DataFrame.
        :param columns: List of columns to apply VAR on.
        :param lags: Number of lags to use in the VAR model.
        :return: Forecasted values and model summary.
        """
        # Prepare data for VAR model
        model_data = df[columns].dropna()

        # Fit VAR model
        model = VAR(model_data)
        model_fit = model.fit(lags)

        # Forecasting the next 10 periods
        forecast = model_fit.forecast(model_data.values[-lags:], steps=10)

        # Plot the result for each series
        plt.figure(figsize=(10, 6))
        for i, col in enumerate(columns):
            plt.subplot(len(columns), 1, i + 1)
            plt.plot(model_data.index, model_data[col], label="Actual")
            plt.plot(pd.date_range(model_data.index[-1], periods=11, freq='D')[1:], forecast[:, i], label="Forecast",
                     color='red')
            plt.title(f'{col} Forecast')
            plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

        return forecast, model_fit
