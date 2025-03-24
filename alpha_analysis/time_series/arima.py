import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from typing import Union


class ARIMAModel:
    """
    Класс для работы с моделями AR, MA, ARMA и ARIMA.
    """

    def __init__(self, order: tuple = (1, 0, 1)):
        """
        :param order: Параметры модели (p, d, q)
        """
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, data: Union[pd.Series, np.ndarray]):
        """
        Обучение модели ARIMA
        :param data: Временной ряд
        """
        self.model = ARIMA(data, order=self.order)
        self.fitted_model = self.model.fit()
        return self.fitted_model

    def forecast(self, steps: int = 1):
        """
        Прогнозирование будущих значений
        :param steps: Число шагов вперед
        """
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")
        return self.fitted_model.forecast(steps=steps)

    def summary(self):
        """
        Возвращает сводку обученной модели
        """
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")
        return self.fitted_model.summary()


# Пример использования
if __name__ == "__main__":
    # Генерируем временной ряд
    np.random.seed(42)
    ts_data = np.cumsum(np.random.randn(100))  # Имитируем случайный процесс

    # Создаем и обучаем модель
    arima = ARIMAModel(order=(2, 1, 2))
    arima.fit(ts_data)

    # Прогнозируем будущее значение
    forecast = arima.forecast(steps=5)
    print(f"Прогноз: {forecast}")

    # Выводим сводку
    print(arima.summary())
