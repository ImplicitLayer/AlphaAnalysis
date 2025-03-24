import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


class VARModel:
    """
    Класс для работы с моделью векторной авторегрессии (VAR).
    """

    def __init__(self, lag_order: int = 1):
        """
        :param lag_order: Количество лагов (p) для VAR модели
        """
        self.lag_order = lag_order
        self.model = None
        self.fitted_model = None

    def fit(self, data: pd.DataFrame):
        """
        Обучение модели VAR.
        :param data: Датафрейм с временными рядами (каждый столбец - отдельный ряд)
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Данные должны быть переданы в формате pandas DataFrame.")

        self.model = VAR(data)
        self.fitted_model = self.model.fit(self.lag_order)
        return self.fitted_model

    def forecast(self, steps: int = 1):
        """
        Прогнозирование будущих значений.
        :param steps: Число шагов вперед
        """
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")

        return self.fitted_model.forecast(self.fitted_model.endog, steps=steps)

    def summary(self):
        """
        Возвращает сводку обученной модели.
        """
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")

        return self.fitted_model.summary()


# Пример использования
if __name__ == "__main__":
    # Генерируем искусственные временные ряды
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "series_1": np.cumsum(np.random.randn(100)),
        "series_2": np.cumsum(np.random.randn(100))
    }, index=dates)

    # Создаем и обучаем модель VAR
    var_model = VARModel(lag_order=2)
    var_model.fit(data)

    # Прогнозируем будущее
    forecast = var_model.forecast(steps=5)
    print(f"Прогноз: \n{forecast}")

    # Выводим сводку
    print(var_model.summary())
