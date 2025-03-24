import numpy as np
import pandas as pd
from arch import arch_model
from typing import Union, Optional


class GARCHModel:
    """
    Класс для работы с моделями ARCH и GARCH.
    """

    def __init__(self, p: int = 1, q: int = 1, model_type: str = "GARCH"):
        """
        :param p: Параметр для авторегрессионной части
        :param q: Параметр для скользящего среднего
        :param model_type: ARCH или GARCH
        """
        self.p = p
        self.q = q
        self.model_type = model_type
        self.model = None
        self.fitted_model = None

    def fit(self, data: Union[pd.Series, np.ndarray]):
        """
        Обучение модели ARCH/GARCH
        :param data: Временной ряд
        """
        if self.model_type.upper() == "ARCH":
            self.model = arch_model(data, vol="ARCH", p=self.p)
        else:
            self.model = arch_model(data, vol="GARCH", p=self.p, q=self.q)

        self.fitted_model = self.model.fit(disp="off")
        return self.fitted_model

    def forecast(self, horizon: int = 1):
        """
        Прогнозирование будущей волатильности
        :param horizon: Число шагов вперед
        """
        if self.fitted_model is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")
        return self.fitted_model.forecast(horizon=horizon).variance[-1:]

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
    ts_data = np.random.randn(100)  # Имитация случайных изменений

    # Создаем и обучаем модель GARCH
    garch = GARCHModel(p=1, q=1, model_type="GARCH")
    garch.fit(ts_data)

    # Прогнозируем будущую волатильность
    forecast = garch.forecast(horizon=5)
    print(f"Прогноз волатильности: {forecast}")

    # Выводим сводку
    print(garch.summary())
