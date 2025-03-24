import numpy as np
import pandas as pd
import statsmodels.api as sm
from hmmlearn import hmm
from typing import Optional, Union, Tuple


class KalmanFilterModel:
    """
    Реализация фильтра Калмана для одномерных временных рядов.
    """

    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, data: pd.Series):
        """
        Обучает модель на временном ряде.

        :param data: Временной ряд (pandas Series)
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Данные должны быть pandas Series")

        # Создаем модель состояния с локальным уровнем (Local Level Model)
        self.model = sm.tsa.UnobservedComponents(data, level='local level')
        self.result = self.model.fit()

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Прогнозирует будущее значение временного ряда.

        :param steps: Число шагов вперед
        :return: Предсказанные значения
        """
        if self.result is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")

        return self.result.forecast(steps=steps)

    def summary(self):
        """Возвращает сводку модели"""
        if self.result is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")

        return self.result.summary()


class HiddenMarkovModel:
    """
    Реализация скрытой марковской модели (HMM).
    """

    def __init__(self, n_states: int = 2):
        """
        :param n_states: Количество скрытых состояний
        """
        self.n_states = n_states
        self.model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag")

    def fit(self, data: np.ndarray):
        """
        Обучает модель HMM.

        :param data: Данные в формате numpy array (размерность [samples, features])
        """
        if not isinstance(data, np.ndarray):
            raise ValueError("Данные должны быть numpy array")

        self.model.fit(data)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Определяет вероятностное распределение состояний.

        :param data: Данные в формате numpy array
        :return: Последовательность скрытых состояний
        """
        return self.model.predict(data)


class BayesianStructuralTimeSeries:
    """
    Байесовский структурный временной ряд (BSTS)
    (Простейшая реализация на основе state-space models из statsmodels)
    """

    def __init__(self):
        self.model = None
        self.result = None

    def fit(self, data: pd.Series):
        """
        Обучает BSTS-модель.

        :param data: Временной ряд (pandas Series)
        """
        if not isinstance(data, pd.Series):
            raise ValueError("Данные должны быть pandas Series")

        # Простая локальная линейная модель
        self.model = sm.tsa.UnobservedComponents(data, level="local linear trend")
        self.result = self.model.fit()

    def forecast(self, steps: int = 1) -> np.ndarray:
        """
        Прогноз будущих значений.

        :param steps: Число шагов вперед
        :return: Предсказанные значения
        """
        if self.result is None:
            raise ValueError("Сначала обучите модель с помощью .fit()")

        return self.result.forecast(steps=steps)


# Пример использования
if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

    # Фильтр Калмана
    kf = KalmanFilterModel()
    kf.fit(data)
    print("Kalman Forecast:", kf.forecast(steps=5))

    # Скрытые марковские модели
    hmm_data = np.column_stack([data.values])
    hmm_model = HiddenMarkovModel(n_states=3)
    hmm_model.fit(hmm_data)
    print("HMM States:", hmm_model.predict(hmm_data))

    # Байесовский структурный временной ряд
    bsts = BayesianStructuralTimeSeries()
    bsts.fit(data)
    print("BSTS Forecast:", bsts.forecast(steps=5))
