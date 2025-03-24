import numpy as np


class Portfolio:
    def __init__(self, returns, weights):
        """
        Инициализация портфеля.

        :param returns: Массив доходностей активов (например, N x M, где N — количество периодов, M — количество активов)
        :param weights: Список весов активов в портфеле (например, [0.5, 0.3, 0.2] для трех активов)
        """
        self.returns = np.array(returns)
        self.weights = np.array(weights)
        self.n_assets = len(weights)
        self.n_periods = len(returns)

        # Рассчитываем ковариационную матрицу
        self.cov_matrix = np.cov(self.returns.T)

    def expected_return(self):
        """
        Расчет ожидаемой доходности портфеля.

        :return: Ожидаемая доходность портфеля
        """
        return np.sum(self.weights * np.mean(self.returns, axis=0))

    def portfolio_volatility(self):
        """
        Расчет волатильности (стандартного отклонения) портфеля.

        :return: Волатильность портфеля
        """
        return np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights)))

    def portfolio_variance(self):
        """
        Расчет дисперсии портфеля.

        :return: Дисперсия портфеля
        """
        return np.dot(self.weights.T, np.dot(self.cov_matrix, self.weights))

    def portfolio_sharpe_ratio(self, risk_free_rate=0.0):
        """
        Расчет коэффициента Шарпа для портфеля.

        :param risk_free_rate: Безрисковая ставка (по умолчанию 0)
        :return: Коэффициент Шарпа портфеля
        """
        excess_return = self.expected_return() - risk_free_rate
        return excess_return / self.portfolio_volatility()

    def covariance_matrix(self):
        """
        Получение ковариационной матрицы активов.

        :return: Ковариационная матрица
        """
        return self.cov_matrix

    def maximum_drawdown(self):
        """
        Расчет максимальной просадки портфеля.

        :return: Максимальная просадка
        """
        cumulative_returns = np.cumprod(1 + self.returns, axis=0)  # Кумулятивный доход
        cumulative_returns = np.sum(cumulative_returns, axis=1)  # Кумулятивный доход для всего портфеля
        peak = np.maximum.accumulate(cumulative_returns)  # Максимальный пик
        drawdown = (cumulative_returns - peak) / peak  # Просадка
        max_drawdown = np.min(drawdown)  # Максимальная просадка

        return max_drawdown

    def portfolio_return(self, initial_value=1):
        """
        Расчет доходности портфеля за весь период с учетом начальной стоимости портфеля.

        :param initial_value: Начальная стоимость портфеля (по умолчанию 1)
        :return: Конечная стоимость портфеля
        """
        cumulative_returns = np.cumprod(1 + np.sum(self.returns * self.weights, axis=1))  # Кумулятивный доход портфеля
        final_value = initial_value * cumulative_returns[-1]
        return final_value

    def correlation_matrix(self):
        """
        Расчет корреляционной матрицы активов.

        :return: Корреляционная матрица
        """
        return np.corrcoef(self.returns.T)

    def rolling_volatility(self, window=30):
        """
        Расчет волатильности портфеля с использованием скользящего окна.

        :param window: Размер окна для скользящей волатильности
        :return: Массив волатильности для каждого окна
        """
        rolling_volatility = np.array([
            np.std(np.sum(self.returns[i - window:i] * self.weights, axis=1)) for i in range(window, self.n_periods)
        ])
        return rolling_volatility

    def random_portfolio_simulation(self, num_simulations=1000):
        """
        Симуляция случайных портфелей для анализа оптимальных весов.

        :param num_simulations: Количество случайных портфелей
        :return: Массив с доходностями, рисками (волатильность) и коэффициентами Шарпа для случайных портфелей
        """
        results = np.zeros((3, num_simulations))  # Массив для хранения доходности, рисков и коэффициентов Шарпа
        for i in range(num_simulations):
            random_weights = np.random.random(self.n_assets)
            random_weights /= np.sum(random_weights)  # Нормируем веса, чтобы их сумма была 1

            portfolio = Portfolio(self.returns, random_weights)
            results[0, i] = portfolio.expected_return()  # Доходность
            results[1, i] = portfolio.portfolio_volatility()  # Волатильность
            results[2, i] = portfolio.portfolio_sharpe_ratio()  # Коэффициент Шарпа

        return results


# Пример использования

if __name__ == "__main__":
    # Пример случайных данных (доходности активов за несколько периодов)
    np.random.seed(42)  # Для воспроизводимости
    returns = np.random.normal(0.05, 0.02, (252, 3))  # Генерация случайных доходностей для 3 активов за 252 дня (1 год)

    # Веса активов в портфеле
    weights = [0.4, 0.4, 0.2]

    # Создание объекта портфеля
    portfolio = Portfolio(returns, weights)

    # Ожидаемая доходность портфеля
    print(f"Ожидаемая доходность портфеля: {portfolio.expected_return():.4f}")

    # Волатильность портфеля
    print(f"Волатильность портфеля: {portfolio.portfolio_volatility():.4f}")

    # Коэффициент Шарпа
    print(f"Коэффициент Шарпа портфеля: {portfolio.portfolio_sharpe_ratio():.4f}")

    # Ковариационная матрица
    print(f"Ковариационная матрица:\n{portfolio.covariance_matrix()}")

    # Максимальная просадка
    print(f"Максимальная просадка: {portfolio.maximum_drawdown():.4f}")

    # Доходность портфеля с начальной стоимостью 1
    print(f"Доходность портфеля (начальная стоимость 1): {portfolio.portfolio_return():.4f}")

    # Корреляционная матрица
    print(f"Корреляционная матрица:\n{portfolio.correlation_matrix()}")

    # Волатильность с использованием скользящего окна (например, окно 30 дней)
    rolling_volatility = portfolio.rolling_volatility(window=30)
    print(f"Волатильность с использованием скользящего окна (30 дней):\n{rolling_volatility}")

    # Симуляция случайных портфелей
    simulation_results = portfolio.random_portfolio_simulation(num_simulations=1000)
    print(f"Пример случайных портфелей (доходность, волатильность, Шарп):\n{simulation_results[:, :5]}")
