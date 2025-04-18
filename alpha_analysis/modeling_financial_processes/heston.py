import numpy as np


class HestonModel:
    def __init__(self, mu, kappa, theta, sigma_v, rho, S0, v0):
        """
        Инициализация модели Хестона.

        :param mu: Дрейф актива (средняя доходность)
        :param kappa: Скорость возвращения волатильности к среднему
        :param theta: Долгосрочное среднее значение волатильности
        :param sigma_v: Волатильность волатильности
        :param rho: Корреляция между винеров процессами для цены и волатильности
        :param S0: Начальная цена актива
        :param v0: Начальная волатильность
        """
        self.mu = mu  # Дрейф актива
        self.kappa = kappa  # Скорость возвращения волатильности к среднему
        self.theta = theta  # Долгосрочное среднее значение волатильности
        self.sigma_v = sigma_v  # Волатильность волатильности
        self.rho = rho  # Корреляция между винеров процессами
        self.S0 = S0  # Начальная цена актива
        self.v0 = v0  # Начальная волатильность

    def simulate(self, T, dt, n_steps):
        """
        Симуляция траектории цены актива и его волатильности с использованием модели Хестона.

        :param T: Время (например, 1 год)
        :param dt: Шаг по времени
        :param n_steps: Количество шагов (например, 252 для ежедневных шагов)

        :return: Массив с симулированными ценами актива и волатильностью
        """
        n = int(T / dt)  # Число шагов по времени

        # Массивы для хранения значений цены и волатильности
        S = np.zeros(n_steps)
        v = np.zeros(n_steps)

        # Инициализация начальных значений
        S[0] = self.S0
        v[0] = self.v0

        # Генерация случайных величин для винеров процессов
        for i in range(1, n_steps):
            # Генерация двух зависимых винеров процессов (по корреляции rho)
            z1 = np.random.normal(0, 1)
            z2 = np.random.normal(0, 1)
            dz1 = z1
            dz2 = self.rho * z1 + np.sqrt(1 - self.rho ** 2) * z2

            # Изменение волатильности и цены актива
            dv = self.kappa * (self.theta - v[i - 1]) * dt + self.sigma_v * np.sqrt(v[i - 1]) * dz2
            v[i] = v[i - 1] + dv

            # Коррекция волатильности, чтобы она не была отрицательной
            v[i] = max(v[i], 0)

            dS = self.mu * S[i - 1] * dt + np.sqrt(v[i - 1]) * S[i - 1] * dz1
            S[i] = S[i - 1] + dS

        return S, v


# Пример использования

if __name__ == "__main__":
    # Параметры для модели Хестона
    mu = 0.05  # Дрейф актива (средняя доходность)
    kappa = 2.0  # Скорость возвращения волатильности к среднему
    theta = 0.04  # Долгосрочное среднее значение волатильности
    sigma_v = 0.1  # Волатильность волатильности
    rho = -0.5  # Корреляция между винеров процессами
    S0 = 100  # Начальная цена актива
    v0 = 0.04  # Начальная волатильность

    # Создание объекта модели Хестона
    heston = HestonModel(mu, kappa, theta, sigma_v, rho, S0, v0)

    # Симуляция траектории
    T = 1  # Время до окончания (1 год)
    dt = 1 / 252  # Шаг по времени (252 торговых дня в году)
    n_steps = 252  # Количество шагов (252 для ежедневных шагов)

    # Генерация симулированных траекторий
    S_sim, v_sim = heston.simulate(T, dt, n_steps)

    # Результаты симуляции
    print("Симулированные цены актива:", S_sim)
    print("Симулированные волатильности:", v_sim)
