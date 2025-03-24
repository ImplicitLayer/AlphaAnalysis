import numpy as np


class JumpDiffusionModel:
    def __init__(self, mu, sigma, lambda_, jump_mean, jump_vol, S0):
        """
        Инициализация модели с прыжками в диффузии.

        :param mu: Дрейф (средняя доходность)
        :param sigma: Волатильность
        :param lambda_: Интенсивность пуассоновского процесса (среднее количество прыжков)
        :param jump_mean: Средний размер прыжка
        :param jump_vol: Волатильность прыжков
        :param S0: Начальная цена актива
        """
        self.mu = mu  # Дрейф
        self.sigma = sigma  # Волатильность
        self.lambda_ = lambda_  # Интенсивность пуассоновского процесса
        self.jump_mean = jump_mean  # Средний размер прыжка
        self.jump_vol = jump_vol  # Волатильность прыжков
        self.S0 = S0  # Начальная цена актива

    def simulate(self, T, dt, n_steps):
        """
        Симуляция траектории цены актива с прыжками в диффузии.

        :param T: Время (например, 1 год)
        :param dt: Шаг по времени
        :param n_steps: Количество шагов (например, 252 для ежедневных шагов)

        :return: Массив с симулированными ценами актива
        """
        n = int(T / dt)  # Число шагов по времени

        # Массив для хранения значений цены
        S = np.zeros(n_steps)
        S[0] = self.S0  # Начальное значение

        # Генерация случайных величин для винеров процесса и пуассоновских прыжков
        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))  # Генерация случайной величины dW_t (диффузия)
            jump = 0

            # Случайная величина для пуассоновского процесса (прыжки)
            if np.random.poisson(self.lambda_ * dt) > 0:
                # Генерация величины прыжка
                jump = np.random.normal(self.jump_mean, self.jump_vol)

            # Изменение цены
            dS = self.mu * S[i - 1] * dt + self.sigma * S[i - 1] * dW + jump * S[i - 1]
            S[i] = S[i - 1] + dS

        return S


# Пример использования

if __name__ == "__main__":
    # Параметры для модели с прыжками в диффузии
    mu = 0.05  # Дрейф (средняя доходность)
    sigma = 0.2  # Волатильность
    lambda_ = 0.1  # Интенсивность пуассоновского процесса
    jump_mean = -0.1  # Средний размер прыжка (например, снижение на 10%)
    jump_vol = 0.3  # Волатильность прыжков
    S0 = 100  # Начальная цена актива

    # Создание объекта модели с прыжками в диффузии
    jump_diffusion = JumpDiffusionModel(mu, sigma, lambda_, jump_mean, jump_vol, S0)

    # Симуляция траектории
    T = 1  # Время до окончания (1 год)
    dt = 1 / 252  # Шаг по времени (252 торговых дня в году)
    n_steps = 252  # Количество шагов (252 для ежедневных шагов)

    # Генерация симулированной траектории
    S_sim = jump_diffusion.simulate(T, dt, n_steps)

    # Результат симуляции
    print(S_sim)
