import numpy as np


class GBM:
    def __init__(self, mu, sigma, S0):
        self.mu = mu       # Дрейф
        self.sigma = sigma # Волатильность
        self.S0 = S0       # Начальная цена актива

    def _drift(self, t, S):
        return self.mu * S   # Дрейф

    def _diffusion(self, t, S):
        return self.sigma * S  # Волатильность

    def simulate(self, T, dt, n_steps):
        """ Симуляция процесса GBM """
        n = int(T / dt)
        S = np.zeros(n_steps)
        S[0] = self.S0
        t = np.linspace(0, T, n_steps)

        for i in range(1, n_steps):
            dW = np.random.normal(0, np.sqrt(dt))  # Стандартное нормальное распределение
            S[i] = S[i-1] + self._drift(t[i-1], S[i-1]) * dt + self._diffusion(t[i-1], S[i-1]) * dW
        return t, S


if __name__ == "__main__":
    # Параметры
    mu = 0.05  # Дрейф (5% годовых)
    sigma = 0.2  # Волатильность (20% годовых)
    S0 = 100  # Начальная цена

    # Создание объекта модели GBM
    gbm_process = GBM(mu, sigma, S0)

    # Симуляция
    T = 1  # 1 год
    dt = 0.01  # Шаг по времени
    n_steps = 1000
    t, S = gbm_process.simulate(T, dt, n_steps)

    print(S)
