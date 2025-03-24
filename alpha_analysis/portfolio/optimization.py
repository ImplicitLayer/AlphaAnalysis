import numpy as np
import pandas as pd
import scipy.optimize as sco


class PortfolioOptimization:
    @staticmethod
    def mean_variance_optimization(returns, risk_free_rate=0.02):
        """
        Portfolio optimization by Modern Portfolio Theory (MPT).

        :param returns: DataFrame with asset returns.
        :param risk_free_rate: Risk free rate.
        :return: Optimal weights, expected return, volatility, Sharpe Ratio.
        """
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_assets = len(mean_returns)

        def portfolio_stats(weights):
            weights = np.array(weights)
            port_return = np.dot(weights, mean_returns)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility
            return np.array([port_return, port_volatility, sharpe_ratio])

        def neg_sharpe(weights):
            return -portfolio_stats(weights)[2]

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        optimal = sco.minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal.x, *portfolio_stats(optimal.x)

    @staticmethod
    def black_litterman(expected_returns, tau, P, Q, omega=None):
        """
        Implementation of the Black-Litterman model.

        :param expected_returns: Vector of expected asset returns.
        :param tau: Uncertainty coefficient of market returns.
        :param P: Matrix representing investor views of asset returns.
        :param Q: Expected returns according to the investor's views.
        :param omega: Covariance matrix of views (if None, automatically).
        :return: Adjusted expectations of returns.
        """
        cov_matrix = expected_returns.cov()
        pi = expected_returns.mean()  # Market equilibrium returns

        if omega is None:
            omega = np.diag(np.diag(P @ cov_matrix @ P.T)) * tau

        inv_cov = np.linalg.inv(cov_matrix * tau)
        inv_omega = np.linalg.inv(omega)

        mu_bl = np.linalg.inv(inv_cov + P.T @ inv_omega @ P) @ (inv_cov @ pi + P.T @ inv_omega @ Q)
        return mu_bl

    @staticmethod
    def risk_parity(returns):
        """
        Portfolio optimization by Risk Parity.

        :param returns: DataFrame with asset returns.
        :return: Optimal asset weights.
        """
        cov_matrix = returns.cov()
        num_assets = len(cov_matrix)

        def risk_budget_objective(weights):
            portfolio_variance = weights.T @ cov_matrix @ weights
            marginal_risk_contributions = cov_matrix @ weights
            risk_contributions = weights * marginal_risk_contributions
            return np.sum((risk_contributions - portfolio_variance / num_assets) ** 2)

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        optimal = sco.minimize(risk_budget_objective, initial_guess, method='SLSQP', bounds=bounds,
                               constraints=constraints)
        return optimal.x

    @staticmethod
    def conditional_var(returns, alpha=0.05):
        """
        Conditional Value-at-Risk (CVaR) optimization.

        :param returns: DataFrame with asset returns.
        :param alpha: Significance level for CVaR.
        :return: Optimal weights based on CVaR.
        """
        # Calculate the quantile for the VaR at the given confidence level
        var = np.percentile(returns, (1 - alpha) * 100)
        cvar = np.mean(returns[returns <= var])

        # Minimize CVaR by optimization
        def cvar_objective(weights):
            portfolio_returns = np.dot(returns, weights)
            portfolio_cvar = np.mean(portfolio_returns[portfolio_returns <= var])
            return portfolio_cvar

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(len(returns.columns)))
        initial_guess = len(returns.columns) * [1. / len(returns.columns)]

        optimal = sco.minimize(cvar_objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal.x

    @staticmethod
    def monte_carlo_simulation(returns, num_simulations=10000, risk_free_rate=0.02):
        """
        Метод Монте-Карло для моделирования оптимального портфеля.

        :param returns: DataFrame с доходностями активов.
        :param num_simulations: Количество симуляций.
        :param risk_free_rate: Безрисковая ставка (для вычисления коэффициента Шарпа).
        :return: Кортеж (матрица результатов, матрица весов).
        """

        num_assets = len(returns.columns)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        # Матрица для хранения результатов: доходность, волатильность, Sharpe Ratio
        results = np.zeros((3, num_simulations))

        # Отдельная матрица для хранения весов активов
        weights_record = np.zeros((num_assets, num_simulations))

        for i in range(num_simulations):
            # Генерация случайных весов так, чтобы их сумма была равна 1
            weights = np.random.dirichlet(np.ones(num_assets), size=1).flatten()

            # Расчет портфельной доходности
            port_return = np.dot(weights, mean_returns)

            # Расчет портфельной волатильности (стандартное отклонение)
            port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

            # Расчет коэффициента Шарпа
            sharpe_ratio = (port_return - risk_free_rate) / port_volatility

            # Запись результатов
            results[0, i] = port_return
            results[1, i] = port_volatility
            results[2, i] = sharpe_ratio

            # Сохранение весов
            weights_record[:, i] = weights

        return results, weights_record

    @staticmethod
    def active_share(returns, max_active_assets=5):
        """
        Active Share optimization, limiting the number of active assets.

        :param returns: DataFrame with asset returns.
        :param max_active_assets: Maximum number of active assets allowed in the portfolio.
        :return: Optimal weights that respect the Active Share constraint.
        """
        num_assets = len(returns.columns)

        # Objective: Minimize the number of active assets
        def active_share_objective(weights):
            num_active_assets = np.count_nonzero(weights > 0)
            return num_active_assets

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        optimal = sco.minimize(active_share_objective, initial_guess, method='SLSQP', bounds=bounds,
                               constraints=constraints)
        optimal_weights = optimal.x

        # Ensure the number of active assets is within the limit
        if np.count_nonzero(optimal_weights > 0) > max_active_assets:
            # Zero out the least significant assets
            sorted_indices = np.argsort(optimal_weights)
            optimal_weights[sorted_indices[:num_assets - max_active_assets]] = 0

        return optimal_weights

    @staticmethod
    def tax_aware_optimization(returns, tax_rate=0.2):
        """
        Tax-aware portfolio optimization.

        :param returns: DataFrame with asset returns.
        :param tax_rate: Tax rate to be applied on returns.
        :return: Optimal portfolio weights considering taxes.
        """
        mean_returns = returns.mean()
        cov_matrix = returns.cov()

        num_assets = len(mean_returns)

        def tax_adjusted_return(weights):
            # Adjust returns by the tax rate
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_return_after_tax = portfolio_return * (1 - tax_rate)
            return portfolio_return_after_tax

        def neg_sharpe(weights):
            portfolio_return = tax_adjusted_return(weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio

        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = num_assets * [1. / num_assets]

        optimal = sco.minimize(neg_sharpe, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal.x


# Пример использования
if __name__ == "__main__":
    # Генерация случайных доходностей для 3 активов за 252 дня
    np.random.seed(42)
    returns = pd.DataFrame(np.random.normal(0.05, 0.02, (252, 3)), columns=["Asset 1", "Asset 2", "Asset 3"])

    # Оптимизация с использованием модели MPT
    optimal_weights, exp_return, vol, sharpe = PortfolioOptimization.mean_variance_optimization(returns)
    print(f"Optimal Weights: {optimal_weights}")
    print(f"Expected Return: {exp_return}")
    print(f"Volatility: {vol}")
    print(f"Sharpe Ratio: {sharpe}")

    # Оптимизация с использованием Risk Parity
    risk_parity_weights = PortfolioOptimization.risk_parity(returns)
    print(f"Risk Parity Weights: {risk_parity_weights}")

    # Симуляция случайных портфелей с использованием Монте-Карло
    monte_carlo_results = PortfolioOptimization.monte_carlo_simulation(returns)
    print(f"Simulated Portfolio Results (first 5 portfolios):\n{monte_carlo_results}")

    # Оптимизация с учетом Active Share
    active_share_weights = PortfolioOptimization.active_share(returns, max_active_assets=2)
    print(f"Active Share Weights: {active_share_weights}")

    # Оптимизация с учетом налогов
    tax_aware_weights = PortfolioOptimization.tax_aware_optimization(returns, tax_rate=0.2)
    print(f"Tax-Aware Weights: {tax_aware_weights}")
