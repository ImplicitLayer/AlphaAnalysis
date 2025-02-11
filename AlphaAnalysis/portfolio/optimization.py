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
        pi = expected_returns.mean()  # Рыночные ожидания доходностей

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
