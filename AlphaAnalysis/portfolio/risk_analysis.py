import numpy as np
import pandas as pd


class RiskAnalysis:
    @staticmethod
    def value_at_risk(returns, confidence_level=0.95):
        """
        Calculation of VaR (Value at Risk) with specified confidence level.

        :param returns: DataFrame with asset returns.
        :param confidence_level: Confidence level (e.g. 0.95 for 95% VaR).
        :return: VaR estimate.
        """
        return np.percentile(returns, 100 * (1 - confidence_level))

    @staticmethod
    def conditional_value_at_risk(returns, confidence_level=0.95):
        """
        Calculating CVaR (Conditional Value at Risk), also known as Expected Shortfall.

        :param returns: DataFrame with asset returns.
        :param confidence_level: Confidence level.
        :return: CVaR estimate.
        """
        var = RiskAnalysis.value_at_risk(returns, confidence_level)
        return returns[returns <= var].mean()

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Sharpe Ratio calculation to estimate returns per unit of risk.

        :param returns: DataFrame with portfolio returns.
        :param risk_free_rate: Risk free rate.
        :return: Sharpe Ratio value.
        """
        excess_returns = returns - risk_free_rate / len(returns)
        return excess_returns.mean() / excess_returns.std()

    @staticmethod
    def max_drawdown(cumulative_returns):
        """
        Calculation of Maximum Drawdown.

        :param cumulative_returns: Cumulative portfolio returns.
        :return: Maximum drawdown.
        """
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    @staticmethod
    def portfolio_volatility(returns, weights):
        """
        Calculation of portfolio volatility.

        :param returns: DataFrame with asset returns.
        :param weights: Weights of assets in the portfolio.
        :return: Portfolio volatility.
        """
        cov_matrix = returns.cov()
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
