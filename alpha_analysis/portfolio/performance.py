import numpy as np
import pandas as pd


class PortfolioPerformance:
    @staticmethod
    def portfolio_return(weights, returns):
        """
        Рассчитывает среднюю доходность портфеля.

        :param weights: Веса активов в портфеле.
        :param returns: DataFrame с доходностями активов.
        :return: Средняя доходность портфеля.
        """
        return np.dot(weights, returns.mean())

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.02):
        """
        Рассчитывает коэффициент Шарпа.

        :param returns: DataFrame с доходностями портфеля.
        :param risk_free_rate: Безрисковая ставка (по умолчанию 2%).
        :return: Коэффициент Шарпа.
        """
        excess_return = returns.mean() - risk_free_rate
        return excess_return / returns.std()

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0.02):
        """
        Рассчитывает коэффициент Сортино (учитывает только отрицательную волатильность).

        :param returns: DataFrame с доходностями портфеля.
        :param risk_free_rate: Безрисковая ставка.
        :return: Коэффициент Сортино.
        """
        excess_return = returns.mean() - risk_free_rate
        downside_risk = np.std(returns[returns < 0])
        return excess_return / downside_risk

    @staticmethod
    def max_drawdown(cumulative_returns):
        """
        Рассчитывает максимальную просадку портфеля.

        :param cumulative_returns: DataFrame с накопленной доходностью.
        :return: Максимальная просадка.
        """
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

    @staticmethod
    def calmar_ratio(returns):
        """
        Рассчитывает коэффициент Кэлмара (отношение доходности к максимальной просадке).

        :param returns: DataFrame с доходностями портфеля.
        :return: Коэффициент Кэлмара.
        """
        cumulative_returns = (1 + returns).cumprod()
        max_dd = PortfolioPerformance.max_drawdown(cumulative_returns)
        return returns.mean() / abs(max_dd)

    @staticmethod
    def information_ratio(portfolio_returns, benchmark_returns):
        """
        Рассчитывает коэффициент информационного преимущества (Information Ratio).

        :param portfolio_returns: DataFrame с доходностями портфеля.
        :param benchmark_returns: DataFrame с доходностями бенчмарка.
        :return: Коэффициент информационного преимущества.
        """
        active_return = portfolio_returns - benchmark_returns
        tracking_error = active_return.std()
        return active_return.mean() / tracking_error

    @staticmethod
    def beta(portfolio_returns, market_returns):
        """
        Рассчитывает бета-коэффициент портфеля.

        :param portfolio_returns: DataFrame с доходностями портфеля.
        :param market_returns: DataFrame с доходностями рынка.
        :return: Бета-коэффициент.
        """
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    @staticmethod
    def jensens_alpha(portfolio_returns, market_returns, risk_free_rate=0.02):
        """
        Рассчитывает Alpha (Jensen’s Alpha) — избыточную доходность портфеля.

        :param portfolio_returns: DataFrame с доходностями портфеля.
        :param market_returns: DataFrame с доходностями рынка.
        :param risk_free_rate: Безрисковая ставка.
        :return: Значение Jensen's Alpha.
        """
        beta = PortfolioPerformance.beta(portfolio_returns, market_returns)
        expected_return = risk_free_rate + beta * (market_returns.mean() - risk_free_rate)
        return portfolio_returns.mean() - expected_return

    @staticmethod
    def treynor_ratio(portfolio_returns, market_returns, risk_free_rate=0.02):
        """
        Рассчитывает коэффициент Трейнора (Treynor Ratio).

        :param portfolio_returns: DataFrame с доходностями портфеля.
        :param market_returns: DataFrame с доходностями рынка.
        :param risk_free_rate: Безрисковая ставка.
        :return: Коэффициент Трейнора.
        """
        beta = PortfolioPerformance.beta(portfolio_returns, market_returns)
        return (portfolio_returns.mean() - risk_free_rate) / beta

    @staticmethod
    def value_at_risk(returns, confidence_level=0.95):
        """
        Рассчитывает Value-at-Risk (VaR) с заданным уровнем доверия.

        :param returns: DataFrame с доходностями портфеля.
        :param confidence_level: Уровень доверия (по умолчанию 95%).
        :return: Значение VaR.
        """
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        return -var_percentile


if __name__ == "__main__":
    # Генерируем случайные данные доходностей для портфеля и рынка
    np.random.seed(42)
    portfolio_returns = pd.Series(np.random.normal(0.01, 0.05, 100))
    market_returns = pd.Series(np.random.normal(0.008, 0.04, 100))

    # Веса активов в портфеле
    weights = np.array([0.4, 0.3, 0.3])

    # Доходности активов
    asset_returns = pd.DataFrame({
        "Asset 1": np.random.normal(0.01, 0.05, 100),
        "Asset 2": np.random.normal(0.02, 0.07, 100),
        "Asset 3": np.random.normal(0.015, 0.06, 100),
    })

    # Вычисляем метрики
    print("Средняя доходность портфеля:", PortfolioPerformance.portfolio_return(weights, asset_returns))
    print("Sharpe Ratio:", PortfolioPerformance.sharpe_ratio(portfolio_returns))
    print("Sortino Ratio:", PortfolioPerformance.sortino_ratio(portfolio_returns))
    print("Max Drawdown:", PortfolioPerformance.max_drawdown((1 + portfolio_returns).cumprod()))
    print("Calmar Ratio:", PortfolioPerformance.calmar_ratio(portfolio_returns))
    print("Information Ratio:", PortfolioPerformance.information_ratio(portfolio_returns, market_returns))
    print("Beta:", PortfolioPerformance.beta(portfolio_returns, market_returns))
    print("Jensen’s Alpha:", PortfolioPerformance.jensens_alpha(portfolio_returns, market_returns))
    print("Treynor Ratio:", PortfolioPerformance.treynor_ratio(portfolio_returns, market_returns))
    print("Value-at-Risk (95% confidence):", PortfolioPerformance.value_at_risk(portfolio_returns))
