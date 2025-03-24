import numpy as np


class RiskManagement:
    def __init__(self, capital, risk_per_trade=0.01, max_drawdown=0.2):
        """
        Initialization of risk management system.

        :param capital: Initial capital.
        :param risk_per_trade: Percentage of risk per trade (e.g. 1% = 0.01).
        :param max_drawdown: Maximum drawdown of the portfolio (20% = 0.2).
        """
        self.initial_capital = capital
        self.capital = capital
        self.risk_per_trade = risk_per_trade
        self.max_drawdown = max_drawdown
        self.max_capital = capital

    def calculate_position_size(self, stop_loss, entry_price):
        """
        Calculates the position size based on the specified risk per trade.

        :param stop_loss: Stop loss price.
        :param entry_price: Entry price.
        :return: Position size in asset units.
        """
        risk_amount = self.capital * self.risk_per_trade
        position_size = risk_amount / abs(entry_price - stop_loss)
        return position_size

    def update_capital(self, pnl):
        """
        Updates the capital after the transaction.

        :param pnl: Profit or loss from the transaction.
        """
        self.capital += pnl
        self.max_capital = max(self.max_capital, self.capital)

    def check_drawdown(self):
        """
        Checks if the portfolio has reached the critical drawdown limit.

        :return: True if the drawdown limit is reached, otherwise False.
        """
        drawdown = (self.max_capital - self.capital) / self.max_capital
        return drawdown >= self.max_drawdown

    def calculate_var(self, returns, confidence_level=0.95):
        """
        Calculation of Value at Risk (VaR) by historical data_preprocessing.

        :param returns: Array of returns.
        :param confidence_level: Confidence level (e.g. 95%).
        :return: VaR value.
        """
        if len(returns) == 0:
            return 0
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var

    def calculate_cvar(self, returns, confidence_level=0.95):
        """
        Calculation of Conditional Value at Risk (CVaR).

        :param returns: Array of returns.
        :param confidence_level: Confidence level (e.g. 95%).
        :return: CVaR value.
        """
        var = self.calculate_var(returns, confidence_level)
        cvar = np.mean([r for r in returns if r <= var])
        return cvar
