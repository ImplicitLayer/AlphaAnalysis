import pandas as pd
import numpy as np


class Backtester:
    def __init__(self, data, signals, initial_balance=10000, commission=0.001, slippage=0.0005):
        """
        Class for backtesting trading strategies.

        :param data: DataFrame with columns ['Date', 'Price']
        :param signals: DataFrame with columns ['Date', 'Signal'], where Signal = {-1, 0, 1}
        :param initial_balance: Initial capital
        :param commission: Commission per trade (in percent, 0.001 = 0.1%)
        :param slippage: Slippage (in percent)
        """
        self.data = data.copy()
        self.signals = signals.copy()
        self.balance = initial_balance
        self.position = 0
        self.commission = commission
        self.slippage = slippage
        self.trades = []

    def run_backtest(self):
        """ Starts backtesting and calculates profitability. """
        merged = self.data.merge(self.signals, on='Date', how='left').fillna(0)
        merged['Signal'] = merged['Signal'].shift(1)  # Запаздывание сигнала на 1 день

        for i in range(1, len(merged)):
            price = merged.loc[merged.index[i], 'Price']
            signal = merged.loc[merged.index[i], 'Signal']

            if signal == 1 and self.position == 0:  # Покупка
                self.position = self.balance / (price * (1 + self.slippage))
                self.balance = 0
                self.trades.append(('BUY', price, merged.index[i]))

            elif signal == -1 and self.position > 0:  # Продажа
                self.balance = self.position * price * (1 - self.slippage) * (1 - self.commission)
                self.position = 0
                self.trades.append(('SELL', price, merged.index[i]))

        # Финальная оценка портфеля
        final_value = self.balance + (self.position * merged.iloc[-1]['Price'] if self.position > 0 else 0)
        return final_value

    def get_trades(self):
        """ Returns a list of transactions. """
        return pd.DataFrame(self.trades, columns=['Type', 'Price', 'Date'])
