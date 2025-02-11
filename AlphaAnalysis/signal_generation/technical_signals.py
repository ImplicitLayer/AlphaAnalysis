import pandas as pd
import numpy as np


class TechnicalSignals:
    def __init__(self, data):
        """
        Class for generating trading signals based on technical indicators.

        :param data: DataFrame with columns [‘Date’, ‘Open’, ‘High’, ‘Low’, ‘Close’, ‘Volume’]
        """
        self.data = data.copy()

    def moving_average_crossover(self, short_window=10, long_window=50):
        """
        Moving averages (SMA) crossover signal.
        Buy if the short SMA crosses the long SMA upwards.
        Sell if the short SMA crosses the long SMA downwards.
        """
        self.data['SMA_Short'] = self.data['Close'].rolling(window=short_window).mean()
        self.data['SMA_Long'] = self.data['Close'].rolling(window=long_window).mean()
        self.data['SMA_Signal'] = np.where(self.data['SMA_Short'] > self.data['SMA_Long'], 1, -1)
        return self.data[['Date', 'SMA_Signal']]

    def rsi_signal(self, period=14, overbought=70, oversold=30):
        """
        Signal based on RSI.
        Buy if RSI is below the oversold level.
        Sell if RSI is above the overbought level.
        """
        delta = self.data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).rolling(window=period, min_periods=1).mean()
        avg_loss = pd.Series(loss).rolling(window=period, min_periods=1).mean()

        rs = avg_gain / (avg_loss + 1e-10)  # добавляем маленькое число, чтобы избежать деления на ноль
        self.data['RSI'] = 100 - (100 / (1 + rs))
        self.data['RSI_Signal'] = np.where(self.data['RSI'] < oversold, 1,
                                           np.where(self.data['RSI'] > overbought, -1, 0))
        return self.data[['Date', 'RSI_Signal']]

    def macd_signal(self, fast=12, slow=26, signal=9):
        """
        Signal based on MACD.
        Buy if MACD crosses the signal line upwards.
        Sell if MACD crosses the signal line downwards.
        """
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_Signal'] = np.where(self.data['MACD'] > self.data['MACD_Signal'], 1, -1)
        return self.data[['Date', 'MACD_Signal']]

    def bollinger_bands_signal(self, period=20, std_dev=2):
        """
        Signal based on Bollinger Bands.
        Buy if the price is below the lower boundary.
        Sell if the price is above the upper boundary.
        """
        self.data['Middle_Band'] = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        self.data['Upper_Band'] = self.data['Middle_Band'] + std_dev * std
        self.data['Lower_Band'] = self.data['Middle_Band'] - std_dev * std
        self.data['BB_Signal'] = np.where(self.data['Close'] < self.data['Lower_Band'], 1,
                                          np.where(self.data['Close'] > self.data['Upper_Band'], -1, 0))
        return self.data[['Date', 'BB_Signal']]

    def atr_signal(self, period=14, threshold=1.5):
        """
        Signal based on ATR (Average True Range).
        Buy if ATR is above the average value.
        Sell if ATR is below the average value.
        """
        high_low = self.data['High'] - self.data['Low']
        high_close = np.abs(self.data['High'] - self.data['Close'].shift(1))
        low_close = np.abs(self.data['Low'] - self.data['Close'].shift(1))

        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        self.data['ATR'] = tr.rolling(window=period).mean()
        atr_mean = self.data['ATR'].mean()
        self.data['ATR_Signal'] = np.where(self.data['ATR'] > atr_mean * threshold, 1, -1)
        return self.data[['Date', 'ATR_Signal']]

    def stochastic_signal(self, k_period=14, d_period=3, overbought=80, oversold=20):
        """
        Signal based on stochastic oscillator.
        Buy if %K crosses %D from below in the oversold zone.
        Sell if %K crosses %D from above in the overbought zone.
        """
        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()

        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / (high_max - low_min + 1e-10))
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=d_period).mean()

        self.data['Stoch_Signal'] = np.where(
            (self.data['Stoch_K'] > self.data['Stoch_D']) & (self.data['Stoch_K'] < oversold), 1,
            np.where((self.data['Stoch_K'] < self.data['Stoch_D']) & (self.data['Stoch_K'] > overbought), -1, 0))
        return self.data[['Date', 'Stoch_Signal']]

    def generate_signals(self):
        """
        Generate all signals and combine them into one DataFrame.
        """
        sma = self.moving_average_crossover()
        rsi = self.rsi_signal()
        macd = self.macd_signal()
        bb = self.bollinger_bands_signal()
        atr = self.atr_signal()
        stoch = self.stochastic_signal()

        signals = sma.merge(rsi, on='Date').merge(macd, on='Date').merge(bb, on='Date').merge(atr, on='Date').merge(
            stoch, on='Date')
        return signals
