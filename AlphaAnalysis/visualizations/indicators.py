import pandas as pd
import numpy as np


class Indicators:
    @staticmethod
    def moving_average(df, column, window=14):
        """
        Calculates the moving average for a given column.
        :param df: Input DataFrame.
        :param column: Column to calculate moving average on.
        :param window: Window size for moving average.
        :return: DataFrame with moving average column.
        """
        df[f'ma_{column}_{window}'] = df[column].rolling(window=window).mean()
        return df

    @staticmethod
    def relative_strength_index(df, column, window=14):
        """
        Calculates the Relative Strength Index (RSI).
        :param df: Input DataFrame.
        :param column: Column to calculate RSI on.
        :param window: Window size for RSI calculation.
        :return: DataFrame with RSI column.
        """
        delta = df[column].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df[f'rsi_{column}_{window}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def bollinger_bands(df, column, window=20, num_std=2):
        """
        Calculates Bollinger Bands.
        :param df: Input DataFrame.
        :param column: Column to calculate Bollinger Bands on.
        :param window: Window size for moving average.
        :param num_std: Number of standard deviations for bands.
        :return: DataFrame with Bollinger Bands columns.
        """
        rolling_mean = df[column].rolling(window=window).mean()
        rolling_std = df[column].rolling(window=window).std()
        df[f'bb_upper_{column}'] = rolling_mean + (rolling_std * num_std)
        df[f'bb_lower_{column}'] = rolling_mean - (rolling_std * num_std)
        return df

    @staticmethod
    def moving_average_convergence_divergence(df, column, short_window=12, long_window=26, signal_window=9):
        """
        Calculates the MACD indicator.
        :param df: Input DataFrame.
        :param column: Column to calculate MACD on.
        :param short_window: Short EMA window.
        :param long_window: Long EMA window.
        :param signal_window: Signal line EMA window.
        :return: DataFrame with MACD and Signal Line columns.
        """
        short_ema = df[column].ewm(span=short_window, adjust=False).mean()
        long_ema = df[column].ewm(span=long_window, adjust=False).mean()
        df['macd'] = short_ema - long_ema
        df['macd_signal'] = df['macd'].ewm(span=signal_window, adjust=False).mean()
        return df

    @staticmethod
    def average_true_range(df, high_column, low_column, close_column, window=14):
        """
        Calculates the Average True Range (ATR).
        :param df: Input DataFrame.
        :param high_column: High price column.
        :param low_column: Low price column.
        :param close_column: Close price column.
        :param window: Window size for ATR calculation.
        :return: DataFrame with ATR column.
        """
        df['H-L'] = df[high_column] - df[low_column]
        df['H-PC'] = abs(df[high_column] - df[close_column].shift(1))
        df['L-PC'] = abs(df[low_column] - df[close_column].shift(1))
        df['TrueRange'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df['ATR'] = df['TrueRange'].rolling(window=window).mean()
        return df

    @staticmethod
    def stochastic_oscillator(df, high_column, low_column, close_column, window=14):
        """
        Calculates the Stochastic Oscillator.
        :param df: Input DataFrame.
        :param high_column: High price column.
        :param low_column: Low price column.
        :param close_column: Close price column.
        :param window: Window size for the calculation.
        :return: DataFrame with Stochastic Oscillator column.
        """
        highest_high = df[high_column].rolling(window=window).max()
        lowest_low = df[low_column].rolling(window=window).min()
        df['stochastic'] = 100 * (df[close_column] - lowest_low) / (highest_high - lowest_low)
        return df
