import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import requests
import os


class DataLoader:
    def __init__(self, alpha_vantage_api_key=None):
        self.alpha_vantage_api_key = alpha_vantage_api_key

    def load_from_yahoo(self, ticker, start=None, end=None, interval='1d'):
        """
        Loads historical data from Yahoo Finance.
        :param ticker: Stock ticker (e.g., 'AAPL').
        :param start: Start date (e.g., '2020-01-01').
        :param end: End date (e.g., '2024-01-01').
        :param interval: Interval ('1d', '1wk', '1mo').
        :return: DataFrame with data.
        """
        try:
            data = yf.download(ticker, start=start, end=end, interval=interval)
            data.reset_index(inplace=True)
            return data
        except Exception as e:
            print(f"Error loading data from Yahoo Finance: {e}")
            return None

    def load_from_alpha_vantage(self, ticker, interval='1day', outputsize='compact'):
        """
        Loads historical data from Alpha Vantage.
        :param ticker: Stock ticker (e.g., 'AAPL').
        :param interval: Interval ('1min', '5min', '15min', '30min', '60min', 'daily').
        :param outputsize: 'compact' (100 records) or 'full' (all records).
        :return: DataFrame with data.
        """
        if not self.alpha_vantage_api_key:
            print("Alpha Vantage API key is not set.")
            return None

        ts = TimeSeries(key=self.alpha_vantage_api_key, output_format='pandas')
        try:
            if interval in ['1min', '5min', '15min', '30min', '60min']:
                data, _ = ts.get_intraday(symbol=ticker, interval=interval, outputsize=outputsize)
            else:
                data, _ = ts.get_daily(symbol=ticker, outputsize=outputsize)
            data.rename(columns=lambda x: x[3:], inplace=True)  # Remove '1. ' in column names
            return data
        except Exception as e:
            print(f"Error loading data from Alpha Vantage: {e}")
            return None

    def load_from_binance(self, symbol, interval='1d', limit=100):
        """
        Loads data from Binance.
        :param symbol: Trading pair (e.g., 'BTCUSDT').
        :param interval: Interval ('1m', '5m', '1h', '1d').
        :param limit: Number of recent candles.
        :return: DataFrame with data.
        """
        url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
        try:
            response = requests.get(url)
            data = response.json()
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
                                             'quote_asset_volume', 'trades', 'taker_base', 'taker_quote', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(
                float)
            return df
        except Exception as e:
            print(f"Error loading data from Binance: {e}")
            return None

    def load_from_csv(self, file_path):
        """
        Loads data from a CSV file.
        :param file_path: File path.
        :return: DataFrame with data.
        """
        if not os.path.exists(file_path):
            print("File not found.")
            return None

        try:
            df = pd.read_csv(file_path, parse_dates=True, index_col=0)
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
