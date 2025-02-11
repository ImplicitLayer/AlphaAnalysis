import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class FeatureEngineering:
    @staticmethod
    def add_moving_average(df, column, window=5):
        """
        Adds a moving average column.
        :param df: Input DataFrame.
        :param column: Column to calculate moving average on.
        :param window: Window size for moving average.
        :return: DataFrame with moving average column.
        """
        df[f'ma_{column}_{window}'] = df[column].rolling(window=window).mean()
        return df

    @staticmethod
    def add_exponential_moving_average(df, column, span=5):
        """
        Adds an exponential moving average column.
        :param df: Input DataFrame.
        :param column: Column to calculate EMA on.
        :param span: Span for EMA calculation.
        :return: DataFrame with EMA column.
        """
        df[f'ema_{column}_{span}'] = df[column].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def add_volatility(df, column, window=5):
        """
        Adds a volatility column (standard deviation over a window).
        :param df: Input DataFrame.
        :param column: Column to calculate volatility on.
        :param window: Window size for volatility calculation.
        :return: DataFrame with volatility column.
        """
        df[f'volatility_{column}_{window}'] = df[column].rolling(window=window).std()
        return df

    @staticmethod
    def add_log_return(df, column):
        """
        Adds a log return column.
        :param df: Input DataFrame.
        :param column: Column to calculate log return on.
        :return: DataFrame with log return column.
        """
        df[f'log_return_{column}'] = np.log(df[column] / df[column].shift(1))
        return df

    @staticmethod
    def add_polynomial_features(df, columns, degree=2):
        """
        Adds polynomial features for given columns.
        :param df: Input DataFrame.
        :param columns: List of column names to create polynomial features for.
        :param degree: Degree of polynomial features.
        :return: DataFrame with polynomial features.
        """
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_features = poly.fit_transform(df[columns])
        feature_names = poly.get_feature_names_out(columns)
        df_poly = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        return pd.concat([df, df_poly], axis=1)
