import pandas as pd
import numpy as np


class DataCleaning:
    @staticmethod
    def remove_missing_values(df, threshold=0.5):
        """
        Removes columns with missing values above the given threshold.
        :param df: Input DataFrame.
        :param threshold: Maximum allowed fraction of missing values in a column (default is 0.5).
        :return: Cleaned DataFrame.
        """
        missing_fraction = df.isnull().mean()
        columns_to_keep = missing_fraction[missing_fraction <= threshold].index
        return df[columns_to_keep].dropna()

    @staticmethod
    def fill_missing_values(df, method='ffill'):
        """
        Fills missing values using the specified method.
        :param df: Input DataFrame.
        :param method: Method to fill missing values ('ffill', 'bfill', or 'mean').
        :return: DataFrame with filled values.
        """
        if method == 'mean':
            return df.fillna(df.mean())
        elif method in ['ffill', 'bfill']:
            return df.fillna(method=method)
        else:
            raise ValueError("Invalid method. Choose from 'ffill', 'bfill', or 'mean'.")

    @staticmethod
    def remove_outliers(df, method='zscore', threshold=3):
        """
        Removes outliers from numerical columns using Z-score or IQR method.
        :param df: Input DataFrame.
        :param method: Method to detect outliers ('zscore' or 'iqr').
        :param threshold: Threshold for detecting outliers.
        :return: Cleaned DataFrame.
        """
        df_cleaned = df.copy()
        if method == 'zscore':
            z_scores = np.abs((df_cleaned - df_cleaned.mean()) / df_cleaned.std())
            df_cleaned = df_cleaned[(z_scores < threshold).all(axis=1)]
        elif method == 'iqr':
            Q1 = df_cleaned.quantile(0.25)
            Q3 = df_cleaned.quantile(0.75)
            IQR = Q3 - Q1
            df_cleaned = df_cleaned[
                ~((df_cleaned < (Q1 - threshold * IQR)) | (df_cleaned > (Q3 + threshold * IQR))).any(axis=1)]
        else:
            raise ValueError("Invalid method. Choose from 'zscore' or 'iqr'.")
        return df_cleaned

    @staticmethod
    def normalize_data(df, method='minmax'):
        """
        Normalizes numerical columns using Min-Max scaling or Z-score standardization.
        :param df: Input DataFrame.
        :param method: Normalization method ('minmax' or 'zscore').
        :return: Normalized DataFrame.
        """
        df_normalized = df.copy()
        if method == 'minmax':
            df_normalized = (df_normalized - df_normalized.min()) / (df_normalized.max() - df_normalized.min())
        elif method == 'zscore':
            df_normalized = (df_normalized - df_normalized.mean()) / df_normalized.std()
        else:
            raise ValueError("Invalid method. Choose from 'minmax' or 'zscore'.")
        return df_normalized
