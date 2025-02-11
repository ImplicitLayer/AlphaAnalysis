import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class PlotUtils:
    @staticmethod
    def plot_time_series(df, column, title="Time Series Plot"):
        """
        Plots a time series graph.
        :param df: Input DataFrame with a datetime index.
        :param column: Column to plot.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df[column], label=column, color='blue')
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()

    @staticmethod
    def plot_histogram(data, xlabel="x", bins=30, title="Histogram"):
        """
        Plots a histogram for a given column.
        :param data: Input data.
        :param xlabel: label in x axis.
        :param bins: Number of bins.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        sns.histplot(data, bins=bins, kde=True, color='blue')
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_correlation_matrix(df, title="Correlation Matrix"):
        """
        Plots a heatmap of the correlation matrix.
        :param df: Input DataFrame.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_scatter(x_data, y_data, title="Scatter Plot"):
        """
        Plots a scatter plot between two columns.
        :param x_data: X-axis data.
        :param y_data: Y-axis data.
        :param title: Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        sns.scatterplot(x=x_data, y=y_data, color='blue')
        plt.title(title)
        plt.show()
