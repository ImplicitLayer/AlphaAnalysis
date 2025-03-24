import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.tsa.stattools import acf, pacf
from sklearn.feature_selection import RFE, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.decomposition import PCA
import xgboost as xgb
import pywt
import shap


class FeatureGenerator:
    """
    Генерация признаков для временных рядов.
    """

    @staticmethod
    def rolling_features(data: pd.Series, window: int = 5) -> pd.DataFrame:
        """
        Рассчитывает скользящие статистические признаки.

        :param data: Временной ряд (pandas Series)
        :param window: Длина окна для расчета признаков
        :return: DataFrame с новыми признаками
        """
        features = pd.DataFrame(index=data.index)
        features["rolling_mean"] = data.rolling(window=window).mean()
        features["rolling_std"] = data.rolling(window=window).std()
        features["rolling_min"] = data.rolling(window=window).min()
        features["rolling_max"] = data.rolling(window=window).max()
        return features.fillna(0)

    @staticmethod
    def autocorrelation_features(data: pd.Series, nlags: int = 10) -> pd.DataFrame:
        """
        Рассчитывает автокорреляционные признаки (ACF, PACF).

        :param data: Временной ряд (pandas Series)
        :param nlags: Количество лагов для вычисления
        :return: DataFrame с ACF и PACF признаками
        """
        features = pd.DataFrame(index=[0])
        acf_values = acf(data, nlags=nlags)
        pacf_values = pacf(data, nlags=nlags)

        for i in range(1, nlags + 1):
            features[f"acf_lag_{i}"] = acf_values[i]
            features[f"pacf_lag_{i}"] = pacf_values[i]

        return features.fillna(0)

    @staticmethod
    def seasonality_features(data: pd.Series, period: int = 7) -> pd.DataFrame:
        """
        Добавляет сезонные признаки (среднее, стандартное отклонение за период).

        :param data: Временной ряд (pandas Series)
        :param period: Период сезонности (например, 7 для недельной)
        :return: DataFrame с сезонными признаками
        """
        features = pd.DataFrame(index=data.index)
        features["seasonal_mean"] = data.rolling(window=period).mean()
        features["seasonal_std"] = data.rolling(window=period).std()
        return features.fillna(0)

    @staticmethod
    def volatility_features(data: pd.Series, window: int = 5) -> pd.DataFrame:
        """
        Рассчитывает волатильность (коэффициент вариации).

        :param data: Временной ряд (pandas Series)
        :param window: Длина окна
        :return: DataFrame с признаками волатильности
        """
        features = pd.DataFrame(index=data.index)
        features["volatility"] = data.rolling(window=window).std() / data.rolling(window=window).mean()
        return features.fillna(0)

    @staticmethod
    def fourier_features(data: pd.Series, periods: list = [7, 30]) -> pd.DataFrame:
        """
        Извлекает Fourier-признаки для сезонности.

        :param data: Временной ряд (pandas Series)
        :param periods: Список периодов (например, [7, 30] для недельной и месячной сезонности)
        :return: DataFrame с Fourier-признаками
        """
        features = pd.DataFrame(index=data.index)
        for period in periods:
            features[f"sin_{period}"] = np.sin(2 * np.pi * data.index.dayofyear / period)
            features[f"cos_{period}"] = np.cos(2 * np.pi * data.index.dayofyear / period)
        return features.fillna(0)

    @staticmethod
    def wavelet_features(data: pd.Series, wavelet: str = "db4") -> pd.DataFrame:
        """
        Извлекает wavelet-признаки с помощью дискретного вейвлет-преобразования.

        :param data: Временной ряд (pandas Series)
        :param wavelet: Тип вейвлета
        :return: DataFrame с вейвлет-признаками
        """
        coeffs = pywt.wavedec(data, wavelet, level=3)
        features = {f"wavelet_{i}": np.mean(np.abs(coeff)) for i, coeff in enumerate(coeffs)}

        return pd.DataFrame(features, index=[0])


class FeatureSelector:
    """
    Отбор признаков для временных рядов.
    """

    @staticmethod
    def select_k_best(X: pd.DataFrame, y: pd.Series, k: int = 5) -> pd.DataFrame:
        """
        Отбор K лучших признаков с помощью F-статистики.

        :param X: Матрица признаков
        :param y: Целевая переменная
        :param k: Количество признаков для отбора
        :return: DataFrame с отобранными признаками
        """
        selector = SelectKBest(score_func=f_regression, k=k)
        selector.fit(X, y)
        selected_features = X.columns[selector.get_support()]
        return X[selected_features]

    @staticmethod
    def recursive_feature_elimination(X: pd.DataFrame, y: pd.Series, n_features: int = 5) -> pd.DataFrame:
        """
        Рекурсивный отбор признаков (RFE) с помощью случайного леса.

        :param X: Матрица признаков
        :param y: Целевая переменная
        :param n_features: Количество отбираемых признаков
        :return: DataFrame с отобранными признаками
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(model, n_features_to_select=n_features)
        selector.fit(X, y)
        selected_features = X.columns[selector.support_]
        return X[selected_features]

    @staticmethod
    def shap_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Оценка важности признаков с помощью SHAP.

        :param X: Матрица признаков
        :param y: Целевая переменная
        :return: DataFrame с оценкой важности
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        shap_df = pd.DataFrame({'feature': X.columns, 'importance': np.abs(shap_values.values).mean(axis=0)})
        return shap_df.sort_values(by="importance", ascending=False)

    @staticmethod
    def lasso_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Отбор признаков с помощью Lasso (L1-регуляризация).

        :param X: Матрица признаков
        :param y: Целевая переменная
        :return: DataFrame с отобранными признаками
        """
        lasso = LassoCV(cv=5).fit(X, y)
        selected_features = X.columns[np.abs(lasso.coef_) > 1e-5]
        return X[selected_features]

    @staticmethod
    def xgboost_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Оценка важности признаков с помощью XGBoost.

        :param X: Матрица признаков
        :param y: Целевая переменная
        :return: DataFrame с важностью признаков
        """
        model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100)
        model.fit(X, y)
        importance = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_})
        return importance.sort_values(by="importance", ascending=False)

    @staticmethod
    def pca_feature_reduction(X: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        Снижение размерности признаков с помощью PCA.

        :param X: Матрица признаков
        :param n_components: Число главных компонент
        :return: DataFrame с преобразованными признаками
        """
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        return pd.DataFrame(X_pca, columns=[f"PCA_{i+1}" for i in range(n_components)])


if __name__ == "__main__":
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
    data = pd.Series(np.cumsum(np.random.randn(100)), index=dates)

    feature_gen = FeatureGenerator()
    rolling_feats = feature_gen.rolling_features(data)
    seasonality_feats = feature_gen.seasonality_features(data)
    volatility_feats = feature_gen.volatility_features(data)

    X = pd.concat([rolling_feats, seasonality_feats, volatility_feats], axis=1)
    y = data.shift(-1).fillna(0)

    selector = FeatureSelector()
    selected_features = selector.lasso_feature_selection(X, y)
    print("Выбранные признаки (Lasso):", selected_features.columns.tolist())

    pca_features = selector.pca_feature_reduction(X, n_components=3)
    print("PCA-признаки:", pca_features.head())
