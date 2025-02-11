import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import catboost as cb


class MLModels:
    @staticmethod
    def random_forest(df, features, target, test_size=0.2, n_estimators=100, random_state=42):
        """
        Train and forecast using Random Forest Regressor.
        :param df: Input DataFrame.
        :param features: List of feature column names.
        :param target: Target column name.
        :param test_size: Proportion of the dataset to be used for testing.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Random seed for reproducibility.
        :return: Trained model, predictions, and MAE.
        """
        X = df[features]
        y = df[target]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # Initialize the model
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def xgboost_model(df, features, target, test_size=0.2, learning_rate=0.1, n_estimators=100, max_depth=3):
        """
        Train and forecast using XGBoost.
        :param df: Input DataFrame.
        :param features: List of feature column names.
        :param target: Target column name.
        :param test_size: Proportion of the dataset to be used for testing.
        :param learning_rate: Learning rate for boosting.
        :param n_estimators: Number of boosting rounds.
        :param max_depth: Maximum depth of each tree.
        :return: Trained model, predictions, and MAE.
        """
        X = df[features]
        y = df[target]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def catboost_model(df, features, target, test_size=0.2, learning_rate=0.1, iterations=1000, depth=6):
        """
        Train and forecast using CatBoost.
        :param df: Input DataFrame.
        :param features: List of feature column names.
        :param target: Target column name.
        :param test_size: Proportion of the dataset to be used for testing.
        :param learning_rate: Learning rate for boosting.
        :param iterations: Number of boosting iterations.
        :param depth: Depth of the trees.
        :return: Trained model, predictions, and MAE.
        """
        X = df[features]
        y = df[target]

        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = cb.CatBoostRegressor(learning_rate=learning_rate, iterations=iterations, depth=depth, cat_features=[],
                                     silent=True)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def feature_importance(model, features):
        """
        Plot feature importance for the trained model.
        :param model: Trained machine learning model.
        :param features: List of feature column names.
        :return: Feature importance plot.
        """
        import matplotlib.pyplot as plt

        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
        elif isinstance(model, xgb.XGBRegressor):
            importances = model.feature_importances_
        elif isinstance(model, cb.CatBoostRegressor):
            importances = model.get_feature_importance()
        else:
            raise ValueError("Unsupported model type.")

        # Create a DataFrame for feature importances
        feature_importance_df = pd.DataFrame({
            'feature': features,
            'importance': importances
        })
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

        # Plot feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()

        return feature_importance_df
