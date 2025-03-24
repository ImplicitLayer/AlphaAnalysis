import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import xgboost as xgb
import matplotlib.pyplot as plt


class MLModels:
    @staticmethod
    def random_forest(X, y, test_size=0.2, n_estimators=100, random_state=42):
        """
        Train and forecast using Random Forest Regressor.
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :param n_estimators: Number of trees in the forest.
        :param random_state: Random seed for reproducibility.
        :return: Trained model, predictions, and MAE.
        """
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
    def xgboost_model(X, y, test_size=0.2, learning_rate=0.1, n_estimators=100, max_depth=3):
        """
        Train and forecast using XGBoost.
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :param learning_rate: Learning rate for boosting.
        :param n_estimators: Number of boosting rounds.
        :param max_depth: Maximum depth of each tree.
        :return: Trained model, predictions, and MAE.
        """
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
    def linear_regression(X, y, test_size=0.2):
        """
        Train and forecast using Linear Regression.
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :return: Trained model, predictions, and MAE.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = LinearRegression()

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def knn(X, y, test_size=0.2, n_neighbors=2):
        """
        Train and forecast using k-Nearest Neighbors.
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :param n_neighbors: Number of neighbors to use.
        :return: Trained model, predictions, and MAE.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = KNeighborsRegressor(n_neighbors=n_neighbors)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def decision_tree(X, y, test_size=0.2, max_depth=None):
        """
        Train and forecast using Decision Tree Regressor.
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :param max_depth: Maximum depth of the tree.
        :return: Trained model, predictions, and MAE.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = DecisionTreeRegressor(max_depth=max_depth)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, predictions)

        return model, predictions, mae

    @staticmethod
    def svm(X, y, test_size=0.2, kernel='rbf', C=1.0):
        """
        Train and forecast using Support Vector Machine (SVM).
        :param X: Features matrix.
        :param y: Target vector.
        :param test_size: Proportion of the dataset to be used for testing.
        :param kernel: Kernel type to use in SVM.
        :param C: Regularization parameter.
        :return: Trained model, predictions, and MAE.
        """
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Initialize the model
        model = SVR(kernel=kernel, C=C)

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
        if isinstance(model, RandomForestRegressor):
            importances = model.feature_importances_
        elif isinstance(model, xgb.XGBRegressor):
            importances = model.feature_importances_
        elif isinstance(model, DecisionTreeRegressor):
            importances = model.feature_importances_
        elif isinstance(model, KNeighborsRegressor):
            importances = [1 / len(features)] * len(features)  # Equal importance for all features in KNN
        elif isinstance(model, LinearRegression):
            importances = model.coef_
        elif isinstance(model, SVR):
            importances = model.coef0
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


# Пример использования:
if __name__ == "__main__":
    # Пример данных
    data = {
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [10, 20, 30, 40, 50]
    }

    df = pd.DataFrame(data)
    X = df[['feature1', 'feature2']]
    y = df['target']

    # Пример использования классов
    model, predictions, mae = MLModels.random_forest(X, y)
    print("Random Forest MAE:", mae)

    model, predictions, mae = MLModels.xgboost_model(X, y)
    print("XGBoost MAE:", mae)

    model, predictions, mae = MLModels.linear_regression(X, y)
    print("Linear Regression MAE:", mae)

    model, predictions, mae = MLModels.knn(X, y)
    print("KNN MAE:", mae)

    model, predictions, mae = MLModels.decision_tree(X, y)
    print("Decision Tree MAE:", mae)

    model, predictions, mae = MLModels.svm(X, y)
    print("SVM MAE:", mae)

    # Визуализация важности признаков для RandomForest
    MLModels.feature_importance(model, X.columns)
