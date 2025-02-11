import pandas as pd
import numpy as np
from ml_models import MLModels
from deep_learning_models import DeepLearningModels
from sklearn.model_selection import train_test_split


class AutoML:
    @staticmethod
    def find_best_model(df, features, target, use_deep_learning=False):
        """
        Automatically finds the best model for the given dataset.
        :param df: Input DataFrame.
        :param features: List of feature column names.
        :param target: Target column name.
        :param use_deep_learning: Whether to include deep learning models in the search.
        :return: Best model, its predictions, and MAE score.
        """
        X = df[features]
        y = df[target]

        # Split the data into training, validation, and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

        results = {}

        # ML Models
        models = {
            "RandomForest": MLModels.random_forest,
            "XGBoost": MLModels.xgboost_model,
            "CatBoost": MLModels.catboost_model,
        }

        for name, model_func in models.items():
            model, predictions, mae = model_func(df, features, target)
            results[name] = {"model": model, "predictions": predictions, "mae": mae}
            print(f"{name} MAE: {mae}")

        # Deep Learning Models (if enabled)
        if use_deep_learning:
            # Reshape for deep learning models (samples, timesteps, features)
            X_train_dl = np.expand_dims(X_train.values, axis=1)
            X_val_dl = np.expand_dims(X_val.values, axis=1)
            X_test_dl = np.expand_dims(X_test.values, axis=1)

            deep_models = {
                "LSTM": DeepLearningModels.create_lstm_model,
                "GRU": DeepLearningModels.create_gru_model,
                "Transformer": DeepLearningModels.create_transformer_model,
            }

            # Define layers config for DL models
            lstm_gru_config = [
                {"units": 50, "activation": "relu", "return_sequences": True},
                {"units": 50, "activation": "relu", "return_sequences": False},
            ]
            transformer_config = [{"num_heads": 4, "ff_dim": 64, "num_layers": 2}]

            for name, model_func in deep_models.items():
                if name in ["LSTM", "GRU"]:
                    model = model_func(input_shape=X_train_dl.shape[1:], layers_config=lstm_gru_config, dropout_rate=0.2)
                else:
                    model = model_func(input_shape=X_train_dl.shape[1:], layers_config=transformer_config, dropout_rate=0.2)

                trained_model = DeepLearningModels.train_model(model, X_train_dl, y_train, X_val_dl, y_val)
                mae, predictions = DeepLearningModels.evaluate_model(trained_model, np.expand_dims(X_test.values, axis=1), y_test)

                results[name] = {"model": trained_model, "predictions": predictions, "mae": mae}
                print(f"{name} MAE: {mae}")

        # Find the best model
        best_model_name = min(results, key=lambda k: results[k]["mae"])
        best_model_info = results[best_model_name]

        print(f"\nBest Model: {best_model_name} with MAE {best_model_info['mae']}")
        return best_model_info["model"], best_model_info["predictions"], best_model_info["mae"]
