from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


class DeepLearningModels:
    @staticmethod
    def create_lstm_model(input_shape, layers_config, dropout_rate=0.2, learning_rate=0.001):
        """
        Create and compile an LSTM model with customizable layer configurations.
        :param input_shape: Shape of the input data (number of features, number of timesteps).
        :param layers_config: List of dictionaries defining each layer's configuration.
        :param dropout_rate: Dropout rate to prevent overfitting.
        :param learning_rate: Learning rate for the Adam optimizer.
        :return: Compiled LSTM model.
        """
        model = Sequential()

        for i, layer_config in enumerate(layers_config):
            units = layer_config.get('units', 50)
            activation = layer_config.get('activation', 'relu')
            return_sequences = layer_config.get('return_sequences', True) if i < len(layers_config) - 1 else False

            if i == 0:
                model.add(LSTM(units=units, activation=activation, input_shape=input_shape,
                               return_sequences=return_sequences))
            else:
                model.add(LSTM(units=units, activation=activation, return_sequences=return_sequences))

            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        model.add(Dense(1))  # Output layer

        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    @staticmethod
    def create_gru_model(input_shape, layers_config, dropout_rate=0.2, learning_rate=0.001):
        """
        Create and compile a GRU model with customizable layer configurations.
        :param input_shape: Shape of the input data (number of features, number of timesteps).
        :param layers_config: List of dictionaries defining each layer's configuration.
        :param dropout_rate: Dropout rate to prevent overfitting.
        :param learning_rate: Learning rate for the Adam optimizer.
        :return: Compiled GRU model.
        """
        model = Sequential()

        for i, layer_config in enumerate(layers_config):
            units = layer_config.get('units', 50)
            activation = layer_config.get('activation', 'relu')
            return_sequences = layer_config.get('return_sequences', True) if i < len(layers_config) - 1 else False

            if i == 0:
                model.add(
                    GRU(units=units, activation=activation, input_shape=input_shape, return_sequences=return_sequences))
            else:
                model.add(GRU(units=units, activation=activation, return_sequences=return_sequences))

            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))

        model.add(Dense(1))  # Output layer

        # Compile model
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
        return model

    @staticmethod
    def create_transformer_model(input_shape, layers_config, dropout_rate=0.2):
        """
        Create and compile a Transformer model with customizable layer configurations.
        :param input_shape: Shape of the input data (number of features, number of timesteps).
        :param layers_config: List of dictionaries defining each layer's configuration.
        :param dropout_rate: Dropout rate to prevent overfitting.
        :return: Compiled Transformer model.
        """
        inputs = Input(shape=input_shape)
        x = inputs

        for layer_config in layers_config:
            num_heads = layer_config.get('num_heads', 4)
            ff_dim = layer_config.get('ff_dim', 64)
            num_layers = layer_config.get('num_layers', 2)

            for _ in range(num_layers):
                x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)(x, x)
                x = tf.keras.layers.Dropout(dropout_rate)(x)
                x = tf.keras.layers.LayerNormalization()(x)
                x = tf.keras.layers.Dense(ff_dim, activation='relu')(x)
                x = tf.keras.layers.LayerNormalization()(x)

        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.Dense(1)(x)  # Output layer
        model = tf.keras.models.Model(inputs, x)

        model.compile(optimizer=Adam(), loss='mean_squared_error')
        return model

    @staticmethod
    def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """
        Train the model with early stopping.
        :param model: The deep learning model to train.
        :param X_train: Training input data.
        :param y_train: Training target data.
        :param X_val: Validation input data.
        :param y_val: Validation target data.
        :param epochs: Number of epochs to train.
        :param batch_size: Batch size for training.
        :return: Trained model.
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                  callbacks=[early_stopping], verbose=1)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        """
        Evaluate the model on test data.
        :param model: Trained model.
        :param X_test: Test input data.
        :param y_test: Test target data.
        :return: MAE (Mean Absolute Error).
        """
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        return mae, predictions

    @staticmethod
    def plot_predictions(y_test, predictions):
        """
        Plot true vs predicted values.
        :param y_test: Actual target values.
        :param predictions: Predicted target values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='True Values')
        plt.plot(predictions, label='Predictions')
        plt.legend()
        plt.title('True vs Predicted Values')
        plt.show()
