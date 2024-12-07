import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

model_path = "./models/"


class EcomobLSTMmodel:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        def custom_loss(y_true, y_pred):
            vehicle_counts = tf.reduce_sum(y_true[:, :-1], axis=1)
            speeds = y_true[:, -1]
            predicted_speeds = y_pred[:, -1]
            speed_loss = tf.where(
                vehicle_counts >= tf.reduce_mean(vehicle_counts),
                (speeds - predicted_speeds) ** 2,
                tf.abs(predicted_speeds - speeds),
            )
            count_loss = tf.reduce_mean(
                tf.square(y_true[:, :-1] - y_pred[:, :-1]), axis=1
            )
            return tf.reduce_mean(speed_loss + count_loss)

        if model_path is not None:
            print(model_path)
            self.model = tf.keras.models.load_model(
                model_path, custom_objects={"self.custom_loss": custom_loss}
            )

    def normalize_data(self, data):
        # Normalize data
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def unnormalize_data(self, data):
        # Unormalize data
        unscaled_data = self.scaler.inverse_transform(data)
        return unscaled_data

    def create_dataset(self, data, look_back=1):
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i : (i + look_back), :]
            X.append(a)
            Y.append(data[i + look_back, :])
        return np.array(X), np.array(Y)

    def train_model(
        self, data, lr=0.01, epochs=20, batch_size=1, look_back=1, save=True
    ):

        def custom_loss(y_true, y_pred):
            vehicle_counts = tf.reduce_sum(y_true[:, :-1], axis=1)
            speeds = y_true[:, -1]
            predicted_speeds = y_pred[:, -1]
            speed_loss = tf.where(
                vehicle_counts >= tf.reduce_mean(vehicle_counts),
                (speeds - predicted_speeds) ** 2,
                tf.abs(predicted_speeds - speeds),
            )
            count_loss = tf.reduce_mean(
                tf.square(y_true[:, :-1] - y_pred[:, :-1]), axis=1
            )
            return tf.reduce_mean(speed_loss + count_loss)

        df = data.copy()

        scaled_data = self.normalize_data(df)
        X, Y = self.create_dataset(scaled_data, look_back)
        X_train = X.reshape(X.shape[0], look_back, df.shape[1])
        Y_train = Y

        # Build the LSTM model
        model = Sequential(
            [
                LSTM(
                    40,
                    return_sequences=True,
                    input_shape=(look_back, df.shape[1]),
                ),
                LSTM(40),
                Dense(df.shape[1], activation="relu"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss)

        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

        if save:
            model_name = f"{model_path}lstm_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.keras"
            model.save(model_name)

        self.model = model

    def predict_next_24hr(self, data, look_back=1):
        df = data.copy()

        scaled_data = self.normalize_data(data)
        last_known_data = scaled_data[-look_back:]

        predictions = []
        for _ in range(24):
            last_known_data = last_known_data.reshape((1, look_back, df.shape[1]))
            prediction = self.model.predict(last_known_data)
            prediction = np.abs(prediction)
            predictions.append(prediction[0])
            last_known_data = np.append(
                last_known_data[:, 1:, :],
                prediction.reshape((1, 1, df.shape[1])),
                axis=1,
            )

        # Invert the prediction to the original scaled
        predictions = self.unnormalize_data(predictions)
        prediction_dates = pd.date_range(start=df.index[-1], periods=24, freq="h")

        # Create DataFrame for the predictions
        predictions_df = pd.DataFrame(
            predictions, index=prediction_dates, columns=data.columns
        )
        predictions_df.reset_index(inplace=True)
        predictions_df.rename(columns={"index": "DateTime"}, inplace=True)
        return predictions_df

    def predict_steps(self, data, steps, look_back=1):
        df = data.copy()

        scaled_data = self.normalize_data(data)
        last_known_data = scaled_data[-look_back:]

        predictions = []
        for _ in range(steps):
            last_known_data = last_known_data.reshape((1, look_back, df.shape[1]))
            prediction = self.model.predict(last_known_data)
            prediction = np.abs(prediction)
            predictions.append(prediction[0])
            last_known_data = np.append(
                last_known_data[:, 1:, :],
                prediction.reshape((1, 1, df.shape[1])),
                axis=1,
            )

        # Invert the prediction to the original scaled
        predictions = self.unnormalize_data(predictions)
        prediction_dates = pd.date_range(start=df.index[-1], periods=steps, freq="h")

        # Create DataFrame for the predictions
        predictions_df = pd.DataFrame(
            predictions, index=prediction_dates, columns=data.columns
        )
        predictions_df.reset_index(inplace=True)
        predictions_df.rename(columns={"index": "DateTime"}, inplace=True)
        return predictions_df
