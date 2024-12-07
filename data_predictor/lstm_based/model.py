import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


# Custom Loss Function
def custom_loss(y_true, y_pred):
    """
    Function to calculate vehicle count loss and speed loss.

    Parameters:
        y_true(tensor): true values
        y_pred(tensor): predicted values

    Returns:
        loss: total loss combining count loss and speed loss
    """
    vehicle_counts = tf.reduce_sum(y_true[:, :-1], axis=1)
    speeds = y_true[:, -1]
    predicted_speeds = y_pred[:, -1]
    speed_loss = tf.where(
        vehicle_counts >= tf.reduce_mean(vehicle_counts),
        (speeds - predicted_speeds) ** 2,
        tf.abs(predicted_speeds - speeds),
    )
    count_loss = tf.reduce_mean(tf.square(y_true[:, :-1] - y_pred[:, :-1]), axis=1)
    return tf.reduce_mean(speed_loss + count_loss)


# LSTM Prediction Model trained on Ecomob data
class EcomobLSTMmodel:
    def __init__(self, model_path=None):
        """
        LSTM Traffic Intensity Prediction Model trained on Ecomob data.

        Parameters:
            model_path(string): the path to the pre-trained model.
        """
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        if model_path is not None:
            self.model = tf.keras.models.load_model(
                model_path, custom_objects={"self.custom_loss": custom_loss}
            )

    def __normalize_data(self, data):
        """
        Function to normalize data

        Parameters:
            data(DataFrame): the unormalized dataset

        Return:
            scaled_data(DataFrame): the normalized dataset
        """
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def __unnormalize_data(self, data):
        """
        Function to unnormalize data

        Parameters:
            data(DataFrame): the normalized dataset

        Return:
            unscaled_data(DataFrame): the unormalized dataset
        """
        unscaled_data = self.scaler.inverse_transform(data)
        return unscaled_data

    def __create_dataset(self, data, look_back=1):
        """
        Transform a time series into a prediction dataset

        Parameters:
            data(DataFrame): the normalized dataset
            look_back(int): the size of window for prediction
        Return:
            X(numpy.array): time series input for prediction
            Y(numpy.array): true time seires
        """
        X, Y = [], []
        for i in range(len(data) - look_back):
            a = data[i : (i + look_back), :]
            X.append(a)
            Y.append(data[i + look_back, :])
        return np.array(X), np.array(Y)

    def train_model(
        self,
        data,
        units=40,
        lr=0.01,
        epochs=20,
        batch_size=1,
        look_back=1,
        model_name=None,
    ):
        """
        Train LSTM Prediction model

        Parameters:
            data(DataFrame): time series dataset
            units(int): number of hidden units
            lr(float): learning rate
            epochs(int): number of epoch
            batch_size(int): batch size
            look_back(int): the size of time window for prediction
            model_name(string): the output model name
        """
        df = data.copy()

        # Prepare data
        scaled_data = self.__normalize_data(df)
        X, Y = self.__create_dataset(scaled_data, look_back)
        X_train = X.reshape(X.shape[0], look_back, df.shape[1])
        Y_train = Y

        # Build the LSTM model
        model = Sequential(
            [
                LSTM(
                    units,
                    return_sequences=True,
                    input_shape=(look_back, df.shape[1]),
                ),
                LSTM(units),
                Dense(df.shape[1], activation="relu"),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=lr), loss=custom_loss)

        # Train the LSTM model
        model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

        # Save LSTM model
        if model_name is not None:
            model_name = (
                f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.keras"
            )
            model.save(model_name)

        self.model = model

    def predict_steps(self, data, steps, look_back=1):
        """
        Generate predictions

        Parameters:
            data(DataFrame): time seires data
            steps(int): the number of next records
            look_back(int): the size of time window for prediction
        """
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
