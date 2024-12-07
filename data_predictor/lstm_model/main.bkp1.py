import numpy as np
import pandas as pd
from pandas.core.api import DataFrame
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

import requests

import io
import pickle
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
import uvicorn
from typing import Optional

import logging

# Logging Config
logging.basicConfig(filename="main.log", level=logging.DEBUG)


# Global Variable
ecomob_model = None


def fetch_data_c2jn_broker(
    client_id: str, client_secret: str, area_code: str, timeAt: str
):
    auth_url = (
        "https://sso.c2jn.fr/auth/realms/smart-city/protocol/openid-connect/token"
    )
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    body = {
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "client_credentials",
    }

    try:
        response = requests.post(auth_url, headers=headers, data=body)
        logging.debug(response.status_code)
        response_json = response.json()
        access_token = response_json["access_token"]

        api_url = f"https://api-gw.stellio.c2jn.fr/ngsi-ld/v1/temporal/entities/urn:ngsi-ld:TrafficFlowObserved:{area_code}?timerel=after&timeAt={timeAt}&options=temporalValues"
        logging.debug(api_url)
        headers = {
            "Authorization": f"Bearer {access_token}",
            "NGSILD-Tenant": "urn:ngsi-ld:tenant:smart-city",
        }

        try:
            response = requests.get(api_url, headers=headers)
            logging.debug(response.status_code)
            data = response.json()
            return data
        except requests.HTTPError:
            print("Failed to get data.")

    except requests.HTTPError:
        print("Failed to authenticate.")


def parse_ecomob_data(data):
    df = pd.DataFrame(
        columns=[
            "DateTime",
            "HeavyVehicle Value",
            "CommercialVehicle Value",
            "LightVehicle Value",
            "CongestionIndex",
            "CO2Equivalent",
            "AverageVehicleSpeed",
        ]
    )

    # Extract data from json data
    heavy_vehicle_data = data["https://vocab.egm.io/flow"][0]["values"]
    commercial_vehicle_data = data.get("https://vocab.egm.io/flow", [{}])[2].get(
        "values", []
    )
    light_vehicle_data = data.get("https://vocab.egm.io/flow", [{}])[3].get(
        "values", []
    )
    congestion_index_data = data.get(
        "https://smartdatamodels.org/dataModel.Transportation/congestionIndex", {}
    ).get("values", [])
    co2_equivalent_data = data.get(
        "https://smartdatamodels.org/dataModel.Transportation/co2Equivalent", {}
    ).get("values", [])
    average_vehicle_speed_data = data.get(
        "https://smartdatamodels.org/dataModel.Transportation/averageVehicleSpeed", {}
    ).get("values", [])

    # Prepare data to append to DataFrame
    for entry in light_vehicle_data:
        date_time = entry[1]
        heavy_weight_value = heavy_weight_value = next(
            (item[0] for item in heavy_vehicle_data if item[1] == date_time), None
        )
        commercial_vehicle_value = next(
            (item[0] for item in commercial_vehicle_data if item[1] == date_time), None
        )
        light_vehicle_value = next(
            (item[0] for item in light_vehicle_data if item[1] == date_time), None
        )
        congestion_index = next(
            (item[0] for item in congestion_index_data if item[1] == date_time), None
        )
        co2_equivalent = next(
            (item[0] for item in co2_equivalent_data if item[1] == date_time), None
        )
        average_vehicle_speed = next(
            (item[0] for item in average_vehicle_speed_data if item[1] == date_time),
            None,
        )

        # Append to DataFrame
        df.loc[len(df)] = {
            # "RoadID": road_id,
            "DateTime": date_time,
            "HeavyVehicle Value": heavy_weight_value,
            "CommercialVehicle Value": commercial_vehicle_value,
            "LightVehicle Value": light_vehicle_value,
            "CongestionIndex": congestion_index,
            "CO2Equivalent": co2_equivalent,
            "AverageVehicleSpeed": average_vehicle_speed,
        }
    return df


def train_Ecomob_LSTM_model(
    ecomob_data: DataFrame, epochs=20, batch_size=1, look_back=72, save=False
):
    global ecomob_model

    df = ecomob_data.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
    df.set_index("DateTime", inplace=True)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Function to create dataset matrix
    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i : (i + look_back), :]
            X.append(a)
            Y.append(dataset[i + look_back, :])
        return np.array(X), np.array(Y)

    X, Y = create_dataset(scaled_data, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = X.reshape((X.shape[0], look_back, df.shape[1]))
    Y_train = Y

    # Custom loss function to handle the relationship between vehicle counts and speeds
    def custom_loss(y_true, y_pred):
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

    # Build the LSTM model
    model = Sequential(
        [
            LSTM(60, return_sequences=True, input_shape=(look_back, df.shape[1])),
            LSTM(60),
            Dense(df.shape[1]),
        ]
    )
    model.compile(optimizer=Adam(learning_rate=0.01), loss=custom_loss)

    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    if save:
        model_name = f"lstm_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.keras"
        model.save(model_name)
    ecomob_model = model


def predict_next_day_with_Ecomob_model(model, ecomob_data, look_back, model_path=None):
    df = ecomob_data.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
    df.set_index("DateTime", inplace=True)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    def custom_loss(y_true, y_pred):
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

    if model_path is not None:
        model = tf.keras.models.load_model(
            model_path, custom_objects={"custom_loss": custom_loss}
        )

    predictions = []

    print("Shape: ", df.shape[1])
    print("Scale shape before: ", scaled_data.shape)

    last_known_data = scaled_data[-look_back:]

    print("Sclae shape after", scaled_data.shape)

    for _ in range(24):
        last_known_data = last_known_data.reshape((1, look_back, df.shape[1]))
        prediction = model.predict(last_known_data)
        prediction = np.abs(prediction)
        predictions.append(prediction[0])
        last_known_data = np.append(
            last_known_data[:, 1:, :], prediction.reshape((1, 1, df.shape[1])), axis=1
        )

    # Invert the prediction to the original scaled
    predictions = scaler.inverse_transform(predictions)
    prediction_dates = pd.date_range(start=df.index[-1], periods=24, freq="h")

    # Create DataFrame for the predictions
    predictions_df = pd.DataFrame(
        predictions, index=prediction_dates, columns=df.columns
    )
    predictions_df.reset_index(inplace=True)
    predictions_df.rename(columns={"index": "DateTime"}, inplace=True)
    return predictions_df


def data_convert_to_ngsild(data: DataFrame, area_code: str):
    df = data.copy()

    def format_datetime(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Initialize the main structure
    traffic_data = {
        "id": f"urn:ngsi-ld:TrafficFlowObserved:{area_code}",
        "type": "TrafficFlowObserved",
        "refRoad": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:Road:{area_code}",
        },
        "temporalResolution": {"type": "Property", "value": "PT1H"},
        "predictedAverageVehicleSpeed": [],
        "predictedCongestionIndex": [],
        "predictedFlow": [],
        "PredictedCo2Equivalent": [],
        "@context": [
            "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld"
        ],
    }

    # Populate data structure
    for index, row in df.iterrows():
        observed_at = format_datetime(row["DateTime"])

        # Populate the averageVehicleSpeed
        traffic_data["predictedAverageVehicleSpeed"].append(
            {
                "type": "Property",
                "value": row["AverageVehicleSpeed"],
                "observedAt": observed_at,
                "unitCode": "KMH",
            }
        )

        # Populate the congestionIndex
        traffic_data["predictedCongestionIndex"].append(
            {
                "type": "Property",
                "value": row["CongestionIndex"],
                "observedAt": observed_at,
                "unitCode": "P1",
            }
        )

        # Populate the co2Equivalent
        traffic_data["PredictedCo2Equivalent"].append(
            {
                "type": "Property",
                "value": row["CO2Equivalent"],
                "observedAt": observed_at,
                "unitCode": "TNE",
            }
        )

        # Populate the predicted flow for each vehicle type, use appropriate column names from your DataFrame
        traffic_data["predictedFlow"].extend(
            [
                {
                    "type": "Property",
                    "value": row["LightVehicle Value"]
                    + row["CommercialVehicle Value"]
                    + row[
                        "HeavyVehicle Value"
                    ],  # This should be a column in your DataFrame
                    "observedAt": observed_at,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:All",
                    "vehicleType": {"type": "Property", "value": "All"},
                },
                {
                    "type": "Property",
                    "value": row["LightVehicle Value"],
                    "observedAt": observed_at,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:LightVehicle",
                    "vehicleType": {"type": "Property", "value": "Light Vehicle"},
                },
                {
                    "type": "Property",
                    "value": row["CommercialVehicle Value"],
                    "observedAt": observed_at,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:CommercialVehicle",
                    "vehicleType": {"type": "Property", "value": "Commercial Vehicle"},
                },
                {
                    "type": "Property",
                    "value": row["HeavyVehicle Value"],
                    "observedAt": observed_at,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:HeavyWeight",
                    "vehicleType": {"type": "Property", "value": "Heavy Weight"},
                },
            ]
        )
        return traffic_data


# FastAPI
app = FastAPI()


@app.get("/train_ecomob_model")
async def train_ecomob_model(
    client_id: str,
    client_secret: str,
    area_code: str,
    timeAt: str,
    epochs=20,
    batch_size=1,
    look_back=72,
):
    background_tasks = BackgroundTasks()

    if (
        not client_id
        or not client_secret
        or not area_code
        or not timeAt
        or not epochs
        or not batch_size
        or not look_back
    ):
        raise HTTPException(status_code=400, detail="Missing required parameters.")

    data = fetch_data_c2jn_broker(client_id, client_secret, area_code, timeAt)
    if data is not None:
        parse_data = parse_ecomob_data(data)
        background_tasks.add_task(
            train_Ecomob_LSTM_model,
            parse_data,
            epochs=20,
            batch_size=1,
            look_back=72,
            save=True,
        )

        # ecomob_model = train_Ecomob_LSTM_model(
        #     parse_data, epochs=20, batch_size=1, look_back=72, save=True
        # )
        return {"message": "Training model for Ecomob data has started."}
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/predict_next_day_with_ecomob_model")
async def predict_next_day_with_ecomob_model(
    client_id: str,
    client_secret: str,
    area_code: str,
    timeAt: str,
    look_back=72,
    model_path: Optional[str] = None,
):
    global ecomob_model

    data = fetch_data_c2jn_broker(client_id, client_secret, area_code, timeAt)
    if data is None:
        return {"message": "Failed to get previous data or data not available."}
    parse_data = parse_ecomob_data(data)

    if model_path is not None:
        predictions_df = predict_next_day_with_Ecomob_model(
            None,
            parse_data,
            look_back=72,
            model_path=model_path,
        )
    else:
        if ecomob_model is None:
            return {"message": "No pretrained model available."}
        else:
            predictions_df = predict_next_day_with_Ecomob_model(
                ecomob_model,
                parse_data,
                look_back=72,
            )

    ngsild_data = data_convert_to_ngsild(predictions_df, area_code)
    return ngsild_data


if __name__ == "__main__":
    client_id = "imt"
    client_secret = "B9ujTPZipkJcRG9xFuxTMFsp3TvdmHzN"
    area_code = "ILM_92130_2385"
    timeAt = "2023-05-01T07:00:00Z"

    data = fetch_data_c2jn_broker(client_id, client_secret, area_code, timeAt)
    parse_data = parse_ecomob_data(data)
    # # ecomob_model = train_Ecomob_LSTM_model(
    # #    parse_data, epochs=20, batch_size=1, look_back=72, save=True
    # # )
    predictions_df = predict_next_day_with_Ecomob_model(
        None,
        parse_data,
        look_back=72,
        model_path="lstm_mdoel_2024-04-23_13-39-57.keras",
    )
    ngsild_data = data_convert_to_ngsild(predictions_df, area_code)
    print(ngsild_data)
    # uvicorn.run(app, host="0.0.0.0", port=9000)
