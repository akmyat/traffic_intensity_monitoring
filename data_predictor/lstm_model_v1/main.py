from operator import ge
from model import EcomobLSTMmodel, model_path

import requests

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
import uvicorn
from typing import Optional

import os
import sys
from datetime import datetime
import pandas as pd
from pandas.core.api import DataFrame

import uuid
import json

# Global
MODEL_PATH = os.environ.get("MODEL_PATH", None)
CLIENT_ID = os.environ.get("CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", None)
ACCESS_TOKEN = None
MODEL_ID = os.environ.get("MODEL_ID", None)


# Ecomob
def authenticate(client_id, client_secret):
    global ACCESS_TOKEN

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

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Authentication Failed.")

        response_json = response.json()
        access_token = response_json["access_token"]
        ACCESS_TOKEN = access_token
        return True
    except HTTPException:
        print("Authentication Failed.")
        return False


def fetch_data_c2jn_broker(area_code: str):
    global ACCESS_TOKEN
    authenticate(CLIENT_ID, CLIENT_SECRET)
    access_token = ACCESS_TOKEN

    if access_token is None:
        print("Authenticate first")
        return None

    # api_url = f"https://api-gw.stellio.c2jn.fr/ngsi-ld/v1/temporal/entities/urn:ngsi-ld:TrafficFlowObserved:{area_code}?timerel=after&timeAt={timeAt}&options=temporalValues"
    api_url = f"https://api-gw.stellio.c2jn.fr/ngsi-ld/v1/temporal/entities/urn:ngsi-ld:TrafficFlowObserved:{area_code}?options=temporalValues"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "NGSILD-Tenant": "urn:ngsi-ld:tenant:smart-city",
    }

    try:
        response = requests.get(api_url, headers=headers)

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get data.")

        data = response.json()
        return data

    except HTTPException:
        print("Failed to get data.")
        return None


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
    df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
    df.set_index("DateTime", inplace=True)
    return df


def convert_to_ngsild_format(row, model_id):

    observed_at = row["DateTime"].strftime("%Y-%m-%dT%H:%M:%SZ")

    traffic_data = {
        "predictedFlow": [],
        "@context": [
            "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld"
        ],
    }

    traffic_data["predictedFlow"].extend(
        [
            {
                "type": "Property",
                "value": row["LightVehicle Value"]
                + row["CommercialVehicle Value"]
                + row["HeavyVehicle Value"],
                "unitCode": "E50",
                "datasetId": f"urn:ngsi-ld:Dataset:All:{model_id}",
                "observedAt": observed_at,
                "vehicleType": {"type": "Property", "value": "All"},
                "predictedBy": {
                    "type": "Relationship",
                    "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
                },
            },
            {
                "type": "Property",
                "value": row["LightVehicle Value"],
                "unitCode": "E50",
                "datasetId": f"urn:ngsi-ld:Dataset:LightVehicle:{model_id}",
                "observedAt": observed_at,
                "vehicleType": {"type": "Property", "value": "Light Vehicle"},
                "predictedBy": {
                    "type": "Relationship",
                    "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
                },
            },
            {
                "type": "Property",
                "value": row["CommercialVehicle Value"],
                "unitCode": "E50",
                "datasetId": f"urn:ngsi-ld:Dataset:CommercialVehicle:{model_id}",
                "observedAt": observed_at,
                "vehicleType": {"type": "Property", "value": "Commercial Vehicle"},
                "predictedBy": {
                    "type": "Relationship",
                    "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
                },
            },
            {
                "type": "Property",
                "value": row["HeavyVehicle Value"],
                "unitCode": "E50",
                "datasetId": f"urn:ngsi-ld:Dataset:HeavyWeight:{model_id}",
                "observedAt": observed_at,
                "vehicleType": {"type": "Property", "value": "Heavy Weight"},
                "predictedBy": {
                    "type": "Relationship",
                    "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
                },
            },
        ]
    )

    traffic_data["predictedAverageVehicleSpeed"] = {
        "type": "Property",
        "value": row["AverageVehicleSpeed"],
        "observedAt": observed_at,
        "unitCode": "KMH",
        "predictedBy": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
        },
    }

    traffic_data["predictedCongestionIndex"] = {
        "type": "Property",
        "value": row["CongestionIndex"],
        "observedAt": observed_at,
        "unitCode": "P1",
        "predictedBy": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
        },
    }

    traffic_data["predictedCo2Equivalent"] = {
        "type": "Property",
        "value": row["CO2Equivalent"],
        "observedAt": observed_at,
        "unitCode": "TNE",
        "predictedBy": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:DataServiceProcessing:TrafficFlowPrediction:{model_id}",
        },
    }

    ngsild_data = str(json.dumps(traffic_data, indent=4))

    return ngsild_data


def send_prediction_to_broker(area_code, ngsild_data):
    global ACCESS_TOKEN, CLIENT_ID, CLIENT_SECRET

    if not authenticate(CLIENT_ID, CLIENT_SECRET):
        return None

    access_token = ACCESS_TOKEN
    api_url = f"https://api-gw.stellio.c2jn.fr/ngsi-ld/v1/entities/urn:ngsi-ld:TrafficFlowObserved:{area_code}/attrs"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "NGSILD-Tenant": "urn:ngsi-ld:tenant:smart-city",
        "Content-Type": "application/ld+json",
    }

    try:
        response = requests.post(api_url, headers=headers, data=ngsild_data)
        if response.status_code != 204:
            print(response.status_code)
            print(response.json())
            raise HTTPException(
                status_code=400, detail="Failed to send data to broker."
            )
        print("Successfuly sent data to broker.")
    except HTTPException:
        print("Failed to send data to broker.")


# FastAPI
app = FastAPI()


@app.get("/predict_next_24hr_with_ecomob_model")
async def predict_next_24hr_with_ecomob_model(
    area_code: str,
    timeAt: str,
    look_back=72,
):
    data = fetch_data_c2jn_broker(area_code=area_code, timeAt=timeAt)

    if data is not None:
        parse_data = parse_ecomob_data(data)

        ecomob_model = EcomobLSTMmodel(model_path=MODEL_PATH)
        predictions = ecomob_model.predict_next_24hr(
            parse_data, look_back=int(look_back)
        )

        ngsild_data = data_convert_to_ngsild(
            predictions, area_code=area_code, model_id=MODEL_ID
        )
        return ngsild_data
    else:
        return {"message": "Failed to fetch data."}


@app.get("/predict_with_ecomob_model")
async def predict_with_ecomob_model(
    area_code: str,
    future_datetime: str,
    look_back=72,
):
    """
    Predict Traffic Information for the given Date Time.
    """
    data = fetch_data_c2jn_broker(area_code=area_code)

    if data is not None:
        parse_data = parse_ecomob_data(data)

        # ecomob_model = EcomobLSTMmodel()
        # ecomob_model.train_model(
        #     parse_data, lr=0.01, epochs=20, batch_size=1, look_back=72, save=True
        # )

        data_end_datetime = datetime.fromisoformat(str(parse_data.index.max()))
        predict_datetime = datetime.fromisoformat(future_datetime[:-1])

        steps = int((predict_datetime - data_end_datetime).total_seconds() / 3600)

        ecomob_model = EcomobLSTMmodel(model_path=MODEL_PATH)
        predictions = ecomob_model.predict_steps(
            parse_data, steps=steps, look_back=int(look_back)
        )

        ngsild_data = None
        for index, row in predictions.iterrows():
            model_id = MODEL_ID
            ngsild_data = convert_to_ngsild_format(row, model_id)
            # send_prediction_to_broker(area_code, ngsild_data)

        return ngsild_data


if __name__ == "__main__":
    if (
        MODEL_PATH is None
        or CLIENT_ID is None
        or CLIENT_SECRET is None
        or MODEL_ID is None
    ):
        if MODEL_PATH is None:
            print("MODEL_PATH is not defined")
        if CLIENT_ID is None:
            print("CLIENT_ID is not defined")
        if CLIENT_SECRET is None:
            print("CLIENT_SECRET is not defined")
        if MODEL_ID is None:
            print("MODEL_ID is not defined")
        sys.exit()

    auth_status = authenticate(CLIENT_ID, CLIENT_SECRET)
    if auth_status:
        print("Authentication Success.")
    else:
        sys.exit()

    uvicorn.run(app, host="0.0.0.0", port=9000)
