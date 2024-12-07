import requests
import os
import sys
from datetime import datetime
import pandas as pd
from pandas.core.api import DataFrame

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


if __name__ == "__main__":
    if CLIENT_ID is None or CLIENT_SECRET is None:
        if CLIENT_ID is None:
            print("CLIENT_ID is not defined")
        if CLIENT_SECRET is None:
            print("CLIENT_SECRET is not defined")
        sys.exit()

    auth_status = authenticate(CLIENT_ID, CLIENT_SECRET)
    if auth_status:
        print("Authentication Success.")
        # print("Access Token: ", ACCESS_TOKEN)
    else:
        sys.exit()

    area_code = "ILM_92130_7767"
    future_datetime = "2024-05-30T23:00:00Z"
    look_back = 72

    data = fetch_data_c2jn_broker(area_code=area_code)

    if data is not None:
        parse_data = parse_ecomob_data(data)

    parse_data.to_csv(f"{area_code}_dataset.csv")
