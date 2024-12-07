import json
import requests
import pandas as pd
from fastapi import HTTPException


class EcomobDataProcessing:
    def __init__(self, client_id, client_secrect, model_id):
        self.client_id = client_id
        self.client_secret = client_secrect
        self.model_id = model_id
        self.access_token = None

    def authenticate(self):
        auth_url = (
            "https://sso.c2jn.fr/auth/realms/smart-city/protocol/openid-connect/token"
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        body = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }
        try:
            response = requests.post(auth_url, headers=headers, data=body)

            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Authentication Failed.")

            response_json = response.json()
            self.access_token = response_json["access_token"]
            return True
        except HTTPException:
            print("Authentication Failed.")
            return False

    def fetch_data_c2jn_broker(self, area_code: str):
        if not self.authenticate():
            return None

        access_token = self.access_token
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

    def parse_ecomob_data(self, data):
        df = pd.DataFrame(
            columns=[
                "DateTime",
                "Heavy_Vehicle",
                "Commercial_Vehicle",
                "Light_Vehicle",
                "CongestionIndex",
                "CO2Equivalent",
                "AverageVehicleSpeed",
            ]
        )

        # Extract data from json data
        heavy_vehicle_data = data["https://vocab.egm.io/flow"][0]["values"]
        all_vehicle_data = data.get("https://vocab.egm.io/flow", [{}])[1].get(
            "values", []
        )        
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
            "https://smartdatamodels.org/dataModel.Transportation/averageVehicleSpeed",
            {},
        ).get("values", [])

        # Prepare data to append to DataFrame
        for entry in light_vehicle_data:
            date_time = entry[1]
            heavy_weight_value  = next(
                (item[0] for item in heavy_vehicle_data if item[1] == date_time), None
            )
            all_vehicle_value  = next(
                (item[0] for item in all_vehicle_data if item[1] == date_time), None
            )            
            commercial_vehicle_value = next(
                (item[0] for item in commercial_vehicle_data if item[1] == date_time),
                None,
            )
            light_vehicle_value = next(
                (item[0] for item in light_vehicle_data if item[1] == date_time), None
            )
            congestion_index = next(
                (item[0] for item in congestion_index_data if item[1] == date_time),
                None,
            )
            co2_equivalent = next(
                (item[0] for item in co2_equivalent_data if item[1] == date_time), None
            )
            average_vehicle_speed = next(
                (
                    item[0]
                    for item in average_vehicle_speed_data
                    if item[1] == date_time
                ),
                None,
            )

            # Append to DataFrame
            df.loc[len(df)] = {
                # "RoadID": road_id,
                "DateTime": date_time,
                "Heavy_Vehicle": heavy_weight_value,
                "All_Vehicle": all_vehicle_value,
                "Commercial_Vehicle": commercial_vehicle_value,
                "Light_Vehicle": light_vehicle_value,
                "CongestionIndex": congestion_index,
                "CO2Equivalent": co2_equivalent,
                "AverageVehicleSpeed": average_vehicle_speed,
            }
        df["DateTime"] = pd.to_datetime(df["DateTime"]).dt.tz_localize(None)
        df.set_index("DateTime", inplace=True)
        return df

    def convert_to_ngsild_format(self, row):
        model_id = self.model_id

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
                    "value": row["All_Vehicle"],
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
                    "value": row["Light_Vehicle"],
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
                    "value": row["Commercial_Vehicle"],
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
                    "value": row["Heavy_Vehicle"],
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

    def send_prediction_to_broker(self, area_code, ngsild_data):

        if not self.authenticate():
            return None

        access_token = self.access_token
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
