import requests
import pandas as pd
import json

road_id = "ILM_92130_2385"
filename = f"./{road_id}_dataset.csv"


def create(ngsild_data):
    url = f"http://192.168.10.220:8080/ngsi-ld/v1/entities"
    headers = {"Content-Type": "application/ld+json"}

    ngsild_data = json.dumps(ngsild_data)
    response = requests.post(url, headers=headers, data=ngsild_data)
    print(response.status_code)


def patch(ngsild_data, road_id):
    url = f"http://192.168.10.220:8080/ngsi-ld/v1/entities/urn:ngsi-ld:TrafficFlowObserved:{road_id}/attrs"
    headers = {
        "Content-Type": "application/ld+json",
        "Link": "<https://easy-global-market.github.io/c2jn-data-m      odels/jsonld-contexts/c2jn-compound.jsonld>",
    }

    ngsild_data = json.dumps(ngsild_data)
    response = requests.patch(url, headers=headers, data=ngsild_data)
    print(response.status_code)


df = pd.read_csv(filename)
df["DateTime"] = pd.to_datetime(df["DateTime"])
df.set_index("DateTime", inplace=True)


first_row = True
for index, row in df.iterrows():
    observedAt = index.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    co2equivalent = row["CO2Equivalent"]
    avgVehicleSpeed = row["AverageVehicleSpeed"]
    congestionIndex = row["CongestionIndex"]
    flowHeavy = row["HeavyVehicle Value"]
    flowLight = row["LightVehicle Value"]
    flowCommercial = row["CommercialVehicle Value"]
    flowAll = flowHeavy + flowLight + flowCommercial

    if first_row:
        first_row = False
        ngsild_data = {
            "id": f"urn:ngsi-ld:TrafficFlowObserved:{road_id}",
            "type": "TrafficFlowObserved",
            "title": {
                "type": "Property",
                "value": f"Traffic au point de mesure {road_id}",
            },
            "temporalResolution": {"type": "Property", "value": "PT1H"},
            "refRoad": {
                "type": "Relationship",
                "object": f"urn:ngsi-ld:Road:{road_id}",
            },
            "co2Equivalent": {
                "type": "Property",
                "value": co2equivalent,
                "unitCode": "TNE",
                "observedAt": observedAt,
            },
            "congestionIndex": {
                "type": "Property",
                "value": congestionIndex,
                "unitCode": "P1",
                "observedAt": observedAt,
            },
            "averageVehicleSpeed": {
                "type": "Property",
                "value": avgVehicleSpeed,
                "unitCode": "KMH",
                "observedAt": observedAt,
            },
            "flow": [
                {
                    "type": "Property",
                    "value": flowAll,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:All",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowLight,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:LightVehicle",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowHeavy,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:HeavyVehicle",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowCommercial,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:CommercialVehicle",
                    "observedAt": observedAt,
                },
            ],
            "@context": "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld",
        }
        create(ngsild_data)
    else:
        ngsild_data = {
            "co2Equivalent": {
                "type": "Property",
                "value": co2equivalent,
                "unitCode": "TNE",
                "observedAt": observedAt,
            },
            "congestionIndex": {
                "type": "Property",
                "value": congestionIndex,
                "unitCode": "P1",
                "observedAt": observedAt,
            },
            "averageVehicleSpeed": {
                "type": "Property",
                "value": avgVehicleSpeed,
                "unitCode": "KMH",
                "observedAt": observedAt,
            },
            "flow": [
                {
                    "type": "Property",
                    "value": flowAll,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:All",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowLight,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:LightVehicle",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowHeavy,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:HeavyVehicle",
                    "observedAt": observedAt,
                },
                {
                    "type": "Property",
                    "value": flowCommercial,
                    "unitCode": "E50",
                    "datasetId": "urn:ngsi-ld:Dataset:CommercialVehicle",
                    "observedAt": observedAt,
                },
            ],
            "@context": "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld",
        }
        patch(ngsild_data, road_id)
