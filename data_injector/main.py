import io
import pandas as pd
from datetime import datetime

import requests
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from concurrent.futures import ThreadPoolExecutor

import pycountry


def substitute_values(row):
    datetime_obj = datetime.strptime(row["Datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
    datetime_str = datetime_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

    country_code = pycountry.countries.get(name=row["Country"]).alpha_2
    postal_code = str(row["Postal_Code"])
    city_code = "".join([part[0] for part in row["City"].split("-")]).upper()
    road = row["Road"].replace(" ", "")

    return {
        "id": f"urn:ngsi-ld:TrafficFlowObserved:{country_code}-{city_code}-{postal_code}-{road}",
        "type": "TrafficFlowObserved",
        "refRoad": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:Road:{country_code}-{city_code}-{postal_code}-{road}",
        },
        "temporalResolution": {"type": "Property", "value": "PT1H"},
        "averageVehicleSpeed": {
            "type": "Property",
            "observedAt": datetime_str,
            "value": row["speed"],
            "unitCode": "KMH",
        },
        "intensity": {
            "type": "Property",
            "observedAt": datetime_str,
            "value": row["Intensity"],
            "unitCode": "P1",
        },
        "@context": [
            "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld"
        ],
    }


def push_to_broker(row, broker_address, broker_port):
    jsonld_data = row.copy()
    del jsonld_data["@context"]

    url = f"http://{broker_address}:{str(broker_port)}/ngsi-ld/v1/entities"
    headers = {
        "Link": '<https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    response = requests.post(url, headers=headers, json=jsonld_data)
    if response.status_code != 201:
        if response.status_code == 409:
            url = f"http://{broker_address}:{str(broker_port)}/ngsi-ld/v1/entities/{jsonld_data['id']}/attrs"
            headers = {
                "Link": '<https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
            }
            data = {
                "averageVehicleSpeed": jsonld_data["averageVehicleSpeed"],
                "intensity": jsonld_data["intensity"],
            }
            response = requests.patch(url, headers=headers, json=data)
            if response.status_code != 204:
                return {
                    "message": "Error! Failed to PATCH data."
                }  # TODO: ERROR HANDLING


def process_dataframe(df, broker_address, broker_port):
    for _, row in df.iterrows():
        push_to_broker(row["JSON-LD"], broker_address, broker_port)


# FastAPI
app = FastAPI()


@app.post("/upload-csv-to-broker")
async def upload_csv_to_broker(
    broker_address: str, broker_port: int, file: UploadFile = File(...)
):
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode()))

        # Extract average vehicle speed and traffic intensity data
        if not set(
            ["Datetime", "speed", "Intensity", "Country", "City", "Road", "Postal_Code"]
        ).issubset(set(df.columns)):
            raise ValueError("Does not have necessary columns in csv file.")

        df = df[
            ["Datetime", "speed", "Intensity", "Country", "City", "Road", "Postal_Code"]
        ]

        # Convert to json-ld format
        df["JSON-LD"] = df.apply(substitute_values, axis=1)

        # Push to context broker
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(process_dataframe, df, broker_address, broker_port)

        return {"message": "File is being processed in the background"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    finally:
        await file.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
