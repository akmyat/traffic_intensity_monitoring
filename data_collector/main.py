from asyncio import tasks
from logging import raiseExceptions
import os
import re
import sys
import csv
import asyncio
import requests
import pycountry
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn

# Global Variable
loop_status = False
task = None


# Traffic data collector
def fetch_data(
    api_key: str, origin_coordinates: str, destination_coordinates: str
) -> dict:
    """
    Fetch data from Google API about traffic and return information about traffic condition and intensity.

    Parameters:
    api_key (str): Google API Access Token
    origin_coordinates (str): Coordinates of origin location
    destination_coordinates (str): Coordinates of destination location

    Returns:
    dict: The dictionary which contains traffic condition and intensity
    """
    api_url = f"https://maps.googleapis.com/maps/api/distancematrix/json?units=imperial&traffic_model=best_guess&departure_time=now&origins={origin_coordinates}&destinations={destination_coordinates}&key={api_key}"

    response = requests.get(api_url)
    data = response.json()

    # Get the UTC time when the data is accessed
    date_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    origin_address = data["origin_addresses"][0]
    (_, origin_road, origin_postal_code, origin_city, origin_country) = re.match(
        r"(\d+)\s([\w\s]+),\s(\d+)\s([\w\-]+),\s([\w]+)", origin_address
    ).groups()

    destination_address = data["destination_addresses"][0]
    (
        _,
        destination_road,
        destination_postal_code,
        destination_city,
        destination_country,
    ) = re.match(
        r"(\d+)\s([\w\s]+),\s(\d+)\s([\w\-]+),\s([\w]+)",
        destination_address,
    ).groups()

    # Extract distance, duration and calculate speed
    distance_text = data["rows"][0]["elements"][0]["distance"]["text"]
    if "ft" in distance_text:
        distance_ft = float(
            data["rows"][0]["elements"][0]["distance"]["text"].replace(" ft", "")
        )
        # convert to distnace in miles (1 ft = 0.0003048 km)
        distance_km = distance_ft * 0.0003048
    if "mi" in distance_text:
        distance_mi = float(
            data["rows"][0]["elements"][0]["distance"]["text"].replace(" mi", "")
        )
        # convert to distnace in miles (1 mile = 1.6 km)
        distance_km = distance_mi * 1.6

    # duration in mins
    duration_in_mins = float(
        data["rows"][0]["elements"][0]["duration_in_traffic"]["text"]
        .replace(" mins", "")
        .replace(" min", "")
    )
    # convert duration to hour
    duration_in_h = duration_in_mins / 60
    # Speed
    speed_km_h = distance_km / duration_in_h

    # Determine traffic condition and traffic intensity base on speed
    traffic_condition = None
    traffic_intensity = None
    if speed_km_h > 20:
        traffic_condition = "Low Traffic"
        traffic_intensity = 0
    elif 15 <= speed_km_h <= 20:
        traffic_condition = "Moderate Traffic"
        traffic_intensity = 0.4 + 0.03 * (speed_km_h - 15)
    elif 11 <= speed_km_h <= 15:
        traffic_condition = "Heavy Traffic"
        traffic_intensity = 0.7 - 0.05 * (speed_km_h - 11)
    elif 5 <= speed_km_h <= 10:
        traffic_condition = "Congested Traffic"
        traffic_intensity = 0.9 - 0.01 * (speed_km_h - 5)
    else:
        traffic_condition = "Congested Traffic"
        traffic_intensity = 1

    # Verify origin and destination are on same road
    road = ""
    if origin_road == destination_road:
        road = origin_road
    else:
        print("Origin and Destination are not on the same road")
        sys.exit(1)

    # Verify origin and destination are in same city
    city = ""
    if origin_city == destination_city:
        city = origin_city
    else:
        print("Origin and Destination are not in same city")
        sys.exit(1)

    # Verify origin and destination are in same country
    country = ""
    if origin_country == destination_country:
        country = origin_country
    else:
        print("Origin and Destination are not in same country")
        sys.exit(1)

    # Verify origin and destination postal code are same
    postal_code = ""
    if origin_postal_code == destination_postal_code:
        postal_code = origin_postal_code
    else:
        print("Origin and Destination Postal Code is different")
        sys.exit(1)

    data = {
        "Datetime": date_time,
        "Origin_Coordinates": origin_coordinates,
        "Destination_Coordinates": destination_coordinates,
        "PostalAddress": postal_code,
        "Road": road,
        "City": city,
        "Country": country,
        "Trip_Distance(km)": distance_km,
        "Trip_Total_Duration(h)": duration_in_h,
        "Speed(kmh)": speed_km_h,
        "Traffic_Condition": traffic_condition,
        "Traffic_Intensity": traffic_intensity,
    }
    return data


def write_to_csv(data, filename):
    header_to_write = [
        "Datetime",
        "Origin Coordinates",
        "Destination Coordinates",
        "PostalAddress",
        "Road",
        "City",
        "Country",
        "Trip Distance(km)",
        "Trip Total Duration(h)",
        "Speed(kmh)",
        "Traffic Condition",
        "Traffic Intensity",
    ]
    data_to_write = list(data.values())

    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        if not file_exists:
            writer.writerow(header_to_write)

        writer.writerow(data_to_write)


def convert_to_jsonld_format(data):
    date_time_obj = datetime.strptime(data["Datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
    observed_date_time = date_time_obj.strftime("%Y-%m-%dT%H:%M:%SZ")

    country = pycountry.countries.get(name=data["Country"]).alpha_2
    postal_code = data["PostalAddress"]
    city = "".join([part[0] for part in data["City"].split("-")]).upper()
    road = data["Road"].replace(" ", "")

    jsonld_dict = {
        "id": f"urn:ngsi-ld:TrafficFlowObserved:{country}-{city}-{postal_code}-{road}",
        "type": "TrafficFlowObserved",
        "refRoad": {
            "type": "Relationship",
            "object": f"urn:ngsi-ld:Road:{country}-{city}-{postal_code}-{road}",
        },
        "temporalResolution": {"type": "Property", "value": "PT1H"},
        "averageVehicleSpeed": {
            "type": "Property",
            "observedAt": f"{observed_date_time}",
            "value": float(data["Speed(kmh)"]),
            "unitCode": "KMH",
        },
        "intensity": {
            "type": "Property",
            "observedAt": f"{observed_date_time}",
            "value": float(data["Traffic_Intensity"]),
            "unitCode": "P1",
        },
        "@context": [
            "https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld"
        ],
    }
    return jsonld_dict


# FastAPI
app = FastAPI()


@app.get("/traffic-data")
async def get_traffic_data(
    api_key: str, origin_coordinates: str, destination_coordinates: str
):
    if not api_key or not origin_coordinates or not destination_coordinates:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    try:
        data = fetch_data(api_key, origin_coordinates, destination_coordinates)
        return data
    except:
        return {"message": "Failed to fetch data."}


@app.get("/traffic-data-ngsild")
async def get_traffic_data_ngsild(
    api_key: str, origin_coordinates: str, destination_coordinates: str
):
    if not api_key or not origin_coordinates or not destination_coordinates:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    try:
        data = fetch_data(api_key, origin_coordinates, destination_coordinates)
        jsonld_data = convert_to_jsonld_format(data)
        return jsonld_data
    except:
        return {"message": "Failed to fetch data."}


@app.get("/traffic-data-csv")
async def get_traffic_data(
    api_key: str, origin_coordinates: str, destination_coordinates: str
):
    if not api_key or not origin_coordinates or not destination_coordinates:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    try:
        data = fetch_data(api_key, origin_coordinates, destination_coordinates)
        write_to_csv(data, "/tmp/csv/data.csv")
        return data
    except:
        return {"message": "Failed to fetch data."}


@app.get("/traffic-data-broker")
async def get_traffic_data_broker(
    api_key: str,
    origin_coordinates: str,
    destination_coordinates: str,
    broker_address: str,
    broker_port: int,
):
    if (
        not api_key
        or not origin_coordinates
        or not destination_coordinates
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missing required parameters")

    data = fetch_data(api_key, origin_coordinates, destination_coordinates)
    jsonld_data = convert_to_jsonld_format(data)
    del jsonld_data["@context"]

    url = f"http://{broker_address}:{str(broker_port)}/ngsi-ld/v1/entities"
    headers = {
        "Link": '<https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"',
    }
    response = requests.post(url, headers=headers, json=jsonld_data)
    try:
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
                    print("\n\n204\n\n")
                    raise ValueError(response.json())
            else:
                raise ValueError(response.json())
    except ValueError:
        print(response.json())
        return {"message": "Failed to fetch data"}
    except:
        return {"message": "Failed to fetch data"}

    return jsonld_data


async def traffic_data_logging_task(
    api_key: str,
    origin_coordinates: str,
    destination_coordinates: str,
    time_delta: int,
    broker_address: str,
    broker_port: int,
):
    global loop_status
    while loop_status:
        print("OK")
        await get_traffic_data_broker(
            api_key,
            origin_coordinates,
            destination_coordinates,
            broker_address,
            broker_port,
        )
        await asyncio.sleep(time_delta * 60)


@app.get("/start-traffic-data-logging")
async def start_traffic_data_logging(
    background_tasks: BackgroundTasks,
    api_key: str,
    origin_coordinates: str,
    destination_coordinates: str,
    time_delta: int,
    broker_address: str,
    broker_port: int,
):
    global loop_status, task
    if loop_status == True:
        raise HTTPException(status_code=400, detail="Data logging is already running.")
    loop_status = True
    task = background_tasks.add_task(
        traffic_data_logging_task,
        api_key,
        origin_coordinates,
        destination_coordinates,
        time_delta,
        broker_address,
        broker_port,
    )
    return {"message": "Traffic data logging started."}


@app.get("/stop-traffic-data-logging")
async def stop_traffic_data_logging():
    global loop_status
    if not loop_status:
        raise HTTPException(status_code=400, detail="Data logging is not running.")
    loop_status = False
    return {"message": "Traffic data logging stopped."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
