import os
import re
import sys
import csv
import requests
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException
import uvicorn


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

    # Extract Addresses
    origin_address = data["origin_addresses"][0]
    origin_road, origin_postal_code, origin_city, origin_country = re.match(
        r"^(\d+\s[A-Za-z\s]+),\s(\d{5})\s([A-Za-z\-]+),\s([A-Za-z]+)$", origin_address
    ).groups()

    destination_address = data["destination_addresses"][0]
    (
        destination_road,
        destination_postal_code,
        destination_city,
        destination_country,
    ) = re.match(
        r"^(\d+\s[A-Za-z\s]+),\s(\d{5})\s([A-Za-z\-]+),\s([A-Za-z]+)$",
        destination_address,
    ).groups()

    # Extract distance, duration and calculate speed
    # distance in miles
    distance_mi = float(
        data["rows"][0]["elements"][0]["distance"]["text"].replace(" mi", "")
    )
    # convert to distnace in miles (1 mile = 1.6 km)
    distance_km = distance_mi * 1.6
    # duration in mins
    duration_in_mins = float(
        data["rows"][0]["elements"][0]["duration_in_traffic"]["text"].replace(
            " mins", ""
        )
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

    # Verify origin and destination are in same city and country
    city = ""
    country = ""
    if origin_city == destination_city:
        city = origin_city
    else:
        print("Origin and Destination are not in same city")
        sys.exit(1)
    if origin_country == destination_country:
        country = origin_country
    else:
        print("Origin and Destination are not in same country")
        sys.exit(1)

    data = {
        "Datetime": date_time,
        "Origin_Coordinates": origin_coordinates,
        "Origin_Road": origin_road,
        "Destination_Coordinates": destination_coordinates,
        "Destination_Road": destination_road,
        "City": city,
        "Country": country,
        "Trip_Distance(miles)": distance_mi,
        "Trip_Total_Duration(mins)": duration_in_mins,
        "Trip_Distance(km)": distance_km,
        "Trip_Total_Duration(h)": duration_in_h,
        "Traffic_Condition": traffic_condition,
        "Traffic_Intensity": traffic_intensity,
    }
    return data


def write_to_csv(data, filename):
    header_to_write = [
        "Datetime",
        "Origin Coordinates",
        "Origin Road",
        "Destination Coordinates",
        "Destination Road",
        "City",
        "Country",
        "Trip Distance(miles)",
        "Trip Total Duration(mins)",
        "Trip Distance(km)",
        "Trip Total Duration(h)",
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


def convert_to_jsonld_format(data, time_delta):
    date_time_obj = datetime.strptime(data["Datetime"], "%Y-%m-%dT%H:%M:%S.%fZ")
    sub_delta_date_time_obj = date_time_obj - timedelta(minutes=time_delta)
    date_observed_from = sub_delta_date_time_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    date_observed_to = date_time_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    date_observed = date_time_obj.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    city = data["City"].replace("-", "_")
    id = f"urn:ngsi-ld:TrafficFlowObserved:TrafficFlowObserved-{city}-{date_observed}"

    jsonld_dict = {
        "id": id,
        "type": "TrafficFlowObserved",
        "dateObserved": {
            "type": "Property",
            "value": date_observed,
        },
        "dateObservedFrom": {
            "type": "Property",
            "Value": {
                "@type": "DateTime",
                "@value": date_observed_from,
            },
        },
        "dateObservedTo": {
            "type": "Property",
            "Value": {
                "@type": "DateTime",
                "@value": date_observed_to,
            },
        },
        "intensity": {
            "type": "Property",
            "value": float(data["Traffic_Intensity"]),
        },
        "location": {
            "type": "GeoProperty",
            "value": {
                "type": "LineString",
                "coordinates": (
                    (data["Origin_Coordinates"]),
                    (data["Destination_Coordinates"]),
                ),
            },
        },
        "address": {
            "type": "Property",
            "value": {
                "type": "PostalAddress",
                "addressCountry": data["Country"],
                "addressLocality": city,
                "streetAddress": data["Origin_Road"],
            },
        },
        "@context": [
            "https://uri.etsi.org/ngsi-ld/v1/ngsi-ld-core-context.jsonld",
            "https://schema.lab.fiware.org/ld/context",
        ],
    }
    return jsonld_dict


# FastAPI
app = FastAPI()


@app.get("/traffic-data/")
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


@app.get("/traffic-data-ngsild/")
async def get_traffic_data_ngsild(
    api_key: str, origin_coordinates: str, destination_coordinates: str, time_delta: int
):
    if (
        not api_key
        or not origin_coordinates
        or not destination_coordinates
        or not time_delta
    ):
        raise HTTPException(status_code=400, detail="Missing required parameters")

    try:
        data = fetch_data(api_key, origin_coordinates, destination_coordinates)
        jsonld_data = convert_to_jsonld_format(data, time_delta)
        return jsonld_data
    except:
        return {"message": "Failed to fetch data."}


@app.get("/traffic-data-csv/")
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
