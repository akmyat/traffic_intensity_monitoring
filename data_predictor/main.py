import pandas as pd
from pandas.core.api import DataFrame
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

import requests

import io
import pickle
from datetime import datetime

from fastapi import FastAPI, HTTPException, Response
import uvicorn

# Global
intensity_model = None
avgvspeed_model = None


def fetch_data(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    url = (
        f"http://{broker_address}:{str(broker_port)}/ngsi-ld/v1/temporal/entities/{id}"
    )
    headers = {
        "Link": '<https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
    }
    params = {"timerel": "between", "timeAt": start_time, "endTimeAt": end_time}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code != 200:
        return None
    return response.json()


def extract_data(data: dict):
    # Extract average Vehicle Speed data
    extract_avgVSpeed_data = {
        "Datetime": [],
        "averageVehicleSpeed": [],
    }
    for item in data["averageVehicleSpeed"]:
        extract_avgVSpeed_data["Datetime"].append(item["observedAt"])
        extract_avgVSpeed_data["averageVehicleSpeed"].append(item["value"])

    avgVSpeed_dataframe = pd.DataFrame(extract_avgVSpeed_data)
    avgVSpeed_dataframe["Datetime"] = pd.to_datetime(avgVSpeed_dataframe["Datetime"])

    # Extract traffic intensity data
    extract_intensity_data = {
        "Datetime": [],
        "intensity": [],
    }
    for item in data["intensity"]:
        extract_intensity_data["Datetime"].append(item["observedAt"])
        extract_intensity_data["intensity"].append(item["value"])

    intensity_dataframe = pd.DataFrame(extract_intensity_data)
    intensity_dataframe["Datetime"] = pd.to_datetime(intensity_dataframe["Datetime"])

    # combine average vehicle speed and traffic intensity data
    combined_dataframe = pd.merge(
        avgVSpeed_dataframe, intensity_dataframe, on="Datetime"
    )
    combined_dataframe.set_index("Datetime", inplace=True)
    return combined_dataframe


def train_traffic_intensity(intensity_data: DataFrame, save=False):
    # Pre-processing
    data = intensity_data.copy()
    data["intensity"] = pd.to_numeric(data["intensity"], errors="coerce")
    data["intensity"].ffill(inplace=True)
    daily_data = data["intensity"].resample("D").mean()

    # ARIMA model
    model = ARIMA(daily_data, order=(1, 0, 1))
    model = model.fit()

    if save:
        model_name = f"arima_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        with open(model_name, "wb") as pkl:
            pickle.dump(model, pkl)

    return model


def predict_next_day_traffic_intensity(model, daily_data, save=False):
    forecast_24hr = model.get_forecast(steps=24).predicted_mean
    next_24hr_timestamps = pd.date_range(
        start=daily_data.index[-1], periods=25, freq="H"
    )[1:]

    forecast_24hr_df = pd.DataFrame(
        {"Datetime": next_24hr_timestamps, "intensity": forecast_24hr}
    )
    forecast_24hr_df["Datetime"] = forecast_24hr_df["Datetime"].dt.tz_localize(None)

    # Save forecast
    if save:
        filename = datetime.now().strftime("forecast_intensity_%Y%m%d.csv")
        forecast_24hr_df.to_csv(filename, index=False)
        print(f"Forecast plot saved as {filename}")

    return forecast_24hr_df


def plot_next_day_traffic_intensity(forecast_24hr_df, save=False):
    plt.figure(figsize=(12, 6))
    plt.plot(
        forecast_24hr_df["Datetime"],
        forecast_24hr_df["intensity"],
        "ro-",
        label="Forecasted Intensity",
    )
    plt.legend()
    plt.title("Traffic Intensity 24-hour Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Intensity")
    plt.grid(True)

    # Save the plot with the current date for uniqueness
    if save:
        plot_filename = datetime.now().strftime("forecast_intensity_%Y%m%d_plot.png")
        plt.savefig(plot_filename)
        print(f"Forecast plot saved as {plot_filename}")
    else:
        plt.show()
    plt.close()


def train_average_vehicle_speed(avgvspeed_data: DataFrame, save=False):
    # Pre-processing
    data = avgvspeed_data.copy()
    data["averageVehicleSpeed"] = pd.to_numeric(
        data["averageVehicleSpeed"], errors="coerce"
    )
    data["averageVehicleSpeed"].ffill(inplace=True)
    daily_data = data["averageVehicleSpeed"].resample("D").mean()

    # ARIMA model
    model = ARIMA(daily_data, order=(1, 0, 1))
    model = model.fit()

    if save:
        model_name = (
            f"arima_model_avgspeed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl"
        )
        with open(model_name, "wb") as pkl:
            pickle.dump(model, pkl)

    return model


def predict_next_day_average_vehicle_speed(model, daily_data, save=False):
    forecast_24hr = model.get_forecast(steps=24).predicted_mean
    next_24hr_timestamps = pd.date_range(
        start=daily_data.index[-1], periods=25, freq="H"
    )[1:]

    forecast_24hr_df = pd.DataFrame(
        {"Datetime": next_24hr_timestamps, "averageVehicleSpeed": forecast_24hr}
    )
    forecast_24hr_df["Datetime"] = forecast_24hr_df["Datetime"].dt.tz_localize(None)

    # Save forecast
    if save:
        filename = datetime.now().strftime("forecast_average_vehicle_speed_%Y%m%d.csv")
        forecast_24hr_df.to_csv(filename, index=False)
        print(f"Forecast plot saved as {filename}")

    return forecast_24hr_df


def plot_next_day_average_vehicle_speed(forecast_24hr_df, save=False):
    plt.figure(figsize=(12, 6))
    plt.plot(
        forecast_24hr_df["Datetime"],
        forecast_24hr_df["averageVehicleSpeed"],
        "ro-",
        label="Forecasted average vehicle speed",
    )
    plt.legend()
    plt.title("Average Vehicle Speed 24-hour Forecast")
    plt.xlabel("Datetime")
    plt.ylabel("Average Vehilce Speed")
    plt.grid(True)

    # Save the plot with the current date for uniqueness
    if save:
        plot_filename = datetime.now().strftime(
            "forecast_average_vehicle_speed_%Y%m%d_plot.png"
        )
        plt.savefig(plot_filename)
        print(f"Forecast plot saved as {plot_filename}")
    else:
        plt.show()
    plt.close()


# FastAPI
app = FastAPI()


@app.get("/get_csv")
async def get_csv(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        stream = io.StringIO()
        combined_dataframe = extract_data(data)
        combined_dataframe.to_csv(stream)
        stream.seek(0)
        return Response(
            content=stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=data.csv"},
        )
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/train_traffic_intensity_model")
async def train_traffic_intensity_model(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global intensity_model

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["averageVehicleSpeed"])
        intensity_model = train_traffic_intensity(daily_data, save=True)
        return {"message": "Successfuly train model for traffic intensity prediction."}
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/train_average_vehicle_speed_model")
async def train_average_vehicle_speed_model(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global avgvspeed_model

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["intensity"])
        avgvspeed_model = train_average_vehicle_speed(daily_data, save=True)
        return {
            "message": "Successfuly train model for average vehicle speed prediction."
        }
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/predict_traffic_intensity")
async def predict_traffic_intensity(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global intensity_model

    if intensity_model is None:
        return {"message": "No pretrained model available."}

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        stream = io.StringIO()
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["averageVehicleSpeed"])
        forecast_24hr_df = predict_next_day_traffic_intensity(
            intensity_model, daily_data, save=True
        )
        forecast_24hr_df.to_csv(stream)
        stream.seek(0)
        return Response(
            content=stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=forecast_24hr.csv"},
        )
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/predict_average_vehicle_speed")
async def predict_average_vehicle_speed(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global avgvspeed_model

    if avgvspeed_model is None:
        return {"message": "No pretrained model available."}

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        stream = io.StringIO()
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["averageVehicleSpeed"])
        forecast_24hr_df = predict_next_day_average_vehicle_speed(
            avgvspeed_model, daily_data, save=True
        )
        forecast_24hr_df.to_csv(stream)
        stream.seek(0)
        return Response(
            content=stream.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=forecast_24hr.csv"},
        )
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/plot_traffic_intensity")
async def predict_traffic_intensity(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global intensity_model

    if intensity_model is None:
        return {"message": "No pretrained model available."}

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        stream = io.BytesIO()
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["averageVehicleSpeed"])
        forecast_24hr_df = predict_next_day_traffic_intensity(
            intensity_model, daily_data, save=True
        )

        plt.figure(figsize=(12, 6))
        plt.plot(
            forecast_24hr_df["Datetime"],
            forecast_24hr_df["intensity"],
            "ro-",
            label="Forecasted Intensity",
        )
        plt.legend()
        plt.title("Traffic Intensity 24-hour Forecast")
        plt.xlabel("Datetime")
        plt.ylabel("Intensity")
        plt.grid(True)
        plt.savefig(stream, format="png")

        stream.seek(0)
        return Response(
            content=stream.read(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=forecast_24hr.png"},
        )
    else:
        return {"message": "Failed to get data or data not available."}


@app.get("/plot_average_vehicle_speed")
async def predict_average_vehicle_speed(
    start_time: str, end_time: str, id: str, broker_address: str, broker_port: int
):
    global avgvspeed_model

    if avgvspeed_model is None:
        return {"message": "No pretrained model available."}

    if (
        not start_time
        or not end_time
        or not id
        or not broker_address
        or not broker_port
    ):
        raise HTTPException(status_code=400, detail="Missiing required parameters.")

    data = fetch_data(
        start_time,
        end_time,
        id,
        broker_address,
        broker_port,
    )

    if data is not None:
        stream = io.BytesIO()
        combined_dataframe = extract_data(data)
        daily_data = combined_dataframe.drop(columns=["averageVehicleSpeed"])
        forecast_24hr_df = predict_next_day_average_vehicle_speed(
            avgvspeed_model, daily_data, save=True
        )

        plt.figure(figsize=(12, 6))
        plt.plot(
            forecast_24hr_df["Datetime"],
            forecast_24hr_df["averageVehicleSpeed"],
            "ro-",
            label="Forecasted average vehicle speed",
        )
        plt.legend()
        plt.title("Average Vehicle Speed 24-hour Forecast")
        plt.xlabel("Datetime")
        plt.ylabel("Average Vehilce Speed")
        plt.grid(True)
        plt.savefig(stream, format="png")

        stream.seek(0)
        return Response(
            content=stream.read(),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=forecast_24hr.png"},
        )
    else:
        return {"message": "Failed to get data or data not available."}


# plot_next_day_average_vehicle_speed(forecast_24hr_df, save=True)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
