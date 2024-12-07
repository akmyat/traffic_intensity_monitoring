from model import EcomobLSTMmodel
from dataprocessing import EcomobDataProcessing

import uvicorn
from fastapi import FastAPI

import os
import sys
from datetime import datetime

CLIENT_ID = os.environ.get("CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", None)
MODEL_ID = os.environ.get("MODEL_ID", None)

# FastAPI
app = FastAPI()


@app.get("/predict_with_ecomob_model")
async def predict_with_ecomob_model(
    road_code: str,
    date_time: str,
    look_back=72,
):
    """
    Predict Traffic Information until the given Date Time.

    Parameters:
        road_code(string): the code name of road
        date_time(string): the desire date time
        look_back(int): the time window for prediction
    Return:
        PredictionTrafficInformation(string): traffic information in ngsi-ld format
    """
    ecomob_dp = EcomobDataProcessing(CLIENT_ID, CLIENT_SECRET, MODEL_ID)

    data = ecomob_dp.fetch_data_c2jn_broker(area_code=road_code)
    if data is not None:
        parse_data = ecomob_dp.parse_ecomob_data()

        data_end_datetime = datetime.fromisoformat(str(parse_data.index.max()))
        predict_datetime = datetime.fromisoformat(date_time[:-1])

        steps = int((predict_datetime - data_end_datetime).total_seconds() / 3600)

        ecomob_model = EcomobLSTMmodel(model_path=f"./models/{road_code}.keras")
        predictions = ecomob_model.predict_steps(
            parse_data, steps=steps, look_back=int(look_back)
        )

        ngsild_data = None
        for _, row in predictions.iterrows():
            ngsild_data = ecomob_dp.convert_to_ngsild_format(row)
            # ecomob_dp.send_prediction_to_broker(area_code, ngsild_data)

        return ngsild_data


if __name__ == "__main__":
    if CLIENT_ID is None or CLIENT_SECRET is None or MODEL_ID is None:
        if CLIENT_ID is None:
            print("CLIENT ID is not defined")
        if CLIENT_SECRET is None:
            print("CLIENT SECRET is not defined")
        if MODEL_ID is None:
            print("MODEL ID is not defined")

        sys.exit()

    uvicorn.run(app, host="0.0.0.0", port=9000)
