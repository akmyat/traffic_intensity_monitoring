import os
import sys
from model import EcomobLSTMmodel
from dataprocessing import EcomobDataProcessing

CLIENT_ID = os.environ.get("CLIENT_ID", None)
CLIENT_SECRET = os.environ.get("CLIENT_SECRET", None)
MODEL_ID = os.environ.get("MODEL_ID", None)
ROAD_ID = os.environ.get("ROAD_ID", None)

if CLIENT_ID is None or CLIENT_SECRET is None or MODEL_ID is None or ROAD_ID is None:
    if CLIENT_ID is None:
        print("CLIENT ID is not defined")
    if CLIENT_SECRET is None:
        print("CLIENT SECRET is not defined")
    if MODEL_ID is None:
        print("MODEL ID is not defined")
    if ROAD_ID is None:
        print("ROAD ID is not defined")

    sys.exit()

ecomob_dp = EcomobDataProcessing(CLIENT_ID, CLIENT_SECRET, MODEL_ID)
ecomob_model = EcomobLSTMmodel()

data = ecomob_dp.fetch_data_c2jn_broker(area_code=ROAD_ID)
if data is not None:
    parse_data = ecomob_dp.parse_ecomob_data(data)

    ecomob_model.train_model(
        parse_data,
        40,
        lr=0.01,
        epochs=20,
        batch_size=1,
        look_back=72,
        model_name=f"./models/{ROAD_ID}.keras",
    )
