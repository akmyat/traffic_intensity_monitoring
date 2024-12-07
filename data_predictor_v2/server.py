import os
import re
import glob
import pandas as pd
from apscheduler.schedulers.background import BackgroundScheduler
from c2jnBrokerCommunicator import C2JNBrokerCommunicator
from dataprocessing import EcomobDataProcessing
from flask import Flask, render_template
import plotly.io as pio
import plotly.graph_objects as go
from plotly._subplots import make_subplots

road_codes = [
    "ILM_92130_2385",
    "ILM_92130_3794",
    "ILM_92130_6164",
    "ILM_92130_7720",
    "ILM_92130_3209",
    "ILM_92130_4925",
    "ILM_92130_6253",
    "ILM_92130_7767",
    "ILM_92130_3219",
    "ILM_92130_4947",
    "ILM_92130_6556",
    "ILM_92130_7786",
]

######### CONTEXT BROKER COMMUNICATION #########
broker_gateway = os.environ.get('BROKER_GATEWWAY', "api-gw.stellio.c2jn.fr")
client_id = os.environ.get('CLIENT_ID',"imt")
client_secret = os.environ.get('CLIENT_SECRET',"B9ujTPZipkJcRG9xFuxTMFsp3TvdmHzN")

def download_data():
    global broker_gateway
    global client_id
    global client_secret
    global road_codes
    
    brokerCommunicator = C2JNBrokerCommunicator(broker_gateway, client_id, client_secret)
    ecomob_dp = EcomobDataProcessing()
        
    for road in road_codes:
        data = brokerCommunicator.fetch_traffic_data(road)
        
        data_frame = ecomob_dp.parse_ecomob_data(data)
        
        data_frame.to_csv(f'./data/{road}.csv')
        
######### COMBINE DATA #########
def get_data():
    global road_codes
    
    data_frames = []
    for road_code in road_codes:
        filename = f"./data/{road_code}.csv"
        data = pd.read_csv(filename, index_col='DateTime', parse_dates=True)
        data_frames.append(data.rename(columns=lambda x: f"{x}_{road_code}"))
    merged_df = pd.concat(data_frames, axis=1)

    for road_code in road_codes:
        filename = f"./data/{road_code}.csv"
        sub_data = merged_df.filter(like=road_code)
        columns = [column.replace(f'_{road_code}', '') for column in sub_data.columns]
        sub_data.columns = columns
        sub_data.to_csv(f'./dataset/{road_code}.csv')
    processed_files = glob.glob('./dataset/*.csv')
    
    traffic_data = {
        "Heavy_Vehicle": {},
        "Light_Vehicle": {},
        "Commercial_Vehicle": {},
        "All_Vehicle": {},
        "Congestion_Index": {},
        "CO2_Equivalent": {},
        "Average_Vehicle_Speed": {}
    }
    
    for road_code in road_codes:
        filename = f"./dataset/{road_code}.csv"
        data = pd.read_csv(filename, index_col='DateTime', parse_dates=True)
        
        traffic_data["Heavy_Vehicle"][road_code] = data["Heavy_Vehicle"]
        traffic_data["Light_Vehicle"][road_code] = data["Light_Vehicle"]
        traffic_data["Commercial_Vehicle"][road_code] = data["Commercial_Vehicle"]
        traffic_data["All_Vehicle"][road_code] = data["All_Vehicle"]
        traffic_data["Congestion_Index"][road_code] = data["CongestionIndex"]
        traffic_data["CO2_Equivalent"][road_code] = data["CO2Equivalent"]
        traffic_data["Average_Vehicle_Speed"][road_code] = data["AverageVehicleSpeed"]
        
    return traffic_data

def plot_time_series_data(traffic_data, road_codes):
    # Create initial plot with the first area code
    initial_area_code = road_codes[0]

    # Create the figure
    fig = go.Figure()

    # Light Vehicle Trace
    fig.add_trace(
        go.Scatter(
            x=traffic_data['Light_Vehicle'][initial_area_code].index,
            y=traffic_data['Light_Vehicle'][initial_area_code],
            mode='lines',
            name='Light Vehicle Count',
            line=dict(color='orange'),
            visible='legendonly'  # Initially hide this trace
        )
    )
    # Commercial Vehicle Trace
    fig.add_trace(
        go.Scatter(
            x=traffic_data['Commercial_Vehicle'][initial_area_code].index,
            y=traffic_data['Commercial_Vehicle'][initial_area_code],
            mode='lines',
            name='Commercial Vehicle Count',
            line=dict(color='cyan'),
            visible='legendonly'  # Initially hide this trace
        )
    )
    # Heavy Vehicle Trace
    fig.add_trace(
        go.Scatter(
            x=traffic_data['Heavy_Vehicle'][initial_area_code].index,
            y=traffic_data['Heavy_Vehicle'][initial_area_code],
            mode='lines',
            name='Heavy Vehicle Count',
            line=dict(color='blue'),
            visible='legendonly'  # Initially hide this trace
        )
    )
    # All Vehicle Trace
    fig.add_trace(
        go.Scatter(
            x=traffic_data['All_Vehicle'][initial_area_code].index,
            y=traffic_data['All_Vehicle'][initial_area_code],
            mode='lines',
            name='All Vehicle Count',
            line=dict(color='purple')
        )
    )

    # Congestion Index
    fig.add_trace(
        go.Scatter(
            x=traffic_data['Congestion_Index'][initial_area_code].index,
            y=traffic_data['Congestion_Index'][initial_area_code],
            mode='lines',
            name='Congestion Index',
            line=dict(color='red'),
            visible='legendonly'  # Initially hide this trace
        )
    )

    # CO2 Equivalent
    fig.add_trace(
        go.Scatter(
            x=traffic_data['CO2_Equivalent'][initial_area_code].index,
            y=traffic_data['CO2_Equivalent'][initial_area_code],
            mode='lines',
            name='CO2 Equivalent',
            line=dict(color='green'),
            visible='legendonly'  # Initially hide this trace
        )
    )

    # Average Vehicle Speed
    fig.add_trace(
        go.Scatter(
            x=traffic_data['Average_Vehicle_Speed'][initial_area_code].index,
            y=traffic_data['Average_Vehicle_Speed'][initial_area_code],
            mode='lines',
            name='Average Vehicle Speed',
            line=dict(color='yellow'),
            visible='legendonly'  # Initially hide this trace
        )
    )
    
        
    # Update layout with title and xaxis configurations
    fig.update_layout(
        title='ECOMOB Traffic Data',
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(
                        count=1, label='1m', step='month', stepmode='backward'
                    ),
                    dict(
                        count=6, label='6m', step='month', stepmode='backward'
                    ),
                    dict(count=1, label='YTD', step='year', stepmode='todate'),
                    dict(
                        count=1, label='1y', step='year', stepmode='backward'
                    ),
                    dict(step='all'),
                ]
            ),
            rangeslider=dict(visible=True),
            type='date',
        ),
    )

    # Dropdown menu
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(label=f"Road Code {code}",
                         method="update",
                         args=[{"y": [
                                    traffic_data['Light_Vehicle'][code],
                                    traffic_data['Commercial_Vehicle'][code],
                                    traffic_data['Heavy_Vehicle'][code],
                                    traffic_data['All_Vehicle'][code],
                                    traffic_data['Congestion_Index'][code],
                                    traffic_data['CO2_Equivalent'][code],
                                    traffic_data['Average_Vehicle_Speed'][code]                                    
                                ]}]
                    ) for code in road_codes
                ],
                direction="down",
                pad={"r": 30, "t": 0},
                showactive=True,
                x=1,
                xanchor="left",
                y=1.5,
                yanchor="top"
            ),
        ]
    )

    # fig.show()
    return fig

scheduler = BackgroundScheduler()
scheduler.add_job(func=download_data, trigger="interval", hours=24)
scheduler.start()

app = Flask(__name__)

@app.route("/")
def index():
    global road_codes
    
    traffic_data = get_data()
    
    fig = plot_time_series_data(traffic_data, road_codes)
    plot_html = pio.to_html(fig, full_html=False)
    
    return render_template('index.html', plot_html=plot_html)

if __name__ == "__main__":    
    app.run(debug=True)