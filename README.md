# traffic_intensity_monitoring

# Contributors

Contributors for this project are:
- Myra Alvi
- Aung Kaung Myat

# Tasks
|Tasks|Contributor|
|-----|-----------|
|Data Collector|Myra Alvai|
|Traffic Predictor|Myra Alvai|
|Data Injector|Aung Kaung MYAT|
|Containerization|Aung Kaung MYAT|

# Requirements
- Stellio Context Broker [version 2.12.0](https://github.com/stellio-hub/stellio-context-broker/releases/tag/2.12.0)
- Curl
- Python 3.10
- Docker
- Docker Compose
- Google Developer API Key

# Documentation

### Guide for Data Collector
The docker image for data collector can be download using the following command.
```bash
docker pull akmyat/traffic_intensity_monitoring:data_collector_alpha
```
To run,
```bash
docker run -it --rm --network host akmyat/traffic_intensity_monitoring:data_collector_alpha
```

To use the API, we need Google API Key, coordinates for origin location, coordinates for destination location. If you would like to push data directly to Stellio Context Broker, broker address and port is needed.

The origin and destination should be same road, city, and country.

To get data as dictionary key-value pairs,
```bash
curl -X GET "http://app-host-domain-or-ip:7000/traffic-data?api_key={Google_API_KEY}&origin_coordinates={ORIGIN_COORDINATES}&destination_coordinates={DESTINATION_COORDINATE}"
```

To get data as JSON-LD format,
```bash
curl -X GET "http://app-host-domain-or-ip:7000/traffic-data-ngsild?api_key={Google_API_KEY}&origin_coordinates={ORIGIN_COORDINATES}&destination_coordinates={DESTINATION_COORDINATE}"
```

To push data to Context Broker,
```bash
curl -X GET "http://app-host-domain-or-ip:7000/traffic-data-broker?api_key={Google_API_KEY}&origin_coordinates={ORIGIN_COORDINATES}&destination_coordinates={DESTINATION_COORDINATE}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To start logging data to broker, frequency is require,
```bash
curl -X GET "http://app-host-domain-or-ip:7000/start-traffic-data-logging?api_key={Google_API_KEY}&origin_coordinates={ORIGIN_COORDINATES}&destination_coordinates={DESTINATION_COORDINATE}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}&time_delta={FREQUENCY_IN_MINS}"
```

To stop logging data to broker,
```bash
curl -X GET "http://app-host-domain-or-ip:7000/stop-traffic-data-logging"
```

### Guid for Data Injector
The docker image for data injector can be download using the following command.
```bash
docker pull akmyat/traffic_intensity_monitoring:data_injector_alpha
```
To run,
```bash
docker run -it --rm --network host akmyat/traffic_intensity_monitoring:data_injector_alpha
```

To inject data from csv file to context broker,
```bash
curl -X POST "http://app-host-domain-or-ip:8000/upload-csv-to-broker?broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}" -F "file=@{/path/to/file.csv}"
```

### Guide for Data Predictor
The docker image for data predictor can be download using the following command.
```bash
docker pull akmyat/traffic_intensity_monitoring:data_predictor_alpha
```
To run,
```bash
docker run -it --rm --network host akmyat/traffic_intensity_monitoring:data_predictor_alpha
```

To predict next day's traffic intensity or average vehicle speed, we need to train the model first.

To train a model, we need to provide history data by specifying the start and end date in UTC format(e.g. YYYY-MM-DDTHH:MM:SSZ)

To get data for specified date time from context broker,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/get_csv?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To train model for traffic intensity prediction,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/train_traffic_intensity_model?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To train model for average vehicle speed prediction,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/train_average_vehicle_speed_model?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To predict next day's traffic intensity,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/predict_traffic_intensity?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To predict next day's average vehicle speed,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/predict_average_vehicle_speed?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To plot the results of traffic intensity prediction,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/plot_traffic_intensity?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```

To plot the results of average vehicle speed prediction,
```bash
curl -X GET "http://app-host-domain-or-ip:9000/plot_average_vehicle_speed?start_time={START_TIME}&end_time={END_TIME}&id={ID}&broker_address={BROKER_ADDRESS}&broker_port={BROKER_PORT}"
```