import requests

road_code = "ILM_92130_2385"
url = f"http://192.168.10.220:8080/ngsi-ld/v1/temporal/entities/urn:ngsi-ld:TrafficFlowObserved:{road_code}"

headers = {
    "Link": '<https://easy-global-market.github.io/c2jn-data-models/jsonld-contexts/c2jn-compound.jsonld>; rel="http://www.w3.org/ns/json-ld#context"; type="application/ld+json"'
}

try:
    response = requests.get(url, headers=headers)
    response.raise_for_status()
except requests.exceptions.HTTPError:
    print(response.json())

data = response.json()
print(data)
