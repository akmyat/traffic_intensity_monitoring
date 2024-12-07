import logging
from typing import Optional, Dict, List

import requests

logging.basicConfig(level=logging.DEBUG)


class C2JNBrokerCommunicator:
    def __init__(self, broker_gateway: str, client_id: str, client_secret: str) -> None:
        """
        Initialize the values.

        Parameters:
            broker_gateway(str): address of the broker gateway
            client_id(str): Client Identity
            client_secret(str): Client Password
        """
        self.broker_gateway = broker_gateway
        self.client_id = client_id
        self.client_secret = client_secret

        self.acess_token = None

        self.stateholder_types = {
            "ServiceProvider": "status.user==urn:ngsi-ld:ServiceProvider",
            "PlatformProvider": "status.user==urn:ngsi-ld:PlatformProvider",
            "Customer": "user==urn:ngsi-ld:Customer",
        }

    def __authenticate(self):
        """
        Authenticate to C2JN Context Broker and obtain Access Token.

        Return:
            True(bool): authentication successful
            False(bool): authentication unsuccessful
        """
        auth_url = (
            "https://sso.c2jn.fr/auth/realms/smart-city/protocol/openid-connect/token"
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        }

        try:
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logging.error("Authentication to Context Broker Failed")
            return False

        self.access_token = response.json()["access_token"]
        return True

    def fetch_traffic_data(self, road_code: str) -> Optional[Dict]:
        """
        Fetch the Ecomob Traffic data from the C2JN Stellio Context Broker.

        Parameters:
            road_code(str): the code name of the road in Issy-les-Moulineaux
        Return:
            None: if it is failed to authenticated or failed to fetch data
            traffic_data(dict): ecomob traffic data for the specified road
        """
        if not self.__authenticate():
            return None

        api_url = f"https://{self.broker_gateway}/ngsi-ld/v1/temporal/entities/urn:ngsi-ld:TrafficFlowObserved:{road_code}?options=temporalValues"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "NGSILD-Tenant": "urn:ngsi-ld:tenant:smart-city",
        }

        try:
            response = requests.get(api_url, headers=headers)
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            logging.error("Failed to get the traffic data from Context Broker")
            return None

        return response.json()

    def put_predicted_traffic_data(self, road_code: str, predicted_data: dict) -> bool:
        """
        Put the predicted traffic data to Context Broker.

        Parameters:
            road_code(str): the code name of the road in Issy-les-Moulineaux
            predicted_data(dict): the predicted traffic data
        Return:
            False: if it is failed to authenticated or failed to send predicted data to Context Broker
            True: if sending predicted data to Context Broker successful
        """
        if not self.__authenticate():
            return False

        api_url = f"https://{self.broker_gateway}/ngsi-ld/v1/entities/urn:ngsi-ld:TrafficFlowObserved:{road_code}/attrs"
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "NGSILD-Tenant": "urn:ngsi-ld:tenant:smart-city",
            "Content-Type": "application/ld+json",
        }

        try:
            response = requests.post(api_url, headers=headers, data=predicted_data)
            if response.status_code != 204:
                raise requests.exceptions.HTTPError(
                    f"Unexpected status code: {response.status_code}", response=response
                )
        except requests.exceptions.HTTPError:
            logging.error("Failed to put predicted traffic data to Context Broker")
            return False

        return True