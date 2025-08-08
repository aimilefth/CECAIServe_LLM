import os
import time
import json
import requests
import base_client

class MyClient(base_client.BaseClient):
    def __init__(self, address):
        super().__init__(address)

    def send_request(self, url, dataset_path):
        # dataset_path points to prompts.json: { "prompts": ["...", "..."] }
        with open(dataset_path, 'r') as f:
            payload = json.load(f)
        headers = {'Content-Type': 'application/json'}
        start = time.time()
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        end = time.time()
        if response.status_code != 200:
            raise Exception(f'Inference request failed with status code {response.status_code}')
        latency_s = end - start
        return response, latency_s

    def manage_response(self, response):
        data = json.loads(response.content)
        with open(self.output, 'w') as out:
            out.write(json.dumps(data, indent=2))