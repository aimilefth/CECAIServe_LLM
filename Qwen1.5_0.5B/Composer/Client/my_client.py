import os
import time
import json
import requests
import logging # Import the logging module
import base_client

class MyClient(base_client.BaseClient):
    def __init__(self, address):
        super().__init__(address)

    def send_request(self, url, dataset_path):
        # dataset_path points to prompts.json: { "prompts": ["...", "..."] }
        with open(dataset_path, 'r') as f:
            payload = json.load(f)

        # --- START: New logic for handling DATASET_SIZE ---

        # 1. Get the DATASET_SIZE environment variable. Use .get() to avoid errors if it's not set.
        dataset_size_str = os.environ.get('DATASET_SIZE')

        if dataset_size_str:
            try:
                # 2. Try to convert the variable to an integer.
                num_prompts = int(dataset_size_str)
                
                # 3. Only slice if the number is positive.
                if num_prompts > 0:
                    original_count = len(payload['prompts'])
                    # 4. Slice the list of prompts to the desired size.
                    payload['prompts'] = payload['prompts'][:num_prompts]
                    new_count = len(payload['prompts'])
                    logging.info(f"DATASET_SIZE is set to {num_prompts}. Sending {new_count}/{original_count} prompts.")
                else:
                    # Handle cases where DATASET_SIZE is 0 or negative
                    logging.info("DATASET_SIZE is not a positive integer. Sending all prompts.")
            except ValueError:
                # 5. If conversion fails (e.g., it's the default "Default" string), log a warning and proceed with all prompts.
                logging.warning(
                    f"DATASET_SIZE ('{dataset_size_str}') is not a valid integer. Sending all prompts."
                )
        
        # --- END: New logic for handling DATASET_SIZE ---

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