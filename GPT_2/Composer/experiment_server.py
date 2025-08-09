import os
import json
import numpy as np
from flask import Response
import base_server

class BaseExperimentServer(base_server.BaseServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.experiment_configs = {}
        self.set_experiment_configs()

    def set_experiment_configs(self):
        # For LLMs we don't need image shapes, but we keep placeholders for the base API
        self.experiment_configs['expected_input'] = (None,)  # one dim: string prompt
        self.experiment_configs['expected_output'] = (None,) # string output
        self.experiment_configs['list_prompts'] = []

    def decode_input(self, indata):
        # Expect JSON: { "prompts": [str, ...], "gen": { optional overrides } }
        try:
            payload = json.loads(indata.decode('utf-8'))
        except Exception:
            # Allow raw string as single prompt
            payload = {"prompts": [indata.decode('utf-8')]}
        prompts = payload.get('prompts', [])
        if isinstance(prompts, str):
            prompts = [prompts]
        self.experiment_configs['list_prompts'] = prompts

        # merge any generation overrides into env/defaults by simply storing here;
        # platform server (Transformers) already reads env; we’ll pass via self if needed later.
        gen = payload.get('gen', {})
        self.experiment_configs['gen_overrides'] = gen

        run_total = len(prompts)
        decoded_input = prompts
        return decoded_input, run_total

    def create_and_preprocess(self, decoded_input, run_total):
        # Nothing to preprocess; just batch prompts for THR mode
        bs = self.server_configs['BATCH_SIZE']
        if self.server_configs['SERVER_MODE'] == 0:
            # LAT: expect single prompt
            if run_total != 1:
                raise AssertionError(f"Latency mode expects exactly 1 prompt, got {run_total}")
            return decoded_input[0]
        # THR: split into batches of size BATCH_SIZE (list of lists)
        dataset = [decoded_input[i:i+bs] for i in range(0, run_total, bs)]
        return dataset

    def postprocess(self, exp_output, run_total):
        # exp_output is np.array(dtype=object) of generated texts (length run_total)
        outputs = list(exp_output.tolist())
        # Map back to prompt strings
        result = []
        for i, text in enumerate(outputs):
            result.append({"prompt": self.experiment_configs['list_prompts'][i], "generated_text": text})
        return result

    def encode_output(self, output):
        # Return JSON dict: { "<prompt>": "<generated_text>", ... }
        out_dict = {}
        for item in output:
            out_dict[item['prompt']] = item['generated_text']
        return out_dict

    def send_response(self, encoded_output):
        return Response(response=json.dumps(encoded_output), status=200, mimetype="application/json")
