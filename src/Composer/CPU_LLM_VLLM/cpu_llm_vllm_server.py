"""
CPU + Transformers server (text-generation) for LLMs.
Mirrors the TF2AIF server shape: init_kernel, warm_up, experiment_single/multiple, platform_pre/postprocess.
"""

import os
import time
import numpy as np
import logging
import experiment_server
import power_scraper
import utils
from vllm import LLM, SamplingParams

class CpuLlmVllmServer(experiment_server.BaseExperimentServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.server_configs['NUM_THREADS'] = int(os.environ.get('NUM_THREADS', '-1'))
        self.server_configs['NUM_CPUS']   = os.environ.get('NUM_CPUS', 'Default')
        self.server_configs['CPU_SET']    = os.environ.get('CPU_SET', 'Default')

        # LLM configs
        self.server_configs['MAX_LENGTH']      = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['TEMPERATURE']     = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']           = float(os.environ.get('TOP_P', '1.0'))

        self.llm = None
        self.sampling_params = None
        self.power_scraper = None

        self.init_kernel()
        self.warm_up()

    def init_kernel(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']
        self.llm = LLM(model=model_path, )
        self.sampling_params = SamplingParams(
            n=1,
            temperature = self.server_configs['TEMPERATURE'],
            top_p = self.server_configs['TOP_P'],
            max_tokens = self.server_configs['MAX_LENGTH'],
        )
        self.power_scraper = power_scraper.power_scraper()
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time: {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        _ = self.llm.generate(["Hello"], self.sampling_params)
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def experiment_single(self, input, run_total=1):
        # input is a single prompt string
        outs = self.llm.generate([input], self.sampling_params)
        text = outs[0].outputs[0].text
        return np.array([text], dtype=object)

    def experiment_multiple(self, dataset, run_total):
        # dataset is a list of batched lists of strings (created by experiment_server)
        output_list = []
        
        for batch_prompts in dataset:
            outs = self.llm.generate(batch_prompts, self.sampling_params)
            gen_texts = [o.outputs[0].text for o in outs]
            output_list.append(np.array(gen_texts, dtype=object))

        # Concatenate all batch results into a single NumPy array
        if not output_list:
            return np.array([], dtype=object)
            
        start = time.perf_counter()
        exp_output = np.concatenate(output_list, axis=0)
        end = time.perf_counter()
        self.log(f"Concat output time: {(end - start) * 1000:.2f} ms")
        
        # Ensure the final output size matches the total number of prompts
        return exp_output[:run_total]

    def platform_preprocess(self, data):
        """Preprocess the input, specific to the platform requirement, not the experiment ones. Used on create_and_preprocess as the last step."""
        return data

    def platform_postprocess(self, data):
        """Postprocess the output, specific to the platform requirement, not the experiment ones. Used on postprocess as the first step."""
        return data

    def dummy_inference(self):
        _ = _ = self.llm.generate(["Ping"], self.sampling_params)

    def scrape_power_data(self):
        power = self.power_scraper.get_power()
        return power
    