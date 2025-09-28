"""
Author: Aimilios Leftheriotis
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module defines the AgxOrinLlmVllmServer class, a specialized server for the NVIDIA Jetson AGX Orin using the vLLM inference engine. It is tailored for high-throughput LLM serving on this edge device.

Overview:
- The AgxOrinLlmVllmServer class utilizes the vLLM engine for batched, high-throughput text generation on the AGX Orin's integrated GPU.
- It is designed to load and execute vLLM-compatible Hugging Face models, with GPU memory utilization constraints to ensure stability on the edge platform.
- It implements abstract methods for model lifecycle management, inference, and power measurement specific to the AGX Orin.

Classes:
- AgxOrinLlmVllmServer: A server managing a `vllm.LLM` engine on an AGX Orin device.

Methods:
- __init__(self, logger): Initializes the server, prepares the vLLM engine, and instantiates the Jetson-specific power scraper.
- init_inference_engine(self): Configures and creates the `vllm.LLM` engine, setting `gpu_memory_utilization` to prevent out-of-memory errors on the device.
- warm_up(self): Executes a single inference with a dummy prompt to initialize the vLLM engine and its GPU workers.
- inference_single(self, input, run_total=1): Runs inference for a single prompt.
- inference_batch(self, dataset, run_total): Runs inference for a dataset of prompts, leveraging vLLM's batching on the Orin GPU.
- platform_preprocess(self, data): Placeholder for platform-specific preprocessing.
- platform_postprocess(self, data): Calculates generation metrics using the vLLM tokenizer to count generated tokens.
- dummy_inference(self): Executes a dummy inference workload for power measurement.
- scrape_power_data(self): Interfaces with `power_scraper` to collect power metrics from the AGX Orin hardware.

This file is a core component of the Composer's platform-specific logic and is NOT meant to be edited directly.
"""

import os
import time
import numpy as np
import logging
import model_server
import power_scraper
import utils
from vllm import LLM, SamplingParams

class AgxOrinLlmVllmServer(model_server.ModelServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.server_configs['PRECISION']  = os.environ['PRECISION']
        # LLM configs
        self.server_configs['MAX_LENGTH']      = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['TEMPERATURE']     = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']           = float(os.environ.get('TOP_P', '1.0'))

        self.llm = None
        self.sampling_params = None
        self.power_scraper = None

        self.init_inference_engine()
        self.warm_up()

    def init_inference_engine(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']

        vllm_dtype = map_precision_to_vllm_dtype(self.server_configs['PRECISION'])
        self.log(f"Setting vLLM model dtype to: {vllm_dtype}")
        
        self.llm = LLM(
            model=model_path,
            dtype=vllm_dtype,
            gpu_memory_utilization=0.7, # It limits used RAM, so that it doesnt explode in AGX
        )
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

    def inference_single(self, input, run_total=1):
        # input is a single prompt string
        outs = self.llm.generate([input], self.sampling_params)
        text = outs[0].outputs[0].text
        return np.array([text], dtype=object)

    def inference_batch(self, dataset, run_total):
        # dataset is a list of batched lists of strings (created by model_server)
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
        """Preprocess the input, specific to the platform requirement, not the inference ones. Used on create_and_preprocess as the last step."""
        return data

    def platform_postprocess(self, data):
        """
        Calculate generation metrics without altering timings or outputs.
        'data' is the array of generated texts.
        We compute per-prompt generated tokens, total tokens, and avg tokens/sec
        over the model 'inference' time, then return 'data' unchanged.
        """
        try:
            # Retrieve the tokenizer from the vLLM engine
            tokenizer = self.llm.get_tokenizer()

            # Ensure we have a plain list of strings for processing
            if isinstance(data, np.ndarray):
                gen_texts = data.tolist()
            else:
                gen_texts = list(data)

            # Count generated tokens for each prompt's output
            # We use add_special_tokens=False to count only the content tokens
            gen_tokens_per_prompt = [
                len(tokenizer.encode(t, add_special_tokens=False)) for t in gen_texts
            ]

            total_generated_tokens = int(sum(gen_tokens_per_prompt))

            # Use only the core model execution time for tokens/sec calculation
            exp_time = float(self.inference_timings_s.get('inference') or 0.0)
            avg_tokens_per_second = (total_generated_tokens / exp_time) if exp_time > 0 else 0.0

            # Persist the calculated metrics to the metrics dictionary
            self.inference_metrics_s['gen_tokens_per_prompt'] = gen_tokens_per_prompt
            self.inference_metrics_s['total_generated_tokens'] = total_generated_tokens
            self.inference_metrics_s['avg_tokens_per_second'] = avg_tokens_per_second

        except Exception as e:
            # It's crucial not to fail the entire request due to a metrics calculation error.
            self.log(f"platform_postprocess metrics computation failed: {e}")

        # Return the original data to ensure the rest of the pipeline remains unaffected
        return data

    def dummy_inference(self):
        _ = self.llm.generate(["Ping"], self.sampling_params)

    def scrape_power_data(self):
        power = self.power_scraper.get_power()
        return power
    
def map_precision_to_vllm_dtype(precision_str):
    """
    Map env PRECISION to a vLLM-compatible dtype string.
    FP32 -> 'float32'
    FP16 -> 'float16'
    BF16 -> 'bfloat16'
    auto -> 'auto'

    Because xformers are not installed, FP32 will fail here!
    """
    s = str(precision_str).strip().lower()
    mapping = {
        'fp32': 'float32',
        'float32': 'float32',
        'fp16': 'float16',
        'float16': 'float16',
        'bf16': 'bfloat16',
        'bfloat16': 'bfloat16',
        'auto': 'auto',
    }
    # Use .get() with a default 'auto' to handle unknown inputs gracefully
    return mapping.get(s, 'auto')