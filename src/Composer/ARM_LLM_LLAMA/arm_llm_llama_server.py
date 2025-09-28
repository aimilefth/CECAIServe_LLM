"""
Author: Aimilios Leftheriotis
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module defines the ArmLlmLlamaServer class, a specialized server for running GGUF models on ARM CPUs using the llama-cpp-python library. It inherits from ModelServer and handles the specifics of CPU-based inference on ARM architecture.

Overview:
- The ArmLlmLlamaServer utilizes `llama-cpp` for efficient, CPU-bound inference of LLMs on ARM platforms (e.g., NVIDIA Jetson CPU).
- It is designed to load and execute models in the GGUF format.
- It implements abstract methods to manage the model's lifecycle, execute inference, and collect platform-specific power metrics.

Classes:
- ArmLlmLlamaServer: A server that manages a `llama_cpp.Llama` inference
  engine on an ARM CPU.

Methods:
- __init__(self, logger): Initializes the server, prepares the llama.cpp engine, and instantiates the power scraper for the ARM device.
- init_inference_engine(self): Configures and creates the `llama_cpp.Llama` inference object from a GGUF model file for ARM architecture.
- warm_up(self): Executes a short inference with a dummy prompt to load the model and prepare the engine.
- inference_single(self, input, run_total=1): Runs inference for a single prompt.
- inference_batch(self, dataset, run_total): Runs inference sequentially for a dataset of multiple prompts.
- platform_preprocess(self, data): Placeholder for platform-specific preprocessing.
- platform_postprocess(self, data): Calculates generation metrics using token counts directly provided by the llama-cpp-python inference output.
- dummy_inference(self): Executes a dummy inference workload for power measurement.
- scrape_power_data(self): Interfaces with `power_scraper` to collect real-time power metrics from the ARM hardware.

This file is a core component of the Composer's platform-specific logic and is NOT meant to be edited directly.
"""

import os
import time
import numpy as np
import model_server
import power_scraper
import utils

from llama_cpp import Llama

class ArmLlmLlamaServer(model_server.ModelServer):
    def __init__(self, logger):
        super().__init__(logger)
        # LLM configs
        self.server_configs['MAX_LENGTH']  = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['TEMPERATURE'] = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']       = float(os.environ.get('TOP_P', '1.0'))

        self.llm = None
        self.power_scraper = None

        self._current_request_token_counts = None # Used to get the generated token counts

        self.init_inference_engine()
        self.warm_up()

    def init_inference_engine(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']
        # Construct Llama
        self.llm = Llama(
            model_path=model_path,
            n_ctx=self.server_configs['MAX_LENGTH'],
        )
        self.power_scraper = power_scraper.power_scraper()
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time: {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        _ = self.llm(
            "Hello",
            max_tokens=64,
            temperature=self.server_configs['TEMPERATURE'],
            top_p=self.server_configs['TOP_P'],
        )
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def _generate_one(self, prompt: str) -> (str, int):
        """Generates a single completion and returns the text and token count."""
        out = self.llm(
            prompt,
            max_tokens=None, # Let it run until eos or max_length
            temperature=self.server_configs['TEMPERATURE'],
            top_p=self.server_configs['TOP_P'],
        )
        try:
            txt = out["choices"][0]["text"]
            # Extract token count directly from the usage stats
            completion_tokens = out["usage"]["completion_tokens"]
        except Exception:
            # Fallback: best-effort stringify
            txt = str(out)
            completion_tokens = 0 # Cannot determine token count
        return txt, completion_tokens

    def inference_single(self, input, run_total=1):
        """Handles a single prompt request."""
        # input is a single prompt string
        text, token_count = self._generate_one(input)
        # Store the token count for platform_postprocess
        self._current_request_token_counts = [token_count]
        return np.array([text], dtype=object)

    def inference_batch(self, dataset, run_total):
        """Handles batched prompt requests."""
        # dataset is a list of batched lists of strings
        output_list = []
        token_counts = []
        for batch_prompts in dataset:
            batch_texts = []
            for p in batch_prompts:
                text, count = self._generate_one(p)
                batch_texts.append(text)
                token_counts.append(count)
            output_list.append(np.array(batch_texts, dtype=object))
        
        # Store all token counts for platform_postprocess
        self._current_request_token_counts = token_counts

        if not output_list:
            return np.array([], dtype=object)

        start = time.perf_counter()
        exp_output = np.concatenate(output_list, axis=0)
        end = time.perf_counter()
        self.log(f"Concat output time: {(end - start) * 1000:.2f} ms")
        return exp_output[:run_total]

    def platform_preprocess(self, data):
        return data

    def platform_postprocess(self, data):
        """
        Calculate generation metrics using token counts gathered during inference.
        """
        try:
            # Use the token counts we saved during the inference run
            gen_tokens_per_prompt = self._current_request_token_counts

            total_generated_tokens = int(sum(gen_tokens_per_prompt))

            # Use only the core model execution time for tokens/sec calculation
            exp_time = float(self.inference_timings_s.get('inference') or 0.0)
            avg_tokens_per_second = (total_generated_tokens / exp_time) if exp_time > 0 else 0.0

            # Persist the calculated metrics to the metrics dictionary
            self.inference_metrics_s['gen_tokens_per_prompt'] = gen_tokens_per_prompt
            self.inference_metrics_s['total_generated_tokens'] = total_generated_tokens
            self.inference_metrics_s['avg_tokens_per_second'] = avg_tokens_per_second

        except Exception as e:
            self.log(f"platform_postprocess metrics computation failed: {e}")
        
        # Clear the temporary list for the next request
        self._current_request_token_counts = None
        
        # Return the original data to ensure the rest of the pipeline remains unaffected
        return data

    def dummy_inference(self):
        _ = self.llm(
            "Ping",
            max_tokens=64,
            temperature=self.server_configs['TEMPERATURE'],
            top_p=self.server_configs['TOP_P'],
        )

    def scrape_power_data(self):
        return self.power_scraper.get_power()
