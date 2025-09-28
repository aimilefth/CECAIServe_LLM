"""
Author: Aimilios Leftheriotis
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module defines the AgxOrinLlmTrtServer class, a specialized server for the NVIDIA Jetson AGX Orin using the TensorRT-LLM inference engine. It is designed for high-performance, optimized LLM serving on edge devices.

Overview:
- The AgxOrinLlmTrtServer class utilizes the TensorRT-LLM API (`tensorrt_llm.LLM`) to build and run highly optimized TensorRT engines on the AGX Orin's GPU.
- It is designed to load standard Hugging Face models and can build the TensorRT engine on the fly, a process that can be time-consuming.
- It implements memory management configurations (`KvCacheConfig`) to ensure stable operation on the resource-constrained edge device.
- It implements abstract methods for model lifecycle, inference, and power measurement.

Classes:
- AgxOrinLlmTrtServer: A server that manages a TensorRT-LLM engine on an AGX Orin device.

Methods:
- __init__(self, logger): Initializes the server, prepares the TensorRT-LLM engine, and instantiates the Jetson-specific power scraper.
- init_inference_engine(self): Configures and creates the `tensorrt_llm.LLM` object, including build and cache configurations, which may trigger an engine build.
- warm_up(self): Executes a single inference with a dummy prompt to ensure the TensorRT engine is fully loaded and ready for requests.
- inference_single(self, input, run_total=1): Runs inference for a single prompt.
- inference_batch(self, dataset, run_total): Runs inference for a dataset of multiple prompts.
- platform_preprocess(self, data): Placeholder for platform-specific preprocessing.
- platform_postprocess(self, data): Calculates generation metrics by tokenizing the generated text using the engine's tokenizer.
- dummy_inference(self): Executes a dummy inference workload for power measurement.
- scrape_power_data(self): Interfaces with `power_scraper` to collect power metrics from the AGX Orin hardware.
- _prepare_generation(self, prompts): Prepares prompts and dynamic `SamplingParams` to handle variable input lengths and maximize generation length within device constraints.

This file is a core component of the Composer's platform-specific logic and is NOT meant to be edited directly.
"""

import os
import time
import numpy as np
import model_server
import power_scraper
import utils
from typing import List, Tuple
from tensorrt_llm import LLM, BuildConfig, SamplingParams
from tensorrt_llm.plugin import PluginConfig
from tensorrt_llm.llmapi import KvCacheConfig

class AgxOrinLlmTrtServer(model_server.ModelServer):
    def __init__(self, logger):
        super().__init__(logger)
        if LLM is None:
            raise ImportError("tensorrt_llm is not installed. Please check the environment.")

        # LLM configs
        self.server_configs['MAX_LENGTH']  = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['TEMPERATURE'] = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']       = float(os.environ.get('TOP_P', '1.0'))
        self.server_configs['PRECISION']   = os.environ.get('PRECISION', 'auto') 
        self.server_configs['FREE_GPU_MEMORY_FRACTION'] = float(os.environ.get('FREE_GPU_MEMORY_FRACTION', '0.2'))
        self.llm = None
        self.build_config = None
        self.power_scraper = None

        self.init_inference_engine()
        self.warm_up()

    def init_inference_engine(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']
        # Overcome this issue https://github.com/NVIDIA/TensorRT-LLM/issues/1515
        self.plugin_config = PluginConfig()
        self.plugin_config.context_fmha_type = 0
        self.build_config = BuildConfig(
            max_seq_len=self.server_configs['MAX_LENGTH'],
            opt_batch_size=self.server_configs['BATCH_SIZE'],
            max_batch_size=self.server_configs['BATCH_SIZE'],
            plugin_config=self.plugin_config,
        )
        self.kv_cfg = KvCacheConfig(
            free_gpu_memory_fraction=self.server_configs['FREE_GPU_MEMORY_FRACTION'],
        )
        self.server_configs['PLUGIN_CONFIG'] = self.plugin_config
        self.server_configs['BUILD_CONFIG'] = self.build_config
        self.server_configs['KV_CONFIG'] = self.kv_cfg
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype=map_precision_to_trt_dtype(self.server_configs['PRECISION']),
            build_config=self.build_config,
            kv_cache_config=self.kv_cfg,
        )

        self.power_scraper = power_scraper.power_scraper()
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time (includes potential TRT engine build): {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        prompts, sparams = self._prepare_generation(["Hello world!"])
        _ = self.llm.generate(prompts, sparams)  # per-prompt SamplingParams
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def inference_single(self, input, run_total=1):
        # input is a single prompt string
        prompts, sparams = self._prepare_generation([input])
        outputs = self.llm.generate(prompts, sparams)
        text = outputs[0].outputs[0].text
        # Return as np.array of dtype=object to fit the framework's numpy expectations
        return np.array([text], dtype=object)

    def inference_batch(self, dataset, run_total):
        # dataset is a list of batched lists of strings
        output_list = []
        for batch_prompts in dataset:
            prompts, sparams = self._prepare_generation(batch_prompts)
            outputs = self.llm.generate(prompts, sparams)
            gen_texts = [o.outputs[0].text for o in outputs]
            output_list.append(np.array(gen_texts, dtype=object))

        if not output_list:
            return np.array([], dtype=object)
        exp_output = np.concatenate(output_list, axis=0)
        return exp_output[:run_total]

    def platform_preprocess(self, data):
        return data

    def platform_postprocess(self, data):
        """
        Calculate generation metrics by tokenizing the final generated text strings.
        This aligns the logic with all other server implementations.
        """
        try:
            # The tokenizer is a HF Tokenizer object wrapped by the TRT-LLM API
            tokenizer = self.llm.tokenizer

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
        prompts, sparams = self._prepare_generation(["Ping"])
        _ = self.llm.generate(prompts, sparams)

    def scrape_power_data(self):
        return self.power_scraper.get_power()

    def _encode_batch(self, prompts: List[str]):
        """
        Returns (input_ids_list, lengths). Uses the TRT-LLM-wrapped HF tokenizer.
        """
        enc = self.llm.tokenizer.batch_encode_plus(
            prompts,
            add_special_tokens=True,
            return_attention_mask=False,
            truncation=False,   # we'll manage truncation ourselves to be explicit
        )
        ids_list = enc["input_ids"]
        lens = [len(ids) for ids in ids_list]
        return ids_list, lens

    def _prepare_generation(self, prompts: List[str]) -> Tuple[List[str], List[SamplingParams]]:
        """
        Ensures each prompt fits (<= max_seq_len - 1 tokens) and builds a SamplingParams
        with max_tokens = max_seq_len - prompt_len for each prompt.
        """
        max_seq_len = self.server_configs['MAX_LENGTH']
        max_input_len = max_seq_len - 1  # TRT-LLM typically requires at least 1 slot for generation
        ids_list, lens = self._encode_batch(prompts)

        final_prompts: List[str] = []
        params_list: List[SamplingParams] = []

        for orig_text, ids, L in zip(prompts, ids_list, lens):
            # If prompt is too long, trim from the left (tokenizer defaults are left-truncation/padding)
            if L > max_input_len:
                ids = ids[-max_input_len:]
                # Decode back to text to feed LLM a string that will re-tokenize to the same ids
                # Keep special tokens (skip_special_tokens=False) to preserve structure.
                trimmed_text = self.llm.tokenizer.decode(ids, skip_special_tokens=False)
                prompt_text = trimmed_text
                L = len(ids)  # == max_input_len
            else:
                prompt_text = orig_text

            # How many new tokens can we legally generate?
            max_new = max_seq_len - L
            # Safety: we want at least 1 token if possible.
            if max_new < 1:
                # This only happens if L == max_seq_len. Since we trimmed to max_seq_len-1 above,
                # we should never get here; still, guard for robustness.
                max_new = 1

            params_list.append(
                SamplingParams(
                    temperature=self.server_configs['TEMPERATURE'],
                    top_p=self.server_configs['TOP_P'],
                    max_tokens=int(max_new),
                )
            )
            final_prompts.append(prompt_text)

        return final_prompts, params_list

def map_precision_to_trt_dtype(precision_str: str) -> str:
    """
    Map env PRECISION to a TensorRT-LLM-compatible dtype string.
    FP16 -> 'float16'
    BF16 -> 'bfloat16'
    FP32 -> 'float32'
    auto -> 'auto'
    """
    s = str(precision_str).strip().lower()
    mapping = {
        'fp16': 'float16',
        'float16': 'float16',
        'bf16': 'bfloat16',
        'bfloat16': 'bfloat16',
        'fp32': 'float32',
        'float32': 'float32',
        'auto': 'auto',
    }
    # Use .get() with a default 'auto' to handle unknown inputs gracefully
    return mapping.get(s, 'auto')