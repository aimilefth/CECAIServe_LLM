"""
Author: Aimilios Leftheriotis
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module defines the GpuLlmTrfServer class, which provides a specialized server implementation for NVIDIA GPUs using the Hugging Face Transformers library within the CECAIServe framework. It inherits from the project-specific ModelServer and handles the hardware-level details of model inference.

Overview:
- The GpuLlmTrfServer class utilizes the Hugging Face Transformers `pipeline` for text generation, accelerated on an NVIDIA GPU.
- It is designed to load and execute standard Hugging Face language models.
- It handles GPU device selection and memory management implicitly through PyTorch.
- It implements the abstract methods from its parent classes to manage the model's lifecycle, execute inference, and collect hardware-specific power metrics.

Classes:
- GpuLlmTrfServer: A platform-specific server that manages a Transformers pipeline on a GPU device.

Methods:
- __init__(self, logger): Initializes the server, prepares for the inference engine, and instantiates the power scraper.
- init_inference_engine(self): Configures and creates the Transformers text-generation pipeline (`transformers.pipeline`) for GPU execution.
- warm_up(self): Executes a single inference with a dummy prompt to prepare the pipeline and ensure the model is fully loaded onto the GPU.
- inference_single(self, input, run_total=1): Runs inference for a single prompt.
- inference_batch(self, dataset, run_total): Runs inference for a dataset of multiple prompts.
- platform_preprocess(self, data): A placeholder for any platform-specific preprocessing.
- platform_postprocess(self, data): Calculates generation metrics like tokens per second by tokenizing the generated text.
- dummy_inference(self): Executes a dummy inference workload, used by the base server to measure power consumption during the inference phase.
- scrape_power_data(self): Interfaces with the `power_scraper` module to collect real-time power metrics from the NVIDIA GPU hardware.

This file is a core component of the Composer's platform-specific logic and is NOT meant to be edited directly.
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPU_DEVICE_INDEX"]

import time
import numpy as np
import logging
import transformers
import torch
import model_server
import power_scraper
import utils

class GpuLlmTrfServer(model_server.ModelServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.server_configs['PRECISION'] = os.environ['PRECISION']
        # LLM configs
        self.server_configs['GPU_DEVICE_INDEX'] = int(os.environ['GPU_DEVICE_INDEX'])
        self.server_configs['MAX_LENGTH']      = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['DO_SAMPLE']       = utils.strtobool(os.environ.get('DO_SAMPLE', 'True'))
        self.server_configs['TEMPERATURE']     = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']           = float(os.environ.get('TOP_P', '1.0'))

        self.generator = None
        self.tokenizer = None
        self.pipeline  = None
        self.power_scraper = None

        self.init_inference_engine()
        self.warm_up()

    def init_inference_engine(self):
        start = time.perf_counter()
        self._log_cuda_order_torch()
        model_path = self.server_configs['MODEL_PATH']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        # FIX: Explicitly set pad_token if it's not present (for open-ended generation)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        # Build a pipeline for batched generation
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.generator,
            tokenizer=self.tokenizer,
            torch_dtype=map_precision_to_dtype(self.server_configs['PRECISION']),
            device=0 #GPU
        )
        self.power_scraper = power_scraper.power_scraper(self.server_configs['GPU_DEVICE_INDEX'])
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time: {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        _ = self.pipeline("Hello", max_length=128, truncation=True, max_new_tokens=None,)
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def inference_single(self, input, run_total=1):
        # input is a single prompt string
        out = self.pipeline(
            input,
            max_length=self.server_configs['MAX_LENGTH'],
            do_sample=self.server_configs['DO_SAMPLE'],
            temperature=self.server_configs['TEMPERATURE'],
            top_p=self.server_configs['TOP_P'],
            num_return_sequences=1,
            truncation=True,
            max_new_tokens=None,
            return_full_text=False,
        )
        # Return as np.array of dtype=object to fit the framework's numpy expectations
        text = out[0]['generated_text']
        return np.array([text], dtype=object)

    def inference_batch(self, dataset, run_total):
        # dataset is a list of batched lists of strings (created by model_server)
        output_list = []
        
        for batch_prompts in dataset:
            out = self.pipeline(
                batch_prompts,
                max_length=self.server_configs['MAX_LENGTH'],
                do_sample=self.server_configs['DO_SAMPLE'],
                temperature=self.server_configs['TEMPERATURE'],
                top_p=self.server_configs['TOP_P'],
                num_return_sequences=1,
                truncation=True,
                max_new_tokens=None,
                return_full_text=False,
            )
            # The output is a list of lists. Each inner list contains one dictionary.
            gen_texts = [result[0]['generated_text'] for result in out]
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
        'data' is the array/list of generated texts (since return_full_text=False).
        We compute per-prompt generated tokens, total tokens, and avg tokens/sec
        over the model 'inference' time, then return 'data' unchanged.
        """
        try:
            # Ensure we have a plain list of strings
            if isinstance(data, np.ndarray):
                gen_texts = data.tolist()
            else:
                gen_texts = list(data)

            # Count generated tokens per prompt (prompt is not included in generated_text)
            gen_tokens_per_prompt = [
                len(self.tokenizer.encode(t, add_special_tokens=False)) for t in gen_texts
            ]

            total_generated_tokens = int(sum(gen_tokens_per_prompt))

            # Use model time only (as requested): self.inference_timings_s['inference']
            exp_time = float(self.inference_timings_s.get('inference') or 0.0)
            avg_tokens_per_second = (total_generated_tokens / exp_time) if exp_time > 0 else 0.0

            # Persist to the metrics dict (JSON-serializable)
            self.inference_metrics_s['gen_tokens_per_prompt'] = gen_tokens_per_prompt
            self.inference_metrics_s['total_generated_tokens'] = total_generated_tokens
            self.inference_metrics_s['avg_tokens_per_second'] = avg_tokens_per_second

        except Exception as e:
            # Never fail the request because of metrics; just log and proceed
            self.log(f"platform_postprocess metrics computation failed: {e}")

        # Return the same data so the rest of the pipeline is unchanged
        return data

    def dummy_inference(self):
        _ = self.pipeline("Ping", max_length=128, truncation=True, max_new_tokens=None,)

    def scrape_power_data(self):
        power = self.power_scraper.get_power()
        return power
    
    def _log_cuda_order_torch(self,):
        import os, torch
        self.log(f"CUDA_DEVICE_ORDER = {os.getenv('CUDA_DEVICE_ORDER')}")
        self.log(f"CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}")
        self.log(f"NVIDIA_VISIBLE_DEVICES = {os.getenv('NVIDIA_VISIBLE_DEVICES')}")
        n = torch.cuda.device_count()
        self.log(f"torch sees {n} GPU(s)")
        for i in range(n):
            props = torch.cuda.get_device_properties(i)
            self.log(f"[torch] cuda:{i}  name={props.name}  cc={props.major}.{props.minor}  mem={props.total_memory/1024**3:.1f} GiB")


def map_precision_to_dtype(precision_str):
    """
    Map env PRECISION to a torch dtype (or 'auto').
    FP32 -> torch.float32
    FP16 -> torch.float16
    BF16 -> torch.bfloat16
    auto -> 'auto'
    """
    s = str(precision_str).strip().lower()
    mapping = {
        'fp32': torch.float32,
        'float32': torch.float32,
        'fp16': torch.float16,
        'float16': torch.float16,
        'bf16': torch.bfloat16,
        'bfloat16': torch.bfloat16,
        'auto': 'auto',
    }
    return mapping.get(s, 'auto')