"""
CPU + Transformers server (text-generation) for LLMs.
Mirrors the TF2AIF server shape: init_kernel, warm_up, experiment_single/multiple, platform_pre/postprocess.
"""

import os
import time
import numpy as np
import logging
import transformers
import torch
import experiment_server
import power_scraper
import utils

class CpuLlmTrfServer(experiment_server.BaseExperimentServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.server_configs['NUM_THREADS'] = int(os.environ.get('NUM_THREADS', '-1'))
        self.server_configs['NUM_CPUS']   = os.environ.get('NUM_CPUS', 'Default')
        self.server_configs['CPU_SET']    = os.environ.get('CPU_SET', 'Default')

        # LLM configs
        self.server_configs['MAX_LENGTH']      = int(os.environ.get('MAX_NEW_TOKENS', '1024'))
        self.server_configs['DO_SAMPLE']       = utils.strtobool(os.environ.get('DO_SAMPLE', 'True'))
        self.server_configs['TEMPERATURE']     = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']           = float(os.environ.get('TOP_P', '1.0'))

        self.generator = None
        self.tokenizer = None
        self.pipeline  = None
        self.power_scraper = None

        self.set_threads()
        self.init_kernel()
        self.warm_up()

    def init_kernel(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        self.generator = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        # Build a pipeline for batched generation
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.generator,
            tokenizer=self.tokenizer,
            device=-1 # CPU
        )
        self.power_scraper = power_scraper.power_scraper()
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time: {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        _ = self.pipeline("Hello", max_new_tokens=64)
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def experiment_single(self, input, run_total=1):
        # input is a single prompt string
        out = self.pipeline(
            input,
            max_length=self.server_configs['MAX_NEW_TOKENS'],
            do_sample=self.server_configs['DO_SAMPLE'],
            temperature=self.server_configs['TEMPERATURE'],
            top_p=self.server_configs['TOP_P'],
            num_return_sequences=1
        )
        # Return as np.array of dtype=object to fit the framework's numpy expectations
        text = out[0]['generated_text']
        return np.array([text], dtype=object)

    def experiment_multiple(self, dataset, run_total):
        # dataset is list of batched lists of strings (created by experiment_server)
        output_list = []
        iterations = run_total // self.server_configs['BATCH_SIZE']
        for i, batch_prompts in enumerate(dataset):
            # pad last partial batch to BATCH_SIZE (framework expects fixed shape handling)
            if i == iterations:
                pad = self.server_configs['BATCH_SIZE'] - len(batch_prompts)
                if pad > 0:
                    batch_prompts = batch_prompts + [""] * pad
            out = self.pipeline(
                batch_prompts,
                max_new_tokens=self.server_configs['MAX_NEW_TOKENS'],
                do_sample=self.server_configs['DO_SAMPLE'],
                temperature=self.server_configs['TEMPERATURE'],
                top_p=self.server_configs['TOP_P'],
                num_return_sequences=1
            )
            gen_texts = [o['generated_text'] for o in out]
            valid = self.server_configs['BATCH_SIZE'] if i != iterations else (run_total - iterations*self.server_configs['BATCH_SIZE'])
            output_list.append(np.array(gen_texts[:valid], dtype=object))

        # concatenate along first axis
        start = time.perf_counter()
        exp_output = np.concatenate(output_list, axis=0)
        end = time.perf_counter()
        self.log(f"Concat output time: {(end - start) * 1000:.2f} ms")
        return exp_output

    def platform_preprocess(self, data):
        """Preprocess the input, specific to the platform requirement, not the experiment ones. Used on create_and_preprocess as the last step."""
        return data

    def platform_postprocess(self, data):
        """Postprocess the output, specific to the platform requirement, not the experiment ones. Used on postprocess as the first step."""
        return data

    def dummy_inference(self):
        _ = self.pipeline("Ping", max_new_tokens=32)

    def scrape_power_data(self):
        power = self.power_scraper.get_power()
        return power
    
    def set_threads(self):
        if(self.server_configs['NUM_THREADS'] != -1): # -1 Acts as Default
            self.server_configs['THREADS'] = self.server_configs['NUM_THREADS']
            torch.set_num_threads(self.server_configs['THREADS'])
        elif(self.server_configs['NUM_CPUS'] != 'Default'):
            self.server_configs['THREADS'] = int(self.server_configs['NUM_CPUS'])
            torch.set_num_threads(self.server_configs['THREADS'])
        else:
            self.server_configs['THREADS'] = -1

    def _set_torch_threads(self):
        threads = self.server_configs['NUM_THREADS']
        if threads != -1:
            torch.set_num_threads(threads)
            os.environ['OMP_NUM_THREADS'] = str(threads)
            self.log(f"Set torch/OMP threads to {threads}")
        elif self.server_configs['NUM_CPUS'] != 'Default':
            n = int(self.server_configs['NUM_CPUS'])
            torch.set_num_threads(n)
            os.environ['OMP_NUM_THREADS'] = str(n)
            self.log(f"Set torch/OMP threads from NUM_CPUS to {n}")
        else:
            self.log("Torch threads left to default")