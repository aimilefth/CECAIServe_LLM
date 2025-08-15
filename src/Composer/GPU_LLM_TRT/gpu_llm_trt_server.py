"""
GPU + TensorRT-LLM server for text-generation.
"""
import os
import time
import numpy as np
import experiment_server
import power_scraper
import utils
from typing import List, Tuple
from tensorrt_llm.hlapi import LLM, BuildConfig, SamplingParams


class GpuLlmTrtServer(experiment_server.BaseExperimentServer):
    def __init__(self, logger):
        super().__init__(logger)
        if LLM is None:
            raise ImportError("tensorrt_llm is not installed. Please check the environment.")

        # LLM configs
        self.server_configs['GPU_DEVICE_INDEX'] = int(os.environ['GPU_DEVICE_INDEX'])
        self.server_configs['MAX_LENGTH']  = int(os.environ.get('MAX_LENGTH', '1024'))
        self.server_configs['TEMPERATURE'] = float(os.environ.get('TEMPERATURE', '1.0'))
        self.server_configs['TOP_P']       = float(os.environ.get('TOP_P', '1.0'))
        # Set this to match the correct NVIDIA GPU ORDERING. Solves this issue https://github.com/LostRuins/koboldcpp/issues/1023
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        self.llm = None
        self.build_config = None
        self.power_scraper = None

        self.init_kernel()
        self.warm_up()

    def init_kernel(self):
        start = time.perf_counter()
        model_path = self.server_configs['MODEL_PATH']
        self.build_config = BuildConfig()
        self.build_config.max_seq_len = self.server_configs['MAX_LENGTH']
        self.build_config.opt_batch_size = self.server_configs['BATCH_SIZE']
        self.server_configs['BUILD_CONFIG'] = self.build_config
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            build_config=self.build_config,
        )

        self.power_scraper = power_scraper.power_scraper(self.server_configs['GPU_DEVICE_INDEX'])
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

    def experiment_single(self, input, run_total=1):
        # input is a single prompt string
        prompts, sparams = self._prepare_generation([input])
        outputs = self.llm.generate(prompts, sparams)
        text = outputs[0].outputs[0].text
        # Return as np.array of dtype=object to fit the framework's numpy expectations
        return np.array([text], dtype=object)

    def experiment_multiple(self, dataset, run_total):
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
