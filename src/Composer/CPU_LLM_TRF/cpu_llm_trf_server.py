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
from transformers.generation.streamers import BaseStreamer  # HF >= 4.26

class TimingStreamer(BaseStreamer):
    """
    Records per-token arrival times during generation for a batch.
    Works with HF generate(streamer=...).
    """
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.start_time = None          # set right before calling generate
        self.first_token_time = [None] * batch_size
        self.token_times = [[] for _ in range(batch_size)]
        self.end_time = None

    def put(self, value):
        """
        value: Tensor or list with shape (batch_size,) per decoding step.
        Called once per new token.
        """
        now = time.perf_counter()
        if isinstance(value, torch.Tensor):
            # value shape is (batch_size,) in greedy/sampling modes
            value = value.tolist()
        # record a timestamp for each sequence in the batch
        for i in range(min(len(value), self.batch_size)):
            if self.first_token_time[i] is None:
                self.first_token_time[i] = now
            self.token_times[i].append(now)

    def end(self):
        self.end_time = time.perf_counter()

    def metrics_for_batch(self):
        """
        Returns a list of dicts (one per sequence) with:
        - ttft_s
        - per_token_intervals_s: list of Δt between consecutive tokens
        - tokens_per_second
        - generated_tokens (count, includes EOS if produced)
        """
        out = []
        for i in range(self.batch_size):
            ft = self.first_token_time[i]
            times = self.token_times[i]
            if ft is None or not times or self.end_time is None or self.start_time is None:
                out.append({
                    "ttft_s": None,
                    "per_token_intervals_s": [],
                    "tokens_per_second": None,
                    "generated_tokens": 0
                })
                continue

            # TTFT = time from "start" to first produced token
            ttft = ft - self.start_time

            # Δt between tokens
            intervals = [t2 - t1 for t1, t2 in zip(times[:-1], times[1:])]

            # throughput after first token
            gen_window_s = max(self.end_time - ft, 1e-9)
            tps = len(times) / gen_window_s  # average tokens/sec after TTFT

            out.append({
                "ttft_s": ttft,
                "per_token_intervals_s": intervals,
                "tokens_per_second": tps,
                "generated_tokens": len(times),
            })
        return out

class CpuLlmTrfServer(experiment_server.BaseExperimentServer):
    def __init__(self, logger):
        super().__init__(logger)
        self.server_configs['NUM_THREADS'] = int(os.environ.get('NUM_THREADS', '-1'))
        self.server_configs['NUM_CPUS']   = os.environ.get('NUM_CPUS', 'Default')
        self.server_configs['CPU_SET']    = os.environ.get('CPU_SET', 'Default')
        self.server_configs['PRECISION'] = os.environ['PRECISION']
        # LLM configs
        self.server_configs['MAX_LENGTH']      = int(os.environ.get('MAX_LENGTH', '1024'))
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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generator = transformers.AutoModelForCausalLM.from_pretrained(model_path)
        self.generator.eval()
        # Build a pipeline for batched generation
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.generator,
            tokenizer=self.tokenizer,
            torch_dtype=map_precision_to_dtype(self.server_configs['PRECISION']),
            device=-1 # CPU
        )
        # optional: dtype by precision
        dtype = map_precision_to_dtype(self.server_configs['PRECISION'])
        self.generator.to(dtype=dtype)

        self.power_scraper = power_scraper.power_scraper()
        end = time.perf_counter()
        self.once_timings['init'] = end - start
        self.log(f"Initialize time: {self.once_timings['init'] * 1000:.2f} ms")

    def warm_up(self):
        start = time.perf_counter()
        _ = self.pipeline("Hello", max_length=128, truncation=True, max_new_tokens=None,)
        end = time.perf_counter()
        self.once_timings['warm_up'] = end - start
        self.log(f"Warmup time: {self.once_timings['warm_up'] * 1000:.2f} ms")

    def experiment_single(self, input, run_total=1):
        texts, metrics = self._generate_with_metrics([input])
        # attach metrics to inference_metrics_s for observability/logging
        m = metrics[0]
        self.inference_metrics_s['ttft_s'] = m['ttft_s']
        self.inference_metrics_s['tokens_per_second'] = m['tokens_per_second']
        self.inference_metrics_s['mean_time_between_tokens_s'] = (
            sum(m['per_token_intervals_s']) / len(m['per_token_intervals_s'])
            if m['per_token_intervals_s'] else None
        )
        self.inference_metrics_s['generated_tokens'] = m['generated_tokens']
        return np.array([texts[0]], dtype=object)

    def experiment_multiple(self, dataset, run_total):
        output_list = []
        all_ttft = []
        all_tps = []
        all_mean_tbt = []
        all_generated_tokens = []
        total_prompts = 0

        for batch_prompts in dataset:
            texts, metrics = self._generate_with_metrics(batch_prompts)
            output_list.append(np.array(texts, dtype=object))
            for m in metrics:
                all_ttft.append(m['ttft_s'])
                all_tps.append(m['tokens_per_second'])
                mean_tbt = (sum(m['per_token_intervals_s']) / len(m['per_token_intervals_s'])
                            if m['per_token_intervals_s'] else None)
                all_mean_tbt.append(mean_tbt)
                all_generated_tokens.append(m["generated_tokens"])
                total_prompts += 1

        if output_list:
            exp_output = np.concatenate(output_list, axis=0)[:run_total]
        else:
            exp_output = np.array([], dtype=object)

        def _avg(xs):
            xs = [x for x in xs if x is not None]
            return (sum(xs) / len(xs)) if xs else None

        self.inference_metrics_s['per_prompt_ttft_s'] = all_ttft
        self.inference_metrics_s['per_prompt_tokens_per_second'] = all_tps
        self.inference_metrics_s['per_prompt_mean_time_between_tokens_s'] = all_mean_tbt
        self.inference_metrics_s['per_prompt_generated_tokens'] = all_generated_tokens
        self.inference_metrics_s['avg_ttft_s'] = _avg(all_ttft)
        self.inference_metrics_s['avg_tokens_per_second'] = _avg(all_tps)
        self.inference_metrics_s['avg_time_between_tokens_s'] = _avg(all_mean_tbt)

        return exp_output

    def platform_preprocess(self, data):
        """Preprocess the input, specific to the platform requirement, not the experiment ones. Used on create_and_preprocess as the last step."""
        return data

    def platform_postprocess(self, data):
        """Postprocess the output, specific to the platform requirement, not the experiment ones. Used on postprocess as the first step."""
        return data

    def dummy_inference(self):
        _ = self.pipeline("Ping", max_length=128, truncation=True, max_new_tokens=None,)

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
    
    # ---- NEW helper that returns (texts, per_prompt_metrics) ----
    def _generate_with_metrics(self, prompts, max_new_tokens=None):
        # defaults/overrides
        do_sample   = self.server_configs['DO_SAMPLE']
        temperature = self.server_configs['TEMPERATURE']
        top_p       = self.server_configs['TOP_P']
        max_len     = self.server_configs['MAX_LENGTH']
        if max_new_tokens is None:
            # prefer new-tokens cap if you were previously using max_length
            max_new_tokens = max(16, min(256, max_len))

        batch = self.tokenizer(prompts, return_tensors="pt", padding=True)
        # CPU
        input_ids = batch["input_ids"]
        attn_mask = batch.get("attention_mask", None)

        streamer = TimingStreamer(batch_size=len(prompts))
        streamer.start_time = time.perf_counter()

        with torch.no_grad():
            outputs = self.generator.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_length=max_len,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,  # <-- timestamps per token
                return_dict_in_generate=True,
                output_scores=False,
            )
        streamer.end()  # defensively mark end (HF will call it too)

        # decode
        sequences = outputs.sequences  # shape (batch, seq_len)
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)

        # compute metrics per prompt
        per_prompt = streamer.metrics_for_batch()
        return texts, per_prompt

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