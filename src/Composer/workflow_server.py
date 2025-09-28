"""
Author: Aimilios Leftheriotis
Affiliations: Microlab@NTUA, VLSILab@UPatras

This module defines the WorkflowServer, an abstract base class that provides the
core structure and workflow for all inference servers within the CECAIServe
project. It serves as the engine for the Accelerated Inference Service Containers (ASCIs).

Overview:
- The WorkflowServer class is designed to be extended by two levels of subclasses,
  following a Template Method design pattern.
- The `get_inference()` method defines the high-level skeleton of the inference
  workflow, orchestrating a sequence of abstract steps.
- Subclasses are responsible for implementing the concrete logic for these steps,
  allowing for specialization based on the AI experiment and the target hardware.

Architectural Pattern & Subclass Responsibilities:
1.  **Template Method Pattern**: The `get_inference()` method calls a series of
    methods (`decode_input`, `preprocess`, `inference_single`, etc.)
    in a fixed order. Subclasses must provide implementations for these methods.

2.  **Experiment-Specific Subclass (`model_server.ModelServer`)**:
    This class inherits from `WorkflowServer` and is responsible for implementing
    data-handling logic that is specific to the AI task (e.g., image classification)
    but independent of the hardware. This includes:
    - `decode_input`, `preprocess`, `postprocess`, `encode_output`.

3.  **Platform-Specific Subclass (`{AccInf-Platform}_server.py`)**:
    This class inherits from `ModelServer` and implements the logic
    tied directly to the hardware and inference framework (e.g., `GpuTfServer`,
    `AlveoServer`). This includes:
    - `init_inference_engine`, `warm_up`, `inference_single`, `inference_batch`,
      `platform_preprocess`, `platform_postprocess`, and `scrape_power_data`.

Core Functionalities Provided by WorkflowServer:
-   **Configuration Management**: Initializes server configurations, model details,
    and AI characteristics from environment variables.
-   **Inference Workflow Orchestration**: Manages the end-to-end inference
    process from input decoding to output encoding.
-   **Performance Timing**: Meticulously times each stage of the inference pipeline
    (e.g., preprocessing, inference, postprocessing) to provide detailed latency breakdowns.
-   **Metric Calculation & Storage**: Calculates key performance indicators like total
    latency and throughput, and stores them in a limited-size list for retrieval
    via the API.
-   **Power Measurement**: Implements a sophisticated mechanism to measure power
    consumption across different operational phases (idle, preprocessing, inference,
    postprocessing) by running dummy workloads while a separate process scrapes power data.
-   **Logging & Redis Integration**: Provides centralized logging and optional
    integration with Redis for real-time metrics monitoring.

Note:
- This file defines the foundational logic and should NOT be edited directly.
  Custom behavior should be implemented in the appropriate subclass.
"""

import os
import time
from dotenv import load_dotenv
import numpy as np
import utils  # Custom module for utility functions
import multiprocessing
import threading
import datetime
import logging

class WorkflowServer:
    """
    This is the foundational class for server operations across different platforms and experiments.
    It provides methods for managing the server's workflow, from data input to response generation, 
    and also for metrics, logging, and Redis connections.
    """
    def __init__(self, my_logger):
        """Initialize server configurations, metrics, timings, and AI characteristics."""
        self.logger = my_logger
        self.my_redis = None
        self.my_metrics_list = utils.LimitedList(int(os.environ['METRICS_LIST_SIZE']))

        # Configuration settings for the server
        self.server_configs = {
            'MODEL_PATH': os.environ['MODEL_NAME'],
            'BATCH_SIZE': int(os.environ['BATCH_SIZE']),
            'SEND_METRICS': utils.strtobool(os.environ['SEND_METRICS']),
            'AIF_timestamp': datetime.datetime.utcnow().isoformat(),
            'SERVER_MODE': utils.decode_server_mode(os.environ['SERVER_MODE']),  # 0 == LAT, 1 == THR
            'DEBUG_MODE': utils.strtobool(os.environ['DEBUG_MODE']),
            'NODE_NAME': os.environ['NODE_NAME'] # Where server is run, optional
        }

        # Timings related to server operations
        self.once_timings = {
            'init': None,
            'warm_up': None
        }
        self.inference_timings_s = {
            'decode_input': None,
            'preprocess': None,
            'reshape_input': None,
            'inference': None,
            'reshape_output': None,
            'postprocess': None,
            'encode_output': None,
            'full_inference': None,
            'redis_send': None,
            'save_metrics': None
        }
        self.inference_metrics_s = {
            'total_latency': None,
            'inference_latency': None,
            'preprocessing_latency': None,
            'postprocessing_latency': None,
            'throughput': None,
            'dataset_size': None,
            'batch_size': self.server_configs['BATCH_SIZE']
        }

        # Metrics related to power
        self.power_metrics = {

        }

        # Characteristics of the AI Framework
        self.aif_characteristics = {
            'app_name': os.environ['APP_NAME'],
            'network_name': os.environ['NETWORK_NAME'],
            'device': os.environ['AI_DEVICE'],
            'precision': os.environ['PRECISION']
        }
        # Power metrics dummy variables
        self.preprocessing_dummy = None
        self.postprocessing_dummy = None

        self.create_redis()  # Create a Redis connection if required
        self.load_env_variables()

    def log(self, string_message):
        """
        Log a message using the server's logger.
        If 'DEBUG_MODE' is enabled in the server configuration, the message is also printed to the console.
        """
        self.logger.info(string_message)
        if(self.server_configs['DEBUG_MODE']):
            print(string_message)
    
    def load_env_variables(self):
        """Load environment variables from a specified .env file on ENV_FILE env variable."""
        load_dotenv(dotenv_path=os.environ['ENV_FILE'])

    def set_experiment_configs(self):
        """Define experiment configurations. Must be overridden by model_server.py (ModelServer)."""
        raise NotImplementedError('Forgot to overload set_experiment_configs. Must be overridden by model_server.py (ModelServer).')

    def create_redis(self):
        """Attempt to create a Redis connection."""
        if self.server_configs['SEND_METRICS']:
            try:
                utils.read_node_env_variables()
                self.my_redis = utils.create_redis()
                self.log('Created Redis connection')
            except Exception as e:
                self.log(f"Could not create Redis connection: {str(e)}")

    def init_inference_engine(self):
        """
        Initialize one-time AI-framework/platform pair-specific server operations. Must be overridden by {pair}_server.py ({Pair}Server).
        The function should calculate the elapsed time, saving it at self.once_timings['init'], and log it accordingly.
        """
        raise NotImplementedError('Forgot to overload init_inference_engine. Must be overridden by {pair}_server.py ({Pair}Server).')

    def warm_up(self):
        """
        Run first-time platform-specific server operations. Must be overridden by {pair}_server.py ({Pair}Server).
        The function should calculate the elapsed time, saving it at self.once_timings['warm_up'], and log it accordingly.
        """
        raise NotImplementedError('Forgot to overload warm_up. Must be overridden by {pair}_server.py ({Pair}Server).')

    def send_response(self, encoded_output):
        """Send a response after processing. Must be overridden by model_server.py (ModelServer)."""
        raise NotImplementedError('Forgot to overload send_response. Must be overridden by model_server.py (ModelServer).')

    def get_inference(self, indata):
        """
        Handle the entire inference process, including:
        - Decoding input
        - Data preprocessing
        - Inference execution
        - Data postprocessing
        - Encoding output
        """

        # Starting timer for the full inference process
        full_start = time.perf_counter()

        # Useful for power metrics
        if self.preprocessing_dummy is None:
            self.preprocessing_dummy = indata

        # Executing the input decoding process
        decode_input_start = time.perf_counter()
        decoded_input, run_total = self.decode_input(indata=indata)
        assert not (self.server_configs['SERVER_MODE'] == 0 and run_total > 1), \
            f"AssertionError: server_mode is 0 and run_total is > 1, got server_mode: {self.server_configs['SERVER_MODE']} and run_total: {run_total}"
        self.log(f"Dataset size: {run_total}")
        decode_input_end = time.perf_counter()
        self.inference_timings_s['decode_input'] = decode_input_end - decode_input_start
        self.log(f"Decode Input time: {self.inference_timings_s['decode_input']:.9f} s")

        # Executing the dataset creation and preprocessing
        preprocess_start = time.perf_counter()
        dataset = self.preprocess(decoded_input=decoded_input, run_total=run_total)
        preprocess_end = time.perf_counter()
        self.inference_timings_s['preprocess'] = preprocess_end - preprocess_start
        self.log(f"Create and Preprocess time: {self.inference_timings_s['preprocess']:.9f} s")
        
        # Reshaping input data if in Latency Server Mode (self.server_configs['SERVER_MODE'] == 0)
        reshape_input_start = time.perf_counter()
        if self.server_configs['SERVER_MODE'] == 0:
            dataset = self.reshape_input(input=dataset)
        reshape_input_end = time.perf_counter()
        self.inference_timings_s['reshape_input'] = reshape_input_end - reshape_input_start
        self.log(f"Reshape input time: {self.inference_timings_s['reshape_input']:.9f} s")

        # Running the inference based on the server mode
        inference_start = time.perf_counter()
        if self.server_configs['SERVER_MODE'] == 0:
            assert self.server_configs['BATCH_SIZE'] == 1, \
                f"Batch size should be equal to 1 in cases where server_mode == 0, got {self.server_configs['BATCH_SIZE']}"
            exp_output = self.inference_single(input=dataset, run_total=run_total)
        elif self.server_configs['SERVER_MODE'] == 1:
            exp_output = self.inference_batch(dataset=dataset, run_total=run_total)
        else:
            raise ValueError(f"Server Mode is neither 0 nor 1 (LAT or THR), got {self.server_configs['SERVER_MODE']}")
        inference_end = time.perf_counter()
        self.inference_timings_s['inference'] = inference_end - inference_start
        self.log(f"Inference time: {self.inference_timings_s['inference']:.9f} s")

        # Reshaping the inference output
        reshape_output_start = time.perf_counter()
        exp_output = self.reshape_output(exp_output=exp_output, run_total=run_total)
        reshape_output_end = time.perf_counter()
        self.inference_timings_s['reshape_output'] = reshape_output_end - reshape_output_start
        self.log(f"Reshape output time: {self.inference_timings_s['reshape_output']:.9f} s")
        
        # Useful for power metrics
        if self.postprocessing_dummy is None:
            self.postprocessing_dummy = (exp_output, run_total)

        # Post-processing the inference output
        postprocess_start = time.perf_counter()
        output = self.postprocess(exp_output=exp_output, run_total=run_total)
        postprocess_end = time.perf_counter()
        self.inference_timings_s['postprocess'] = postprocess_end - postprocess_start
        self.log(f"Postprocess time: {self.inference_timings_s['postprocess']:.9f} s")

        # Encoding the final output
        encode_output_start = time.perf_counter()
        encoded_output = self.encode_output(output=output)
        encode_output_end = time.perf_counter()
        self.inference_timings_s['encode_output'] = encode_output_end - encode_output_start
        self.log(f"Encode Output time: {self.inference_timings_s['encode_output']:.9f} s")

        # Calculating and storing the full elapsed time for the inference
        full_end = time.perf_counter()
        self.inference_timings_s['full_inference'] = full_end - full_start

        # Various post-inference operations
        self.benchmarks(run_total=run_total)
        self.redis_create_send()
        self.save_metrics()
        self.prints()
        return encoded_output

    def decode_input(self, indata):
        """
        Decode input data. Must be overridden by model_server.py (ModelServer).
        indata is the input from the request. 
        Output needs to be: return decoded_input, run_total
        - decoded_input: the data (in any format).
        - run_total: the number of data items.
        decoded_input becomes the input for preprocess, which is also defined in model_server.py.
        """
        raise NotImplementedError('Forgot to overload decode_input. Must be overridden by model_server.py (ModelServer).')

    def preprocess(self, decoded_input, run_total):
        """
        Create and preprocess the dataset. Must be overridden by model_server.py (ModelServer).
        Gets decoded_input from decode_input() and outputs dataset.
        dataset NEEDS to be:
        a) a numpy.array in Latency Server Mode (self.server_configs['SERVER_MODE'] == 0).
        b) a tf.data.Dataset in Throughput Server Mode (self.server_configs['SERVER_MODE'] == 1).
        """
        raise NotImplementedError('Forgot to overload preprocess. Must be overridden by model_server.py (ModelServer).')

    def reshape_input(self, input):
        """
        Reshape input data if necessary, works only in Latency Server Mode (self.server_configs['SERVER_MODE'] == 0).
        Takes a numpy array as input and returns a reshaped numpy array as output.
        It reshapes the input based on the `expected_input` field in the self.experiment_configs in the set_experiment_configs method of model_server.py.
        self.experiment_configs['expected_input'] must be batched input, e.g., (None, 224, 224, 3).
        """
        if 'expected_input' in self.experiment_configs:
            batched_expected_input = (self.server_configs['BATCH_SIZE'],) + self.experiment_configs['expected_input'][1:]
            input = np.reshape(input, batched_expected_input)
        return input

    def inference_single(self, input, run_total=1):
        """
        Execute the inference for single input data. Works only in Latency Server Mode (self.server_configs['SERVER_MODE'] == 0).
        Must be overridden by {pair}_server.py ({Pair}Server).
        Takes a numpy.array as input and returns a numpy.array as output.
        """
        raise NotImplementedError('Forgot to overload inference_single. Must be overridden by {pair}_server.py ({Pair}Server).')

    def inference_batch(self, dataset, run_total):
        """
        Execute the inference for multiple input data. Works only in Throughput Server Mode (self.server_configs['SERVER_MODE'] == 1).
        Must be overridden by {pair}_server.py ({Pair}Server).
        Takes a tf.data.Dataset as input and returns a numpy array as output.
        """
        raise NotImplementedError('Forgot to overload inference_batch. Must be overridden by {pair}_server.py ({Pair}Server).')

    def reshape_output(self, exp_output, run_total):
        """
        Reshape inference output if necessary.
        Takes a numpy array as input and returns a reshaped numpy array as output.
        It reshapes the output based on the `expected_output` field in the self.experiment_configs in the set_experiment_configs method of model_server.py.
        self.experiment_configs['expected_output'] must be batched output, e.g., (None, 1000).
        """
        if 'expected_output' in self.experiment_configs:
            batched_expected_output = (run_total,) + self.experiment_configs['expected_output'][1:]
            exp_output = np.reshape(exp_output, batched_expected_output)
        return exp_output
    
    def postprocess(self, exp_output, run_total):
        """
        Post-process the inference output. Must be overridden by model_server.py (ModelServer).
        Input is a numpy array.
        output becomes the input for encode_output (in any format that fits).
        """
        raise NotImplementedError('Forgot to overload postprocess. Must be overridden by model_server.py (ModelServer).')

    def encode_output(self, output):
        """
        Encode the processed output.
        Takes the input from postprocess and passes the encoded_output to the send_response function.
        """
        raise NotImplementedError('Forgot to overload encode_output. Must be overridden by model_server.py (ModelServer).')
    
    def benchmarks(self, run_total):
        """Calculate various benchmarks based on inference metrics."""
        assert isinstance(self.inference_timings_s['inference'], float), \
            f"This variable is not float, it is {type(self.inference_timings_s['inference']).__name__} with value {self.inference_timings_s['inference']}"
        self.inference_metrics_s['total_latency'] = self.inference_timings_s['full_inference']
        self.inference_metrics_s['inference_latency'] = self.inference_timings_s['inference']
        self.inference_metrics_s['preprocessing_latency'] = self.inference_timings_s['decode_input'] + self.inference_timings_s['preprocess'] + self.inference_timings_s['reshape_input']
        self.inference_metrics_s['postprocessing_latency'] = self.inference_timings_s['reshape_output'] + self.inference_timings_s['postprocess'] + self.inference_timings_s['encode_output']
        self.inference_metrics_s['throughput'] = run_total / self.inference_metrics_s['total_latency']
        self.inference_metrics_s['dataset_size'] = run_total

    def redis_create_send(self):
        """Create metric data and send it to Redis."""
        if self.server_configs['SEND_METRICS']:
            create_start = time.perf_counter()
            # Passing None means that the function will get the value from the environment
            keys_metrics_tuples = utils.fill_redis_verbose(aif_characteristics_dict=self.aif_characteristics,
                            AIF_timestamp=self.server_configs['AIF_timestamp'], inference_metrics_dict=self.inference_metrics_s)
            create_end = time.perf_counter()
            send_start = time.perf_counter()
            try:
                utils.send_redis_verbose(self.my_redis, keys_metrics_tuples)
            except Exception as e:
                self.log(f"Could not send data to Redis: {str(e)}")
            send_end = time.perf_counter()
            create_elapsed_time = create_end - create_start
            send_elapsed_time = send_end - send_start
            self.inference_timings_s['redis_create'] = create_elapsed_time
            self.inference_timings_s['redis_send'] = send_elapsed_time
            self.log(f"Redis Create and Send time: {create_elapsed_time:.9f} s , {send_elapsed_time:.9f} s")
        else:
            self.log('Send Redis Metrics -> False')

    def save_metrics(self):
        """Save metrics on the server, to be requested from the metrics endpoint."""
        start = time.perf_counter()
        self.my_metrics_list.append(utils.create_metric_dictionary(self.inference_metrics_s))
        end = time.perf_counter()
        elapsed_time = end - start
        self.inference_timings_s['save_metrics'] = elapsed_time
        self.log(f"Save metrics time: {elapsed_time:.9f} s")

    def get_metrics(self, number = None):
        # Get all the internal data and filter them, allowing only json serializable
        all_configs = [
            utils.filter_jsonable_dict(self.aif_characteristics),
            utils.filter_jsonable_dict(self.server_configs),
            # self.experiment_configs is too verbose
            #utils.filter_jsonable_dict(self.experiment_configs),
            utils.filter_jsonable_dict(self.once_timings)
        ]        
        if(number is None):
            return all_configs + self.my_metrics_list
        else:
            # [-number:] gives the latest number metrics
            return all_configs + self.my_metrics_list[-number:]
    
    def prints(self):
        """Print latency breakdown and throughput metrics."""
        self.log(' ')
        # Construct the latency breakdown string in seconds
        latency_breakdown = (
            f"{self.inference_metrics_s['total_latency']:.9f} s "
            f"({self.inference_metrics_s['preprocessing_latency']:.9f} + "
            f"{self.inference_metrics_s['inference_latency']:.9f} + "
            f"{self.inference_metrics_s['postprocessing_latency']:.9f})"
        )

        # Print the total latency and its components
        self.log(f"\tTotal Latency (preprocessing + inference + postprocessing): \t{latency_breakdown}")
        self.log(f"\tThroughput: \t{self.inference_metrics_s['throughput']:.2f} fps")
        self.log(f"\tDataset Size: \t{self.inference_metrics_s['dataset_size']}")
        
    def platform_preprocess(self, data):
        """Preprocess the input, specific to the platform requirement, not the inference ones. Must be overridden by {pair}_server.py ({Pair}Server)."""
        raise NotImplementedError('Forgot to overload platform_preprocess. Must be overridden by {pair}_server.py ({Pair}Server).')

    def platform_postprocess(self, data):
        """Postprocess the output, specific to the platform requirement, not the inference ones. Must be overridden by {pair}_server.py ({Pair}Server)."""
        raise NotImplementedError('Forgot to overload platform_postprocess. Must be overridden by {pair}_server.py ({Pair}Server).')

    def log_all_server_values(self):
        """Log all the server values for debugging purposes."""
        try:
            self.log(f"server_configs: {self.server_configs}")
            self.log(f"experiment_configs: {self.experiment_configs}")
            self.log(f"once_timings: {self.once_timings}")
            self.log(f"inference_timings_s: {self.inference_timings_s}")
            self.log(f"inference_metrics_s: {self.inference_metrics_s}")
            self.log(f"aif_characteristics: {self.aif_characteristics}")
            self.log(f"power_metrics: {self.power_metrics}")
        except Exception as e:
            self.log(f"While trying to log_all_server_values, got this error: {e}")
            print(f"While trying to log_all_server_values, got this error: {e}")

    def dummy_inference(self):
        raise NotImplementedError('Forgot to overload dummy_inference. Must be overridden by {pair}_server.py ({Pair}Server).')

    def dummy_preprocessing(self):
        #raise NotImplementedError('Forgot to overload dummy_preprocessing. Must be overridden by {pair}_server.py ({Pair}Server).')
        if self.preprocessing_dummy is None:
            raise AssertionError('Preprocessing dummy is None, run an inference before power metrics')
        # Executing the input decoding process
        decoded_input, run_total = self.decode_input(indata=self.preprocessing_dummy)
        dataset = self.preprocess(decoded_input=decoded_input, run_total=run_total)
        if self.server_configs['SERVER_MODE'] == 0:
            dataset = self.reshape_input(input=dataset)      

    def dummy_postprocessing(self):
        #raise NotImplementedError('Forgot to overload dummy_postprocessing. Must be overridden by {pair}_server.py ({Pair}Server).')
        if self.postprocessing_dummy is None:
            raise AssertionError('Postprocessing dummy is None, run an inference before power metrics')
        output = self.postprocess(exp_output=self.postprocessing_dummy[0], run_total=self.postprocessing_dummy[1])
        encoded_output = self.encode_output(output=output)

    def continuous_dummy_function(self, event, error_flag, function_name:str = None):
        # if you get deadlocks, remove the prints (think is thread-unsafe)
        if(function_name == 'inference'):
            dummy_function = self.dummy_inference
        elif(function_name == 'preprocessing'):
            dummy_function = self.dummy_preprocessing
        elif(function_name == 'postprocessing'):
            dummy_function = self.dummy_postprocessing
        else:
            error_flag.set()
            raise AssertionError(f'Function name is wrong, got: {function_name}. Should be inference, preprocessing or postprocessing')
        counter = 0
        total_time_s = 0
        while not event.is_set():
            start = time.perf_counter()
            try:
                dummy_function()
            except Exception as e: 
                print(f'self.continuous_dummy_function() got this error: {e}')
                self.log(f'self.continuous_dummy_function() got this error: {e}')
                error_flag.set()
                return
            end = time.perf_counter()
            total_time_s = total_time_s + end - start
            counter = counter + 1
        print(f"Number of dummy {function_name}: {counter}, time running: {total_time_s:.3f} s")
        self.log(f"Number of dummy {function_name}: {counter}, time running: {total_time_s:.3f} s")
        return

    def scrape_power_data(self):
        raise NotImplementedError('scrape_power_data. Must be overridden by {pair}_server.py ({Pair}Server).')

    def collect_power_data(self, event, error_flag, power_scraping_seconds, power_waiting_seconds, power_data_queue):
        time.sleep(power_waiting_seconds) # To get correct power data
        power_data_list = []
        start = time.perf_counter()
        end = start
        while(end - start < power_scraping_seconds):
            try:
                power_data_list.append(self.scrape_power_data())
            except Exception as e:
                print(f'self.scrape_power_data() got this error: {e}')
                error_flag.set()
            end = time.perf_counter()
        event.set()
        power_data_queue.put(power_data_list)
        return

    def process_power_data(self, idle_power_data_list, preprocess_power_data_list, inference_power_data_list, postprocess_power_data_list):
        # Helper function to create a structured dictionary from a list of power data
        def create_power_metrics(data_list):
            # Initialize the metrics dictionary with the count of data points
            metrics = {'indexes': len(data_list)}
            # Add each data point with its index as a string key
            for index, power_data in enumerate(data_list):
                metrics[str(index)] = power_data
            return metrics

        # Create a dictionary with power metrics for each stage
        power_metrics_dict = {
            'idle': create_power_metrics(idle_power_data_list),
            'preprocess': create_power_metrics(preprocess_power_data_list),
            'inference': create_power_metrics(inference_power_data_list),
            'postprocess': create_power_metrics(postprocess_power_data_list)
        }

        # Return the complete power metrics dictionary
        return power_metrics_dict

        
    def get_power_metrics(self):

        def run_power_taking_process(name, dummy_thread, power_data_process, running_event, error_flag_event, cooldown_seconds, power_data_queue):
            # Cooldown to reset power
            time.sleep(cooldown_seconds)
            print(f"Collect {name} data starting")
            if(dummy_thread is not None):
                dummy_thread.start()
                self.log(f"{name} dummy_thread.start()")
            
            power_data_process.start()
            self.log(f"{name} power_data_process.start()")

            if(dummy_thread is not None):
                dummy_thread.join()
                self.log(f"{name} dummy_thread.join()")

            power_data_list = power_data_queue.get()
            power_data_process.join()
            self.log(f"{name} power_data_process.join()")

            if(error_flag_event.is_set()):
                self.log(f'Emptying power_data_list of {name} because error_flag_event is set')
                power_data_list = []

            running_event.clear()
            error_flag_event.clear()
            return power_data_list

        # Constants
        POWER_SCRAPING_SECONDS = 10
        POWER_WAITING_SECONDS = 3
        COOLDOWN_SECONDS = 2

        running_event = multiprocessing.Event()
        error_flag_event = multiprocessing.Event()
        power_data_queue = multiprocessing.Queue()
        continuous_dummy_preprocessing_thread = threading.Thread(target=self.continuous_dummy_function, args=(running_event, error_flag_event, 'preprocessing'))
        continuous_dummy_inference_thread = threading.Thread(target=self.continuous_dummy_function, args=(running_event, error_flag_event, 'inference'))
        continuous_dummy_postprocessing_thread = threading.Thread(target=self.continuous_dummy_function, args=(running_event, error_flag_event, 'postprocessing'))
        collect_idle_power_data_process = multiprocessing.Process(target=self.collect_power_data, args=(running_event, error_flag_event, POWER_SCRAPING_SECONDS, POWER_WAITING_SECONDS, power_data_queue))
        collect_preprocessing_power_data_process = multiprocessing.Process(target=self.collect_power_data, args=(running_event, error_flag_event, POWER_SCRAPING_SECONDS, POWER_WAITING_SECONDS, power_data_queue))
        collect_inference_power_data_process = multiprocessing.Process(target=self.collect_power_data, args=(running_event, error_flag_event, POWER_SCRAPING_SECONDS, POWER_WAITING_SECONDS, power_data_queue))
        collect_postprocessing_power_data_process = multiprocessing.Process(target=self.collect_power_data, args=(running_event, error_flag_event, POWER_SCRAPING_SECONDS, POWER_WAITING_SECONDS, power_data_queue))
       
        idle_power_data_list = run_power_taking_process('idle', None, collect_idle_power_data_process, running_event, error_flag_event, COOLDOWN_SECONDS, power_data_queue)
        preprocessing_power_data_list = run_power_taking_process('preprocessing', continuous_dummy_preprocessing_thread, collect_preprocessing_power_data_process, running_event, error_flag_event, COOLDOWN_SECONDS, power_data_queue)
        inference_power_data_list = run_power_taking_process('inference', continuous_dummy_inference_thread, collect_inference_power_data_process, running_event, error_flag_event, COOLDOWN_SECONDS, power_data_queue)
        postprocessing_power_data_list = run_power_taking_process('postprocessing', continuous_dummy_postprocessing_thread, collect_postprocessing_power_data_process, running_event, error_flag_event, COOLDOWN_SECONDS, power_data_queue)
        
        # Process power data
        power_metrics_dict = self.process_power_data(idle_power_data_list, preprocessing_power_data_list, inference_power_data_list, postprocessing_power_data_list)
        self.log("Processed power_metrics")
        self.power_metrics = power_metrics_dict
        return power_metrics_dict
    
    def get_logs(self):
        """
        Returns the content of the server log file.
        """
        log_file_path = None
        # Iterate over the handlers to find the FileHandler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                log_file_path = handler.baseFilename
                break
        if not log_file_path or not os.path.exists(log_file_path):
            return 'Log file not found.'
        try:
            with open(log_file_path, 'r') as log_file:
                log_content = log_file.read()
            return log_content
        except Exception as e:
            return f'Error reading log file: {e}'
