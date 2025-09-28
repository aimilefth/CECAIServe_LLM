import pynvml as nvml
from pyJoules.energy_meter import measure_energy
from pyJoules.handler import EnergyHandler, UnconsistantSamplesError
from pyJoules.energy_trace import EnergyTrace, EnergySample
import time
import datetime

class DictHandler(EnergyHandler):
    """
    Handle energy samples and convert them into a dictionary without using Pandas.
    """
    def __init__(self):
        super().__init__()  # Initialize the parent EnergyHandler class

    def get_single_dictionary(self) -> dict:
        """
        Returns the data of a single energy sample as a dictionary.
        Asserts that there is exactly one energy sample.
        """
        if not self.traces:
            raise NoSampleProcessedError("No samples have been processed.")

        # Use the inherited _flaten_trace method
        flattened_trace = self._flaten_trace()
        samples = list(flattened_trace)

        if len(samples) != 1:
            raise ValueError(f"Expected exactly one sample, but got {len(samples)}.")

        sample = samples[0]
        result = {
            'timestamp': sample.timestamp,
            'tag': sample.tag,
            'duration': sample.duration,
        }
        result.update(sample.energy)
        return result

    def reset(self) -> None:
        """
        Resets the handler by clearing all stored traces.
        """
        self.traces = []

class power_scraper:
    def __init__(self, gpu_device_index) -> None:
        self.initialized = False
        self.gpu_handle = None
        self.gpu_device_index = gpu_device_index
        self.dict_handler = DictHandler()

    def gpu_get_power(self):
        # Need to initialize here because when creating new process, it needs to be reinitialized
        if self.initialized is False:
            nvml.nvmlInit()
            self.gpu_handle = nvml.nvmlDeviceGetHandleByIndex(self.gpu_device_index)
        power_dict = {'Power': nvml.nvmlDeviceGetPowerUsage(self.gpu_handle)/1000,
                      'GPU Name': nvml.nvmlDeviceGetName(self.gpu_handle),
                      'GPU Index': self.gpu_device_index}
        return power_dict

    def cpu_get_power(self, interval: float= 0.1):
        @measure_energy(handler=self.dict_handler)
        def waste_time_sleep(interval: float= 0.1):
            time.sleep(interval)
        waste_time_sleep(interval)
        data = self.dict_handler.get_single_dictionary()
        self.dict_handler.reset()
        return data

    def get_power(self):
        data = {'host': self.cpu_get_power(),
                'accel': self.gpu_get_power(),
                'timestamp': datetime.datetime.utcnow().isoformat()}
        return data
