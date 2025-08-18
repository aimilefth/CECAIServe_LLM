import datetime

class power_scraper:
    def __init__(self) -> None:
        self.name = ['VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0', 'VDDQ_VDD2_1V8AO']       
        
        self.address = [0, 0, 0, 1]

        self.channel = [1, 2, 3, 2]

        self.description = ['Total power consumed by GPU and SOC core which supplies to memory subsystem and various engines like nvdec, nvenc, vi, vic, isp etc.', 
                            'Total power consumed by CPU and CV cores i.e. DLA and PVA.',
                            'Power consumed by system 5V rail which supplies to various IOs e.g. HDMI, USB, UPHY, UFS, SDMMC, EMMC, DDR etc. VDDQ_VDD2_1V8AO power is also included in VIN_SYS_5V0 power.',
                            'Power consumed by DDR core, DDR IO and 1V8AO(Always ON power rail).',]
        
    def get_power(self):
        power = {}
        total_power = 0
        for (address, channel, name) in zip(self.address, self.channel, self.name):
            # Values from files are milli
            v = int(get_value_from_read(f'/sys/bus/i2c/drivers/ina3221/1-004{address}/hwmon/hwmon{address+1}/in{channel}_input'))/1000
            i = int(get_value_from_read(f'/sys/bus/i2c/drivers/ina3221/1-004{address}/hwmon/hwmon{address+1}/curr{channel}_input'))/1000
            p = v * i
            temp_dir = {'Voltage': v, 'Current': i, 'Power': p}
            power[name] = temp_dir
            if(self.name is not 'VDDQ_VDD2_1V8AO'):
                # VDDQ_VDD2_1V8AO is included in the VIN_SYS_5V0, according to 
                # https://docs.nvidia.com/jetson/archives/r36.4.3/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html#software-based-power-consumption-modeling
                total_power = total_power + p
        power['Total Power'] = total_power
        power['timestamp'] = datetime.datetime.utcnow().isoformat()
        return power

def get_value_from_read(path):
    try: 
        with open(path, 'r') as device_file:
            return device_file.read()
    except Exception as e:
        print(f"Error in get_value_from_read: {e}")
        return None