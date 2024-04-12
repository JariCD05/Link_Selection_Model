# Adjusting the SatelliteThroughputIntegrated class for the correct behavior of Satellite 2
from input import *
import matplotlib as plt


class SatelliteThroughputCorrected:

    def __init__(self, time):
        self.time = time
        self.num_satellites = 2  # Assuming this is defined somewhere or passed as an argument
        self.throughput_test = [[0 for _ in range(len(self.time))] for _ in range(self.num_satellites)]
    
    def calculate_throughput(self):
        peak_throughput = 2.5e9
        constant_throughput = 0.5e9
        initial_throughput = 0.1
        peak_timestamp_1 = 10
        end_timestamp_2 = 30
        a1 = -peak_throughput / (peak_timestamp_1**2)
        
        for t in range(peak_timestamp_1*2):
            self.throughput_test[0][t] = max(a1 * (t - peak_timestamp_1)**2 + peak_throughput, 0)
        
        a2_start = 10
        a2_peak = (end_timestamp_2 - a2_start) / 2 + a2_start
        a2 = -peak_throughput / ((a2_peak - a2_start)**2)
        
        for t in range(a2_start):
            self.throughput_test[1][t] = initial_throughput
        
        for t in range(a2_start, len(self.throughput_test[1])):
            if t < end_timestamp_2:
                self.throughput_test[1][t] = max(a2 * (t - a2_peak)**2 + peak_throughput, 0)
            else:
                self.throughput_test[1][t] = constant_throughput
    
    def get_throughput(self):
        if not self.throughput_test[0][1]:
            self.calculate_throughput()
        return self.throughput_test


