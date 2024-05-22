# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import math
import csv
import pandas as pd

# Import input parameters and helper functions
from input_old import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_Sat_Applicable_links import applicable_satellites
from JM_Perf_Param_Availability import Availability_performance



class Throughput_performance:
    def __init__(self, time, throughput):
        self.time = time
        self.throughput = throughput
        self.weights_record = []
        self.weighted_values_record = []


    def calculate_throughput_performance_including_decay(self, decay_rate):
        # Initialize throughput_performance with zeros
        
        self.throughput_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]
        weights_records = []
        weighted_values_records = []
        throughput_performance_values_with_decay = []

        for s in range(num_satellites):
            for t in range(len(self.time)):
                if self.throughput[s][t] > 0:
                    future_values = [self.throughput[s][index] for index in range(t, len(self.time)) if self.throughput[s][index] > 0]
                    if future_values:
                        weights = [math.exp(-decay_rate * (index - t)) for index in range(t, len(self.time)) if self.throughput[s][index] > 0]
                        weighted_sum = sum(fv * w for fv, w in zip(future_values, weights))
                        total_weight = sum(weights)
                        self.throughput_performance[s][t] = weighted_sum / total_weight
                        # Collect records for saving
                        weights_records.append({'Time Step': t, 'Weights': weights})
                        weighted_values_records.append({'Time Step': t, 'Weighted Values': [fv * w for fv, w in zip(future_values, weights)]})
                        throughput_performance_values_with_decay.append({'Time Step': t, 'Performance Values': self.throughput_performance})

        # Save to CSV
        weights_df = pd.DataFrame(weights_records)
        weighted_values_df = pd.DataFrame(weighted_values_records)
        throughput_performance_df = pd.DataFrame(throughput_performance_values_with_decay)
        folder_path = 'CSV'
        os.makedirs(folder_path, exist_ok=True)
        weights_df.to_csv(f'{folder_path}/weights_record.csv', index=False)
        weighted_values_df.to_csv(f'{folder_path}/weighted_values_record.csv', index=False)
        throughput_performance_df.to_csv(f'{folder_path}/decay_included_throughput_performance.csv', index=False)

        return self.throughput_performance

    def calculate_throughput_performance(self):
        # Initialize throughput_performance with zeros
        self.throughput_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]
        simple_throughput_values_record = []
    
        for s in range(num_satellites):
            for t in range(len(self.time)):
                if self.throughput[s][t] > 0:
                    future_values = [self.throughput[s][index] for index in range(t, len(self.time)) if self.throughput[s][index] > 0]
                    if future_values:
                        self.throughput_performance[s][t] = sum(future_values) / len(future_values)
                        simple_throughput_values_record.append({'Time Step': t, 'Performance Values': self.throughput_performance})
    
        # Save to CSV
        simple_throughput_df = pd.DataFrame(simple_throughput_values_record)
        folder_path = 'CSV'
        os.makedirs(folder_path, exist_ok=True)
        simple_throughput_df.to_csv(f'{folder_path}/Throughput_performance_record.csv', index=False)
    
        return self.throughput_performance

    def get_weights(self):
        return self.weights_record

    def get_weighted_values(self):
        return self.weighted_values_record


    def calculate_normalized_throughput_performance(self, data, data_rate_ac):
        # Convert data to a NumPy array to make sure we're working with NumPy functionality
        data_array = np.array(data)
        
        # Create a normalized_throughput_performance array filled with zeros
        # with the same shape as data_array
        self.normalized_throughput_performance = np.zeros(data_array.shape)
        
        # Perform element-wise division
        self.normalized_throughput_performance = data_array / data_rate_ac

        return self.normalized_throughput_performance


    def throughput_visualization(self, throughput_performance, normalized_throughput_performance):
        # Converting the throughput visualization plots to scatter plots and removing all zero values

        # Plotting with zero values filtered out in scatter plot format
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # Filter and Scatter Plot 1: Throughput Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(throughput_performance[s]) if v > 0]
            non_zero_values = [v for v in throughput_performance[s] if v > 0]
            axs[0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0].set_title('Throughput Performance $Q_{R} (t_{j})$ with decay rate:' f"{decay_rate}")
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Throughput [bits/($t_{A_{i,e}} - t_{j}$)]')
        axs[0].legend()

        # Filter and Scatter Plot 2: Normalized Throughput Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_throughput_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_throughput_performance[s] if v > 0]
            axs[1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1].set_title('Normalized Throughput Performance $\hat{Q}_{R} (t_{j})$ with decay rate:' f"{decay_rate}")
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Normalized Throughput [-]')
        axs[1].legend()

        plt.tight_layout()
        plt.show()

