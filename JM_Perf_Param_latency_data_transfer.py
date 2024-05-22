# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
#from input_old import *
from JM_INPUT_CONFIG_FILE import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_Sat_Applicable_links import applicable_satellites




class Latency_data_transfer_performance():
    def __init__(self, time, ranges, lenghts, num_satellites, smallest_latency, acquisition_time_steps):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.time = time
        self.ranges = ranges
        self.lenghts = lenghts
        self.num_satellites = num_satellites
        self.smallest_latency = smallest_latency
        self.speed_of_light = speed_of_light
        self.acquisition_time_steps = acquisition_time_steps

        self.latency_data_transfer_performance = [0] * self.num_satellites
        self.normalized_latency_data_transfer_performance = [0] * self.num_satellites
        self.penalized_latency_data_transfer_performance = [0] * self.num_satellites
        self.normalized_penalized_latency_data_transfer_performance = [0] * self.num_satellites

    def calculate_latency_data_transfer_performance(self):
        # Extract ranges for all satellites over time from the applicable_output
 
       
        # Calculate latency performance for each satellite
        for s in range(self.num_satellites):
            future_latencies_values = self.ranges[s] / self.speed_of_light
            self.latency_data_transfer_performance[s] = np.average(future_latencies_values)
        
        #print("Data Latency Performance",self.latency_data_transfer_performance)
        
        return self.latency_data_transfer_performance

    def calculate_normalized_latency_data_transfer_performance(self):

        for s in range(self.num_satellites):
            self.normalized_latency_data_transfer_performance[s] = self.smallest_latency /self.latency_data_transfer_performance[s] 

        #print("normalized data transfer latency", self.normalized_latency_data_transfer_performance)
        
        return self.normalized_latency_data_transfer_performance

    def calculate_penalized_latency_data_transfer_performance(self):
        # Extract ranges for all satellites over time from the applicable_output
 
       
        # Calculate latency performance for each satellite
        for s in range(self.num_satellites):
            future_latencies_values = self.ranges[s] / self.speed_of_light
            self.penalized_latency_data_transfer_performance[s] = np.average(future_latencies_values[self.acquisition_time_steps:])
        

        
        return self.penalized_latency_data_transfer_performance

    def calculate_normalized_penalized_latency_data_transfer_performance(self):

        for s in range(self.num_satellites):
            self.normalized_penalized_latency_data_transfer_performance[s] = self.smallest_latency /self.penalized_latency_data_transfer_performance[s] 

        #print("normalized data transfer latency", self.normalized_latency_data_transfer_performance)
        
        return self.normalized_penalized_latency_data_transfer_performance




















#------------------------------------------------------------------------------------------------------------------------------------------------------------

    def latency_data_transfer_visualization(self, latency_data_transfer, normalized_latency_data_transfer_performance):    
        # Converting the latency visualization plots to scatter plots and removing all zero values

        # Plotting with zero values filtered out in scatter plot format
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # Filter and Scatter Plot 1: Propagation Latency
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(latency_data_transfer[s]) if v > 0]
            non_zero_values = [v for v in latency_data_transfer[s] if v > 0]
            axs[0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0].set_title('Data transfer latency  $Q_{DTL} (t_{j})$')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Propagated averaged latency (s)')
        axs[0].legend()

        # Filter and Scatter Plot 2: Normalized Propagation Latency
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_latency_data_transfer_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_latency_data_transfer_performance[s] if v > 0]
            axs[1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1].set_title('Data transfer latency performance $\hat{Q}_{DTL} (t_{j})$')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Normalized data transfer latency [-]')
        axs[1].legend()

        plt.tight_layout()
        plt.show()











