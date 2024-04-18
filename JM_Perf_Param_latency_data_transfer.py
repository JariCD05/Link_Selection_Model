# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_applicable_links import applicable_links




class Latency_data_transfer_performance():
    def __init__(self, time, link_geometry):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light

    
    
    def calculate_latency_data_transfer_performance(self):
        # Extract ranges for all satellites over time from the applicable_output
        ranges = self.applicable_output['ranges']
       
        # Initialize the latency_data_transfer_performance array
        self.latency_data_transfer = np.zeros((num_satellites, len(self.time)))

        # Calculate the average future latency for each satellite at each time index
        for sat_index in range(num_satellites):
            for time_index in range(len(self.time)):
                # Ensure there's a valid range value before calculating latency
                if ranges[sat_index][time_index]>0:
                    future_latencies = [ranges[sat_index][t] / self.speed_of_light for t in range(time_index, len(self.time)) if ranges[sat_index][t] >0]
                    # Only calculate the average if there are valid future latency values
                    if future_latencies:
                        self.latency_data_transfer[sat_index][time_index] = sum(future_latencies) / len(future_latencies)
            

        return self.latency_data_transfer

    def distance_normalization_data_transfer(self, latency_data_transfer):
        # Convert input latency array to a numpy array for easier manipulation
        latency_data_transfer_performance = np.array(latency_data_transfer)

        # Find the global minimum nonzero latency across all satellites
        global_min_latency = np.min(latency_data_transfer_performance[latency_data_transfer_performance > 0])

        # Initialize normalized latency performance array
        normalized_latency_data_transfer_performance = np.zeros(latency_data_transfer_performance.shape)

        # Normalize the latency values for each satellite based on the global minimum latency
        for sat_index in range(latency_data_transfer_performance.shape[0]):
            for time_index in range(latency_data_transfer_performance.shape[1]):
                if latency_data_transfer_performance[sat_index, time_index] > 0:
                    normalized_latency_data_transfer_performance[sat_index, time_index] =  global_min_latency /latency_data_transfer_performance[sat_index, time_index]
                else:
                    # If original latency is zero, it remains zero in the normalized array
                    normalized_latency_data_transfer_performance[sat_index, time_index] = 0

        return normalized_latency_data_transfer_performance






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











