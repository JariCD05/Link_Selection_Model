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
#from JM_Atmosphere import JM_attenuation, JM_turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_applicable_links import applicable_links



class Availability_performance():
    def __init__(self, time, link_geometry, availability_vector):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light
        self.availability_vector = availability_vector

    def calculate_availability_performance(self):

        self.availability_performance = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Use np.where to find indices where the condition is True (i.e., value is 1)
            one_indices = np.where(self.availability_vector[s] == 1)[0]

            if len(one_indices) == 0:
                # If no '1' is found in the array, skip this satellite
                continue
            
            # Get the index of the first '1'
            first_one_index = one_indices[0]

            # Count the total number of '1's starting from the first_one_index
            total_ones = len(one_indices)

            # Initialize the running sum to the total number of ones initially
            running_sum = total_ones

            # Update the performance list from the point of the first '1'
            for index in range(first_one_index, len(self.time)):
                if self.availability_vector[s][index] == 1:
                    self.availability_performance[s][index] = running_sum
                    running_sum -= 1  # Decrement the running sum for each '1' encountered

        return self.availability_performance
       

    def calculate_normalized_availability_performance(self, data, sats_applicable):
        max_time = np.nansum(sats_applicable, axis=1)
        max_time_max = np.nanmax(max_time)
        self.normalized_availability_performance = data / max_time_max
        #print(len(max_time))
        #print(max_time)
        return self.normalized_availability_performance
    
    def calculate_availability_performance_including_penalty(self, T_acq, step_size_link):
        # Convert T_acq to the number of time steps
        delay_steps = int(np.ceil(T_acq / step_size_link))

        self.availability_performance_including_penalty = [[0 for _ in range(len(self.time))] for _ in range(num_satellites)]

        for s in range(num_satellites):
            # Use np.where to find indices where the condition is True (i.e., value is 1)
            one_indices = np.where(self.availability_vector[s] == 1)[0]

            if len(one_indices) == 0:
                # If no '1' is found in the array, skip this satellite
                continue
            
            # Adjust the start index by the delay
            first_one_index = one_indices[0] + delay_steps

            # Ensure the start index does not exceed the array bounds
            first_one_index = min(first_one_index, len(self.time) - 1)

            # Count the total number of '1's starting from the first_one_index considering the delay
            total_ones = len([idx for idx in one_indices if idx >= first_one_index])

            # Initialize the running sum to the total number of ones initially
            running_sum = total_ones

            # Update the performance list from the point of the first '1' after considering the delay
            for index in range(first_one_index, len(self.time)):
                if self.availability_vector[s][index] == 1:
                    self.availability_performance_including_penalty[s][index] = running_sum
                    running_sum -= 1  # Decrement the running sum for each '1' encountered

        return self.availability_performance_including_penalty
       

    def calculate_normalized_availability_performance_including_penalty(self, data, sats_applicable):
        max_time = np.nansum(sats_applicable, axis=1)
        max_time_max = np.nanmax(max_time)
        self.normalized_availability_performance_including_penalty = data / max_time_max
        
        return self.normalized_availability_performance_including_penalty
    
    def availability_visualization(self, availability_performance, normalized_availability_performance, availability_performance_including_penalty, normalized_availability_performance_including_penalty):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Filter and Plot 1: Availability Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(availability_performance[s]) if v > 0]
            non_zero_values = [v for v in availability_performance[s] if v > 0]
            axs[0, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 0].set_title('Availability Performance $Q_{A} (t_{j})$')
        axs[0, 0].set_xlabel('Time Steps')
        axs[0, 0].set_ylabel('Performance [# timesteps]')
        axs[0, 0].legend()

        # Filter and Plot 2: Normalized Availability Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_availability_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_availability_performance[s] if v > 0]
            axs[0, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 1].set_title('Normalized Availability Performance $\hat{Q}_{A} (t_{j})$')
        axs[0, 1].set_xlabel('Time Steps')
        axs[0, 1].set_ylabel('Normalized Performance [-]')
        axs[0, 1].legend()

        # Filter and Plot 3: Availability Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(availability_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in availability_performance_including_penalty[s] if v > 0]
            axs[1, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 0].set_title('Availability Performance Including Penalty $Q_{A} (t_{j})$ ')
        axs[1, 0].set_xlabel('Time Steps')
        axs[1, 0].set_ylabel('Performance [# timesteps]')
        axs[1, 0].legend()

        # Filter and Plot 4: Normalized Availability Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_availability_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in normalized_availability_performance_including_penalty[s] if v > 0]
            axs[1, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 1].set_title('Normalized Availability Performance Including Penalty $\hat{Q}_{A} (t_{j})$')
        axs[1, 1].set_xlabel('Time Steps')
        axs[1, 1].set_ylabel('Normalized Performance [-]')

        plt.tight_layout()
        plt.show()
  