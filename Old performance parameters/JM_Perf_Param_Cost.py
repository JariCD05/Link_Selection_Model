# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

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




class Cost_performance():
    def __init__(self, time):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere

        self.time = time
        self.speed_of_light = speed_of_light

   
    def calculate_cost_performance(self):

        # Initialize the cost_performance array
        self.cost_performance = np.zeros((num_satellites, len(self.time)))

        # Set cost performance for each satellite at each time index
        for sat_index in range(num_satellites):
            for time_index in range(len(self.time)):
                self.cost_performance[sat_index, time_index] = variable_link_cost_const1

        # Multiply by sats_applicable to zero out non-applicable instances
        self.cost_performance *= self.sats_applicable

        return self.cost_performance

    def calculate_normalized_cost_performance(self, cost_performance):
        # Calculate the denominator for normalization based on maximum costs
        normalization_factor = max(constellation_variable_link_cost) + max(constellation_fixed_link_cost)
        
        # Normalize the cost_performance array
        self.normalized_cost_performance = 1 - (cost_performance / normalization_factor)
        
        return self.normalized_cost_performance

    def calculate_cost_performance_including_penalty(self):
        
        # Initialize the cost_performance array
        self.cost_performance_including_penalty = np.zeros((num_satellites, len(self.time)))

        for sat_index in range(num_satellites):
            for time_index in range(len(self.time)):
                    self.cost_performance_including_penalty[sat_index, time_index] = fixed_link_cost_const1 + variable_link_cost_const1

        self.cost_performance_including_penalty *= self.sats_applicable

        return self.cost_performance_including_penalty

    def calculate_normalized_cost_performance_including_penalty(self, cost_performance_including_penalty):
        # Calculate the normalization factor
        normalization_factor = max(constellation_variable_link_cost)+ max(constellation_fixed_link_cost)

        # Normalize the cost_performance array
        self.normalized_cost_performance_including_penalty = 1 - (cost_performance_including_penalty / normalization_factor)
    
        return self.normalized_cost_performance_including_penalty
    


    def cost_visualization(self, cost_performance, normalized_cost_performance, cost_performance_including_penalty, normalized_cost_performance_including_penalty):
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Filter and Scatter Plot 1: Cost Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(cost_performance[s]) if v > 0]
            non_zero_values = [v for v in cost_performance[s] if v > 0]
            axs[0, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 0].set_title('Cost Performance $q_{C} (t_{j})$')
        axs[0, 0].set_xlabel('Time Steps')
        axs[0, 0].set_ylabel('Cost [€]')
        axs[0, 0].legend()

        # Filter and Scatter Plot 2: Normalized Cost Performance
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_cost_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_cost_performance[s] if v > 0]
            axs[0, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0, 1].set_title('Normalized Cost Performance $\hat{q}_{C} (t_{j})$')
        axs[0, 1].set_xlabel('Time Steps')
        axs[0, 1].set_ylabel('Normalized Cost [-]')
        axs[0, 1].legend()

        # Filter and Scatter Plot 3: Cost Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(cost_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in cost_performance_including_penalty[s] if v > 0]
            axs[1, 0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 0].set_title('Cost Performance Including Penalty $q_{C} (t_{j})$')
        axs[1, 0].set_xlabel('Time Steps')
        axs[1, 0].set_ylabel('Cost w/ Penalty [€]')
        axs[1, 0].legend()

        # Filter and Scatter Plot 4: Normalized Cost Performance Including Penalty
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_cost_performance_including_penalty[s]) if v > 0]
            non_zero_values = [v for v in normalized_cost_performance_including_penalty[s] if v > 0]
            axs[1, 1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1, 1].set_title('Normalized Cost Performance Including Penalty $\hat{q}_{C} (t_{j})$')
        axs[1, 1].set_xlabel('Time Steps')
        axs[1, 1].set_ylabel('Normalized Cost w/ Penalty [-]')
        axs[1, 1].legend()

        plt.tight_layout()
        plt.show()