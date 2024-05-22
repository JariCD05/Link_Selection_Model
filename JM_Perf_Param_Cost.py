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




class Cost_performance():
    def __init__(self, time, lenghts, num_satellites):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.num_satellites = num_satellites
        self.time = time
        self.lenghts = lenghts

        self.cost_performance =[0] * self.num_satellites
        self.normalized_cost_performance =[0] * self.num_satellites
        self.penalized_cost_performance =[0] * self.num_satellites
        self.normalized_penalized_cost_performance =[0] * self.num_satellites

   
    def calculate_cost_performance(self):

        # Set cost performance for each satellite at each time index
        for s in range(self.num_satellites):
                self.cost_performance[s] = variable_link_cost_const1
        #print("Cost Performance", self.cost_performance)
        
        return self.cost_performance

    def calculate_normalized_cost_performance(self):
        # Calculate the denominator for normalization based on maximum costs
        normalization_factor_variable = max(constellation_variable_link_cost) 
        
        for s in range(self.num_satellites):
            self.normalized_cost_performance[s] = 1 - (self.cost_performance[s] / normalization_factor_variable)

        #print("Normalized cost Performance", self.normalized_cost_performance)
        
        return self.normalized_cost_performance

    def calculate_penalized_cost_performance(self):
        
        # Initialize the cost_performance array
        
        for s in range(self.num_satellites):
            self.penalized_cost_performance[s] = fixed_link_cost_const1 + variable_link_cost_const1

        return self.penalized_cost_performance

    def calculate_normalized_penalized_cost_performance(self):
        # Calculate the normalization factor
        normalization_factor_variable = max(constellation_variable_link_cost)
        normalization_factor_fixed = max(constellation_fixed_link_cost)

        # Normalize the cost_performance array
        for s in range(self.num_satellites):
            self.normalized_penalized_cost_performance[s] = (1 - (fixed_link_cost_const1 / normalization_factor_fixed)) +  (1 - (variable_link_cost_const1 / normalization_factor_variable))
    
        return self.normalized_penalized_cost_performance
    


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