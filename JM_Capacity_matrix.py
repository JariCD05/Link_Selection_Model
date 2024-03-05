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
from JM_Visibility_matrix import routing_network


import numpy as np

class CapacityCalculator:
    def __init__(self, los_matrix, BW, SNR_penalty_matrix, BR):
        self.los_matrix = los_matrix
        self.BW = BW
        self.SNR_penalty_matrix = SNR_penalty_matrix
        self.BR = BR
        self.capacity = np.zeros_like(los_matrix)

    def calculate_capacity(self):
        for time_step in range(self.los_matrix.shape[1]):
            for sat in range(self.los_matrix.shape[0]):
                C = self.BW * np.log2(1 + self.SNR_penalty_matrix[sat, time_step])
                if C > self.BR:
                    self.capacity[sat, time_step] = 1

# Example usage:
num_time_instances = 10
num_satellites = 10

los_matrix = np.random.randint(0, 2, size=(num_satellites, num_time_instances))
SNR_penalty_matrix = np.random.uniform(7, 9, size=(num_satellites, num_time_instances))
BW = 1
BR = 3.248

calculator = CapacityCalculator(los_matrix, BW, SNR_penalty_matrix, BR)
calculator.calculate_capacity()

print(calculator.capacity)



