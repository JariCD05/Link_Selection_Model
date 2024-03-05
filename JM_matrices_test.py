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
import matplotlib.pyplot as plt

class Matrices:
    def __init__(self):
        self.LOS_matrix = None
        self.potential_linktime = None
        self.propagation_latency = None
        self.capacity_performance = None
        self.BER_performance = None
        self.normalized_linktime_result = None
        self.normalized_capacity_performance = None
        self.normalized_BER_performance = None
        self.normalized_propagation_latency = None

    def compute_matrices(self):
        # Retrieve matrices from routing_network
        self.LOS_matrix = self.routing_network.los_matrix
        self.potential_linktime = self.routing_network.Potential_link_time
        self.propagation_latency = self.routing_network.propagation_latency

        # Fill in capacity_performance and BER_performance
        correction = np.random.rand()
        random_values = np.random.randint(10 - correction, 10+ correction, size=self.LOS_matrix.shape)

        self.capacity_performance = self.potential_linktime - random_values
        self.BER_performance = self.potential_linktime - random_values



        # Normalize matrices
        self.normalized_linktime_result = self._time_normalization(self.potential_linktime, self.potential_linktime)
        self.normalized_capacity_performance = self._time_normalization(self.capacity_performance, self.potential_linktime)
        self.normalized_BER_performance = self._time_normalization(self.BER_performance, self.potential_linktime)
        self.normalized_propagation_latency = self._distance_normalization(self.propagation_latency, self.potential_linktime)

    def _time_normalization(self, data, potential_linktime):
        max_time = np.max(self.potential_linktime, axis=1)
        return data / max_time[:, np.newaxis]

    def _distance_normalization(self, data, potential_link_time):
        max_latency = np.max(self.propagation_latency, axis=1)
        return 1 - data / max_latency[:, np.newaxis]

    def plot_normalized_data(self, normalized_data, title_prefix):
        num_rows = normalized_data.shape[0]
        fig, axes = plt.subplots(num_rows, 1, figsize=(10, 2*num_rows))

        for i in range(num_rows):
            ax = axes[i]
            ax.plot(normalized_data[i], marker='o', linestyle='-')
            ax.set_title(f'{title_prefix} - satellite {i+1}')
            ax.set_xlabel('Time Index')
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_matrices(self):
        # Plotting normalized link time
        self.plot_normalized_data(self.normalized_linktime_result, 'Normalized Link Time')

        # Plotting normalized capacity
        self.plot_normalized_data(self.normalized_capacity_performance, 'Normalized Capacity')

        # Plotting normalized BER
        self.plot_normalized_data(self.normalized_BER_performance, 'Normalized BER')

        # Plotting normalized propagation latency
        self.plot_normalized_data(self.normalized_propagation_latency, 'Normalized Latency')

