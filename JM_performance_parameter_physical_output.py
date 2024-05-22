# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import math
import csv
from copy import deepcopy

# Import input parameters and helper functions
#from input_old import *
from JM_INPUT_CONFIG_FILE import *
from helper_functions import *


# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence

# Import supporting micro-scale calculation functions
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

#Import aircraft data
from JM_AC_propagation import geometrical_data_AC


# Import link applicabality
from JM_Sat_Applicable_links import applicable_satellites

# Import link applicabality
from JM_Sat_Visible_links import visible_satellites
#from JM_Sat_Visible_links_copy import visible_satellites_copy

# Import propagated applicability
from JM_Sat_Applicable_Total_Output import applicable_sat_propagation

# Import performance paramaters 
from JM_Perf_Param_Availability import Availability_performance
from JM_Perf_Param_BER import ber_performance
from JM_Perf_Param_Latency import Latency_performance
from JM_Perf_Param_Throughput import Throughput_performance
from JM_Perf_Param_Cost import Cost_performance
from JM_Perf_Param_Latency_data_transfer import Latency_data_transfer_performance

# Import mission level
from JM_mission_level import mission_level

# Import link selection
from JM_Link_selection_No_link import link_selection_no_link


# Import dynamic visualisation
from JM_Dynamic_Link_Selection_visualization import Dynamic_link_selection_visualization

# Import mission analysis
from JM_mission_performance import SatelliteLinkMetrics

# Import visualization
from JM_visualizations_mission_level import SatelliteDataVisualizer


class PhysicalOutput:
    def __init__(self, time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps, throughput):
        self.time = time
        self.availability_vector = availability_vector
        self.lengths = lengths
        self.num_satellites = num_satellites
        self.max_length_applicability = max_length_applicability
        self.acquisition_time_steps = acquisition_time_steps
        self.throughput = throughput

        # Initiate performance parameters
        self.Availability_performance_instance = Availability_performance(time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps)
        self.BER_performance_instance = ber_performance(time, throughput, lengths, num_satellites, max_length_applicability, acquisition_time_steps)
        self.Throughput_performance_instance = Throughput_performance(time, throughput, lengths, num_satellites, acquisition_time_steps)

    def calculate_performance_matrices(self):
        # Availability
        self.availability_performance = self.Availability_performance_instance.calculate_availability_performance()
        self.normalized_availability_performance = self.Availability_performance_instance.calculate_normalized_availability_performance()
        self.penalized_availability_performance = self.Availability_performance_instance.calculate_penalized_availability_performance()
        self.normalized_penalized_availability_performance = self.Availability_performance_instance.calculate_normalized_penalized_availability_performance()

        # BER
        self.BER_performance = self.BER_performance_instance.calculate_BER_performance()
        self.normalized_BER_performance = self.BER_performance_instance.calculate_normalized_BER_performance()
        self.penalized_BER_performance = self.BER_performance_instance.calculate_penalized_BER_performance()
        self.normalized_penalized_BER_performance = self.BER_performance_instance.calculate_normalized_penalized_BER_performance()

        # Throughput
        self.throughput_performance = self.Throughput_performance_instance.calculate_throughput_performance()
        self.normalized_throughput_performance = self.Throughput_performance_instance.calculate_normalized_throughput_performance()
        self.penalized_throughput_performance = self.Throughput_performance_instance.calculate_penalized_throughput_performance()
        self.normalized_penalized_throughput_performance = self.Throughput_performance_instance.calculate_normalized_penalized_throughput_performance()

        # Return normalized values and penalized values
        normalized_values = [
            self.normalized_availability_performance,
            self.normalized_BER_performance,
            self.normalized_throughput_performance
        ]

        normalized_penalized_values = [
            self.normalized_penalized_availability_performance,
            self.normalized_penalized_BER_performance,
            self.normalized_penalized_throughput_performance
        ]

        return normalized_values, normalized_penalized_values


