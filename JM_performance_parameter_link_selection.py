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

from JM_performance_parameter_geometrical_output import GeometricalPerformanceParameter
from JM_performance_parameter_physical_output import PhysicalOutput


class PerformanceEvaluator:
    def __init__(self, time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps, throughput, weights, satellite_position_indices, max_satellites, active_satellite, num_opt_head, ranges_split, smallest_latency):
        self.time = time
        self.availability_vector = availability_vector
        self.lengths = lengths
        self.num_satellites = num_satellites
        self.max_length_applicability = max_length_applicability
        self.acquisition_time_steps = acquisition_time_steps
        self.throughput = throughput
        self.weights = weights
        self.satellite_position_indices = satellite_position_indices
        self.max_satellites = max_satellites
        self.active_satellite = active_satellite
        self.num_opt_head = num_opt_head
        self.ranges_split = ranges_split
        self.smallest_latency = smallest_latency

        # Initialize performance parameter classes
        self.physical_output = PhysicalOutput(time, availability_vector, lengths, num_satellites, max_length_applicability, acquisition_time_steps, throughput)
        self.geometrical_performance = GeometricalPerformanceParameter(time, lengths, num_satellites, ranges_split, smallest_latency, acquisition_time_steps)

        # Calculate performance matrices
        self.calculate_performance_matrices()

    def calculate_performance_matrices(self):
        self.physical_normalized_values, self.physical_normalized_penalized_values = self.physical_output.calculate_performance_matrices()
        self.geo_normalized_values, self.geo_normalized_penalized_values = self.geometrical_performance.calculate_performance_matrices()

        # Combined weights and normalized values arrays
        self.normalized_values = self.physical_normalized_values + self.geo_normalized_values
        self.normalized_penalized_values = self.physical_normalized_penalized_values + self.geo_normalized_penalized_values

    def perform_link_selection(self, index):
        if self.num_opt_head == 1:
            link_selection_instance = link_selection_no_link(
                self.num_satellites, self.time, self.normalized_values, self.normalized_penalized_values, self.weights, self.satellite_position_indices, self.max_satellites, self.active_satellite)
        else:
            link_selection_instance = link_selection_no_link(
                self.num_satellites, self.time, self.normalized_values, self.normalized_values, self.weights, self.satellite_position_indices, self.max_satellites, self.active_satellite)

        weighted_scores = link_selection_instance.calculate_weighted_performance(index)
        best_satellite, max_score, activated_satellite_index, activated_satellite_number = link_selection_instance.select_best_satellite(index)

        return best_satellite, max_score, activated_satellite_index, activated_satellite_number


