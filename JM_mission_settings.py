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

# Import aircraft data
from JM_AC_propagation import geometrical_data_AC

# Import link applicability
from JM_Sat_Applicable_links import applicable_satellites

# Import visible satellites
from JM_Sat_Visible_links import visible_satellites
#from JM_Sat_Visible_links_copy import visible_satellites_copy

# Import propagated applicability
from JM_Sat_Applicable_Total_Output import applicable_sat_propagation

# Import performance parameters
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

# Import dynamic visualization
from JM_Dynamic_Link_Selection_visualization import Dynamic_link_selection_visualization

# Import mission analysis
from JM_mission_performance import SatelliteLinkMetrics

# Import visualization
from JM_visualizations_mission_level import SatelliteDataVisualizer


class mission_settings:
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

    def remove_falling_satellites(self, index, list_of_total_scores_all_satellite, satellite_position_indices, active_satellite):
        """
        Check if satellites are falling and mark them for deletion if they are not outperforming the current active satellite.
        
        Parameters:
        - index: The current time step index.
        - list_of_total_scores_all_satellite: List containing the total scores for all satellites.
        - satellite_position_indices: List containing the position indices of satellites.
        - active_satellite: The index of the currently active satellite.
        
        Returns:
        - marked_for_deletion: List of satellite indices marked for deletion.
        """
        marked_for_deletion = []

        for s in range(self.num_satellites):
            satellite_index = satellite_position_indices[s]
            # Ensure the current satellite is not the active one
            if satellite_index != active_satellite:
                # Ensure we have valid scores to compare (not NaN and sufficient historical data)
                current_score = list_of_total_scores_all_satellite[satellite_index][index-1]
                previous_score = list_of_total_scores_all_satellite[satellite_index][index-2]
                previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-3]
                previous_previous_previous_score = list_of_total_scores_all_satellite[satellite_index][index-4]

                # Check if all scores are valid numbers and that the current score is less than the previous scores
                if (not np.isnan(current_score) and not np.isnan(previous_score) and 
                    current_score <= previous_score <= previous_previous_score <= previous_previous_previous_score):
                    # Mark the satellite as inapplicable
                    marked_for_deletion.append(satellite_index)
                    print(f"Satellite {satellite_index + 1} removed from analysis at time index {index}. Reason: Falling satellite performance.")

        return marked_for_deletion

    def remove_unlikely_outperforming_satellites(self, index, satellite_position_indices, active_satellite, normalized_cost_performance, normalized_latency_performance, normalized_latency_data_transfer_performance, list_of_scores_active_satellites):
        """
        Check if the sum of instantaneous performance parameters and the max score for propagated ones is smaller than the previous score.
        If true, mark the satellite for deletion as it is unlikely to outperform the current active satellite.
        
        Parameters:
        - index: The current time step index.
        - satellite_position_indices: List containing the position indices of satellites.
        - active_satellite: The index of the currently active satellite.
        - normalized_cost_performance: Normalized cost performance for all satellites.
        - normalized_latency_performance: Normalized latency performance for all satellites.
        - normalized_latency_data_transfer_performance: Normalized latency data transfer performance for all satellites.
        - list_of_scores_active_satellites: List of scores for active satellites.
        
        Returns:
        - removed_satellites: List of satellite indices removed based on outperforming conditions.
        """
        removed_satellites = []

        for s in range(self.num_satellites):
            satellite_index = satellite_position_indices[s]
            # Check condition and ensure the current satellite isn't the active one
            if (satellite_index != active_satellite and 
                normalized_cost_performance[s] + normalized_latency_performance[s] + normalized_latency_data_transfer_performance[s] + self.weights[0] + self.weights[1] + self.weights[5] < list_of_scores_active_satellites[index-1]):
                removed_satellites.append(satellite_position_indices[s])  # Store only actually removed satellites

        # Print which satellites have been removed
        if removed_satellites:
            print("Satellites removed from the analysis:", removed_satellites)
        else:
            print("No satellites were removed based on potential outperforming current satellite conditions.")

        return removed_satellites

    def update_satellite_lists(self, index, list_of_total_scores_all_satellite, satellite_position_indices, marked_for_deletion):
        """
        Delete the marked satellites from the list of scores and position indices.
        
        Parameters:
        - index: The current time step index.
        - list_of_total_scores_all_satellite: List containing the total scores for all satellites.
        - satellite_position_indices: List containing the position indices of satellites.
        - marked_for_deletion: List of satellite indices marked for deletion.
        
        Returns:
        - updated_list_of_total_scores_all_satellite: Updated list of total scores with falling satellites marked.
        - updated_satellite_position_indices: Updated list of satellite position indices after deletion.
        """
        for marked_satellite in marked_for_deletion:
            satellite_deletion_index = satellite_position_indices.index(marked_satellite)
            list_of_total_scores_all_satellite[marked_satellite][index] = list_of_total_scores_all_satellite[marked_satellite][index-1]
            del satellite_position_indices[satellite_deletion_index]

        return list_of_total_scores_all_satellite, satellite_position_indices

