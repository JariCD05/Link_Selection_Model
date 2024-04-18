# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os

# Import input parameters and helper functions
from input import *
from helper_functions import *
from JM_Visualisations_mission_level import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
#from JM_Atmosphere import JM_attenuation, JM_turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level


class link_selection():
    def find_and_track_active_satellites_with_pandas(self, weights, sats_applicable, performance_matrices, performance_matrices_including_penalty, availability_vector):
        sats_applicable = np.array(sats_applicable)
        availability_vector = np.array(availability_vector)  # Ensure this is a numpy array

        if len(weights) != len(performance_matrices):
            raise ValueError("Mismatch in number of weights and matrices")

        data_columns = ['Time Step', 'Satellite Index', 'Satellite Visible', 'Satellite Available', 
                        'Active Satellite', 'Weighted Values', 'Unweighted Values', 'Availability', 
                        'BER', 'Cost', 'Latency', 'Data Transfer Latency', 'Throughput', 
                        'Weighted Availability', 'Weighted BER', 'Weighted Cost', 'Weighted Latency', 
                        'Weighted Data Transfer Latency', 'Weighted Throughput', 'Values Used']
        data_records = []
        active_satellites = []

        for time_step in range(sats_applicable.shape[1]):
            applicable_at_time_step = sats_applicable[:, time_step]

            current_matrices = []
            current_unweighted_matrices = []
            active_satellite = None  # Initialize active_satellite to None each iteration

            # Track the usage of penalty or normal values
            values_used = []

            for matrix_idx, matrix in enumerate(performance_matrices_including_penalty):
                if active_satellite is not None and applicable_at_time_step[active_satellite]:
                    # Active satellite uses normal values
                    matrix_copy = np.array(matrix[:, time_step])
                    matrix_copy[active_satellite] = performance_matrices[matrix_idx][:, time_step][active_satellite]
                    values_used.append("Normal" if matrix_idx == active_satellite else "Penalty")
                else:
                    # All others use penalty values
                    matrix_copy = matrix[:, time_step]
                    values_used.extend(["Penalty"] * len(matrix_copy))

                current_matrices.append(matrix_copy)
                current_unweighted_matrices.append(performance_matrices[matrix_idx][:, time_step])

            weighted_sum = np.zeros_like(applicable_at_time_step, dtype=float)
            unweighted_sum = np.zeros_like(applicable_at_time_step, dtype=float)
            weighted_parameters = {i: [] for i in range(len(weights))}

            # Calculating weighted and unweighted sums
            for idx, (weight, matrix, unweighted_matrix) in enumerate(zip(weights, current_matrices, current_unweighted_matrices)):
                weighted_matrix = weight * matrix
                weighted_parameters[idx] = weighted_matrix * applicable_at_time_step
                weighted_sum += weighted_parameters[idx]
                unweighted_sum += unweighted_matrix * applicable_at_time_step

            if np.any(np.isfinite(weighted_sum)) and np.nanmax(weighted_sum) > 0:
                active_satellite = np.nanargmax(weighted_sum)
            else:
                active_satellite = "No link"
            active_satellites.append(active_satellite)

            # Store detailed information for this time step
            for sat_idx in range(len(weighted_sum)):
                satellite_visible = 'Yes' if applicable_at_time_step[sat_idx] == 1 else 'No'
                satellite_available = 'Yes' if availability_vector[sat_idx, time_step] == 1 else 'No'
                record = {
                    'Time Step': time_step,
                    'Satellite Index': sat_idx + 1,
                    'Satellite Visible': satellite_visible,
                    'Satellite Available': satellite_available,
                    'Active Satellite': 'Yes' if sat_idx == active_satellite else 'No',
                    'Weighted Values': weighted_sum[sat_idx],
                    'Unweighted Values': unweighted_sum[sat_idx],
                    'Availability': current_unweighted_matrices[0][sat_idx],
                    'BER': current_unweighted_matrices[1][sat_idx],
                    'Cost': current_unweighted_matrices[2][sat_idx],
                    'Latency': current_unweighted_matrices[3][sat_idx],
                    'Latency data transfer': current_unweighted_matrices[4][sat_idx],
                    'Throughput': current_unweighted_matrices[5][sat_idx],
                    'Weighted Availability': weighted_parameters[0][sat_idx],
                    'Weighted BER': weighted_parameters[1][sat_idx],
                    'Weighted Cost': weighted_parameters[2][sat_idx],
                    'Weighted Latency': weighted_parameters[3][sat_idx],
                    'Weighted Data Transfer Latency': weighted_parameters[4][sat_idx],
                    'Weighted Throughput': weighted_parameters[5][sat_idx],
                    'Values Used': "Normal" if sat_idx == active_satellite and active_satellite != "No link" else "Penalty"
                }
                data_records.append(record)

        # Convert list of records to DataFrame
        data_frame = pd.DataFrame(data_records, columns=data_columns)
        return active_satellites, data_frame