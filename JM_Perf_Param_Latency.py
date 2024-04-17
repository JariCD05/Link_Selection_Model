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




class Latency_performance():
    def __init__(self, time, link_geometry):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light

    def calculate_latency_performance(self):
        # Extract ranges for all satellites over time from the applicable_output
        ranges = self.applicable_output['ranges']
        
        # Initialize the propagation_latency array
        self.propagation_latency = np.zeros((num_satellites, len(self.time)))

        # Calculate propagation latency for each satellite at each time index
        for sat_index in range(num_satellites):
            for time_index in range(len(self.time)):
                # Ensure there's a valid range value before calculating latency
                if ranges[sat_index][time_index] is not None:
                    latency_propagation = ranges[sat_index][time_index] / self.speed_of_light
                    self.propagation_latency[sat_index, time_index] = latency_propagation

        return self.propagation_latency

    def distance_normalization_min(self, propagation_latency):
        self.normalized_latency_performance = np.zeros_like(propagation_latency)

        for sat_index in range(num_satellites):
            # Filter out zero values and find the minimum nonzero latency
            non_zero_latencies = propagation_latency[sat_index, propagation_latency[sat_index, :] > 0]
            if len(non_zero_latencies) > 0:
                min_latency = np.min(non_zero_latencies)
                # Normalize the latency values for the current satellite based on the minimum nonzero latency
                for time_index in range(propagation_latency.shape[1]):
                    if propagation_latency[sat_index, time_index] > 0:
                        self.normalized_latency_performance[sat_index, time_index] = min_latency/(propagation_latency[sat_index, time_index])
                    else:
                        # If original latency is zero, it remains zero in the normalized array
                        self.normalized_latency_performance[sat_index, time_index] = 0
            else:
                # If there are no nonzero latencies (which might be unusual), handle as needed
                # This scenario would imply all latencies for this satellite are zero
                # You might choose to set the normalized latency to a specific value, or leave it as zero
                pass

        return self.normalized_latency_performance

    def latency_visualization(self, propagation_latency, normalized_latency_performance):    
        # Converting the latency visualization plots to scatter plots and removing all zero values

        # Plotting with zero values filtered out in scatter plot format
        fig, axs = plt.subplots(2, 1, figsize=(15, 10))

        # Filter and Scatter Plot 1: Propagation Latency
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(propagation_latency[s]) if v > 0]
            non_zero_values = [v for v in propagation_latency[s] if v > 0]
            axs[0].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[0].set_title('Propagation Latency $q_{LAT} (t_{j})$')
        axs[0].set_xlabel('Time Steps')
        axs[0].set_ylabel('Latency (s)')
        axs[0].legend()

        # Filter and Scatter Plot 2: Normalized Propagation Latency
        for s in range(num_satellites):
            non_zero_indices = [i for i, v in enumerate(normalized_latency_performance[s]) if v > 0]
            non_zero_values = [v for v in normalized_latency_performance[s] if v > 0]
            axs[1].scatter(non_zero_indices, non_zero_values, label=f'Sat {s+1}')
        axs[1].set_title('Normalized Propagation Latency $\hat{q}_{LAT} (t_{j})$')
        axs[1].set_xlabel('Time Steps')
        axs[1].set_ylabel('Normalized Latency [-]')
        axs[1].legend()

        plt.tight_layout()
        plt.show()





















#    def plot_normalized_latency_heatmap(self, normalized_latency, time):
#        # Assuming `time` is a 1D array with the same length as the number of columns in `normalized_latency`
#        num_satellites = normalized_latency.shape[0]
#        plt.figure(figsize=(10, 6))
#
#        # Create a heatmap of the normalized latency
#        plt.imshow(normalized_latency, aspect='auto', cmap='viridis', origin='lower',
#                   extent=[time[0], time[-1], 0, num_satellites])
#
#        plt.colorbar(label='Normalized Latency (1 - L/max(L))')
#        plt.xlabel('Time')
#        plt.ylabel('Satellite Index')
#        plt.title('Normalized Latency Performance per Satellite Over Time')
#
#        # Optional: Adjust y-ticks to show integer satellite indices
#        plt.yticks(np.arange(0, num_satellites, 1))
#
#        plt.show()
#
#    def plot_normalized_latency_scatter(self, normalized_latency, num_satellites):
#
#        plt.figure(figsize=(10, 6))
#
#        # Extract the normalized latency values for the specified satellite
#        satellite_latency = normalized_latency[num_satellites, :]
#
#        # Create a scatter plot of the normalized latency for the specified satellite over time
#        plt.scatter(self.time, satellite_latency, color='blue', label=f'Satellite {num_satellites+1}')
#
#        plt.xlabel('Time')
#        plt.ylabel('Normalized Latency (1 - L/max(L))')
#        plt.title(f'Normalized Latency for Satellite {num_satellites+1} Over Time')
#        plt.legend()
#
#        plt.show()



    
#print('')
#print('------------------END-TO-END-LASER-SATCOM-MODEL-------------------------')
##------------------------------------------------------------------------
##------------------------------TIME-VECTORS------------------------------
##------------------------------------------------------------------------
## Macro-scale time vector is generated with time step 'step_size_link'
## Micro-scale time vector is generated with time step 'step_size_channel_level'
#
#t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
#samples_mission_level = len(t_macro)
#t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
#samples_channel_level = len(t_micro)
#print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
#print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)
#
#print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
#print('')
#print('-----------------------------------MISSION-LEVEL-----------------------------------------')
##------------------------------------------------------------------------
##------------------------------------LCT---------------------------------
##------------------------------------------------------------------------
## Compute the sensitivity and compute the threshold
#LCT = terminal_properties()
#LCT.BER_to_P_r(BER = BER_thres,
#               modulation = modulation,
#               detection = detection,
#               threshold = True)
#PPB_thres = PPB_func(LCT.P_r_thres, data_rate)
#
##------------------------------------------------------------------------
##-----------------------------LINK-GEOMETRY------------------------------
##------------------------------------------------------------------------
## Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
## First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
## Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
## Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
#link_geometry = link_geometry()
#link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
#                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
#link_geometry.geometrical_outputs()
## Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
#time = link_geometry.time
#mission_duration = time[-1] - time[0]
## Update the samples/steps at mission level
#samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])
#
#Links_applicable = applicable_links(time=time)
#applicable_output, sats_visibility,sats_applicable = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
#
## After defining the latency_performance class...
#
## Create an instance of the class
#latency_performance_instance = Latency_performance(time, link_geometry)
#
## Now call the method on the instance
#propagation_latency = latency_performance_instance.calculate_latency_performance()
#normalized_latency_performance = latency_performance_instance.distance_normalization_min(propagation_latency=propagation_latency)
#visualization = latency_performance_instance.plot_normalized_latency_heatmap(normalized_latency=normalized_latency_performance, time = time)
#visualization_scatter = latency_performance_instance.plot_normalized_latency_scatter(normalized_latency=normalized_latency_performance, time=time, satellite_index=26)
#
## If you want to print or work with the propagation_latency
#print(len(normalized_latency_performance))
#print(propagation_latency[1][700])
