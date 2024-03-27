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




class Cost_performance():
    def __init__(self, time, link_geometry):
        # Assuming link_geometry.geometrical_output and step_size_link are defined elsewhere
        self.Links_applicable = applicable_links(time=time)
        self.applicable_output, self.sats_visibility, self.sats_applicable = self.Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)
        self.time = time
        self.speed_of_light = speed_of_light

   
    def calculate_cost_performance(self):
        # Assuming self.num_satellites and self.num_time_instances are available
        # If not, you would need to calculate or access these values accordingly
        num_satellites = len(self.applicable_output['ranges']) # or another way to get the number of satellites
        num_time_instances = len(self.time) # Assuming this is how you get time instances

        # Initialize the cost_performance array
        self.cost_performance = np.zeros((num_satellites, num_time_instances))

        # Set cost performance for each satellite at each time index
        for sat_index in range(num_satellites):
            for time_index in range(num_time_instances):
                self.cost_performance[sat_index, time_index] = variable_link_cost_const1

        return self.cost_performance



    def calculate_normalized_cost_performance(self, cost_performance):
        # Calculate the denominator for normalization based on maximum costs
        normalization_factor = max(constellation_variable_link_cost) + max(constellation_fixed_link_cost)
        
        # Normalize the cost_performance array
        self.normalized_cost_performance = 1 - (cost_performance / normalization_factor)
        
        return self.normalized_cost_performance


    
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
#latency_performance_instance = latency_performance(time, link_geometry)
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
