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
from Routing_network import routing_network


#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
#print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
#print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

#print('----------------------------------------------------------------------------')

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
# First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
# Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
link_geometry.geometrical_outputs()
# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
mission_duration = time[-1] - time[0]
# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])


routing_network = routing_network(time=time)
routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_output, time, step_size_link)


#------------------------------------------------------------------------------------------------------------
#----------------------------initiate all matrices----------------------------------------------
LOS_matrix = routing_network.los_matrix
potential_linktime = routing_network.Potential_link_time
propagation_latency = routing_network.propagation_latency
capacity_performance = potential_linktime.copy()
BER_performance = potential_linktime.copy()

#------------------------------------------------------------------------------------------------------------
#----------------------------fill capacity_performance----------------------------------------------

correction = np.random.rand()
random_values = np.random.randint(10 - correction, 10+ correction, size= LOS_matrix.shape)

capacity_performance[LOS_matrix>0] -= random_values[LOS_matrix>0]
BER_performance[LOS_matrix>0] -= random_values[LOS_matrix>0]
#print(LOS_matrix, potential_linktime, propagation_latency, capacity_performance, BER_performance)


#------------------------------------------------------------------------------------------------------------
#----------------------------normalize all matrices----------------------------------------------

def time_normalization(data, potential_linktime):
    max_time = np.max(potential_linktime, axis=1)
    normalized_result = data / max_time[:, np.newaxis]
    
    return normalized_result

normalized_linktime_result = time_normalization(potential_linktime, potential_linktime)
normalized_capacity_performance = time_normalization(capacity_performance, potential_linktime)
normalized_BER_performance = time_normalization(BER_performance, potential_linktime)



def distance_normalization(propagation_latency):
    # Find the maximum value in each row
  
    max_latency = np.max(propagation_latency, axis=1)

    # Divide each row by its corresponding maximum value
    normalized_latency_result = 1- propagation_latency / max_latency[:, np.newaxis]

    return normalized_latency_result

normalized_propagation_latency = distance_normalization(propagation_latency)

###----------------------------------Generate Plots---------------------------------------------------
import matplotlib.pyplot as plt

def plot_normalized_data(normalized_data, title_prefix):
    num_rows = normalized_data.shape[0]
    fig, axes = plt.subplots(num_rows, 1, figsize=(10, 2*num_rows))
    
    for i in range(num_rows):
        ax = axes[i]
        ax.plot(normalized_data[i], marker='o', linestyle='-')
        ax.set_title(f'{title_prefix} - satellite {i+1}')
        ax.set_xlabel('Time Index')
        ax.grid(True)

    plt.tight_layout()
    #plt.show()

# Plotting normalized link time
plot_normalized_data(normalized_linktime_result, 'Normalized Link Time')

# Plotting normalized capacity
plot_normalized_data(normalized_capacity_performance, 'Normalized Capacity')

# Plotting normalized BER
plot_normalized_data(normalized_BER_performance, 'Normalized BER')

# Plotting normalized propagation latency
plot_normalized_data(normalized_propagation_latency, 'Normalized Latency')

###---------------------------------------------------------------------------------------------


