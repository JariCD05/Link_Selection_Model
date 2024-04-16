#This is original Routing network file.
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
from archive.JM_Link_Propagation import link_propagation_test


# In the old model, the geometrical data was put into the Routing_network(), which creates the routing output. So in that scenario the choice which link was selected before its performance was calculated.
# In this case we need to do it the other way around, for the applicable links we need to calculate their potential performance and based on that we need to make a choice which link is selected.
# In the file, JM_link_propagation, the mission level is copied from the model from wieger without the old routing network (as that was the part where the link choice was made). This file needs to calculate the link performance for all applicable links
# Applicable links are links that are visible to the satelite at that specific time interval. It can be the case that in later stages more restrictions will be added to being "applicable" as it can become computational to extensive to analyze the performance output of a couple hundred of links.

# This will be done in def check_link_applicable
# The output of check_link_applicable should be a list of; at timestamp i listing all applicable links. Satellites are the rows, the collumns are the timestemp, filled with either a 1 or a zero
# For the time instances that a satllites is applicable also the geometrical output shalls be stored. Thus a zero needs to be stored if a satellite is not visible, but the geometrical data needs to be stored once it is visible. 

# The output of JM_applicable links is the input for the link propagation file
# we have equal size list with applicability stored and geometrical output, these are stored in applicability_output

# The link_propagation file shall be iterated over time
# for i in range(len(time)):
#   for j in range(len(applicable satellites)):
# this will lead to a case where at all time instances 
# the biggest issue is now that I don't now how to set up the link_propagation file such that it loops overtime at a specific timestamp
# The output of JM_Link_Propagation should be four matrices (latency_performance, throughput_performance, bit_error_rate_performance, availability_performance) at all time instances
# These matrices should have in the rows the applicable satellite WARNING
# WARNING, make sure to assign a number to each satellite at the beginning of the loop of mission level. Namely, the applicability matrix will be updated with each timestamp, but the performance output in JM_Link_propagation should use the same numbering
# WARNING, it can be the case that satellite 2, 5, 6, 10, 12 are not applicable but the other ones are. The JM_Link_propagation should return the performance of satellite 1, 3, 4, 7, 8, 9, 11

# THIS CAN BE DONE ALREADY WITHIN THE JM_LINK_PROPAGATION FILE
# These four matrices have the applicable satellites as rows, and the performance per time index as collumn
# These matrices should should be (Masked/checked) against a requirement/condition
# Based on the check against the condition the output of this file is going to be four matrices with the applicable satellites as rows, and only 1 collumn. Namely the percentual/one-time performance at that specific time instance

# so for time instance i, we have
# latency_performance: matrix stating the latency time to create the link (in ms)
# throughput_performance: matrix showing the sum of the number of time-instances that throughput value is above certain threshold
# bit_error_rate_performance: matrix showing sum of the number of time-instances that bit_error_rate is below a certain value
# availability_performance: matrix showing sum of the number of time-instances that P_rx + penalty is larger than P_R theshold

# in order to be able to compare these matrices with one another, they need to be normilized
# this is done with def time_normalization and distance_normalization

# Output of def time_normalization:
# normalized_availability_performance 
# normalized_throughput_performance 
# normalized_bit_error_rate_performance 

# Output of def distance_normalization:
# normalized_latency_performance

# These four matrices will have the applicable satellites in their rows and the performance values at time instance i as collumn
# These four matrices will be the input for JM_optimization_function 

# Within this JM_optimization_function the four matrices will be multiplied with a weight 
# and the argmax function will decide to take the one with the highest value, and make that satellite active.

# after that point, thus when at timestamp i a satellite x has been made active, the JM_mission_level_setup loop must go to the next index and create the performance values at the timestamp i + 1 
# Once the performance values are calculated for timestamp i+1 and it arrives again at JM_optimization_function a penalty will be applied to all performance values of non-active links based on T_acq 
# Repeat untill mission time has run out
# store all active links over mission time





def distance_normalization(propagation_latency):
    # Find the maximum value in each row
  
    max_latency = np.max(propagation_latency, axis=1)

    # Divide each row by its corresponding maximum value
    normalized_latency_result = 1- propagation_latency / max_latency[:, np.newaxis]

    return normalized_latency_result




def time_normalization(data, potential_linktime):
    max_time = np.max(potential_linktime, axis=1)
    normalized_result = data / max_time[:, np.newaxis]
    
    return normalized_result

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

link_propagation_test = link_propagation_test()
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
geometrical_output = link_geometry.geometrical_outputs()
# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
mission_duration = time[-1] - time[0]
# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])


normalized_latency_propagation = distance_normalization(propagation_latency = link_propagation_test.calculate_latency(geometrical_output=geometrical_output, num_satellites= num_satellites))