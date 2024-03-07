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


# In the old model, the geometrical data was put into the Routing_network(), which creates the routing output. So in that scenario the choice which link was selected before its performance was calculated.
# In this case we need to do it the other way around, for the available links we need to calculate their potential performance and based on that we need to make a choice which link is selected.
# In the file, JM_link_propagation, the mission level is copied from the model from wieger without the old routing network (as that was the part where the link choice was made). This file needs to calculate the link performance for all applicable links
# Applicable links are links that are visible to the satelite at that specific time interval. It can be the case that in later stages more restrictions will be added to being "applicable" as it can become computational to extensive to analyze the performance output of a couple hundred of links.
# This will be done in def check_link_applicable
# The output of check_link_applicable should be a list at timestamp i listing all applicable links. Satellites are the rows, 1 collumn with either a 1 or a zero

# The output of JM_Link_Propagation should be four matrices (latency_performance, throughput_performance, bit_error_rate_performance, availability_performance)
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

# in order to be able to compare these matrices with one another, the need to be normilized
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


def check_link_applicable(geometrical_output):

    index = 0
    elevation_angles = geometrical_output['elevation']
      
    while index < len(time) - acquisition_time:
                # The handover time is initiated with t=0, then the handover procedure starts
                t_handover = 0
                start_elev = []
                sats_in_LOS  = []
                active_link = 'no'

                # Loop through all satellites in constellation
                # Condition of the while loop is:
                # (1) There is no link currently active (active_link == 'no)
                # (2) The current time step has not yet exceeded the maximum simulation time (index < len(time))
                while active_link == 'no' and index < len(time):
                    for i in range(len(geometrical_output['pos SC'])):
                        elev_last = elevation_angles[i][index - 1]
                        elev      = elevation_angles[i][index]

                        # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), AN ELEVATION THAT IS STILL INCREASING (START OF WINDOW) AND T < T_FINAL
                        # Find satellites for an active link, using 3 conditions:
                        # (1) The current satellite elevation is higher that the defined minimum (elev > elevation_min)
                        # (2) The current satellite elevation is increasing (elev > elev_last)
                        if elev > elevation_min and elev > elev_last:
                            start_elev.append(elev)
                            sats_in_LOS.append(i)

def time_normalization(data, potential_linktime):
    max_time = np.max(potential_linktime, axis=1)
    normalized_result = data / max_time[:, np.newaxis]
    
    return normalized_result

normalized_availability_performance = time_normalization()
normalized_throughput_performance = time_normalization()
normalized_bit_error_rate_performance = time_normalization()



def distance_normalization(propagation_latency):
    # Find the maximum value in each row
  
    max_latency = np.max(propagation_latency, axis=1)

    # Divide each row by its corresponding maximum value
    normalized_latency_result = 1- propagation_latency / max_latency[:, np.newaxis]

    return normalized_latency_result

normalized_latency_performance = distance_normalization()