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
from archive.JM_Mission_Level import *

LCT = terminal_properties()
LCT.BER_to_P_r(BER = BER_thres,
               modulation = modulation,
               detection = detection,
               threshold = True)
PPB_thres = PPB_func(LCT.P_r_thres, data_rate)



class performance_parameters():
    def __init__(self, time):
        self.links = np.zeros(len(time))
        self.number_of_links = 0
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    

    def link_time(self, geometrical_output, time, step_size=1.0):

        # Initialize the LOS matrix
        num_satellites = len(geometrical_output['pos SC'])
        JM_Visibility_matrix = routing_network()
        JM_Visibility_matrix.routing()

        self.Potential_link_time = np.zeros((num_satellites, len(time)))


        # Calculate the potential link time of a link
        visibility_index = 0 
        # Calculate future visibility for each satellite
        for sat_index in range(num_satellites):                                                         # check for all satellites within the available satellites
            for visibility_index in range(len(time)): 
                link_time = 0                                                                           #starting potential time is zere
                if self.los_matrix[sat_index, visibility_index] == 0:                                   # if sattelite is not within line of sight, potential time is zero
                    self.Potential_link_time[sat_index,visibility_index] = 0
                else:
                    # Find future instances of visibility
                    future_index = visibility_index                                                     # if not, look at all future time instances and if these are still a 1, add this time to the link time
                    while future_index <len(time):
                        if self.los_matrix[sat_index, future_index] == 1:
                            link_time +=1
                        else:
                            self.Potential_link_time[sat_index, visibility_index] = link_time
                            break
                        future_index+=1
                    
        

        print("Link Time Matrix:")
        print(self.Potential_link_time)


