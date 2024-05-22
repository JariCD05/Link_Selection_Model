import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import input parameters and helper functions
#from input_old import *
from JM_INPUT_CONFIG_FILE import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Old_wieger.Routing_network import routing_network
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level




class geometrical_data_AC():
    def __init__(self, index, time):
        self.links = np.zeros(1)
        self.number_of_links = 0
        self.index = index
        self.time = time
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(1)

        # Visibility related variables
       

    def AC_propagation(self, geometrical_output):  
        self.AC_geometrical_output = {
            'pos AC': [],
            'lon AC': [],
            'lat AC': [],
            'heights AC': [],
            'speeds AC': []
        }
              

        for s in range(num_satellites):
          
            # Append each data point to the respective key in self.AC_geometrical_output
            self.AC_geometrical_output['pos AC'].append(geometrical_output['pos AC'][self.index])
            self.AC_geometrical_output['lon AC'].append(geometrical_output['lon AC'][self.index])
            self.AC_geometrical_output['lat AC'].append(geometrical_output['lat AC'][self.index])
            self.AC_geometrical_output['heights AC'].append(geometrical_output['heights AC'][self.index])
            self.AC_geometrical_output['speeds AC'].append(geometrical_output['speeds AC'][self.index])



        return self.AC_geometrical_output






