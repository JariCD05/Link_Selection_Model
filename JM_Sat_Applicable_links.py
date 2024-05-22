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

class applicable_satellites():
    def __init__(self, index,  time):
        self.links = np.zeros(1)
        self.number_of_links = 0
        self.index = index
        self.time = time
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(1)

        # Applicability related variables
        self.bitmap = pd.read_csv('CSV/mapping.csv').values  # .values converts it to a numpy array
        self.x_positions = [[0 for _ in range(1)] for _ in range(num_satellites)]
        self.y_positions = [[0 for _ in range(1)] for _ in range(num_satellites)]

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    
    def applicability(self, visibility_output):
       
        self.sats_applicable = [[0 for _ in range(1)] for _ in range(num_satellites)]

        # Iterate through each time instance
     
        for s in range(num_satellites):
            zenith = visibility_output['zenith'][s][0]
            azimuth = visibility_output['azimuth'][s][0]
            while zenith <= -1/2*np.pi:
                    zenith +=  np.pi
                    azimuth +=  np.pi
            while zenith  >= 1/2*np.pi:
                    zenith  -=  np.pi
                    azimuth  -=  np.pi

            while azimuth  <= -np.pi:
                    azimuth  += 2 * np.pi
            while azimuth >= np.pi:
                    azimuth  -= 2 * np.pi
            #print(azimuth)
            #print(zenith)
            if np.isnan(zenith) or np.isnan(azimuth):
                self.sats_applicable[s] = np.nan
            else:
                x = int(np.floor(((azimuth + np.pi) * 36) / (2 * np.pi+0.001000001)))
                y = int(np.floor(((zenith + 0.5*np.pi) * 18) / (np.pi+0.001000001)))

                
                # Store x and y in the respective arrays
                self.x_positions[s] = x
                self.y_positions[s] = y
                #print(f"satellite={s}, timestamp={time}, x={x}, y={y}, bitmap_shape={self.bitmap.shape}")

                # check bitmap # here all other conditions can be added such as minimal elevation angle, rising satellite etc
                if self.bitmap[y][x]:
                    self.sats_applicable[s] = 1
                else:
                    self.sats_applicable[s] = np.nan

        return self.sats_applicable


    


