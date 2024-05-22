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


class visible_satellites():
    def __init__(self, index, time):
        self.links = np.zeros(1)
        self.number_of_links = 0
        self.index = index
        self.time = time
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(1)

        # Visibility related variables
        self.sats_visible = [[0 for _ in range(1)] for _ in range(num_satellites)]
    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    
    def visibility(self, geometrical_output, time):
        self.visibility_output = {
            'link number': [],
            'time': [],
            'pos SC': [],
            'lon SC': [],
            'lat SC': [],
            'vel SC': [],
            'heights SC': [],
            'ranges': [],
            'elevation': [],
            'azimuth': [],
            'zenith': [],
            'radial': [],
            'slew rates': [],
            'elevation rates': [],
            'azimuth rates': [],
            'doppler shift': []
        }

        self.visibility_total_output = {}

        # Assuming 'time' is a list with the same length as the number of time instances for each satellite
        # and 'elevation_min' is already defined
        
            
  
        
        
        
        # Iterate through each time instance

        for s in range(num_satellites):
            elevation_angles = geometrical_output['elevation'][s][self.index]
            if elevation_angles > elevation_min:
                self.sats_visible[s] = 1
            else:
                self.sats_visible[s] = np.nan

                  
        for s in range(num_satellites):
            # Initialize lists to collect data for each satellite
            visible_sat_data = {key: [] for key in self.visibility_output.keys()}
            visible_sat_data['link number'] = s + 1  # Assuming link number is simply the satellite index + 1

            
            if self.sats_visible[s] == 1:
                # Satellite is visible, collect geometrical data
                visible_sat_data['time'].append(time)
                visible_sat_data['pos SC'].append(geometrical_output['pos SC'][s][self.index])
                visible_sat_data['lon SC'].append(geometrical_output['lon SC'][s][self.index])
                visible_sat_data['lat SC'].append(geometrical_output['lat SC'][s][self.index])
                visible_sat_data['vel SC'].append(geometrical_output['vel SC'][s][self.index])
                visible_sat_data['heights SC'].append(geometrical_output['heights SC'][s][self.index])
                visible_sat_data['ranges'].append(geometrical_output['ranges'][s][self.index])
                visible_sat_data['elevation'].append(geometrical_output['elevation'][s][self.index])
                visible_sat_data['azimuth'].append(geometrical_output['azimuth'][s][self.index])
                visible_sat_data['zenith'].append(geometrical_output['zenith'][s][self.index])
                visible_sat_data['radial'].append(geometrical_output['radial'][s][self.index])
                visible_sat_data['slew rates'].append(geometrical_output['slew rates'][s][self.index])
                visible_sat_data['elevation rates'].append(geometrical_output['elevation rates'][s][self.index])
                visible_sat_data['azimuth rates'].append(geometrical_output['azimuth rates'][s][self.index])
                visible_sat_data['doppler shift'].append(geometrical_output['doppler shift'][s][self.index])
              
            else:
                # Satellite is not visible, store placeholders
                visible_sat_data['time'].append(time)
                visible_sat_data['pos SC'].append(np.nan)
                visible_sat_data['lon SC'].append(np.nan)
                visible_sat_data['lat SC'].append(np.nan)
                visible_sat_data['vel SC'].append(np.nan)
                visible_sat_data['heights SC'].append(np.nan)
                visible_sat_data['ranges'].append(np.nan)
                visible_sat_data['elevation'].append(np.nan)
                visible_sat_data['azimuth'].append(np.nan)
                visible_sat_data['zenith'].append(np.nan)
                visible_sat_data['radial'].append(np.nan)
                visible_sat_data['slew rates'].append(np.nan)
                visible_sat_data['elevation rates'].append(np.nan)
                visible_sat_data['azimuth rates'].append(np.nan)
                visible_sat_data['doppler shift'].append(np.nan)

           
                
            for key in self.visibility_output:
                self.visibility_output[key].append(visible_sat_data[key])
  
        return self.visibility_output,  self.sats_visible












