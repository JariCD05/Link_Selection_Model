import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input_old import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Old_wieger.Routing_network import routing_network
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

class applicable_sat_propagation():
    def __init__(self, propagated_time,num_satellites):
        self.num_satellites = num_satellites
        self.number_of_links = 0
        self.time = propagated_time
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.bitmap = pd.read_csv('CSV/mapping.csv').values  # .values converts it to a numpy array
        self.x_positions = [[0 for _ in range(1)] for _ in range(num_satellites)]
        self.y_positions = [[0 for _ in range(1)] for _ in range(num_satellites)]

    def sat_propagation(self, visiblity_output, geometrical_output, sats_applicable):
        self.applicable_propagated_output = {
            'propagated_sats_applicable':[],
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

        

        self.propagated_sats_applicable = [[0 for _ in range(len(self.time))] for _ in range(self.num_satellites)]
        
       
             
    
        for s in range(len(self.propagated_sats_applicable)):
            satellite__propagated_data = {key: [] for key in self.applicable_propagated_output.keys()}
            satellite__propagated_data['link number'] = s + 1  # Assuming link number is simply the satellite index + 1

        
            for index in range(len(self.time)):
                if sats_applicable[s][index] == 1:
                    # Satellite is applicable, collect geometrical data
                    satellite__propagated_data['time'].append(self.time[index])
                    satellite__propagated_data['pos SC'].append(geometrical_output['pos SC'][s][index])
                    satellite__propagated_data['lon SC'].append(geometrical_output['lon SC'][s][index])
                    satellite__propagated_data['lat SC'].append(geometrical_output['lat SC'][s][index])
                    satellite__propagated_data['vel SC'].append(geometrical_output['vel SC'][s][index])
                    satellite__propagated_data['heights SC'].append(geometrical_output['heights SC'][s][index])
                    satellite__propagated_data['ranges'].append(geometrical_output['ranges'][s][index])
                    satellite__propagated_data['elevation'].append(geometrical_output['elevation'][s][index])
                    satellite__propagated_data['azimuth'].append(geometrical_output['azimuth'][s][index])
                    satellite__propagated_data['zenith'].append(geometrical_output['zenith'][s][index])
                    satellite__propagated_data['radial'].append(geometrical_output['radial'][s][index])
                    satellite__propagated_data['slew rates'].append(geometrical_output['slew rates'][s][index])
                    satellite__propagated_data['elevation rates'].append(geometrical_output['elevation rates'][s][index])
                    satellite__propagated_data['azimuth rates'].append(geometrical_output['azimuth rates'][s][index])
                    satellite__propagated_data['doppler shift'].append(geometrical_output['doppler shift'][s][index])

                else:
                    # Satellite is not applicable, store placeholders
                    satellite__propagated_data['time'].append(self.time[index])

                    satellite__propagated_data['pos SC'].append(np.nan)
                    satellite__propagated_data['lon SC'].append(np.nan)
                    satellite__propagated_data['lat SC'].append(np.nan)
                    satellite__propagated_data['vel SC'].append(np.nan)
                    satellite__propagated_data['heights SC'].append(np.nan)
                    satellite__propagated_data['ranges'].append(np.nan)
                    satellite__propagated_data['elevation'].append(np.nan)
                    satellite__propagated_data['azimuth'].append(np.nan)
                    satellite__propagated_data['zenith'].append(np.nan)
                    satellite__propagated_data['radial'].append(np.nan)
                    satellite__propagated_data['slew rates'].append(np.nan)
                    satellite__propagated_data['elevation rates'].append(np.nan)
                    satellite__propagated_data['azimuth rates'].append(np.nan)
                    satellite__propagated_data['doppler shift'].append(np.nan)

            for key in self.applicable_propagated_output:
                self.applicable_propagated_output[key].append(satellite__propagated_data[key])

        return self.applicable_propagated_output, self.propagated_sats_applicable
