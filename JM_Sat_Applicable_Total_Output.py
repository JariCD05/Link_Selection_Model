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

class applicable_sat_propagation():
    def __init__(self, time):
            self.links = np.zeros(len(time))
            self.number_of_links = 0
            self.time = time
            self.total_handover_time = 0 # seconds
            self.total_acquisition_time = 0 # seconds
            self.acquisition = np.zeros(len(time))

            self.bitmap = pd.read_csv('CSV/mapping.csv').values  # .values converts it to a numpy array

            self.x_positions = [[0 for _ in range(len(time))] for _ in range(num_satellites)]
            self.y_positions = [[0 for _ in range(len(time))] for _ in range(num_satellites)]

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    
    def sat_propagation(self, geometrical_output, time):
        self.propagated_applicable_output = {
            'link number': [],
            'time': [],
            'pos AC': [],
            'lon AC': [],
            'lat AC': [],
            'heights AC': [],
            'speeds AC': [],
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

        self.applicable_total_output = {}

        # Assuming 'time' is a list with the same length as the number of time instances for each satellite
        # and 'elevation_min' is already defined
        elevation_angles = geometrical_output['elevation']
        # Initialize the lists with the shape of 'elevation_angles', filled with 0s
       
        self.propagated_sats_applicable = [[0 for _ in range(len(time))] for _ in range(num_satellites)]

        # Iterate through each time instance
        for index in range(len(time)):
            for s in range(len(elevation_angles)):
                elev = elevation_angles[s][index]


                while geometrical_output['zenith'][s][index] < -1/2*np.pi:
                        geometrical_output['zenith'][s][index] +=  np.pi
                        geometrical_output['azimuth'][s][index] +=  np.pi
                while geometrical_output['zenith'][s][index] > 1/2*np.pi:
                        geometrical_output['zenith'][s][index] -=  np.pi
                        geometrical_output['azimuth'][s][index] -=  np.pi

                while geometrical_output['azimuth'][s][index] < -np.pi:
                        geometrical_output['azimuth'][s][index] += 2 * np.pi
                while geometrical_output['azimuth'][s][index] > np.pi:
                        geometrical_output['azimuth'][s][index] -= 2 * np.pi

                if np.isnan(geometrical_output['azimuth'][s][index]) or np.isnan(geometrical_output['zenith'][s][index]):
                    self.propagated_sats_applicable[s][index] = np.nan
                else:
                    x = int(np.floor(((geometrical_output['azimuth'][s][index] + np.pi) * 36) / (2 * np.pi+0.001000001)))
                    y = int(np.floor(((geometrical_output['zenith'][s][index] + 0.5*np.pi) * 18) / (np.pi+0.001000001)))

                    
                    # Store x and y in the respective arrays
                    self.x_positions[s][index] = x
                    self.y_positions[s][index] = y
                    #print(f"satellite={s}, timestamp={index}, x={x}, y={y}, bitmap_shape={self.bitmap.shape}")

                    # check bitmap and elevation angle
                    if self.bitmap[y][x] and elev > elevation_min:
                        self.propagated_sats_applicable[s][index] = 1
                    else:
                        self.propagated_sats_applicable[s][index] = np.nan

        for s in range(len(self.propagated_sats_applicable)):
            # Initialize lists to collect data for each satellite
            satellite_data = {key: [] for key in self.propagated_applicable_output.keys()}
            satellite_data['link number'] = s + 1  # Assuming link number is simply the satellite index + 1


            for index in range(len(time)):
                if self.propagated_sats_applicable[s][index] == 1:
                    # Satellite is visible, collect geometrical data
                    satellite_data['time'].append(time[index])
                    satellite_data['pos AC'].append(geometrical_output['pos AC'][index])
                    satellite_data['lon AC'].append(geometrical_output['lon AC'][index])
                    satellite_data['lat AC'].append(geometrical_output['lat AC'][index])
                    satellite_data['heights AC'].append(geometrical_output['heights AC'][index])
                    satellite_data['speeds AC'].append(geometrical_output['speeds AC'][index])
                    satellite_data['pos SC'].append(geometrical_output['pos SC'][s][index])
                    satellite_data['lon SC'].append(geometrical_output['lon SC'][s][index])
                    satellite_data['lat SC'].append(geometrical_output['lat SC'][s][index])
                    satellite_data['vel SC'].append(geometrical_output['vel SC'][s][index])
                    satellite_data['heights SC'].append(geometrical_output['heights SC'][s][index])
                    satellite_data['ranges'].append(geometrical_output['ranges'][s][index])
                    satellite_data['elevation'].append(geometrical_output['elevation'][s][index])
                    satellite_data['azimuth'].append(geometrical_output['azimuth'][s][index])
                    satellite_data['zenith'].append(geometrical_output['zenith'][s][index])
                    satellite_data['radial'].append(geometrical_output['radial'][s][index])
                    satellite_data['slew rates'].append(geometrical_output['slew rates'][s][index])
                    satellite_data['elevation rates'].append(geometrical_output['elevation rates'][s][index])
                    satellite_data['azimuth rates'].append(geometrical_output['azimuth rates'][s][index])
                    satellite_data['doppler shift'].append(geometrical_output['doppler shift'][s][index])

                else:
                    # Satellite is not visible, store placeholders
                    satellite_data['time'].append(time[index])
                    satellite_data['pos AC'].append(np.nan)
                    satellite_data['lon AC'].append(np.nan)
                    satellite_data['lat AC'].append(np.nan)
                    satellite_data['heights AC'].append(np.nan)
                    satellite_data['speeds AC'].append(np.nan)
                    satellite_data['pos SC'].append(np.nan)
                    satellite_data['lon SC'].append(np.nan)
                    satellite_data['lat SC'].append(np.nan)
                    satellite_data['vel SC'].append(np.nan)
                    satellite_data['heights SC'].append(np.nan)
                    satellite_data['ranges'].append(np.nan)
                    satellite_data['elevation'].append(np.nan)
                    satellite_data['azimuth'].append(np.nan)
                    satellite_data['zenith'].append(np.nan)
                    satellite_data['radial'].append(np.nan)
                    satellite_data['slew rates'].append(np.nan)
                    satellite_data['elevation rates'].append(np.nan)
                    satellite_data['azimuth rates'].append(np.nan)
                    satellite_data['doppler shift'].append(np.nan)

            for key in self.propagated_applicable_output:
                self.propagated_applicable_output[key].append(satellite_data[key])

        return self.propagated_applicable_output,  self.propagated_sats_applicable