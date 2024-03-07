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

class routing_network():


        
    
    
    def routing(self, geometrical_output, time, step_size=1.0):
       
    
        # This method computes the routing sequence of the links between AIRCRAFT and SATELLITES in constellation.
        # This model uses only geometric data dictionary as INPUT with 10 KEYS (pos_SC, vel_SC, h_SC, range, slew_rate, elevation, zenith, azimuth, radial, doppler shift)
        # Each KEY has a value with shape (NUMBER_OF_PLANES, NUMBER_SATS_PER_PLANE, LEN(TIME)) (for example: 10x10x3600 for 10 planes, 10 sats and 1 hour with 1s time steps)

        # OUTPUT of the model must be same geometric data dictionary as INPUT, with KEYs of shape LEN(TIME). Meaning that there is only one vector for all KEYS.

        # This option is the DEFAULT routing model. Here, link is available above a minimum elevation angle.
        # When the current link goes beyond this minimum, the next link is searched. This will be the link with the lowest (and rising) elevation angle.

        # Initial array where all orbital trajectories are stored
        self.routing_output = {
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

        self.routing_total_output = {}

        # Initialize the LOS matrix
        num_satellites = len(geometrical_output['pos SC'])
        self.los_matrix = np.zeros((num_satellites, len(time)))
        index = 0
        elevation_angles = geometrical_output['elevation']

        start_elev = []
        sats_in_LOS  = []
        for index in range(len(time)):
            # Loop through all satellites in constellation
            # Condition of the while loop is:
         
            # (1) The current time step has not yet exceeded the maximum simulation time (index < len(time))
            for i in range(len(geometrical_output['pos SC'])):
                elev_last = elevation_angles[i][index - 1]
                elev      = elevation_angles[i][index]

                    # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), 
                    # Find satellites for an active link, using the condition:
                    # (1) The current satellite elevation is higher that the defined minimum (elev > elevation_min)
                if elev > elevation_min :
                    start_elev.append(elev)
                    sats_in_LOS.append(i)
                    self.los_matrix[i, index] = 1




                # If there are satellites that satisfy the above conditions, a new link will be chosen
                # If no satellite satisfies the above conditions, the last step will be repeated for the next time step
                if len(sats_in_LOS) > 0:
                    print("Satellites within Line of Sight:")
                    for s in sats_in_LOS:
                        print("Satellite ", s)

                

                    # During an active link, loop through the indices and evaluate for each indice if the elevation angle satisfies condition:
                    # Current elevation angle is higher than minimum elevation angle (defined in input.py)
                    index_start_window = index
                    while current_elevation > elevation_min and index < len(time):
                        current_elevation = elevation_angles[sats_in_LOS][index]
                        index += 1


                    #self.links[index_start_window:index] = self.number_of_links
                    #mask = self.links > 0

                    #self.routing_output['link number'].append(self.number_of_links)
                    self.routing_output['time'].append(np.array(time[index_start_window:index]))
                    self.routing_output['pos AC'].append(geometrical_output['pos AC'][index_start_window:index])
                    self.routing_output['lon AC'].append(geometrical_output['lon AC'][index_start_window:index])
                    self.routing_output['lat AC'].append(geometrical_output['lat AC'][index_start_window:index])
                    self.routing_output['heights AC'].append(geometrical_output['heights AC'][index_start_window:index])
                    self.routing_output['speeds AC'].append(geometrical_output['speeds AC'][index_start_window:index])

                    self.routing_output['pos SC'].append(geometrical_output['pos SC'][sats_in_LOS][index_start_window:index])
                    self.routing_output['lon SC'].append(geometrical_output['lon SC'][sats_in_LOS][index_start_window:index])
                    self.routing_output['lat SC'].append(geometrical_output['lat SC'][sats_in_LOS][index_start_window:index])
                    self.routing_output['vel SC'].append(geometrical_output['vel SC'][sats_in_LOS][index_start_window:index])
                    self.routing_output['heights SC'].append(geometrical_output['heights SC'][sats_in_LOS][index_start_window:index])
                    self.routing_output['ranges'].append(geometrical_output['ranges'][sats_in_LOS][index_start_window:index])
                    self.routing_output['elevation'].append(geometrical_output['elevation'][sats_in_LOS][index_start_window:index])
                    self.routing_output['azimuth'].append(geometrical_output['azimuth'][sats_in_LOS][index_start_window:index])
                    self.routing_output['zenith'].append(geometrical_output['zenith'][sats_in_LOS][index_start_window:index])
                    self.routing_output['radial'].append(geometrical_output['radial'][sats_in_LOS][index_start_window:index])
                    self.routing_output['slew rates'].append(geometrical_output['slew rates'][sats_in_LOS][index_start_window:index])
                    self.routing_output['elevation rates'].append(geometrical_output['elevation rates'][sats_in_LOS][index_start_window:index])
                    self.routing_output['azimuth rates'].append(geometrical_output['azimuth rates'][sats_in_LOS][index_start_window:index])
                    self.routing_output['doppler shift'].append(geometrical_output['doppler shift'][sats_in_LOS][index_start_window:index])

                index += 1

        

        self.routing_total_output['time'] = flatten(self.routing_output['time'])
        self.routing_total_output['pos AC'] = flatten(self.routing_output['pos AC'])
        self.routing_total_output['lon AC'] = flatten(self.routing_output['lon AC'])
        self.routing_total_output['lat AC'] = flatten(self.routing_output['lat AC'])
        self.routing_total_output['heights AC'] = flatten(self.routing_output['heights AC'])
        self.routing_total_output['speeds AC'] = flatten(self.routing_output['speeds AC'])

        self.routing_total_output['pos SC'] = flatten(self.routing_output['pos SC'])
        self.routing_total_output['lon SC'] = flatten(self.routing_output['lon SC'])
        self.routing_total_output['lat SC'] = flatten(self.routing_output['lat SC'])
        self.routing_total_output['vel SC'] = flatten(self.routing_output['vel SC'])
        self.routing_total_output['heights SC'] = flatten(self.routing_output['heights SC'])
        self.routing_total_output['ranges'] = flatten(self.routing_output['ranges'])
        self.routing_total_output['elevation'] = flatten(self.routing_output['elevation'])
        self.routing_total_output['azimuth'] = flatten(self.routing_output['azimuth'])
        self.routing_total_output['zenith'] = flatten(self.routing_output['zenith'])
        self.routing_total_output['radial'] = flatten(self.routing_output['radial'])
        self.routing_total_output['slew rates'] = flatten(self.routing_output['slew rates'])
        self.routing_total_output['elevation rates'] = flatten(self.routing_output['elevation rates'])
        self.routing_total_output['azimuth rates'] = flatten(self.routing_output['azimuth rates'])
        self.routing_total_output['doppler shift'] = flatten(self.routing_output['doppler shift'])

        self.comm_time = len(self.routing_total_output['time']) * step_size

        self.frac_comm_time = self.comm_time / time[-1]
        print('ROUTING MODEL')
        print('------------------------------------------------')
        print('Optimization of max. link time and max. elevation')
        print('Number of links             : ' + str(self.number_of_links))
        print('Average link time           : ' + str(np.round(self.comm_time/self.number_of_links/60, 3))+' min')
        print('Total acquisition time      : ' + str(self.total_acquisition_time/60)+' min')
        print('Fraction of total link time : ' + str(self.frac_comm_time))
        print('------------------------------------------------')

        
        print("Current state of routing_output:", self.routing_output)

        return self.routing_output, self.routing_total_output




routing_network = routing_network()
routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_outputs, time, step_size_link)

total_time = len(time)*step_size_link
comm_time = len(flatten(routing_output['time']))*step_size_link

