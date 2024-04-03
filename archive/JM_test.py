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


class routing_network():
    def __init__(self, time):
        self.time = time
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
            'doppler shift': [],
            'visibility_duration': []  # New key to store visibility duration
        }

    def routing(self, geometrical_output):
        num_planes, num_sats_per_plane, num_time_steps = geometrical_output['pos_SC'].shape

        for t in range(num_time_steps):
            # Initialize routing output for this time step
            time_step_output = {
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
                'doppler shift': [],
                'visibility_duration': []  # New key to store visibility duration
            }

            for plane_idx in range(num_planes):
                for sat_idx in range(num_sats_per_plane):
                    # Check if the satellite is visible at this time step
                    if geometrical_output['elevation'][plane_idx][sat_idx][t] > elevation_min:
                        # If visible, add its data to the output for this time step
                        time_step_output['link number'].append(plane_idx * num_sats_per_plane + sat_idx)
                        time_step_output['time'].append(self.time[t])
                        time_step_output['pos AC'].append(geometrical_output['pos_AC'][plane_idx][t])
                        time_step_output['lon AC'].append(geometrical_output['lon_AC'][plane_idx][t])
                        time_step_output['lat AC'].append(geometrical_output['lat_AC'][plane_idx][t])
                        time_step_output['heights AC'].append(geometrical_output['h_AC'][plane_idx][t])
                        time_step_output['speeds AC'].append(geometrical_output['speed_AC'][plane_idx][t])
                        time_step_output['pos SC'].append(geometrical_output['pos_SC'][plane_idx][sat_idx][t])
                        time_step_output['lon SC'].append(geometrical_output['lon_SC'][plane_idx][sat_idx][t])
                        time_step_output['lat SC'].append(geometrical_output['lat_SC'][plane_idx][sat_idx][t])
                        time_step_output['vel SC'].append(geometrical_output['vel_SC'][plane_idx][sat_idx][t])
                        time_step_output['heights SC'].append(geometrical_output['h_SC'][plane_idx][sat_idx][t])
                        time_step_output['ranges'].append(geometrical_output['range'][plane_idx][sat_idx][t])
                        time_step_output['elevation'].append(geometrical_output['elevation'][plane_idx][sat_idx][t])
                        time_step_output['azimuth'].append(geometrical_output['azimuth'][plane_idx][sat_idx][t])
                        time_step_output['zenith'].append(geometrical_output['zenith'][plane_idx][sat_idx][t])
                        time_step_output['radial'].append(geometrical_output['radial'][plane_idx][sat_idx][t])
                        time_step_output['slew rates'].append(geometrical_output['slew_rate'][plane_idx][sat_idx][t])
                        time_step_output['elevation rates'].append(geometrical_output['elevation_rate'][plane_idx][sat_idx][t])
                        time_step_output['azimuth rates'].append(geometrical_output['azimuth_rate'][plane_idx][sat_idx][t])
                        time_step_output['doppler shift'].append(geometrical_output['doppler_shift'][plane_idx][sat_idx][t])

                        # Calculate visibility duration
                        visibility_duration = 1
                        for future_t in range(t + 1, num_time_steps):
                            if geometrical_output['elevation'][plane_idx][sat_idx][future_t] > elevation_min:
                                visibility_duration += 1
                            else:
                                break
                        time_step_output['visibility_duration'].append(visibility_duration)\
                    
                            # Append the output for this time step to the overall routing output
            for key in self.routing_output.keys():
                self.routing_output[key].append(time_step_output[key])

        # Convert lists to numpy arrays
        for key in self.routing_output.keys():
            self.routing_output[key] = np.array(self.routing_output[key])

        # Visualize the LOS matrix
        plt.figure(figsize=(10, 6))
        for sat_idx in range(num_sats_per_plane):
            visibility_durations = self.routing_output['visibility_duration'][:, sat_idx]
            visible_time_steps = np.where(visibility_durations > 0)[0]
            for t in visible_time_steps:
                plt.plot([self.time[t], self.time[t] + visibility_durations[t]], [sat_idx, sat_idx], c='black', label=f'Satellite {sat_idx}')

        plt.xlabel('Time')
        plt.ylabel('Satellite Index')
        plt.title('Future Visibility of Satellites')
        plt.legend()
        plt.grid(True)
        plt.show()

        print(f"Time: {time[t]}")
        for sat_idx, duration in enumerate(visibility_duration):
            if duration > 0:
                print(f"Satellite {sat_idx}: Visible for {duration} time steps")

        return self.routing_output

# Import the routing_network class
from archive.JM_test import routing_network

# Create an instance of the routing_network class
network = routing_network()

# Call the routing method to compute the routing sequence
# Pass the required input parameters to the method
routing_output = network.routing(geometrical_output, time)

# Print the routing output
print("Routing Output:")
print(routing_output)