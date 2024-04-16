import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Routing_network import routing_network
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level

class applicable_links():
    def __init__(self, time):
        self.links = np.zeros(len(time))
        self.number_of_links = 0
        self.time = time
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    
    def applicability(self, geometrical_output, time, step_size=1.0):
        self.applicable_output = {
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
       
        self.sats_applicable = [[0 for _ in range(len(time))] for _ in range(len(elevation_angles))]

        # Iterate through each time instance
        for index in range(len(time)):
            # Then iterate through each satellite
            for s in range(len(elevation_angles)):
                elev = elevation_angles[s][index]

                # Directly set the value in 'sats_applicable'; no need for 'sats_visibility' in this context
                # since it seems to be unused based on your provided code
                if elev > elevation_min:
                    self.sats_applicable[s][index] = 1
                else:
                    self.sats_applicable[s][index] = np.nan

        for s in range(len(self.sats_applicable)):
            # Initialize lists to collect data for each satellite
            satellite_data = {key: [] for key in self.applicable_output.keys()}
            satellite_data['link number'] = s + 1  # Assuming link number is simply the satellite index + 1


            for index in range(len(time)):
                if self.sats_applicable[s][index] == 1:
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

            for key in self.applicable_output:
                self.applicable_output[key].append(satellite_data[key])

        return self.applicable_output,  self.sats_applicable

    def plot_satellite_visibility(self, time):
        # Convert sats_applicable to a NumPy array for easier manipulation
        sats_applicable_np = np.array(self.sats_applicable)
        # Calculate the accumulated visibility (sum along the satellite axis)
        accumulated_visibility = np.sum(sats_applicable_np, axis=0)
        # Prepare the figure and axes for plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        # Plot for visibility of each satellite over time
        axes[0].set_title("Satellite Visibility Over Time")
        for s in range(len(self.sats_applicable)):
            axes[0].plot(time, sats_applicable_np[s, :], label=f'Sat {s+1}')
        axes[0].set_ylabel("Visibility (0 or 1)")
        axes[0].legend(loc='upper right')
        # Plot for accumulated visible satellites over time
        axes[1].set_title("Accumulated Satellites Visible Over Time")
        axes[1].plot(time, accumulated_visibility, color='red', label='Accumulated Visibility')
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Number of Visible Satellites")
        axes[1].legend(loc='upper left')
        plt.show()

    
    
    #def plot_satellite_visibility_scatter(self, time):
    #    # Convert sats_applicable to a NumPy array for easier manipulation
    #    sats_applicable_np = np.array(self.sats_applicable)
    #    
    #    # Calculate the accumulated visibility (sum along the satellite axis)
    #    accumulated_visibility = np.sum(sats_applicable_np, axis=0)
    #    
    #    # Prepare the figure and axes for plotting
    #    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    #    
    #    # Plot for visibility of each satellite over time using scatter plot
    #    axes[0].set_title("Satellite Visibility Over Time")
    #    for s in range(sats_applicable_np.shape[0]):
    #        # Find indices (time points) where satellite s is visible
    #        visible_indices = np.where(sats_applicable_np[s, :] == 1)[0]
    #        # Plot these as scatter points
    #        axes[0].scatter(visible_indices, np.full(visible_indices.shape, s), alpha=0.6, label=f'Sat {s+1}')
    #    axes[0].set_ylabel("Satellite Index")
    #    axes[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
    #    
    #    # Plot for accumulated visible satellites over time
    #    axes[1].set_title("Accumulated Satellites Visible Over Time")
    #    axes[1].plot(time/step_size_link, accumulated_visibility, color='red', label='Accumulated Visibility')
    #    axes[1].set_xlabel("Time")
    #    axes[1].set_ylabel("Number of Visible Satellites")
    #    axes[1].legend(loc='upper left')
    #    
    #    plt.tight_layout()
    #    plt.show()


    def plot_satellite_visibility_scatter_update(self):
        # Convert sats_applicable to a NumPy array for easier manipulation
        sats_applicable_np = np.array(self.sats_applicable)

        # Calculate the accumulated visibility (sum along the satellite axis)
        accumulated_visibility = np.nansum(sats_applicable_np, axis=0)

        # Prepare the figure and axes for plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot for visibility of each satellite over time using scatter plot
        axes[0].set_title("Satellite Visibility Over Time")
        for s in range(sats_applicable_np.shape[0]):
            # Find indices (time points) where satellite s is visible
            visible_indices = np.where(sats_applicable_np[s, :] == 1)[0]
            # Adjusting the y-value by adding 1 for satellite indexing
            y_values = np.full(visible_indices.shape, s + 1)
            # Plot these as scatter points
            axes[0].scatter(visible_indices, y_values, alpha=0.6, label=f'Sat {s+1}')

        axes[0].set_ylabel("Satellite Index")
        axes[0].legend(loc='upper right')

        # Plot for accumulated visible satellites over time using scatter plot
        axes[1].set_title("Accumulated Satellites Visible Over Time")
        # Create scatter plot for accumulated visibility
        # Here we assume 'time' is a sequence of time steps corresponding to 'accumulated_visibility'
        axes[1].scatter(self.time/step_size_link, accumulated_visibility, color='red', label='Accumulated Visibility')
        axes[1].set_xlabel("Time (steps)")
        axes[1].set_ylabel("Number of Visible Satellites")
        axes[1].legend(loc='upper right')

        plt.tight_layout()
        plt.show()




  


