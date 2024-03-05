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
        self.links = np.zeros(len(time))
        self.number_of_links = 0
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------
    

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
        

        index = 0
        elevation_angles = geometrical_output['elevation']

        # Initialize the LOS matrix
        num_satellites = len(geometrical_output['pos SC'])
        self.los_matrix = np.zeros((num_satellites, len(time)))
        self.Potential_link_time = np.zeros((num_satellites, len(time)))

        while index < len(time) :
            # The handover time is initiated with t=0, then the handover procedure starts
            t_handover = 0
            start_elev = []
            sats_in_LOS  = []
            active_link = 'no'

            # Loop through all satellites in constellation
            # Condition of the while loop is:
            # (1) There is no link currently active (active_link == 'no)
            # (2) The current time step has not yet exceeded the maximum simulation time (index < len(time))

            for i in range(len(geometrical_output['pos SC'])):
                elev_last = elevation_angles[i][index - 1]
                elev      = elevation_angles[i][index]

                    # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), 
                    # Find satellites for an active link, using the condition:
                    # (1) The current satellite elevation is higher that the defined minimum (elev > elevation_min)
                if elev > elevation_min :
                    start_elev.append(elev)
                    sats_in_LOS.append(i)
                    self.los_matrix[i, index] = 1  # Set the corresponding entry to 1      
            index +=1


        ##Hacker man - adding a row of zero to see if that help with getting link time of all satellites
        #num_rows, num_cols = self.los_matrix.shape
        #zeros_column = np.zeros((num_rows, 1))
        #self.los_matrix = np.hstack((self.los_matrix, zeros_column))


        #print("LOS Matrix:")
        #print(self.los_matrix)
        
#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------LINK TIME PERFORMANCE PARAMETER-------------------------------------------------------------------------------------------
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
                            break
                        future_index+=1
                    self.Potential_link_time[sat_index, visibility_index] = link_time
    

        #print("Link Time Matrix:")
        #print(self.Potential_link_time)

#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------Plot LINK TIME PERFROMANCE PARAMETERS AND LOS ----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

 #       # Get the number of satellites
 #       num_satellites = self.Potential_link_time.shape[0]
#
 #       # Create separate plots for each satellite
 #       for sat_index in range(num_satellites):
 #           fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure for each satellite
#
 #           # Flatten the potential link time for the current satellite
 #           potential_link_time_sat = self.Potential_link_time[sat_index]
 #           potential_link_time_flat = potential_link_time_sat.flatten()
#
 #           # Generate indices corresponding to each time step
 #           indices = np.arange(len(potential_link_time_flat))
#
 #           # Create the scatter plot for the current satellite
 #           ax.scatter(indices, potential_link_time_flat, label=f'Satellite {sat_index + 1}', marker='o', color='blue', alpha=0.5, s=10)  # Reduced scatter size
 #           ax.set_ylabel('Potential Link Time', fontsize=12)  # Setting font size for y-axis label
 #           ax.set_ylim(-5, 10+max(potential_link_time_flat))  # Setting y-axis limits
#
 #           # Set common x-axis label
 #           ax.set_xlabel('Time Index', fontsize=10)  # Setting font size for x-axis label
 #           
 #           # Show legend with label "Satellite <index>"
 #           ax.legend(loc='upper right', fontsize=10)
#
 #           # Adjust layout
 #           plt.tight_layout()
#
 #           # Show plot
 #           #plt.show()

# Example usage:
# Assuming Potential_link_time is a 3D numpy array
# Potential_link_time.shape = (num_satellites, rows_per_satellite, columns_per_satellite)
# You would instantiate YourClass with this array
# your_object = YourClass(Potential_link_time)
# your_object.plot_data()


# Example usage:
# Assuming Potential_link_time is a 3D numpy array
# Potential_link_time.shape = (num_satellites, rows_per_satellite, columns_per_satellite)
# You would instantiate YourClass with this array
# your_object = YourClass(Potential_link_time)
# your_object.plot_data()


#        num_satellites = self.los_matrix.shape[0]
#
#        plt.figure(figsize=(10, 6))
#        for sat_index in range(num_satellites):
#            visible_indices = np.where(self.los_matrix[sat_index] == 1)[0]
#            plt.scatter(visible_indices, [sat_index + 1] * len(visible_indices), c='black', marker='.', label=f'Satellite {sat_index}')  # Adjusted label to start with index
#        plt.scatter([], [], c='black', marker='.', label='Satellite 0')  # Empty scatter plot for legend consistency
#
#        plt.xlabel('Time Index')
#        plt.ylabel('Satellite Index')
#        plt.title('Line of Sight (LOS) Visualization')
#        plt.grid(True)
#        plt.ylim(0, 10)  # Setting y-axis limits
#        #plt.show()

# Example usage:
# Assuming los_matrix is a 2D numpy array representing LOS data
# los_matrix.shape = (num_satellites, num_time_steps)
# You would instantiate YourClass with this array
# your_object = YourClass(los_matrix)
# your_object.visualize_los_matrix()



#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------LATENCY CALCULATION-------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------
        range_index = 0
        ranges = geometrical_output['ranges']
        self.propagation_latency = np.zeros((num_satellites, len(time)))
        for sat_index in range(len(ranges)):
                for range_index in range(len(time)):
                    if self.los_matrix[sat_index, range_index] ==1:
                        latency_propagation = ranges[sat_index][range_index] / speed_of_light
                        self.propagation_latency[sat_index,range_index] = latency_propagation
                    else:
                        latency_propagation = 0 
                        self.propagation_latency[sat_index,range_index] = latency_propagation

        #print("latency")
        #print(self.propagation_latency)
        # Assuming self.propagation_latency is already calculated

        # Get the number of satellites
#        num_satellites = self.propagation_latency.shape[0]
#
#        # Create a scatter plot for each satellite
#        for sat_index in range(num_satellites):
#            # Extract propagation latency for the current satellite
#            latency_satellite = self.propagation_latency[sat_index]
#
#            # Generate indices corresponding to each time step
#            indices = range(len(latency_satellite))
#
#            # Create scatter plot
#            sizes = 20
#            plt.scatter(indices, latency_satellite, label=f'Satellite {sat_index + 1}')
#
#        # Add labels and title
#        plt.xlabel('Time Index')
#        plt.ylabel('Propagation Latency (seconds)')
#        plt.title('Propagation Latency for Each Satellite')
#        plt.legend()
#
#        # Show plot
#        #plt.show()
#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------Matrix creation-------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------Capacity Calculation-------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------




#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------BER Calculation-------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------BER Calculation-------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------



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
        #print('Optimization of max. link time and max. elevation')
        #print('Number of links             : ' + str(self.number_of_links))
        #print('Average link time           : ' + str(np.round(self.comm_time/self.number_of_links/60, 3))+' min')
        #print('Total acquisition time      : ' + str(self.total_acquisition_time/60)+' min')
        #print('Fraction of total link time : ' + str(self.frac_comm_time))
        print('------------------------------------------------')

        

        return self.routing_output, self.routing_total_output, self.los_matrix  
  