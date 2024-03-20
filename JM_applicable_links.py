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
        self.sats_visibility = [[0 for _ in range(len(time))] for _ in range(len(elevation_angles))]
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
                    self.sats_applicable[s][index] = 0

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
                    satellite_data['pos AC'].append(0)
                    satellite_data['lon AC'].append(0)
                    satellite_data['lat AC'].append(0)
                    satellite_data['heights AC'].append(0)
                    satellite_data['speeds AC'].append(0)
                    satellite_data['pos SC'].append(0)
                    satellite_data['lon SC'].append(0)
                    satellite_data['lat SC'].append(0)
                    satellite_data['vel SC'].append(0)
                    satellite_data['heights SC'].append(0)
                    satellite_data['ranges'].append(0)
                    satellite_data['elevation'].append(0)
                    satellite_data['azimuth'].append(0)
                    satellite_data['zenith'].append(0)
                    satellite_data['radial'].append(0)
                    satellite_data['slew rates'].append(0)
                    satellite_data['elevation rates'].append(0)
                    satellite_data['azimuth rates'].append(0)
                    satellite_data['doppler shift'].append(0)

            for key in self.applicable_output:
                self.applicable_output[key].append(satellite_data[key])

        return self.applicable_output, self.sats_visibility, self.sats_applicable

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

    
    
    def plot_satellite_visibility_scatter(self, time):
        # Convert sats_applicable to a NumPy array for easier manipulation
        sats_applicable_np = np.array(self.sats_applicable)
        
        # Calculate the accumulated visibility (sum along the satellite axis)
        accumulated_visibility = np.sum(sats_applicable_np, axis=0)
        
        # Prepare the figure and axes for plotting
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot for visibility of each satellite over time using scatter plot
        axes[0].set_title("Satellite Visibility Over Time")
        for s in range(sats_applicable_np.shape[0]):
            # Find indices (time points) where satellite s is visible
            visible_indices = np.where(sats_applicable_np[s, :] == 1)[0]
            # Plot these as scatter points
            axes[0].scatter(visible_indices, np.full(visible_indices.shape, s), alpha=0.6, label=f'Sat {s+1}')
        axes[0].set_ylabel("Satellite Index")
        axes[0].legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))
        
        # Plot for accumulated visible satellites over time
        axes[1].set_title("Accumulated Satellites Visible Over Time")
        axes[1].plot(time, accumulated_visibility, color='red', label='Accumulated Visibility')
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("Number of Visible Satellites")
        axes[1].legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()




print('')
print('------------------END-TO-END-LASER-SATCOM-MODEL-------------------------')
#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

print('----------------------------------------------------------------------------------MACRO-LEVEL-----------------------------------------------------------------------------------------')
print('')
print('-----------------------------------MISSION-LEVEL-----------------------------------------')
#------------------------------------------------------------------------
#------------------------------------LCT---------------------------------
#------------------------------------------------------------------------
# Compute the sensitivity and compute the threshold
LCT = terminal_properties()
LCT.BER_to_P_r(BER = BER_thres,
               modulation = modulation,
               detection = detection,
               threshold = True)
PPB_thres = PPB_func(LCT.P_r_thres, data_rate)

#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------
# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
# First both AIRCRAFT and SATELLITES are propagated with 'link_geometry.propagate'
# Then, the relative geometrical state is computed with 'link_geometry.geometrical_outputs'
# Here, all links are generated between the AIRCRAFT and each SATELLITE in the constellation
link_geometry = link_geometry()
link_geometry.propagate(time=t_macro, step_size_AC=step_size_AC, step_size_SC=step_size_SC,
                        aircraft_filename=aircraft_filename_load, step_size_analysis=False, verification_cons=False)
link_geometry.geometrical_outputs()
# Initiate time vector at mission level. This is the same as the propagated AIRCRAFT time vector
time = link_geometry.time
mission_duration = time[-1] - time[0]
# Update the samples/steps at mission level
samples_mission_level = number_sats_per_plane * number_of_planes * len(link_geometry.geometrical_output['elevation'])

Links_applicable = applicable_links(time=time)
applicable_output, sats_visibility,sats_applicable = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)

#Links_applicable.visualize_satellite_visibility(sats_applicable = Links_applicable.sats_applicable, time=time, num_satellites=num_satellites)
#print(applicable_output['time'])
#print(len(applicable_output['pos SC']))
#print(len(time))

#Links_applicable.plot_satellite_visibility_scatter(time=time)
#Links_applicable.plot_satellite_visibility(time = time)


#print("Length of visible sattelite array", len(sats_visibility[2]))
#print("Length of applicable sattelite array", len(sats_applicable[]))
#print("time array lenght", len(time))
#print(len(applicable_output["pos SC"][3]))