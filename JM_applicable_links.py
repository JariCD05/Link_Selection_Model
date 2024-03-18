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

        index = 0
        elevation_angles = geometrical_output['elevation']
        self.sats_visibility = [[] for _ in time] 
        self.sats_applicable = [[] for _ in time] 
        for index in range(len(time)):
            start_elev = []
            sats_in_LOS = []
        
            # Loop through all satellites in constellation
            for i in range(len(geometrical_output['pos SC'])):
                # Avoid index out of range for the last element
                elev = elevation_angles[i][index]
        
                if elev > elevation_min:
                    start_elev.append(elev)
                    sats_in_LOS.append(i)
                    self.sats_visibility[index].append(i)

                if elev > elevation_min:
                    self.sats_applicable[index].append(1)
                else:
                    self.sats_applicable[index].append(0)

            for s in sats_in_LOS:
                self.number_of_links += 1
                link_start_index = index
                while index < len(time) - 1 and elevation_angles[s][min(index + 1, len(time) - 1)] > elevation_min:
                    index += 1
                link_end_index = index + 1  # Correctly include the current satellite's last visible index

                self.applicable_output['link number'].append(self.number_of_links)
                self.applicable_output['time'].append(np.array(time[link_start_index:link_end_index]))
                self.applicable_output['pos AC'].append(geometrical_output['pos AC'][link_start_index:link_end_index])
                self.applicable_output['lon AC'].append(geometrical_output['lon AC'][link_start_index:link_end_index])
                self.applicable_output['lat AC'].append(geometrical_output['lat AC'][link_start_index:link_end_index])
                self.applicable_output['heights AC'].append(geometrical_output['heights AC'][link_start_index:link_end_index])
                self.applicable_output['speeds AC'].append(geometrical_output['speeds AC'][link_start_index:link_end_index])
                self.applicable_output['pos SC'].append(geometrical_output['pos SC'][s][link_start_index:link_end_index])
                self.applicable_output['lon SC'].append(geometrical_output['lon SC'][s][link_start_index:link_end_index])
                self.applicable_output['lat SC'].append(geometrical_output['lat SC'][s][link_start_index:link_end_index])
                self.applicable_output['vel SC'].append(geometrical_output['vel SC'][s][link_start_index:link_end_index])
                self.applicable_output['heights SC'].append(geometrical_output['heights SC'][s][link_start_index:link_end_index])
                self.applicable_output['ranges'].append(geometrical_output['ranges'][s][link_start_index:link_end_index])
                self.applicable_output['elevation'].append(geometrical_output['elevation'][s][link_start_index:link_end_index])
                self.applicable_output['azimuth'].append(geometrical_output['azimuth'][s][link_start_index:link_end_index])
                self.applicable_output['zenith'].append(geometrical_output['zenith'][s][link_start_index:link_end_index])
                self.applicable_output['radial'].append(geometrical_output['radial'][s][link_start_index:link_end_index])
                self.applicable_output['slew rates'].append(geometrical_output['slew rates'][s][link_start_index:link_end_index])
                self.applicable_output['elevation rates'].append(geometrical_output['elevation rates'][s][link_start_index:link_end_index])
                self.applicable_output['azimuth rates'].append(geometrical_output['azimuth rates'][s][link_start_index:link_end_index])
                self.applicable_output['doppler shift'].append(geometrical_output['doppler shift'][s][link_start_index:link_end_index])

            if not sats_in_LOS or index == len(time)-1:
                index += 1  # Ensure we always move forward

        self.flatten_applicable_output()
        
        print('Propagation model')
        print('------------------------------------------------')
        print('Propagation of all applicable links')
        print('Number of available links             : ' + str(self.number_of_links))
        #print(self.sats_visibility)




        return self.applicable_output, self.applicable_total_output, self.sats_visibility

    def visualize_satellite_visibility(self, sats_visibility, time, number_sats_per_plane, number_of_planes):
        # Plot 1: Individual satellite visibility
        plt.figure(figsize=(14, 8))
        plt.title('Satellite Visibility Over Time')
        for index, time_point in enumerate(time):
            visible_sats = sats_visibility[index]
            for sat in visible_sats:
                plt.scatter([time_point] * len(visible_sats), [sat + 1 for _ in visible_sats], color='blue', alpha=0.6)
        plt.ylabel('Satellite Number')
        plt.xlabel('Time Index')
        plt.grid(True)
        plt.yticks(range(0, number_sats_per_plane * number_of_planes + 1))
        plt.show()
       

        # Plot 2: Accumulated number of visible satellites over time
        plt.figure(figsize=(14, 8))
        plt.title('Number of Visible Satellites Over Time')
        num_visible_sats = [len(vis) for vis in sats_visibility]  # Number of visible satellites at each time index
        plt.plot(time, num_visible_sats, marker='o', linestyle='-', color='red')
        plt.ylabel('Number of Visible Satellites')
        plt.xlabel('Time Index')
        plt.grid(True)
        plt.show()



    def flatten_applicable_output(self):
        self.applicable_total_output['time'] = flatten(self.applicable_output['time'])
        self.applicable_total_output['pos AC'] = flatten(self.applicable_output['pos AC'])
        self.applicable_total_output['lon AC'] = flatten(self.applicable_output['lon AC'])
        self.applicable_total_output['lat AC'] = flatten(self.applicable_output['lat AC'])
        self.applicable_total_output['heights AC'] = flatten(self.applicable_output['heights AC'])
        self.applicable_total_output['speeds AC'] = flatten(self.applicable_output['speeds AC'])
    
        self.applicable_total_output['pos SC'] = flatten(self.applicable_output['pos SC'])
        self.applicable_total_output['lon SC'] = flatten(self.applicable_output['lon SC'])
        self.applicable_total_output['lat SC'] = flatten(self.applicable_output['lat SC'])
        self.applicable_total_output['vel SC'] = flatten(self.applicable_output['vel SC'])
        self.applicable_total_output['heights SC'] = flatten(self.applicable_output['heights SC'])
        self.applicable_total_output['ranges'] = flatten(self.applicable_output['ranges'])
        self.applicable_total_output['elevation'] = flatten(self.applicable_output['elevation'])
        self.applicable_total_output['azimuth'] = flatten(self.applicable_output['azimuth'])
        self.applicable_total_output['zenith'] = flatten(self.applicable_output['zenith'])
        self.applicable_total_output['radial'] = flatten(self.applicable_output['radial'])
        self.applicable_total_output['slew rates'] = flatten(self.applicable_output['slew rates'])
        self.applicable_total_output['elevation rates'] = flatten(self.applicable_output['elevation rates'])
        self.applicable_total_output['azimuth rates'] = flatten(self.applicable_output['azimuth rates'])
        self.applicable_total_output['doppler shift'] = flatten(self.applicable_output['doppler shift'])

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
applicable_output, applicable_total_output, sats_visibility, sats_applicable = Links_applicable.applicability(link_geometry.geometrical_output, time, step_size_link)

Links_applicable.visualize_satellite_visibility(sats_visibility= Links_applicable.sats_visibility, time=time, number_sats_per_plane=number_sats_per_plane, number_of_planes=number_of_planes)
print(sats_applicable)