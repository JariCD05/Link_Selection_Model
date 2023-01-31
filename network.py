
import Link_budget as LB
import Constellation as SC
import Aircraft as AC
import Turbulence as turb
from constants import *
from helper_functions import *

import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants as cons_tudat


class network():
    def __init__(self, time):
        self.number_of_handovers = 0
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

        self.h_stat = np.zeros(len(time))
        self.Pr_data = np.zeros(len(time))
        self.Ir_data = np.zeros(len(time))
        self.BER_data = np.zeros(len(time))
        self.latency = 0.0

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    def hand_over_strategy(self, number_of_planes, number_sats_per_plane, geometrical_output, time):
        index = 0
        self.handover_output = {}

        pos_SC = np.zeros((len(time), 3))
        vel_SC = np.zeros((len(time), 3))
        heights_SC = np.zeros(len(time))
        ranges = np.zeros(len(time))
        slew_rates = np.zeros(len(time))
        elevation = np.zeros(len(time))
        zenith = np.zeros(len(time))
        azimuth = np.zeros(len(time))
        radial = np.zeros(len(time))
        angular_rates = np.zeros(len(time))


        elevation_angles = geometrical_output['elevation']

        while time[index] < end_time:
            # The handover time is initiated with t=0, then the handover procedure starts
            t_handover = 0
            start_elev = []
            start_sat = []
            # Find a new satellite to start new connection
            while len(start_elev) == 0 and time[index] < end_time:
                for plane in range(number_of_planes):
                    for sat in range(number_sats_per_plane):

                        elev_last = elevation_angles[plane][sat][index - 1]
                        elev = elevation_angles[plane][sat][index]

                        # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), AN ELEVATION THAT IS STILL INCREASING (START OF WINDOW) AND T < T_FINAL
                        if elev > elevation_min and elev > elev_last:
                            start_elev.append(elev)
                            start_sat.append([plane, sat])

                self.total_handover_time += step_size_dim2
                t_handover += step_size_dim2
                index += 1

            if len(start_sat) > 0:
                # CHOOSE THE SATELLITE WITH THE SMALLEST ELEVATION ANGLE
                current_plane = start_sat[np.argmin(start_elev)][0]
                current_sat = start_sat[np.argmin(start_elev)][1]
                current_elevation = start_elev[np.argmin(start_elev)]

                print('------------------------------------------------')
                print('HAND OVER: ', self.number_of_handovers)
                print('------------------------------------------------')
                print('Chosen satellite to in LOS: plane = ', current_plane, ', sat in plane = ', current_sat,
                      ', elevation = ', current_elevation)
                print('Handover time: ', t_handover / 60, 'minutes, Current time: ', time[index])

            else:
                print('------------------------------------------------')
                print('HAND OVER')
                print('------------------------------------------------')
                print('No satellites found, or propagation time has exceeded, t = ', time[index], ', t_end = ', time[-1])

            self.number_of_handovers += 1
            # ------------------------------------------------------------------------
            # ------------------------------ACQUISITION-------------------------------
            # ------------------------------------------------------------------------
            # Perform acquisition and add the acquisition time for computation of total latency
            acquisition_time, acquisition_indices = acquisition()
            self.acquisition[index: index + acquisition_indices] = acquisition_time / acquisition_indices
            self.total_acquisition_time += acquisition_time

            # ------------------------------------------------------------------------
            # -------LOOP-THROUGH-DIFFERENT-SATELLITES-&-ADD-DATA---------------------
            # ------------------------------------------------------------------------
            index_start_window = index
            while current_elevation > elevation_min and time[index] < end_time:
                index += 1
                current_elevation = elevation_angles[current_plane][current_sat][index]

            pos_SC[index_start_window:index] = geometrical_output['pos SC'][current_plane][current_sat][index_start_window:index]
            heights_SC[index_start_window:index] = geometrical_output['heights SC'][current_plane][current_sat][index_start_window:index]
            vel_SC[index_start_window:index] = geometrical_output['vel SC'][current_plane][current_sat][index_start_window:index]
            ranges[index_start_window:index] = geometrical_output['ranges'][current_plane][current_sat][index_start_window:index]
            slew_rates[index_start_window:index] = geometrical_output['slew rates'][current_plane][current_sat][index_start_window:index]
            angular_rates[index_start_window:index] = geometrical_output['angular rates'][current_plane][current_sat][index_start_window:index]
            elevation[index_start_window:index] = geometrical_output['elevation'][current_plane][current_sat][index_start_window:index]
            zenith[index_start_window:index] = geometrical_output['zenith'][current_plane][current_sat][index_start_window:index]
            azimuth[index_start_window:index] = geometrical_output['azimuth'][current_plane][current_sat][index_start_window:index]
            radial[index_start_window:index] = geometrical_output['radial'][current_plane][current_sat][index_start_window:index]

        # print(elevation[index_start_window:index])
        # print(1/0)
        self.handover_output['pos SC'] = pos_SC
        self.handover_output['heights SC'] = heights_SC
        self.handover_output['vel SC'] = vel_SC
        self.handover_output['ranges'] = ranges
        self.handover_output['slew rates'] = slew_rates
        self.handover_output['angular rates'] = angular_rates
        self.handover_output['zenith'] = zenith
        self.handover_output['elevation'] = elevation
        self.handover_output['azimuth'] = azimuth
        self.handover_output['radial'] = radial
        print(elevation[200:300])
        return self.handover_output


    def save_data(self, data_dim1, data_dim2):

        data_elevation = get_data('elevation')

        a_s = (data_dim1[-1] - data_dim1[0]) / len(data_dim1)
        a_0 = data_dim1[1]
        mapping = ((np.rad2deg(data_dim2) - a_0) / a_s).astype(int) + 1
        for i in range(len(mapping)):
            if mapping[i] < 0.0:
                mapping[i] = 0
        mapping = list(mapping)

        data_metrics = ['elevation', 'P_r_0', 'h_scint', 'h_pe', 'h_bw', 'SNR', 'BER', 'number of fades', 'fade time',
                        'fractional fade time']
        self.output = get_data('all', mapping)

    #------------------------------------------------------------------------
    #------------------------PRINT-DATA-&-VISUALIZATION----------------------
    #------------------------------------------------------------------------


    def plot(self, type="trajectories"):
        number_of_planes = self.SC.number_of_planes
        number_of_sats = self.number_sats_per_plane

        if type == "trajectories":
            # Define a 3D figure using pyplot
            fig = plt.figure(figsize=(6,6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

            # Plot fist satellite with label
            # ax.plot(self.states_cons[0][0][:, 1],
            #         self.states_cons[0][0][:, 2],
            #         self.states_cons[0][0][:, 3],
            #             linestyle='-.', linewidth=0.5, color='orange', label='constellation')
            # Plot first satellite filtered data with label
            ax.scatter(self.pos_SC_filtered[0][0][:, 0],
                       self.pos_SC_filtered[0][0][:, 1],
                       self.pos_SC_filtered[0][0][:, 2],
                    linestyle='-.', linewidth=0.5, color='g', label='window \epsilon > '+str(np.rad2deg(elevation_min)))

            # Plot all other satellites in constellation
            for plane in range(1, number_of_planes):
                # ax.plot(self.states_cons[plane][0][:, 1],
                #         self.states_cons[plane][0][:, 2],
                #         self.states_cons[plane][0][:, 3],
                #         linestyle='-.', linewidth=0.5, color='orange')
                ax.scatter(self.pos_SC_filtered[plane][0][:, 0],
                           self.pos_SC_filtered[plane][0][:, 1],
                           self.pos_SC_filtered[plane][0][:, 2],
                        linestyle='-.', linewidth=0.5)

            # Plot aircraft
            ax.plot(self.pos_AC[:, 0],
                    self.pos_AC[:, 1],
                    self.pos_AC[:, 2], color='black', label='aircraft')
            # Draw Earth as a blue dot in origin
            ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue', s=100)

            # # Add the legend and labels, then show the plot
            ax.legend()
            ax.set_xlabel('x [m]', fontsize=15)
            ax.set_ylabel('y [m]', fontsize=15)
            ax.set_zlabel('z [m]', fontsize=15)

        # Plot elevation angles
        elif type == "angles":
            fig_elev, axs_elev = plt.subplots(2, 1, figsize=(6, 6), dpi=125)

            for plane in range(number_of_planes):
                    for sat in range(number_of_sats):
                        axs_elev[0].set_title(f'Elevation angles vs time')
                        # axs_elev[0].plot(self.time_filtered[plane][sat], np.rad2deg(self.elevation_filtered[plane][sat]))
                        # axs_elev[1].plot(self.time_filtered[plane][sat], np.rad2deg(self.azimuth_filtered[plane][sat]))
                        # axs_elev[2].plot(self.time_filtered[plane][sat], np.rad2deg(self.radial_filtered[plane][sat]))
                        # axs_elev[1].plot(self.time, np.rad2deg(self.ranges[plane][sat]))

                        axs_elev[0].plot(self.time, np.ones(len(self.time))*elevation_min)
                        axs_elev[0].plot(self.time, np.rad2deg(self.elevation[plane][sat]))
                        axs_elev[1].plot(self.time, self.ranges[plane][sat])

                        axs_elev[0].set_ylabel('elevation (degrees)')
                        axs_elev[0].set_xlabel('time (seconds)')


    def print(self, plane, sat, index):

        print('------------------------------------------------')
        print('LINK GEOMETRY')
        print('------------------------------------------------')
        print('Range [km]         : ', self.ranges[plane][sat][index])
        print('Zenith angle [deg] : ', np.rad2deg(self.zenith[plane][sat][index]))
        print('Elev. angle  [deg] : ', np.rad2deg(self.elevation[plane][sat][index]))
        print('Slew rate [deg/s]  : ', self.slew_rates[plane][sat][index])