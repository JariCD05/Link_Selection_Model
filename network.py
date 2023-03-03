
from input import *
from helper_functions import *

import numpy as np

class network():
    def __init__(self, time):
        self.number_of_handovers = 0
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    def hand_over_strategy(self, number_of_planes, number_sats_per_plane, geometrical_output, time):
        index = 0
        self.geometrical_output = {}

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

        while index < len(time):
            # The handover time is initiated with t=0, then the handover procedure starts
            t_handover = 0
            start_elev = []
            start_sat = []
            # Find a new satellite to start new connection
            while len(start_elev) == 0 and index < len(time):
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

                # print('------------------------------------------------')
                # print('HAND OVER: ', self.number_of_handovers)
                # print('------------------------------------------------')
                # print('Chosen satellite to in LOS: plane = ', current_plane, ', sat in plane = ', current_sat,
                #       ', elevation = ', current_elevation)
                # print('Handover time: ', t_handover / 60, 'minutes, Current time: ', time[index])

            else:
                print('------------------------------------------------')
                print('HAND OVER')
                print('------------------------------------------------')
                print('No satellites found, or propagation time has exceeded, t = ', time[index-1], ', t_end = ', time[-1])


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
            if index > len(time):
                break
            while current_elevation > elevation_min and index < len(time):
                current_elevation = elevation_angles[current_plane][current_sat][index]
                index += 1

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

        self.geometrical_output['pos SC'] = pos_SC
        self.geometrical_output['heights SC'] = heights_SC
        self.geometrical_output['vel SC'] = vel_SC
        self.geometrical_output['ranges'] = ranges
        self.geometrical_output['slew rates'] = slew_rates
        self.geometrical_output['angular rates'] = angular_rates
        self.geometrical_output['zenith'] = zenith
        self.geometrical_output['elevation'] = elevation
        self.geometrical_output['azimuth'] = azimuth
        self.geometrical_output['radial'] = radial
        return self.geometrical_output


    def save_data(self, data_dim2):

        data_dim1 = get_data('elevation')
        a_s = (data_dim1[-1] - data_dim1[1]) / (len(data_dim1) - 1)
        a_0 = data_dim1[1]
        mapping_lb = ((np.rad2deg(data_dim2) - a_0) / a_s).astype(int)
        mapping_ub = mapping_lb + 1

        for i in range(len(mapping_lb)):
            if mapping_lb[i] < 0.0:
                mapping_lb[i] = 0
                mapping_ub[i] = 0

        mapping_lb = list(mapping_lb)
        mapping_ub = list(mapping_ub)

        self.performance_output = get_data('all', mapping_lb, mapping_ub)

        # print('TEST')
        # print(data_dim1[-1], data_dim1[0])
        # print(a_0, a_s)
        # for i in range(len(data_dim2)):
        #     print(np.rad2deg(data_dim2[i]), self.performance_output[i, 0])
        return self.performance_output

    def variable_data_rate(self):
        # Pr threshold
        P_r = self.performance_output[:, 1]
        Np_r = self.performance_output[:, -2]
        N_p_thres = self.performance_output[:, -4]
        SNR_thres = self.performance_output[:, -5]
        P_r_thres = self.performance_output[:, -6]

        data_rate_thres = data_rate_func(P_r, N_p_thres, eff_quantum_sc)

        # Convert data rate output from constant to variable
        self.performance_output[:,-3] = data_rate_thres
        self.performance_output[:, 1] = P_r_thres
        self.performance_output[:,-2] = N_p_thres
        self.performance_output[:, 8] = SNR_thres
        self.performance_output[:, 9] = BER_thres
        return self.performance_output