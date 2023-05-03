
from input import *
from helper_functions import *

from itertools import chain
import numpy as np

class routing_network():
    def __init__(self, time):
        self.links = np.empty(len(time))
        self.number_of_links = 0
        self.total_handover_time = 0 # seconds
        self.total_acquisition_time = 0 # seconds
        self.acquisition = np.zeros(len(time))

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    # def routing(self, number_of_planes, number_sats_per_plane, geometrical_output, time):
    #     index = 0
    #     self.geometrical_output = {}
    #
    #     pos_SC = np.zeros((len(time), 3))
    #     vel_SC = np.zeros((len(time), 3))
    #     heights_SC = np.zeros(len(time))
    #     ranges = np.zeros(len(time))
    #     slew_rates = np.zeros(len(time))
    #     elevation = np.zeros(len(time))
    #     zenith = np.zeros(len(time))
    #     azimuth = np.zeros(len(time))
    #     radial = np.zeros(len(time))
    #
    #     elevation_angles = geometrical_output['elevation']
    #
    #     while index < len(time):
    #         # The handover time is initiated with t=0, then the handover procedure starts
    #         t_handover = 0
    #         start_elev = []
    #         start_sat = []
    #         # Find a new satellite to start new connection
    #         while len(start_elev) == 0 and index < len(time):
    #             for plane in range(number_of_planes):
    #                 for sat in range(number_sats_per_plane):
    #
    #                     elev_last = elevation_angles[plane][sat][index - 1]
    #                     elev = elevation_angles[plane][sat][index]
    #
    #                     # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), AN ELEVATION THAT IS STILL INCREASING (START OF WINDOW) AND T < T_FINAL
    #                     if elev > elevation_min and elev > elev_last:
    #                         start_elev.append(elev)
    #                         start_sat.append([plane, sat])
    #
    #             self.total_handover_time += step_size_link
    #             t_handover += step_size_link
    #             index += 1
    #
    #         if len(start_sat) > 0:
    #             # CHOOSE THE SATELLITE WITH THE SMALLEST ELEVATION ANGLE
    #             current_plane = start_sat[np.argmin(start_elev)][0]
    #             current_sat = start_sat[np.argmin(start_elev)][1]
    #             current_elevation = start_elev[np.argmin(start_elev)]
    #
    #             # print('------------------------------------------------')
    #             # print('HAND OVER: ', self.number_of_handovers)
    #             # print('------------------------------------------------')
    #             # print('Chosen satellite to in LOS: plane = ', current_plane, ', sat in plane = ', current_sat,
    #             #       ', elevation = ', current_elevation)
    #             # print('Handover time: ', t_handover / 60, 'minutes, Current time: ', time[index])
    #
    #         else:
    #             print('------------------------------------------------')
    #             print('HAND OVER')
    #             print('------------------------------------------------')
    #             print('No satellites found, or propagation time has exceeded, t = ', time[index - 1], ', t_end = ',
    #                   time[-1])
    #
    #         self.number_of_handovers += 1
    #         # ------------------------------------------------------------------------
    #         # ------------------------------ACQUISITION-------------------------------
    #         # ------------------------------------------------------------------------
    #         # Perform acquisition and add the acquisition time for computation of total latency
    #         acquisition_time, acquisition_indices = acquisition()
    #         self.acquisition[index: index + acquisition_indices] = acquisition_time / acquisition_indices
    #         self.total_acquisition_time += acquisition_time
    #
    #         # ------------------------------------------------------------------------
    #         # -------LOOP-THROUGH-DIFFERENT-SATELLITES-&-ADD-DATA---------------------
    #         # ------------------------------------------------------------------------
    #         index_start_window = index
    #         if index > len(time):
    #             break
    #         while current_elevation > elevation_min and index < len(time):
    #             current_elevation = elevation_angles[current_plane][current_sat][index]
    #             index += 1
    #
    #         pos_SC[index_start_window:index] = geometrical_output['pos SC'][current_plane][current_sat][
    #                                            index_start_window:index]
    #         heights_SC[index_start_window:index] = geometrical_output['heights SC'][current_plane][current_sat][
    #                                                index_start_window:index]
    #         vel_SC[index_start_window:index] = geometrical_output['vel SC'][current_plane][current_sat][
    #                                            index_start_window:index]
    #         ranges[index_start_window:index] = geometrical_output['ranges'][current_plane][current_sat][
    #                                            index_start_window:index]
    #         slew_rates[index_start_window:index] = geometrical_output['slew rates'][current_plane][current_sat][
    #                                                index_start_window:index]
    #         elevation[index_start_window:index] = geometrical_output['elevation'][current_plane][current_sat][
    #                                               index_start_window:index]
    #         zenith[index_start_window:index] = geometrical_output['zenith'][current_plane][current_sat][
    #                                            index_start_window:index]
    #         azimuth[index_start_window:index] = geometrical_output['azimuth'][current_plane][current_sat][
    #                                             index_start_window:index]
    #         radial[index_start_window:index] = geometrical_output['radial'][current_plane][current_sat][
    #                                            index_start_window:index]
    #
    #     self.geometrical_output['pos SC'] = pos_SC
    #     self.geometrical_output['heights SC'] = heights_SC
    #     self.geometrical_output['vel SC'] = vel_SC
    #     self.geometrical_output['ranges'] = ranges
    #     self.geometrical_output['slew rates'] = slew_rates
    #     self.geometrical_output['zenith'] = zenith
    #     self.geometrical_output['elevation'] = elevation
    #     self.geometrical_output['azimuth'] = azimuth
    #     self.geometrical_output['radial'] = radial
    #     return self.geometrical_output

    def routing(self, geometrical_output, time, optimization='no'):
        # This method computes the routing sequence of the links between AIRCRAFT and SATELLITES in constellation.
        # This model uses only geometric data dictionary as INPUT with 10 KEYS (pos_SC, vel_SC, h_SC, range, slew_rate, elevation, zenith, azimuth, radial, doppler shift)
        # Each KEY has a value with shape (NUMBER_OF_PLANES, NUMBER_SATS_PER_PLANE, LEN(TIME)) (for example: 10x10x3600 for 10 planes, 10 sats and 1 hour with 1s time steps)

        # OUTPUT of the model must be same geometric data dictionary as INPUT, with KEYs of shape LEN(TIME). Meaning that there is only one vector for all KEYS.

        # This option is the DEFAULT routing model. Here, link is available above a minimum elevation angle.
        # When the current link goes beyond this minimum, the next link is searched. This will be the link with the lowest (and rising) elevation angle.
        if optimization == 'no':
            index = 0
            time_new = []
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
            delta_v = np.zeros(len(time))

            elevation_angles = geometrical_output['elevation']

            while index < len(time):
                # The handover time is initiated with t=0, then the handover procedure starts
                t_handover = 0
                start_elev = []
                start_sat = []
                # Find a new satellite to start new connection
                while len(start_elev) == 0 and index < len(time):
                    for i in range(len(geometrical_output['pos SC'])):
                        elev_last = elevation_angles[i][index - 1]
                        elev = elevation_angles[i][index]

                        # FIND SATELLITES WITH AN ELEVATION ABOVE THRESHOLD (DIRECT LOS), AN ELEVATION THAT IS STILL INCREASING (START OF WINDOW) AND T < T_FINAL
                        if elev > elevation_min and elev > elev_last:
                            start_elev.append(elev)
                            start_sat.append(i)


                    self.total_handover_time += step_size_link
                    t_handover += step_size_link
                    index += 1

                if len(start_sat) > 0:
                    # print('start elev', start_sat, start_elev)
                    # CHOOSE THE SATELLITE WITH THE SMALLEST ELEVATION ANGLE
                    current_sat = start_sat[np.argmin(start_elev)]
                    current_elevation = start_elev[np.argmin(start_elev)]
                    self.number_of_links += 1

                else:
                    break
                # ------------------------------------------------------------------------
                # ------------------------------ACQUISITION-------------------------------
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
                    # print('same satellite', start_sat, current_elevation)
                    current_elevation = elevation_angles[current_sat][index]
                    index += 1

                time_new.append(list(time[index_start_window:index]))
                pos_SC[index_start_window:index] = geometrical_output['pos SC'][current_sat][index_start_window:index]
                heights_SC[index_start_window:index] = geometrical_output['heights SC'][current_sat][index_start_window:index]
                vel_SC[index_start_window:index] = geometrical_output['vel SC'][current_sat][index_start_window:index]
                ranges[index_start_window:index] = geometrical_output['ranges'][current_sat][index_start_window:index]
                slew_rates[index_start_window:index] = geometrical_output['slew rates'][current_sat][index_start_window:index]
                elevation[index_start_window:index] = geometrical_output['elevation'][current_sat][index_start_window:index]
                zenith[index_start_window:index] = geometrical_output['zenith'][current_sat][index_start_window:index]
                azimuth[index_start_window:index] = geometrical_output['azimuth'][current_sat][index_start_window:index]
                radial[index_start_window:index] = geometrical_output['radial'][current_sat][index_start_window:index]
                self.links[index_start_window:index] = self.number_of_links

        mask = heights_SC > 0
        self.geometrical_output['time'] = time[mask]
        self.geometrical_output['link number'] = self.links[mask]
        self.geometrical_output['pos AC'] = geometrical_output['pos AC'][mask]
        self.geometrical_output['pos SC'] = pos_SC[mask]
        self.geometrical_output['heights SC'] = heights_SC[mask]
        self.geometrical_output['vel SC'] = vel_SC[mask]
        self.geometrical_output['ranges'] = ranges[mask]
        self.geometrical_output['slew rates'] = slew_rates[mask]
        self.geometrical_output['zenith'] = zenith[mask]
        self.geometrical_output['elevation'] = elevation[mask]
        self.geometrical_output['azimuth'] = azimuth[mask]
        self.geometrical_output['radial'] = radial[mask]
        self.geometrical_output['doppler shift'] = delta_v[mask]

        print('------------------------------------------------')
        print('ROUTING MODEL')
        print('Number of satellites: ' + str(i+1))
        print('Number of links: '+ str(self.number_of_links))
        print('Total acquisition time: ' + str(self.total_acquisition_time))
        print('------------------------------------------------')
        return self.geometrical_output, mask


    def performance_output_database(self, data_dim2):
        data_dim1 = load_from_database('elevation')
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

        self.performance_output = load_from_database('all', mapping_lb, mapping_ub)
        return self.performance_output

    def performance_output(self, margin, errors, errors_coded):
        self.performance_output = dict()
        self.performance_output['margin1'] = margin[0]
        self.performance_output['margin2'] = margin[1]
        self.performance_output['margin3'] = margin[2]
        self.performance_output['errors'] = errors
        self.performance_output['errors_coded'] = errors_coded
        return self.performance_output

    def variable_data_rate(self,
                           P_r_threshold: np.array,
                           PPB_threshold: np.array):
        # Pr threshold
        P_r = self.performance_output['P_r']
        # Variable data rate
        data_rate_var_BER9 = data_rate_func(P_r, PPB_threshold[0])
        data_rate_var_BER6 = data_rate_func(P_r, PPB_threshold[1])
        data_rate_var_BER3 = data_rate_func(P_r, PPB_threshold[2])
        data_rate_var = [data_rate_var_BER9, data_rate_var_BER6, data_rate_var_BER3]

        # Convert data rate output from constant to variable
        # self.performance_output[:,-3] = data_rate_var
        # self.performance_output['P_r'] = np.ones(np.shape(P_r)) * P_r_threshold
        # self.performance_output['PPB'] = np.ones(np.shape(P_r)) * PPB_threshold
        return data_rate_var