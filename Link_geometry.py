
import Link_budget as LB
import Constellation as SC
import Aircraft as AC
from input import *

import numpy as np
from matplotlib import pyplot as plt
from tudatpy.kernel import constants as cons_tudat


# def geometric_angles(range_array):
#     beta = np.zeros(len(range_array))
#     angle_zen  = np.zeros(len(range_array))
#     angle_elev = np.zeros(len(range_array))
#     angle_azi  = np.zeros(len(range_array))
#
#     for i in range(len(range_array)):
#         angle_zen[i] = np.arccos( (2*cons.R_earth*heights_SC[i] - heights_SC[i]**2 - range_array[i]**2)/ (2*range_array[i]*cons.R_earth))
#         beta[i] = np.arccos(np.cos(lon_AC[i]) * np.cos(lat_AC[i]))
#         # d = np.sqrt(AC.R_earth**2 + (R_earth + SC_heights[i])**2 - 2 * AC.R_earth * (R_earth + SC_heights[i]) * np.cos(beta[i]))
#         # angle_elev[i] = np.rad2deg(np.arccos((R_earth + SC_heights[i]) * np.sin(beta[i]) / d))
#         angle_azi[i] = np.rad2deg(np.arccos(np.tan(lon_AC[i]) * np.cos(beta[i]) * np.sin(beta[i])))
#
#         angle_elev[i] = np.pi - angle_zen[i]
#
#     return np.rad2deg(angle_elev), np.rad2deg(angle_azi), np.rad2deg(angle_zen)



class link_geometry:
    def __init__(self, constellation_type = "LEO_cons"):

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    # # Define set up of all satellites in the constellation
    # def setup(self):
        if constellation_type == "LEO_cons":
            height_init = h_SC
            inc_init = inc_SC
            RAAN_init = np.linspace(0.0, 360.0, number_of_planes)
            TA_init = np.linspace(0.0, 360.0, number_sats_per_plane)
        elif constellation_type == "LEO_1":
            height_init = h_SC
            inc_init = inc_SC
            RAAN_init = 240.0 #* ureg.degree
            TA_init = 0.0 #* ureg.degree
        elif constellation_type == "GEO":
            height_init = 35800.0  # * ureg.kilometer
            inc_init = 0.0  # * ureg.degree
            RAAN_init = 0.0  # * ureg.degree
            TA_init = 0.0  # * ureg.degree

        self.number_of_planes = number_of_planes
        self.number_sats_per_plane = number_sats_per_plane
        self.height_init = height_init
        self.inc_init = inc_init
        self.RAAN_init = RAAN_init
        self.TA_init = TA_init
        self.ECC_init = 0.0 #* ureg.degree
        self.omega_init = 237.5 #* ureg.degree


    def propagate(self,
                  simulation_start_epoch,
                  simulation_end_epoch):

        # ------------------------------------------------------------------------
        # ---------------------INITIATE-AIRCRAFT-CLASS-&-PROPAGATE----------------
        # ------------------------------------------------------------------------

        # Initiate aircraft class
        self.AC = AC.aircraft(lat_init=lat_init_AC,
                              lon_init=lon_init_AC
                              )

        self.pos_AC, self.heights_AC, self.lat_AC, self.lon_AC, self.time = self.AC.propagate(simulation_start_epoch,
                                                                                              simulation_end_epoch,
                                                                                              vel_AC,
                                                                                              # self.states_cons[0][0],
                                                                                              h_AC,
                                                                                              fixed_step_size=step_size_dim2,
                                                                                              method=method_AC)

        # ------------------------------------------------------------------------
        # --------------------INITIATE-SPACECRAFT-CLASS-&-PROPAGATE---------------
        # ------------------------------------------------------------------------
        # Initiate spacecraft class
        self.SC = SC.constellation(sat_setup=constellation_type,
                                   number_of_planes=number_of_planes,
                                   number_sats_per_plane=number_sats_per_plane,
                                   height_init=self.height_init,  # 550.0E3 for Starlink phase 0
                                   inc_init=self.inc_init,  # 53.0 for Starlink phase 0,
                                   RAAN_init=self.RAAN_init,
                                   TA_init=self.TA_init,
                                   ECC_init=self.ECC_init,
                                   omega_init=self.omega_init)

        # Propagate spacecraft and extract time
        self.states_cons, self.dep_var_cons, self.time_SC = self.SC.propagate(simulation_start_epoch = self.time[0],
                                                                               simulation_end_epoch = self.time[-1],
                                                                               method = method_SC,
                                                                               propagator= "Runge-Kutta 4",
                                                                               fixed_step_size= step_size_dim2)
        # self.heights_SC = self.dep_var_cons[:, 0]
        # self.lat_SC = self.dep_var_cons[:, 1]
        # self.lon_SC = self.dep_var_cons[:, 2]
        # self.kepler_SC = self.dep_var_cons[:, -6:]

    # Loop through all satellites and create geometrical outputs
    def geometrical_outputs(self):

        # variables_to_save = ['pos SC', 'heights SC', 'vel SC', 'ranges', 'slew rates', 'angular rates', 'zenith',
        #                      'elevation', 'azimuth', 'radial']
        self.geometrical_output = {}

        # Initiate variable lists
        pos_SC = {}
        heights_SC = {}
        vel_SC = {}
        ranges = {}
        slew_rates = {}
        zenith = {}
        elevation = {}
        azimuth = {}
        radial = {}
        angular_rates = {}
        delta_v = {}

        self.geometrical_output_filtered = {}
        # Initiate variable lists for filtered data
        time_filtered = {}
        ranges_filtered = {}
        pos_SC_filtered = {}
        vel_SC_filtered = {}
        heights_SC_filtered = {}
        zenith_filtered = {}
        elevation_filtered = {}
        azimuth_filtered = {}
        radial_filtered = {}
        slew_rates_filtered = {}
        angular_rates_filtered = {}

        # Loop through orbits for each satellite in constellation
        for plane in range(self.number_of_planes):
            # Initiate variable lists per plane
            pos_SC_per_plane = {}
            heights_SC_per_plane = {}
            vel_SC_per_plane = {}
            ranges_per_plane = {}
            zenith_per_plane = {}
            elevation_per_plane = {}
            azimuth_per_plane = {}
            radial_per_plane = {}
            slew_rates_per_plane = {}
            angular_rates_per_plane = {}
            delta_v_per_plane = {}

            # Initiate variables lists per plane, filtered with min. elevation angle
            time_filt_per_plane = {}
            ranges_filt_per_plane = {}
            pos_SC_filt_per_plane = {}
            vel_SC_filt_per_plane = {}
            heights_SC_filt_per_plane = {}
            zenith_filt_per_plane = {}
            elevation_filt_per_plane = {}
            azimuth_filt_per_plane = {}
            radial_filt_per_plane = {}
            slew_rates_filt_per_plane = {}
            angular_rates_filt_per_plane = {}

            for sat in range(self.number_sats_per_plane):

                # Select data for current satellite
                pos_SC_per_sat =  self.states_cons[plane][sat][:,1:4] #* ureg.meter
                R_SC   = np.sqrt(pos_SC_per_sat[:,0]**2 + pos_SC_per_sat[:,1]**2 + pos_SC_per_sat[:,2]**2)
                vel_SC_per_sat =  self.states_cons[plane][sat][:,-3:] #* ureg.meter / ureg.second
                heights_SC_per_sat = self.dep_var_cons[plane][sat][:,1] #* ureg.meter
                lat_SC = self.dep_var_cons[plane][sat][:,2] #* ureg.rad
                lon_SC = self.dep_var_cons[plane][sat][:,3] #* ureg.rad
                kepler_SC  = self.dep_var_cons[plane][sat][:,-6:]

                pos_SC_per_plane[sat] = pos_SC_per_sat
                heights_SC_per_plane[sat] = heights_SC_per_sat
                vel_SC_per_plane[sat] = vel_SC_per_sat

                # ------------------------------------------------------------------------
                # -----------------------RANGES-BETWEEN-AIRCRAFT-&-SPACECRAFT-------------
                # ------------------------------------------------------------------------

                ranges_per_sat = np.sqrt((pos_SC_per_sat[:,0]-self.pos_AC[:,0])**2 +
                                         (pos_SC_per_sat[:,1]-self.pos_AC[:,1])**2 +
                                         (pos_SC_per_sat[:,2]-self.pos_AC[:,2])**2)
                ranges_per_plane[sat] = ranges_per_sat

                # ------------------------------------------------------------------------
                # ---------------------------GEOMETRIC-ANGLES-----------------------------
                # -----------------------ELEVATION-AZIMUTH-ZENITH-------------------------
                # ------------------------------------------------------------------------

                a = ((heights_SC_per_sat - self.heights_AC) ** 2 +
                     2 * (heights_SC_per_sat - self.heights_AC) * R_earth -
                     ranges_per_sat ** 2) / (2 * ranges_per_sat * R_earth)

                for i in range(len(a)):
                    if a[i] < -1.0:
                        a[i] = -1.0
                    elif a[i] > 1.0:
                        a[i] = 1.0
                zenith_per_sat = np.arccos(a)


                elevation_per_sat = np.pi / 2 - zenith_per_sat

                azimuth_per_sat = np.arctan( abs(np.tan(lon_SC - self.lon_AC)) / np.sin(self.lat_AC))
                # for i in range(len(self.lat_AC)):
                #     lat_ac = self.lat_AC[i]
                #     lon_ac = self.lon_AC[i]
                #     lat_sc = lat_SC[i]
                #     lon_sc = lon_SC[i]
                #     # Aircraft on Northern hemisphere
                #     if lat_ac > 0.0:
                #         # SC is East of AC
                #         if lon_sc > lon_ac:
                #             azimuth[i] = np.pi - azimuth[i]
                #         # SC is West of AC
                #         elif lon_sc < lon_ac:
                #             azimuth[i] = np.pi + azimuth[i]
                #
                #     # Aircraft on Southern hemisphere
                #     elif lat_ac < 0.0:
                #         # SC is East of AC
                #         if lon_sc > lon_ac:
                #             azimuth[i] = azimuth[i]
                #         # SC is West of AC
                #         elif lon_sc < lon_ac:
                #             azimuth[i] = 2*np.pi + azimuth[i]

                radial_per_sat = np.sqrt(elevation_per_sat**2 + azimuth_per_sat**2)

                zenith_per_plane[sat] = zenith_per_sat
                elevation_per_plane[sat] = elevation_per_sat
                azimuth_per_plane[sat] = azimuth_per_sat
                radial_per_plane[sat] = radial_per_sat

                # ------------------------------------------------------------------------
                # ----------------------------SLEW-RATE-&-ANGULAR-RATE--------------------
                # ------------------------------------------------------------------------

                TA = kepler_SC[:, -1]
                # slew_angles = np.arcsin(SC_heights/range_array)
                slew_rates_per_sat = np.zeros(len(TA))
                angular_rates_per_sat = np.zeros(len(TA))
                for i in range(1, len(TA)):
                    dt = self.time[i] - self.time[i - 1]
                    dTA = TA[i] - TA[i - 1]
                    drad = radial_per_sat[i] - radial_per_sat[i - 1]
                    slew_rates_per_sat[i] = dTA / dt
                    angular_rates_per_sat[i] = drad / dt

                slew_rates_per_plane[sat] = slew_rates_per_sat
                angular_rates_per_plane[sat] = angular_rates_per_sat

                # ------------------------------------------------------------------------
                # -----------------------------DOPPLER-SHIFT------------------------------
                # ------------------------------------------------------------------------

                delta_v_per_sat = v * (R_earth + heights_SC_per_sat) * (R_earth + self.heights_AC) * slew_rates_per_sat * np.sin(slew_rates_per_sat * self.time) / \
                                  (speed_of_light * np.sqrt((R_earth + heights_SC_per_sat) ** 2 + (R_earth + self.heights_AC) ** 2 -
                                                            2 * (R_earth + heights_SC_per_sat) * (R_earth + self.heights_AC) * np.cos(slew_rates_per_sat * self.time)))

                delta_v_per_plane[sat] = delta_v_per_sat

                # ------------------------------------------------------------------------
                # --------------FILTER-DATA-FOR-MINIMUM-ELEVATION-REQUIREMENT-------------
                # ------------------------------------------------------------------------

                # Filter data based on constraint requirement (elevation)
                time_filt        = self.time[elevation_per_sat > elevation_min]
                ranges_filt      = ranges_per_sat[elevation_per_sat > elevation_min]
                pos_SC_filt      = pos_SC_per_sat[elevation_per_sat > elevation_min]
                vel_SC_filt      = vel_SC_per_sat[elevation_per_sat > elevation_min]
                heights_SC_filt  = heights_SC_per_sat[elevation_per_sat > elevation_min]
                zenith_filt      = zenith_per_sat[elevation_per_sat > elevation_min]
                elevation_filt   = elevation_per_sat[elevation_per_sat > elevation_min]
                azimuth_filt     = azimuth_per_sat[elevation_per_sat > elevation_min]
                radial_filt      = radial_per_sat[elevation_per_sat > elevation_min]
                slew_rates_filt  = slew_rates_per_sat[elevation_per_sat > elevation_min]
                angular_rates_filt = angular_rates_per_sat[elevation_per_sat > elevation_min]

                # Assign filtered data lists to lists per plane
                pos_SC_filt_per_plane[sat] = pos_SC_filt
                ranges_filt_per_plane[sat] = ranges_filt
                vel_SC_filt_per_plane[sat] = vel_SC_filt
                heights_SC_filt_per_plane[sat] = heights_SC_filt
                time_filt_per_plane[sat] = time_filt
                zenith_filt_per_plane[sat] = zenith_filt
                elevation_filt_per_plane[sat] = elevation_filt
                azimuth_filt_per_plane[sat] = azimuth_filt
                radial_filt_per_plane[sat] = radial_filt
                slew_rates_filt_per_plane[sat] = slew_rates_filt
                angular_rates_filt_per_plane[sat] = angular_rates_filt

            # self.geometrical_output[plane][sat] = ranges_per_plane
            # self.geometrical_output[1][plane] = slew_rates_per_plane
            # self.geometrical_output[2][plane] = zenith_per_plane
            # self.geometrical_output['elevation'][plane] = elevation_per_plane
            # self.geometrical_output['azimuth'][plane] = azimuth_per_plane
            # self.geometrical_output['angular rates'][plane] = angular_rates_per_plane

            # Assign filtered lists per plane to large filtered list
            time_filtered[plane] = time_filt_per_plane
            ranges_filtered[plane] = ranges_filt_per_plane
            slew_rates_filtered[plane] = slew_rates_filt_per_plane
            pos_SC_filtered[plane]  = pos_SC_filt_per_plane
            vel_SC_filtered[plane] = vel_SC_filt_per_plane
            heights_SC_filtered[plane] = heights_SC_filt_per_plane
            zenith_filtered[plane] = zenith_filt_per_plane
            elevation_filtered[plane] = elevation_filt_per_plane
            azimuth_filtered[plane] = azimuth_filt_per_plane
            radial_filtered[plane] = radial_filt_per_plane
            angular_rates_filtered[plane] = angular_rates_per_plane

            # ------------------------------------------------------------------------
            # ----------------------ASSIGN-DATA-TO-GEOMETRICAL-OUTPUT-FILE------------
            # ------------------------------------------------------------------------

            # Assign lists per plane to large list
            pos_SC[plane] = pos_SC_per_plane
            heights_SC[plane] = heights_SC_per_plane
            vel_SC[plane] = vel_SC_per_plane
            ranges[plane] = ranges_per_plane
            slew_rates[plane] = slew_rates_per_plane
            zenith[plane] = zenith_per_plane
            elevation[plane] = elevation_per_plane
            azimuth[plane] = azimuth_per_plane
            radial[plane] = radial_per_plane
            angular_rates[plane] = angular_rates_per_plane
            delta_v[plane] = delta_v_per_plane

        self.geometrical_output_filtered['time'] = time_filtered
        self.geometrical_output_filtered['pos SC'] = pos_SC_filtered
        self.geometrical_output_filtered['heights SC'] = heights_SC_filtered
        self.geometrical_output_filtered['vel SC'] = vel_SC_filtered
        self.geometrical_output_filtered['ranges'] = ranges_filtered
        self.geometrical_output_filtered['slew rates'] = slew_rates_filtered
        self.geometrical_output_filtered['angular rates'] = angular_rates_filtered
        self.geometrical_output_filtered['zenith'] = zenith_filtered
        self.geometrical_output_filtered['elevation'] = elevation_filtered
        self.geometrical_output_filtered['azimuth'] = azimuth_filtered
        self.geometrical_output_filtered['radial'] = radial_filtered

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
        self.geometrical_output['doppler shift'] = delta_v

        return self.geometrical_output

    #------------------------------------------------------------------------
    #------------------------PRINT-DATA-&-VISUALIZATION----------------------
    #------------------------------------------------------------------------


    def plot(self, type="trajectories", sequence = 0.0):

        if type == "trajectories":
            # Define a 3D figure using pyplot
            fig = plt.figure(figsize=(6,6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

            # # Plot fist satellite with label
            # ax.plot(self.states_cons[0][0][:, 1],
            #         self.states_cons[0][0][:, 2],
            #         self.states_cons[0][0][:, 3],
            #         linestyle='-', linewidth=0.5, color='orange', label='constellation')
            #
            # # Plot first satellite filtered data with label
            # ax.scatter(self.geometrical_output_filtered['pos SC'][0][0][:, 0],
            #            self.geometrical_output_filtered['pos SC'][0][0][:, 1],
            #            self.geometrical_output_filtered['pos SC'][0][0][:, 2],
            #            linewidth=0.5, color='g', label='window \epsilon > '+str(np.rad2deg(elevation_min)))

            # Plot all other satellites in constellation
            for plane in range(number_of_planes):
                for sat in range(number_sats_per_plane):
                    ax.plot(self.states_cons[plane][sat][:, 1],
                            self.states_cons[plane][sat][:, 2],
                            self.states_cons[plane][sat][:, 3],
                            linestyle='-', linewidth=0.5, color='orange')
                    ax.scatter(self.geometrical_output_filtered['pos SC'][plane][sat][:, 0],
                               self.geometrical_output_filtered['pos SC'][plane][sat][:, 1],
                               self.geometrical_output_filtered['pos SC'][plane][sat][:, 2],
                               linewidth=0.5)

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
                    for sat in range(number_sats_per_plane):
                        axs_elev[0].set_title(f'Elevation angles vs time')
                        # axs_elev[0].plot(self.time_filtered[plane][sat], np.rad2deg(self.elevation_filtered[plane][sat]))
                        # axs_elev[1].plot(self.time_filtered[plane][sat], np.rad2deg(self.azimuth_filtered[plane][sat]))
                        # axs_elev[2].plot(self.time_filtered[plane][sat], np.rad2deg(self.radial_filtered[plane][sat]))
                        # axs_elev[1].plot(self.time, np.rad2deg(self.ranges[plane][sat]))

                        axs_elev[0].plot(self.time, np.ones(len(self.time))*elevation_min)
                        axs_elev[0].plot(self.time, np.rad2deg(self.geometrical_output['elevation'][plane][sat]))
                        axs_elev[1].plot(self.time, self.geometrical_output['ranges'][plane][sat])

                        axs_elev[0].set_ylabel('elevation (degrees)')
                        axs_elev[0].set_xlabel('time (seconds)')


        elif type == "satellite sequence":
            fig = plt.figure(figsize=(6, 6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('trajectories of satellite sequence with handover strategy')
            ax.plot(sequence[:, 0],
                    sequence[:, 1],
                    sequence[:, 2],
                    linestyle='-.', linewidth=0.5, color='g')
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

    def print(self, plane, sat, index):

        print('------------------------------------------------')
        print('LINK GEOMETRY')
        print('------------------------------------------------')
        print('Range [km]         : ', self.geometrical_output['ranges'][plane][sat][index])
        print('Zenith angle [deg] : ', np.rad2deg(self.geometrical_output['zenith'][plane][sat][index]))
        print('Elev. angle  [deg] : ', np.rad2deg(self.geometrical_output['elevation'][plane][sat][index]))
        print('Slew rate [deg/s]  : ', self.geometrical_output['slew rates'][plane][sat][index])