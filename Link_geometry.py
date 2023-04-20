
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
    def __init__(self):

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


    def propagate(self):

        # ------------------------------------------------------------------------
        # ---------------------INITIATE-AIRCRAFT-CLASS-&-PROPAGATE----------------
        # ------------------------------------------------------------------------

        # Initiate aircraft class
        self.AC = AC.aircraft(lat_init=lat_init_AC,
                              lon_init=lon_init_AC
                              )
        self.pos_AC, self.heights_AC, self.lat_AC, self.lon_AC, self.speed_AC, self.time = self.AC.propagate(start_time,
                                                                                                      end_time,
                                                                                                      # vel_AC,
                                                                                                      # h_AC,
                                                                                                      method=method_AC,
                                                                                                      filename=aircraft_filename)

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
        self.states_cons, self.dep_var_cons, self.time_SC = self.SC.propagate(AC_time=self.time,
                                                                               method = method_SC)
        # self.heights_SC = self.dep_var_cons[:, 0]
        # self.lat_SC = self.dep_var_cons[:, 1]
        # self.lon_SC = self.dep_var_cons[:, 2]
        # self.kepler_SC = self.dep_var_cons[:, -6:]

    # Loop through all satellites and create geometrical outputs
    def geometrical_outputs(self):

        # variables_to_save = ['pos SC', 'heights SC', 'vel SC', 'ranges', 'slew rates', 'zenith',
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
        delta_v_filtered = {}

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
            delta_v_filt_per_plane = {}

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
                # # Correct for refraction index
                # zenith_per_sat = np.arcsin(1 / n_index * np.sin(zenith_per_sat))
                elevation_per_sat = np.pi / 2 - zenith_per_sat
                delta_lon = lon_SC - self.lon_AC
                y = np.sin(delta_lon) * np.cos(lat_SC)
                x = np.cos(self.lat_AC) * np.sin(lat_SC) - np.sin(self.lat_AC) * np.cos(lat_SC) * np.cos(delta_lon)
                azimuth_per_sat = np.arctan2(y, x)
                radial_per_sat = np.sqrt(elevation_per_sat**2 + azimuth_per_sat**2)

                zenith_per_plane[sat] = zenith_per_sat
                elevation_per_plane[sat] = elevation_per_sat
                azimuth_per_plane[sat] = azimuth_per_sat
                radial_per_plane[sat] = radial_per_sat

                dt = np.insert(np.diff(self.time), 0, 0.0)
                d_elevation = np.insert(np.diff(elevation_per_sat), 0, 0.0)
                d_azimuth = np.insert(np.diff(azimuth_per_sat), 0, 0.0)

                # Phase unwrap azimuth vector
                wraps = np.where(np.abs(d_azimuth) > np.pi)[0]
                for i in wraps:
                    if d_azimuth[i] > 0:
                        azimuth_per_sat[i:] -= 2 * np.pi
                    else:
                        azimuth_per_sat[i:] += 2 * np.pi
                d_azimuth = np.insert(np.diff(azimuth_per_sat), 0, 0.0)
                # ------------------------------------------------------------------------
                # --------------------------------SLEW-RATE-RATE--------------------------
                # ------------------------------------------------------------------------
                slew_rates_per_sat = np.sqrt((d_elevation/dt)**2 + (d_azimuth/dt)**2)
                slew_rates_per_sat[0] = 0
                slew_rates_per_plane[sat] = slew_rates_per_sat
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
                mask = elevation_per_sat > elevation_min
                time_filt        = self.time[mask]
                ranges_filt      = ranges_per_sat[mask]
                pos_SC_filt      = pos_SC_per_sat[mask]
                vel_SC_filt      = vel_SC_per_sat[mask]
                heights_SC_filt  = heights_SC_per_sat[mask]
                zenith_filt      = zenith_per_sat[mask]
                elevation_filt   = elevation_per_sat[mask]
                azimuth_filt     = azimuth_per_sat[mask]
                radial_filt      = radial_per_sat[mask]
                slew_rates_filt  = slew_rates_per_sat[mask]
                delta_v_filt     = delta_v_per_sat[mask]

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
                delta_v_filt_per_plane[sat] = delta_v_filt


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
            delta_v_filtered[plane] = delta_v_filt_per_plane

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
            delta_v[plane] = delta_v_per_plane

        self.geometrical_output_filtered['time'] = time_filtered
        self.geometrical_output_filtered['pos SC'] = pos_SC_filtered
        self.geometrical_output_filtered['heights SC'] = heights_SC_filtered
        self.geometrical_output_filtered['vel SC'] = vel_SC_filtered
        self.geometrical_output_filtered['ranges'] = ranges_filtered
        self.geometrical_output_filtered['slew rates'] = slew_rates_filtered
        self.geometrical_output_filtered['zenith'] = zenith_filtered
        self.geometrical_output_filtered['elevation'] = elevation_filtered
        self.geometrical_output_filtered['azimuth'] = azimuth_filtered
        self.geometrical_output_filtered['radial'] = radial_filtered
        self.geometrical_output_filtered['doppler shift'] = delta_v_filtered

        self.geometrical_output['pos AC'] = self.pos_AC
        self.geometrical_output['pos SC'] = pos_SC
        self.geometrical_output['heights SC'] = heights_SC
        self.geometrical_output['vel SC'] = vel_SC
        self.geometrical_output['ranges'] = ranges
        self.geometrical_output['slew rates'] = slew_rates
        self.geometrical_output['zenith'] = zenith
        self.geometrical_output['elevation'] = elevation
        self.geometrical_output['azimuth'] = azimuth
        self.geometrical_output['radial'] = radial
        self.geometrical_output['doppler shift'] = delta_v

        return self.geometrical_output

    #------------------------------------------------------------------------
    #------------------------PRINT-DATA-&-VISUALIZATION----------------------
    #------------------------------------------------------------------------


    def plot(self, type="trajectories", sequence = 0.0, links=0):

        if type == "trajectories":
            # Define a 3D figure using pyplot
            fig = plt.figure(figsize=(6,6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Number of planes: ' + str(number_of_planes) + ', sats per plane: ' + str(number_sats_per_plane))
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

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
            ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='darkgreen', s=100)

            # # Add the legend and labels, then show the plot
            ax.legend()
            ax.set_xlabel('x [m]', fontsize=15)
            ax.set_ylabel('y [m]', fontsize=15)
            ax.set_zlabel('z [m]', fontsize=15)


        # Plot looking angles
        elif type == "angles":
            time_hrs = self.time / 3600
            fig_elev, axs = plt.subplots(1, 1, figsize=(6, 6), dpi=125)
            for plane in range(number_of_planes):
                    for sat in range(number_sats_per_plane):
                        axs.plot(time_hrs, np.rad2deg(self.geometrical_output['elevation'][plane][sat]))
            plt.show()

            samples = number_sats_per_plane * number_of_planes * len(self.geometrical_output['elevation'][0][0])
            fig_elev, axs = plt.subplots(4, 1, figsize=(6, 6), dpi=125)
            for plane in range(number_of_planes):
                    for sat in range(number_sats_per_plane):
                        axs[0].plot(time_hrs, np.rad2deg(self.geometrical_output['elevation'][plane][sat]))

            axs[0].plot(time_hrs, np.rad2deg(self.geometrical_output['elevation'][0][0]))
            axs[1].plot(time_hrs, np.rad2deg(self.geometrical_output['azimuth'][0][0]))
            axs[2].plot(time_hrs, np.rad2deg(self.geometrical_output['radial'][0][0]))
            axs[3].plot(time_hrs, self.geometrical_output['slew rates'][0][0])

            axs[0].set_title(f'Aircraft-Satellite angles vs time \n'
                             f'Samples: '+str(number_sats_per_plane*number_of_planes)+' * '+str(len(self.geometrical_output['elevation'][0][0]))+' = '+str(samples))
            axs[0].plot(time_hrs, np.ones(len(self.time)) * np.rad2deg(elevation_min),
                          label='Minimum elevation=' + str(np.round(np.rad2deg(elevation_min), 2)) + 'deg')

            axs[0].set_ylabel('Elevation (degrees)')
            axs[1].set_ylabel('Azimuth (degrees)')
            axs[2].set_ylabel('Radial (degrees)')
            axs[3].set_ylabel('Slew rate (rad/s)')

            axs[3].set_xlabel('Time (hours)')
            # axs[0].legend(fontsize=15)
            axs[0].grid()
            axs[1].grid()
            axs[2].grid()
            axs[3].grid()


        elif type == "satellite sequence":
            if links > 0:
                comm_time_avg = np.count_nonzero(sequence[:, 0]) * step_size_AC / links / 60.0
            else:
                comm_time_avg = 0
            fig = plt.figure(figsize=(6, 6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title('Number of planes: '+str(number_of_planes)+', sats per plane: '+str(number_sats_per_plane)+'\n'
                         'Number of links: '+str(links)+', average link time (minutes): '+str(np.round(comm_time_avg,2)))
            ax.scatter(sequence[:, 0],
                        sequence[:, 1],
                        sequence[:, 2],
                        linewidth=0.1, label='satellites')
            # Plot aircraft
            ax.plot(self.pos_AC[:, 0],
                    self.pos_AC[:, 1],
                    self.pos_AC[:, 2], color='black', label='aircraft', linewidth = 2)
            # Draw Earth as a blue dot in origin
            ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue', s=100)

            # Plot all other satellites in constellation
            for plane in range(number_of_planes):
                for sat in range(number_sats_per_plane):
                    ax.plot(self.states_cons[plane][sat][:, 1],
                            self.states_cons[plane][sat][:, 2],
                            self.states_cons[plane][sat][:, 3],
                            linestyle='-', linewidth=0.1, color='orange')

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