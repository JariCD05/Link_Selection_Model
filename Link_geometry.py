
import Link_budget as LB
import Constellation as SC
import Aircraft as AC
import Turbulence as turb
import constants as cons

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
    def __init__(self,
                 vel_AC = np.array([0.0, 0.0, 0.0]),  # Velocity profile array (North, East, Down)
                 lat_init_AC = 0.0,
                 lon_init_AC = 0.0,
                 simulation_start_epoch = 0.0,
                 simulation_end_epoch = cons_tudat.JULIAN_DAY,
                 sat_setup = "LEO_cons",
                 fixed_step_size=10.0,
                 SC = SC,
                 AC = AC,
                 elevation_min = 10.0,
                 height_init_AC = 10.0E3):

        #------------------------------------------------------------------------
        #--------------------------DEFINE-SATELLITE-SETUP------------------------
        #------------------------------------------------------------------------

        # Requirement of minimum elevation angle of X degrees
        self.elevation_min = np.deg2rad(elevation_min)
        self.sat_setup = sat_setup
        #"LEO_1" #, GEO

        # Define set up of all satellites in the constellation
        if sat_setup == "LEO_cons":
            number_of_planes = 20
            number_sats_per_plane = 10
            height_init = 550.0E3  # 550.0E3 for Starlink phase 0
            inc_init = 53.0  # 53.0 for Starlink phase 0
            RAAN_init = np.linspace(0.0, 360.0, number_of_planes)
            TA_init = np.linspace(0.0, 360.0, number_sats_per_plane)
        elif sat_setup == "LEO_1":
            number_of_planes = 1
            number_sats_per_plane = 1
            height_init = 550.0E3  # 550.0E3 for Starlink phase 0
            inc_init = 53.0  # 53.0 for Starlink phase 0
            RAAN_init = 240.0
            TA_init = 0.0
        elif sat_setup == "GEO":
            number_of_planes = 1
            number_sats_per_plane = 1
            height_init = 35800.0E3
            inc_init = 0.0
            RAAN_init = 0.0
            TA_init = 0.0

        self.number_of_planes = number_of_planes
        self.number_of_sats = number_sats_per_plane
        # ------------------------------------------------------------------------
        # --------------------INITIATE-SPACECRAFT-CLASS-&-PROPAGATE---------------
        # ------------------------------------------------------------------------

        # Initiate spacecraft class
        self.SC = SC.constellation(simulation_start_epoch,
                              simulation_end_epoch,
                              sat_setup,
                              number_of_planes,
                              number_sats_per_plane,
                              height_init,  # 550.0E3 for Starlink phase 0
                              inc_init,  # 53.0 for Starlink phase 0,
                              RAAN_init,
                              TA_init)

        # Propagate spacecraft and extract time
        self.states_cons, self.dep_var_cons, self.time = self.SC.propagate(method = "tudat",
                                                                       propagator= "Runge-Kutta 4",
                                                                       fixed_step_size= fixed_step_size)
        # self.heights_SC = self.dep_var_cons[:, 0]
        # self.lat_SC = self.dep_var_cons[:, 1]
        # self.lon_SC = self.dep_var_cons[:, 2]
        # self.kepler_SC = self.dep_var_cons[:, -6:]

        # ------------------------------------------------------------------------
        # ---------------------INITIATE-AIRCRAFT-CLASS-&-PROPAGATE----------------
        # ------------------------------------------------------------------------

        # Initiate aircraft class
        self.AC = AC.aircraft(self.time,
                              self.states_cons[0][0],
                              lat_init=lat_init_AC,
                              lon_init=lon_init_AC,
                              height=height_init_AC,
                              fixed_step_size=fixed_step_size
                              )

        self.pos_AC, self.heights_AC, self.lat_AC, self.lon_AC, self.R_AC = self.AC.propagate(vel_AC,
                                                                                              propagator_type="straight")

        # Initiate variable lists
        self.ranges = {}
        self.slew_rates = {}
        self.zenith = {}
        self.elevation = {}
        self.azimuth = {}
        self.radial = {}
        self.angular_rates = {}
        self.doppler = {}

        # Initiate variable lists for filtered data
        self.time_filtered = {}
        self.ranges_filtered = {}
        self.pos_SC_filtered = {}
        self.vel_SC_filtered = {}
        self.heights_SC_filtered = {}
        self.zenith_filtered = {}
        self.elevation_filtered = {}
        self.azimuth_filtered = {}
        self.radial_filtered = {}
        self.slew_rates_filtered = {}
        self.angular_rates_filtered = {}



        # Loop through orbits for each satellite in constellation
        for plane in range(number_of_planes):
            # Initiate variable lists per plane
            ranges_per_plane = {}
            zenith_per_plane = {}
            elevation_per_plane = {}
            azimuth_per_plane = {}
            radial_per_plane = {}
            slew_rates_per_plane = {}
            angular_rates_per_plane = {}
            doppler_per_plane = {}

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

            for sat in range(number_sats_per_plane):

                # Select data for current satellite
                pos_SC =  self.states_cons[plane][sat][:,1:4]
                R_SC   = np.sqrt(pos_SC[:,0]**2 + pos_SC[:,1]**2 + pos_SC[:,2]**2)
                vel_SC =  self.states_cons[plane][sat][:,-3:]
                heights_SC = self.dep_var_cons[plane][sat][:,1]
                lat_SC = self.dep_var_cons[plane][sat][:,2]
                lon_SC = self.dep_var_cons[plane][sat][:,3]
                kepler_SC  = self.dep_var_cons[plane][sat][:,-6:]

                # ------------------------------------------------------------------------
                # -----------------------RANGES-BETWEEN-AIRCRAFT-&-SPACECRAFT-------------
                # ------------------------------------------------------------------------

                ranges = np.sqrt((pos_SC[:,0]-self.pos_AC[:,0])**2 +
                                 (pos_SC[:,1]-self.pos_AC[:,1])**2 +
                                 (pos_SC[:,2]-self.pos_AC[:,2])**2)
                ranges_per_plane[sat] = ranges

                # ------------------------------------------------------------------------
                # ---------------------------GEOMETRIC-ANGLES-----------------------------
                # -----------------------ELEVATION-AZIMUTH-ZENITH-------------------------
                # ------------------------------------------------------------------------
                zenith = np.arccos(((heights_SC - self.heights_AC) ** 2 +
                                                2 * (heights_SC - self.heights_AC) * cons.R_earth -
                                                ranges ** 2) / (2 * ranges * cons.R_earth))
                elevation = np.pi / 2 - zenith

                azimuth = np.arctan( abs(np.tan(lon_SC - self.lon_AC)) / np.sin(self.lat_AC))
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

                radial = np.sqrt(elevation**2 + azimuth**2)
                zenith_per_plane[sat] = zenith
                elevation_per_plane[sat] = elevation
                azimuth_per_plane[sat] = azimuth
                radial_per_plane[sat] = radial

                # ------------------------------------------------------------------------
                # ----------------------------SLEW-RATE-&-ANGULAR-RATE--------------------
                # ------------------------------------------------------------------------

                TA = kepler_SC[:, -1]
                # slew_angles = np.arcsin(SC_heights/range_array)
                slew_rates = np.zeros(len(TA))
                angular_rates = np.zeros(len(TA))
                for i in range(1, len(TA)):
                    dt = self.time[i] - self.time[i - 1]
                    dTA = TA[i] - TA[i - 1]
                    drad = radial[i] - radial[i - 1]
                    slew_rates[i] = dTA / dt
                    angular_rates[i] = drad / dt

                slew_rates_per_plane[sat] = slew_rates
                angular_rates_per_plane[sat] = angular_rates

                # ------------------------------------------------------------------------
                # --------------FILTER-DATA-FOR-MINIMUM-ELEVATION-REQUIREMENT-------------
                # ------------------------------------------------------------------------

                # Filter data based on constraint requirement (elevation)
                time_filt        = self.time[elevation > self.elevation_min]
                ranges_filt      = ranges[elevation > self.elevation_min]
                pos_SC_filt      = pos_SC[elevation > self.elevation_min]
                R_SC_filt        = R_SC[elevation > self.elevation_min]
                vel_SC_filt      = vel_SC[elevation > self.elevation_min]
                heights_SC_filt  = heights_SC[elevation > self.elevation_min]
                lat_SC_filt      = lat_SC[elevation > self.elevation_min]
                lon_SC_filt      = lon_SC[elevation> self.elevation_min]
                kepler_SC_filt   = kepler_SC[elevation > self.elevation_min]
                zenith_filt      = zenith[elevation > self.elevation_min]
                elevation_filt   = elevation[elevation > self.elevation_min]
                azimuth_filt     = azimuth[elevation > self.elevation_min]
                radial_filt      = radial[elevation > self.elevation_min]
                slew_rates_filt  = slew_rates[elevation > self.elevation_min]
                angular_rates_filt = angular_rates[elevation > self.elevation_min]

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

            # Assign lists per plane to large list
            self.ranges[plane] = ranges_per_plane
            self.slew_rates[plane] = slew_rates_per_plane
            self.zenith[plane] = zenith_per_plane
            self.elevation[plane] = elevation_per_plane
            self.azimuth[plane] = azimuth_per_plane
            self.radial[plane] = radial_per_plane
            self.slew_rates[plane] = slew_rates_per_plane
            self.angular_rates[plane] = angular_rates_per_plane

            # Assign filtered lists per plane to large filtered list
            self.time_filtered[plane] = time_filt_per_plane
            self.ranges_filtered[plane] = ranges_filt_per_plane
            self.slew_rates_filtered[plane] = slew_rates_filt_per_plane
            self.pos_SC_filtered[plane]  = pos_SC_filt_per_plane
            self.vel_SC_filtered[plane] = vel_SC_filt_per_plane
            self.heights_SC_filtered[plane] = heights_SC_filt_per_plane
            self.zenith_filtered[plane] = zenith_filt_per_plane
            self.elevation_filtered[plane] = elevation_filt_per_plane
            self.azimuth_filtered[plane] = azimuth_filt_per_plane
            self.radial_filtered[plane] = radial_filt_per_plane
            self.angular_rates_filtered[plane] = angular_rates_per_plane

    # ------------------------------------------------------------------------
    # -----------------------------FUNCTIONS----------------------------------
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # -----------------------------DOPPLER-SHIFT------------------------------
    # ------------------------------------------------------------------------

    def doppler(self, v):
        delta_v = v * (cons.R_earth + self.heights_SC) * (
                cons.R_earth + self.heights_AC) * self.slew_rates * np.sin(self.slew_rates * self.time) \
                      (cons.speed_of_light * np.sqrt(
                          (cons.R_earth + self.heights_SC) ** 2 + (cons.R_earth + self.heights_AC) ** 2 -
                          2 * (cons.R_earth + self.heights_SC)(cons.R_earth + self.heights_AC) * np.cos(
                              self.slew_rates * self.time)))
        return delta_v


    #------------------------------------------------------------------------
    #--------------------------------LASER-INFO-ARRAY------------------------
    #------------------------------------------------------------------------
    # Laser state for propagation between AC and SC
    number_of_steps = 100

    # Laser propagation state function, evaluated between SC and AC, for each time step
    # [time, position x, position y, position, z, position r, position h)
    def state_laser(self, range_array,
                    number_of_steps = 100):
        speed_of_light = 3.0E8
        state_laser = np.zeros((len(self.pos_SC[:, 0]), number_of_steps, 6))
        for i in range(len(range_array)):
            x = np.linspace(self.pos_AC[i, 0], self.pos_SC[i, 0], number_of_steps)
            y = np.linspace(self.pos_AC[i, 1], self.pos_SC[i, 1], number_of_steps)
            z = np.linspace(self.pos_AC[i, 2], self.pos_SC[i, 2], number_of_steps)

            diff_x = x - self.pos_AC[i, 0]
            diff_y = y - self.pos_AC[i, 1]
            diff_z = z - self.pos_AC[i, 2]

            diff_r = np.sqrt(diff_x ** 2 + diff_y ** 2 + diff_z ** 2)
            heights = np.linspace(self.heights_SC[i], self.heights_SC[i], number_of_steps)

            state_laser[i, :, 0] = diff_r / speed_of_light  # time
            state_laser[i, :, 1] = x
            state_laser[i, :, 2] = y
            state_laser[i, :, 3] = z
            state_laser[i, :, 4] = diff_r
            state_laser[i, :, 5] = heights

        return state_laser

    #------------------------------------------------------------------------
    #------------------------PRINT-DATA-&-VISUALIZATION----------------------
    #------------------------------------------------------------------------


    def plot(self, type="trajectories"):
        number_of_planes = self.SC.number_of_planes
        number_of_sats = self.number_of_sats

        if type == "trajectories":
            print(np.shape(self.pos_SC_filtered[0][0][:,0]))
            # Define a 3D figure using pyplot
            fig = plt.figure(figsize=(6,6), dpi=125)
            ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(f'Starlink initial phase configuration', fontsize=40)

            # Plot satellite and aircraft
            ax.plot(self.states_cons[0][0][:, 1], self.states_cons[0][0][:, 2], self.states_cons[0][0][:, 3],
                        linestyle='-.', linewidth=0.5, color='orange', label='constellation')
            ax.scatter(self.pos_SC_filtered[0][0][:, 0], self.pos_SC_filtered[0][0][:, 1], self.pos_SC_filtered[0][0][:, 2],
                    linestyle='-.', linewidth=0.5, color='g', label='window \epsilon > '+str(np.rad2deg(self.elevation_min)))

            for plane in range(1, number_of_planes):
                ax.plot(self.states_cons[plane][0][:, 1], self.states_cons[plane][0][:, 2], self.states_cons[plane][0][:, 3],
                        linestyle='-.', linewidth=0.5, color='orange')
                ax.scatter(self.pos_SC_filtered[plane][0][:, 0], self.pos_SC_filtered[plane][0][:, 1], self.pos_SC_filtered[plane][0][:, 2],
                        linestyle='-.', linewidth=0.5)

            ax.plot(self.pos_AC[:, 0], self.pos_AC[:, 1], self.pos_AC[:, 2], color='black', label='aircraft')
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

                        axs_elev[0].plot(self.time, np.ones(len(self.time))*self.elevation_min)
                        axs_elev[0].plot(self.time, np.rad2deg(self.elevation[plane][sat]))
                        axs_elev[1].plot(self.time, np.rad2deg(self.ranges[plane][sat]))


    def print(self, plane, sat, index):

        print('------------------------------------------------')
        print('LINK GEOMETRY')
        print('------------------------------------------------')
        print('Range [km]         : ', self.ranges[plane][sat][index]/1000)
        print('Zenith angle [deg] : ', np.rad2deg(self.zenith[plane][sat][index]))
        print('Elev. angle  [deg] : ', np.rad2deg(self.elevation[plane][sat][index]))
        print('Slew rate [deg/s]  : ', np.rad2deg(self.slew_rates[plane][sat][index]))