# Load standard modules
import numpy as np
# Uncomment the following to make plots interactive
# %matplotlib widget
from matplotlib import pyplot as plt
from matplotlib import animation

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

import constants as cons

# Load spice kernels
spice.load_standard_kernels()

class constellation:
    def __init__(self,
                 simulation_start_epoch = 0.0,
                 simulation_end_epoch = constants.JULIAN_DAY,
                 sat_setup = "LEO_1",
                 number_of_planes = 72,
                 number_sats_per_plane = 1,
                 height_init = 550.0E3,  # 550.0E3 for Starlink phase 0
                 inc_init = 53.0,  # 53.0 for Starlink phase 0,
                 RAAN_init = 240.0,
                 TA_init = 0.0,
                 ):


        # ------------------------------------------------------------------------
        # --------------------------CHOOSE-SATELLITE-SETUP------------------------
        # ------------------------------------------------------------------------

        # Set simulation start and end epochs
        self.sat_setup = sat_setup
        self.simulation_start_epoch = simulation_start_epoch
        self.simulation_end_epoch = simulation_end_epoch
        self.number_of_planes = number_of_planes
        self.number_sats_per_plane = number_sats_per_plane
        self.height_init = height_init
        self.inc_init = inc_init
        self.RAAN_init = RAAN_init
        self.TA_init = TA_init


    def propagate(self,
                  method = "tudat",
                  propagator = "Runge-Kutta 4",
                  fixed_step_size=10.0
                  ):

        if method == "tudat":

            # Initial array where all orbital trajectories are stored
            self.states_array_planes = {}
            self.dep_var_array_planes = {}

            #------------------------------------------------------------------------
            #--------------------------ENVIRONMENT-SETUP-----------------------------
            #------------------------------------------------------------------------

            # Create default body settings for "Earth"
            bodies_to_create = ["Earth", "Moon", "Mars"]
            Earth_radius = cons.R_earth #m

            # Create default body settings for bodies_to_create, with "Earth"/"J2000" as the global frame origin and orientation
            global_frame_origin = "Earth"
            global_frame_orientation = "J2000"
            # global_frame_orientation = "IAU_Earth"
            body_settings = environment_setup.get_default_body_settings(
                bodies_to_create, global_frame_origin, global_frame_orientation)


            # Create system of bodies (in this case only Earth)
            self.bodies = environment_setup.create_system_of_bodies(body_settings)

            # Add vehicle object to system of bodies
            self.bodies.create_empty_body("sat")

            #------------------------------------------------------------------------
            #--------------------------PROPAGATION-SETUP-----------------------------
            #------------------------------------------------------------------------

            # Define bodies that are propagated
            bodies_to_propagate = ["sat"]

            # Define central bodies of propagation
            central_bodies = ["Earth"]

            # Define accelerations acting on sat
            acceleration_settings_sat = dict(
                Earth=[propagation_setup.acceleration.point_mass_gravity()],
                # Mars =[propagation_setup.acceleration.point_mass_gravity()],
                # Moon = [propagation_setup.acceleration.point_mass_gravity()]
                )
            acceleration_settings = {"sat": acceleration_settings_sat}
            # Create acceleration models
            acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, bodies_to_propagate, central_bodies
            )

            # ------------------------------------------------------------------------
            # --------------LOOP-THROUGH-ALL-SATELLITES-WITHIN-CONSTELLATION----------
            # ------------------------------------------------------------------------

            # Propagate trajectories for each orbit
            for plane in range(self.number_of_planes):
                states_array_sats = {}
                dep_var_array_sats = {}
                for sat in range(self.number_sats_per_plane):

                    if self.sat_setup == "LEO_cons":
                        RAAN_init = self.RAAN_init[plane]
                        TA_init = self.TA_init[sat]
                    elif self.sat_setup == "LEO_1" or self.sat_setup == "GEO":
                        RAAN_init = self.RAAN_init
                        TA_init = self.TA_init

                    # Set initial conditions for the satellite that will be
                    # propagated in this simulation. The initial conditions are given in
                    # Keplerian elements and later on converted to Cartesian elements
                    earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter
                    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
                        gravitational_parameter=earth_gravitational_parameter,
                        semi_major_axis             =(self.height_init+cons.R_earth),
                        eccentricity                =0.0,
                        inclination                 =np.deg2rad(self.inc_init),
                        argument_of_periapsis       =np.deg2rad(235.7),
                        longitude_of_ascending_node =np.deg2rad(RAAN_init),
                        true_anomaly                =np.deg2rad(TA_init))

                    #------------------------------------------------------------------------
                    #--------------------PROPAGATOR/INTEGRATOR-SETTINGS----------------------
                    #------------------------------------------------------------------------

                    # Define dependent variables to save
                    dependent_variables_to_save = [
                        propagation_setup.dependent_variable.altitude("sat", "Earth"),
                        propagation_setup.dependent_variable.latitude("sat", "Earth"),
                        propagation_setup.dependent_variable.longitude("sat", "Earth"),
                        propagation_setup.dependent_variable.keplerian_state("sat", "Earth")
                    ]

                    # Create termination settings
                    termination_condition = propagation_setup.propagator.time_termination(self.simulation_end_epoch)

                    # Create propagation settings
                    propagator_settings = propagation_setup.propagator.translational(
                        central_bodies,
                        acceleration_models,
                        bodies_to_propagate,
                        initial_state,
                        termination_condition,
                        output_variables=dependent_variables_to_save
                    )

                    # Define type of propagator (Default is Runge Kutta 4)
                    # And create numerical integrator settings
                    if propagator == "Runge-Kutta 4":
                        integrator_settings = propagation_setup.integrator.runge_kutta_4(
                            self.simulation_start_epoch, fixed_step_size)
                    else:
                        integrator_settings = propagation_setup.integrator.runge_kutta_4(
                            self.simulation_start_epoch, fixed_step_size)

                #------------------------------------------------------------------------
                #-----------------------------PROPAGATE-ORBIT----------------------------
                #------------------------------------------------------------------------

                    # Create simulation object and propagate the dynamics
                    dynamics_simulator = numerical_simulation.SingleArcSimulator(
                        self.bodies, integrator_settings, propagator_settings)

                    # Extract the resulting state history and convert it to an ndarray
                    states = dynamics_simulator.state_history
                    dependent_variables = dynamics_simulator.dependent_variable_history
                    states_array = result2array(states)
                    dep_var_array = result2array(dependent_variables)

                    dt_array = np.zeros(len(states_array[:,0]))
                    for i in range(1, len(states_array[:, 0])):
                        dt_array[i] = states_array[i, 0] - states_array[i-1, 0]

                    states_array_sats[sat]  = states_array
                    dep_var_array_sats[sat] = dep_var_array

                self.states_array_planes[plane]  = states_array_sats
                self.dep_var_array_planes[plane] = dep_var_array_sats

            self.time = states_array[:, 0]
            return self.states_array_planes, self.dep_var_array_planes, self.time

# # ------------------------------------------------------------------------
# # ------------------------POSITIONING-AIRCRAFT----------------------------
# # ------------------------------------------------------------------------
#
# R_earth = 6367.435E3  # m
# height  = 10.0E3  # m
#
# initial_lon = np.deg2rad(20.0)
# initial_lat = np.deg2rad(45.0)
# initial_pos = np.array([np.cos(initial_lat) * np.cos(initial_lon),
#                         np.cos(initial_lat) * np.sin(initial_lon),
#                         np.sin(initial_lat)]) * (R_earth + height)
#
# AC_pos_array_ECEF = np.ones((len(states_array[:, 0]), 1)) * initial_pos
# AC_lon_array = initial_lon * np.ones((len(states_array[:, 0]), 1))
# AC_lat_array = initial_lat * np.ones((len(states_array[:, 0]), 1))

# rotation_matrix = frame_conversion.tnw_to_inertial_rotation_matrix(initial_pos)



# #------------------------------------------------------------------------
# #------------------------POST-PROCESS-&-VISUALIZATION--------------------
# #------------------------------------------------------------------------

# # plane = 7
# # for plane in range(1, number_of_planes+1):
# #     ax.plot(states_array_planes[plane][:, 1], states_array_planes[plane][:, 2], states_array_planes[plane][:, 3], linestyle='-.')
#
# ax.plot(states_array_planes[plane][:, 1], states_array_planes[plane][:, 2], states_array_planes[plane][:, 3], linestyle='-.')
#
# ax.scatter(AC_pos_array_ECEF[0, 0], AC_pos_array_ECEF[0, 1], AC_pos_array_ECEF[0, 2], label="aircraft", s=20, color='black')
# ax.scatter(0.0, 0.0, 0.0, label="Earth", marker='o', color='blue', s=100)
#
# # Add the legend and labels, then show the plot
# ax.legend()
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')


# plt.show()