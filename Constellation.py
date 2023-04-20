# Load standard modules
import numpy as np
# Uncomment the following to make plots interactive
# %matplotlib widget
from matplotlib import pyplot as plt
from matplotlib import animation

# Load tudatpy modules
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion, frame_conversion
from tudatpy.kernel import constants
from tudatpy.util import result2array

from input import *

# Load spice kernels
spice.load_standard_kernels()

class constellation:
    def __init__(self,
                 sat_setup = "LEO_1",
                 number_of_planes = 72,
                 number_sats_per_plane = 1,
                 height_init = 550.0E3,  # 550.0E3 for Starlink phase 0
                 inc_init = 53.0,  # 53.0 for Starlink phase 0,
                 RAAN_init = 120.0,
                 TA_init = 0.0,
                 ECC_init = 0.0,
                 omega_init = 120.0
                 ):


        # ------------------------------------------------------------------------
        # --------------------------CHOOSE-SATELLITE-SETUP------------------------
        # ------------------------------------------------------------------------

        # Set simulation start and end epochs
        self.sat_setup = sat_setup
        self.number_of_planes = number_of_planes
        self.number_sats_per_plane = number_sats_per_plane
        self.height_init = height_init
        self.inc_init = inc_init
        self.RAAN_init = RAAN_init
        self.TA_init = TA_init
        self.ECC_init = ECC_init
        self.omega_init = omega_init


    def propagate(self,
                  AC_time: np.array,
                  method = "tudat",
                  ):

        simulation_start_epoch = AC_time[0]
        simulation_end_epoch = AC_time[-1]

        if method == "tudat":

            # Initial array where all orbital trajectories are stored
            self.states_array_planes = {}
            self.dep_var_array_planes = {}

            #------------------------------------------------------------------------
            #--------------------------ENVIRONMENT-SETUP-----------------------------
            #------------------------------------------------------------------------

            # Create default body settings for "Earth"
            bodies_to_create = ["Earth", "Moon", "Mars", "Sun"]
            Earth_radius = R_earth #m

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
            self.bodies_to_propagate = ["sat"]

            # Define central bodies of propagation
            self.central_bodies = ["Earth"]

            # Define accelerations acting on sat
            acceleration_settings_sat = dict(
                Earth=[propagation_setup.acceleration.point_mass_gravity()],
                # Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2,2)],
                # Mars =[propagation_setup.acceleration.point_mass_gravity()],
                # Moon = [propagation_setup.acceleration.point_mass_gravity()]
                )
            acceleration_settings = {"sat": acceleration_settings_sat}
            # Create acceleration models
            self.acceleration_models = propagation_setup.create_acceleration_models(
                self.bodies, acceleration_settings, self.bodies_to_propagate, self.central_bodies
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
                    self.earth_gravitational_parameter = self.bodies.get("Earth").gravitational_parameter
                    initial_state = element_conversion.keplerian_to_cartesian_elementwise(
                        gravitational_parameter     =self.earth_gravitational_parameter,
                        semi_major_axis             =(self.height_init + R_earth),
                        eccentricity                =np.deg2rad(self.ECC_init),
                        inclination                 =np.deg2rad(self.inc_init),
                        argument_of_periapsis       =np.deg2rad(self.omega_init),
                        longitude_of_ascending_node =np.deg2rad(RAAN_init),
                        true_anomaly                =np.deg2rad(TA_init))
                        # gravitational_parameter = self.earth_gravitational_parameter,
                        # semi_major_axis = self.height_init + R_earth,
                        # eccentricity = np.deg2rad(self.ECC_init),
                        # inclination = np.deg2rad(self.inc_init),
                        # argument_of_periapsis = np.deg2rad(self.omega_init),
                        # longitude_of_ascending_node = np.deg2rad(RAAN_init),
                        # true_anomaly = np.deg2rad(TA_init))

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
                    self.termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

                    # Create propagation settings
                    self.propagator_settings = propagation_setup.propagator.translational(
                        self.central_bodies,
                        self.acceleration_models,
                        self.bodies_to_propagate,
                        initial_state,
                        self.termination_condition,
                        output_variables=dependent_variables_to_save
                    )

                    # Define type of propagator (Default is Runge Kutta 4)
                    # And create numerical integrator settings
                    if integrator == "Runge Kutta 4":
                        coefficient_set = propagation_setup.integrator.rkf_45
                    elif integrator == "Runge Kutta 78":
                        coefficient_set = propagation_setup.integrator.rkf_78

                    self.integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
                        simulation_start_epoch, step_size_SC, coefficient_set,
                        step_size_SC, step_size_SC,
                        np.inf, np.inf)

                #------------------------------------------------------------------------
                #-----------------------------PROPAGATE-ORBIT----------------------------
                #------------------------------------------------------------------------

                    # Create simulation object and propagate the dynamics
                    dynamics_simulator = numerical_simulation.SingleArcSimulator(
                        self.bodies, self.integrator_settings, self.propagator_settings)

                    # Extract the resulting state history and convert it to an ndarray
                    states = dynamics_simulator.state_history
                    dependent_variables = dynamics_simulator.dependent_variable_history

                # ------------------------------------------------------------------------
                # -------------------------------INTERPOLATE------------------------------
                # ------------------------------------------------------------------------
                    interpolator_settings = interpolators.lagrange_interpolation(8)
                    state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                        states, interpolator_settings)
                    dep_var_interpolator = interpolators.create_one_dimensional_vector_interpolator(
                        dependent_variables, interpolator_settings)

                    states = dict()
                    dependent_variables = dict()
                    for epoch in AC_time:
                        states[epoch] = state_interpolator.interpolate(epoch)
                        dependent_variables[epoch] = dep_var_interpolator.interpolate(epoch)

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
            # CHECK IF THIS IS CORRECT, OTHERWISE:
            # self.time = AC_time
            return self.states_array_planes, self.dep_var_array_planes, self.time

    def verification(self,
                  simulation_start_epoch: float,
                  simulation_end_epoch: float,):
        integrator_ver = "Runge Kutta 78"

        # Set up intitial condition and termination condition
        initial_state = element_conversion.keplerian_to_cartesian_elementwise(
            gravitational_parameter=self.earth_gravitational_parameter,
            semi_major_axis=(self.height_init + R_earth),
            eccentricity=np.deg2rad(self.ECC_init),
            inclination=np.deg2rad(self.inc_init),
            argument_of_periapsis=np.deg2rad(self.omega_init),
            longitude_of_ascending_node=np.deg2rad(self.RAAN_init),
            true_anomaly=np.deg2rad(self.TA_init))
        termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)

        # CREATE TEST MODEL
        dependent_variables_to_save = [propagation_setup.dependent_variable.altitude("sat", "Earth")]

        propagator_settings = propagation_setup.propagator.translational(
            self.central_bodies,
            self.acceleration_models,
            self.bodies_to_propagate,
            initial_state,
            self.termination_condition,
            output_variables=dependent_variables_to_save)

        dynamics_simulator = numerical_simulation.SingleArcSimulator(
            self.bodies, self.integrator_settings, propagator_settings)
        states = dynamics_simulator.state_history
        states = result2array(states)
        dependent_variables = dynamics_simulator.dependent_variable_history
        dependent_variables = result2array(dependent_variables)
        time = states[:, 0]
        # CREATE BENCHMARK MODEL

        # Define integrator settings for benchmark
        time_step_benchmark = step_size_SC
        coefficient_set = propagation_setup.integrator.rkf_78
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            simulation_start_epoch, time_step_benchmark, coefficient_set,
            time_step_benchmark, time_step_benchmark, np.inf, np.inf)

        # Add radiation pressure interface to vehicle
        spacecraft_mass = 100.0
        spacecraft_reference_area = 5
        radiation_pressure_coefficient = 1.2
        radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            "Sun", spacecraft_reference_area, radiation_pressure_coefficient, self.central_bodies)
        environment_setup.add_radiation_pressure_interface(
            self.bodies, "sat", radiation_pressure_settings)
        self.bodies.get_body("sat").set_constant_mass(spacecraft_mass)

        # Create aerodynamic coefficients interface
        drag_coefficient = 1.2
        aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
            spacecraft_reference_area, [drag_coefficient, 0, 0])
        environment_setup.add_aerodynamic_coefficient_interface(
            self.bodies, "sat", aero_coefficient_settings)

        # Define accelerations for benchmark
        acceleration_settings_benchmark = dict(
            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(32, 32)],
                   # propagation_setup.acceleration.aerodynamic()],
            Mars=[propagation_setup.acceleration.point_mass_gravity()],
            Moon=[propagation_setup.acceleration.point_mass_gravity()],
            # Sun =[propagation_setup.acceleration.point_mass_gravity(),
            #       propagation_setup.acceleration.cannonball_radiation_pressure()],
        )

        acceleration_settings_benchmark = {"sat": acceleration_settings_benchmark}
        # Create acceleration models
        acceleration_models_benchmark = propagation_setup.create_acceleration_models(
            self.bodies, acceleration_settings_benchmark, self.bodies_to_propagate, self.central_bodies
        )

        # Create propagation settings
        propagator_settings_benchmark = propagation_setup.propagator.translational(
            self.central_bodies,
            acceleration_models_benchmark,
            self.bodies_to_propagate,
            initial_state,
            self.termination_condition,
            output_variables=dependent_variables_to_save)

        # Propagate benchmark dynamics
        benchmark_dynamics_simulator = numerical_simulation.SingleArcSimulator(
            self.bodies, integrator_settings, propagator_settings_benchmark)
        states_benchmark = benchmark_dynamics_simulator.state_history
        states_benchmark = result2array(states_benchmark)
        dependent_variables_benchmark = benchmark_dynamics_simulator.dependent_variable_history
        dependent_variables_benchmark = result2array(dependent_variables_benchmark)
        time_benchmark = states_benchmark[:, 0]

        # Create interpolator for benchmark results
        interpolator_settings = interpolators.lagrange_interpolation(8)
        benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_dynamics_simulator.state_history, interpolator_settings)

        benchmark_interpolator_height = interpolators.create_one_dimensional_vector_interpolator(
            benchmark_dynamics_simulator.dependent_variable_history, interpolator_settings)

        benchmark_difference = dict()
        benchmark_difference_height = dict()

        # print(1/0)
        for epoch in dynamics_simulator.state_history.keys():
            benchmark_difference[epoch] = dynamics_simulator.state_history[epoch] - benchmark_interpolator.interpolate(epoch)

        for epoch in dynamics_simulator.state_history.keys():
            benchmark_difference_height[epoch] = dynamics_simulator.dependent_variable_history[epoch] - benchmark_interpolator_height.interpolate(epoch)

        diff_benchmark = np.vstack(list(benchmark_difference.values()))
        diff_benchmark_height = np.vstack(list(benchmark_difference_height.values()))
        r = np.sqrt(states[:, 0] ** 2 + states[:, 1] ** 2 + states[:, 2] ** 2) - h_AC
        r_bench = np.sqrt(states_benchmark[:, 0] ** 2 + states_benchmark[:, 1] ** 2 + states_benchmark[:, 2] ** 2) - h_AC
        error_r = np.sqrt(diff_benchmark[:, 0] ** 2 + diff_benchmark[:, 1] ** 2 + diff_benchmark[:, 2] ** 2)
        error_h =  diff_benchmark_height
        time_hrs = [t / 3600 for t in time]
        time_hrs_bench = [t / 3600 for t in time_benchmark]
        T_fs = (wavelength / (4 * np.pi * r)) ** 2
        T_fs_bench = (wavelength / (4 * np.pi * r_bench)) ** 2
        delta_T_fs = abs(T_fs - T_fs_bench)

        fig1, ax = plt.subplots(3, 1, figsize=(20, 17))
        fig1.suptitle('Comparison with benchmark')
        ax[0].plot(time_hrs, error_r)
        ax[0].set_xlim([min(time_hrs), max(time_hrs)])
        ax[0].set_ylabel('$\Delta$ r (m)')
        ax[0].grid()

        ax[1].plot(time_hrs, error_h)
        ax[1].set_xlim([min(time_hrs), max(time_hrs)])
        ax[1].set_ylabel('$\Delta$ h (m)')
        ax[1].grid()

        ax[2].plot(time_hrs, W2dB(T_fs), label='T_fs of test model')
        ax[2].plot(time_hrs_bench, W2dB(T_fs_bench), label='T_fs of benchmark model')
        ax[2].plot(time_hrs, W2dB(delta_T_fs), label= '$\Delta$T_fs over $\Delta$r')
        ax[2].set_xlim([min(time_hrs), max(time_hrs)])
        ax[2].set_ylabel('$\Delta$ Free space loss (dB)')
        ax[2].grid()
        ax[2].legend()


        # ax[2].plot(time / 3600, states_benchmark[:, 1], states_benchmark[:, 2], states_benchmark[:, 3],
        #            label='Benchmark model, integrator: '+str(integrator_ver))
        # ax[2].set_xlim([min(time), max(time)])
        # ax[2].set_ylabel('$\Delta$ r (m)')


        ax[1].set_xlabel('Time (hours)')

        plt.show()


