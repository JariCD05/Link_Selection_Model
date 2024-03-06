# Import standard required tools
import random
from itertools import chain
import numpy as np
from matplotlib import pyplot as plt

# Import input parameters and helper functions
from input import *
from helper_functions import *

# Import classes from other files
from Link_geometry import link_geometry
from Atmosphere import attenuation, turbulence
from LCT import terminal_properties
from Link_budget import link_budget
from bit_level import bit_level
from channel_level import channel_level
from JM_Mission_Level import routing_network
from JM_matrices_test import Matrices 
from scipy.optimize import linprog

#-------LP Formulation----------



# Assuming you've already instantiated the routing_network object

#------------------------------------------------------------------------
#------------------------------TIME-VECTORS------------------------------
#------------------------------------------------------------------------
# Macro-scale time vector is generated with time step 'step_size_link'
# Micro-scale time vector is generated with time step 'step_size_channel_level'

t_macro = np.arange(0.0, (end_time - start_time), step_size_link)
samples_mission_level = len(t_macro)
t_micro = np.arange(0.0, interval_channel_level, step_size_channel_level)
samples_channel_level = len(t_micro)
#print('Macro-scale: Interval=', (end_time - start_time)/60, 'min, step size=', step_size_link, 'sec,  macro-scale steps=', samples_mission_level)
#print('Micro-scale: Interval=', interval_channel_level    , '  sec, step size=', step_size_channel_level*1000, 'msec, micro-scale steps=', samples_channel_level)

#print('----------------------------------------------------------------------------')

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


routing_network = routing_network(time=time)
routing_output, routing_total_output, mask = routing_network.routing(link_geometry.geometrical_output, time, step_size_link)


matrices_instance = Matrices(routing_network)

# Compute matrices and their normalizations
matrices_instance.compute_matrices()

# Plot the matrices
#matrices_instance.plot_matrices()

#------------------------------------------- LP formulation-------------------------------------

class LPFormulation:
    def __init__(self, matrices_instance):
        self.matrices_instance = matrices_instance
        self.num_satellites = matrices_instance.normalized_linktime_result.shape[0]
        self.num_time_steps = matrices_instance.normalized_linktime_result.shape[1]

    def formulate_lp(self):
        # Number of LP variables (equal to the number of matrices)
        num_matrices = 4

        # Flatten the matrices to create LP variables
        matrices_flattened = [
            self.matrices_instance.normalized_linktime_result.flatten(),
            self.matrices_instance.normalized_capacity_performance.flatten(),
            self.matrices_instance.normalized_BER_performance.flatten(),
            self.matrices_instance.normalized_propagation_latency.flatten()
        ]

        # Define the coefficients of the objective function (all set to 1)
        c = [1] * (self.num_time_steps * self.num_satellites * num_matrices)

        # Define the inequality constraints matrix (A_ub)
        # Each row of A_ub corresponds to a time step and satellite combination
        # Each column of A_ub corresponds to an LP variable (matrix)
        A_ub = np.zeros((self.num_time_steps * self.num_satellites, self.num_time_steps * self.num_satellites * num_matrices))
        for i in range(self.num_time_steps):
            for j in range(self.num_satellites):
                row_index = i * self.num_satellites + j
                for k in range(num_matrices):
                    col_index = (i * self.num_satellites + j) * num_matrices + k
                    A_ub[row_index, col_index] = matrices_flattened[k][i * self.num_satellites + j]

        # Define the inequality constraints right-hand side (b_ub)
        b_ub = np.ones(self.num_time_steps * self.num_satellites)

        # Define the bounds for the LP variables (0 <= x_i <= 1)
        bounds = [(0, 1)] * (self.num_time_steps * self.num_satellites * num_matrices)

        # Solve the LP problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        # Reshape the result to obtain the selected matrices
        selected_matrices = np.reshape(result.x, (num_matrices, self.num_satellites, self.num_time_steps))

        # Print the result
        print("Selected Matrices:")
        for i in range(num_matrices):
            print(f"Matrix {i+1}:")
            print(selected_matrices[i])

if __name__ == "__main__":
    # Assuming you've already instantiated the Matrices class and computed the matrices
    matrices_instance = Matrices()
    matrices_instance.compute_matrices()

    # Instantiate LPFormulation class with the matrices_instance
    lp_formulation_instance = LPFormulation(matrices_instance)

    # Start formulating LP
    lp_formulation_instance.formulate_lp()
