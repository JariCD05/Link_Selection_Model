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
from JM_Visibility_matrix import routing_network
from JM_matrices_test import Matrices 


#-------LP Formulation----------
from pulp import LpMaximize, LpProblem, LpVariable

from pulp import LpMaximize, LpProblem, LpVariable, lpSum

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
        # Number of matrices (LP variables)
        num_matrices = 4

        # Flatten the matrices to create LP variables
        matrices_flattened = [
            self.matrices_instance.normalized_linktime_result.flatten(),
            self.matrices_instance.normalized_capacity_performance.flatten(),
            self.matrices_instance.normalized_BER_performance.flatten(),
            self.matrices_instance.normalized_propagation_latency.flatten()
        ]

        # Create LP problem
        prob = LpProblem("LP Problem", LpMaximize)

        # Define LP variables (one for each matrix)
        variables = [LpVariable(f"Matrix_{i}", lowBound=0, upBound=1) for i in range(num_matrices)]

        # Define objective function
        prob += lpSum(variables)

        # Add constraints
        for i in range(self.num_time_steps):
            for j in range(self.num_satellites):
                constraint_expr = lpSum(variables[k] * matrices_flattened[k][i * self.num_satellites + j] for k in range(num_matrices)) <= 1
                prob += constraint_expr

        # Solve LP problem
        prob.solve()

        # Print selected matrices
        print("Selected Matrices:")
        for v in prob.variables():
            print(f"{v.name}: {v.varValue}")

if __name__ == "__main__":
    # Assuming you've already instantiated the Matrices class and computed the matrices
    matrices_instance = Matrices()
    matrices_instance.compute_matrices()

    # Instantiate LPFormulation class with the matrices_instance
    lp_formulation_instance = LPFormulation(matrices_instance)

    # Start formulating LP
    lp_formulation_instance.formulate_lp()
