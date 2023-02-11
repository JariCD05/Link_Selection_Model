import Link_budget as LB
import Turbulence as turb
import network as network
import Terminal as term
import Atmosphere as atm
from constants import *
from helper_functions import *
import Constellation as SC
from Link_geometry import link_geometry

import numpy as np
import sqlite3
from matplotlib import pyplot as plt
from tudatpy.kernel import constants as cons_tudat


#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------

# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
link_geometry = link_geometry(constellation_type=constellation_type)

# link_geometry.setup()
link_geometry.propagate(start_time, end_time)
link_geometry.geometrical_outputs()

# Initiate time for entire communicatin period
time = link_geometry.time

#------------------------------------------------------------------------
#---------------------SATELLITE-NETWORK-STRATEGY-------------------------
#------------------------------------------------------------------------

# The network class is initiated here.
# This class takes the geometrical output of the link_geometry class (which is the output of the aircraft and all satellites)

# It also saves the total time and latencies, due to handovers and acquisition
network = network.network(time)

# The hand_over_strategy function computes a sequence of links between the aircraft and satellites.
# One method is based on minimum required number of handovers. Here, the satellite with the lowest, increasing, elevation angle is found and
# acquisition & communication is started, until the elevation angle decreases below the minimum elevation angle (20 degrees).
# Then a new satellite is found again, until termination time.
# This function thus filters the input and outputs arrays of all geometrical variables over the time interval of dimension 2.
geometrical_output = network.hand_over_strategy(number_of_planes,
                                                number_sats_per_plane,
                                                link_geometry.geometrical_output,
                                                time)


# Here, the geometrical output of the communication link route is mapped with the database.
# The output is an array of performance values for each timestep of dimension 2
performance_output = network.save_data(geometrical_output['elevation'])

# Here, the data rate is converted from a constant value to a variable value
# This is done by setting the fluctuating received power equal to the threshold power and
# computing the corresponding data rate that is needed to increase/decrease this power.
network.variable_data_rate()

# # ------------------------------------------------------------------------
# # ------------------------PLOT-RESULTS-(OPTIONAL)-------------------------
# # ------------------------------------------------------------------------

plot_time_index = 10
link_geometry.plot(type = 'satellite sequence', sequence=geometrical_output['pos SC'])
link_geometry.plot(type='angles')
link_geometry.plot()
# # link_budget.plot(D_ac, D_ac, radial_pos_t, radial_pos_r)

# turbulence_uplink.plot(t = t, plot="scint pdf")
# # turbulence_uplink.plot(t = t, plot="scint vs. zenith", zenith=zenith_angles_a)

# terminal_sc.plot(t = t, plot="pointing")
# terminal_sc.plot(t = t, plot="BER & SNR")

plt.show()