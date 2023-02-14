import Link_budget as LB
import network as network
from input import *
from Link_geometry import link_geometry
import Atmosphere as atm

from matplotlib import pyplot as plt


#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------

# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
link_geometry = link_geometry(constellation_type=constellation_type)

# link_geometry.setup()
link_geometry.propagate(start_time, end_time)
link_geometry.geometrical_outputs()

# Initiate time for entire communication period
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

#------------------------------------------------------------------------
#-------------------------------TURBULENCE-------------------------------
#------------------------------------------------------------------------

# The turbulence class is initiated here. Inside the turbulence class, there are multiple functions that are run.
turbulence_main = atm.turbulence(range=geometrical_output['ranges'], link=link)

# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
# With Cn^2 and r0, the variances for scintillation and beam wander are computed
turbulence_main.windspeed(slew=np.mean(geometrical_output['slew rates']),
                          Vg=speed_AC,
                          wind_model_type=wind_model_type)
turbulence_main.Cn_func(turbulence_model=turbulence_model)
r0 = turbulence_main.r0_func(geometrical_output['elevation'])

# ------------------------------------------------------------------------
# -------------------------LINK-BUDGET-ANALYSIS---------------------------
# ------------------------------------------------------------------------
# The link budget class is initiated here.
link_budget_main = LB.link_budget(geometrical_output['ranges'])
w_r = link_budget_main.beam_spread(r0)

# The power and intensity at the receiver is computed from the link budget.
# All gains, losses and efficiencies are given in the link budget class.
P_r_0 = link_budget_main.P_r_0_func()
I_r_0 = link_budget_main.I_r_0_func()
data_metrics = ['elevation', 'P_r' 'P_r_0', 'h_tot', 'h_att', 'h_scint', 'h_pj', 'h_bw', 'SNR', 'BER',
                'number of fades', 'fade time', 'fractional fade time',
                'P_r threshold', 'SNR threshold', 'Np threshold', 'Data rate', 'Np', 'noise']
# Here, dynamic contributions and the power threshold are added to the static link budget
P_r = performance_output[:, 1]
P_r_0_db = performance_output[:, 2]
Np_r = performance_output[:, -2]
h_att = performance_output[:, 4]
h_scint = performance_output[:, 5]
h_pj = performance_output[:, 6]
h_bw = performance_output[:, 7]
P_r_threshold = performance_output[:, -6]
Np_r_threshold = performance_output[:, -4]
link_budget_main.dynamic_contributions(h_scint,
                                       h_pj,
                                       h_bw,
                                       P_r,
                                       P_r_threshold,
                                       Np_r,
                                       Np_r_threshold)

# print('test-----------------------')
# for i in range(len(P_r_0)):
#     print(P_r_0[i], P_r_0_db[i], W2dB(P_r_threshold[i]), np.rad2deg(geometrical_output['elevation'][i]), performance_output[i, 0])

# With the dynamic contributions and the threshold power, the resulting link margin is computed
link_budget_main.link_margin()



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

link_budget_main.print(t=time, index = plot_time_index, type="link budget", elevation = geometrical_output['elevation'])

plt.show()