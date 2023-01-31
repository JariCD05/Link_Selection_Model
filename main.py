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
#----------------------------IMPORT-DATABASE-----------------------------
#------------------------------------------------------------------------

data_elevation = get_data('elevation')


#------------------------------------------------------------------------
#-----------------------------LINK-GEOMETRY------------------------------
#------------------------------------------------------------------------

# Initiate LINK GEOMETRY class, with inheritance of AIRCRAFT class and CONSTELLATION class
link_geometry = link_geometry(constellation_type=constellation_type)

# link_geometry.setup()
link_geometry.propagate(start_time, end_time, step_size_dim2, h_AC)
link_geometry.geometrical_outputs()

link_geometry.plot(type='angles')
link_geometry.plot()
# plt.show()

# Initiate time for entire communicatin period
time = link_geometry.time

# #------------------------------------------------------------------------
# #---------------------------INITIATE-TERMINALS---------------------------
# #------------------------------------------------------------------------
#
# terminal_ac = term.terminal_properties(D = D_ac,
#                                        eff_quantum = eff_quantum_ac,
#                                        BW = BW_ac,
#                                        modulator = mod_ac,
#                                        detection= detection_ac,
#                                        FOV = FOV_ac,
#                                        delta_wavelength=delta_wavelength_ac,
#                                        M = M_ac,
#                                        F = F_ac,
#                                        )
#
# terminal_sc = term.terminal_properties(D = D_sc,
#                                        eff_quantum = eff_quantum_sc,
#                                        BW = BW_sc,
#                                        modulator = mod_sc,
#                                        detection = detection_sc,
#                                        FOV = FOV_sc,
#                                        delta_wavelength=delta_wavelength_sc,
#                                        M = M_sc,
#                                        F = F_sc,
#                                        )

# Initiate network classe
# This class saves all performance metrics that are modeled throughout the code
# It also saves the total time and latencies, due to handovers and acquisition
network = network.network(time)


# ------------------------------------------------------------------------
# --------------------------HAND-OVER-STRATEGY----------------------------
# ------------------------------------------------------------------------
# A SATELLITE IS CHOSEN FROM ABOVE HANDOVER FUNCTION
# THIS IS PROPAGATED UNTIL ELEVATION ANGLE DECREASES BELOW MINIMUM ELEVATION THRESHOLD, THEN HANDOVER IS INITIATED AGAIN
satellite_sequence = network.hand_over_strategy(number_of_planes,
                                               number_sats_per_plane,
                                               link_geometry.geometrical_output,
                                               time)


# This data is filtered with the handover_strategy() function
# This function selects communication windows and filters data from GEOMETRICAL_OUTPUT to data that only contains values during the communication windows with the current satellite
heights_AC = link_geometry.heights_AC
pos_SC = satellite_sequence['pos SC']
heights_SC = satellite_sequence['heights SC']
vel_SC = satellite_sequence['vel SC']
ranges = satellite_sequence['ranges']
slew_rates = satellite_sequence['slew rates']
angular_rates = satellite_sequence['angular rates']
zenith = satellite_sequence['zenith']
elevation = satellite_sequence['elevation']
azimuth = satellite_sequence['azimuth']
radial = satellite_sequence['radial']

indices = np.nonzero(ranges)
index = indices[0][10]

link_geometry.plot(type = 'satellite sequence', sequence=pos_SC)
link_geometry.print(0, 0, index=index)

network.save_data(data_elevation, elevation)

# # ------------------------------------------------------------------------
# # --------------------------INITIATE-TURBULENCE---------------------------
# # ------------------------------------------------------------------------
# # SET UP TURBULENCE MODEL
# turbulence = atm.turbulence(
#     range=ranges,
#     D_r=D_sc,
#     D_t=D_ac,
#     link='up')
#
# slew_rate = np.mean(abs(slew_rates))
# windspeed_rms = turbulence.windspeed(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
# Cn_HVB = turbulence.Cn(turbulence_model=turbulence_model)
# r0 = turbulence.r0(zenith)

# turbulence_uplink.print()

# # ------------------------------------------------------------------------
# # -------------------------LINK-BUDGET-ANALYSIS---------------------------
# # ------------------------------------------------------------------------
# # SET UP LINK BUDGET (POWER) MODEL
# link_budget = LB.link_budget(ranges,
#                              P_t=P_ac,
#                              eff_t=eff_ac,
#                              eff_r=eff_sc,
#                              eff_coupling=eff_coupling_sc,
#                              D_t=D_sc,
#                              D_r=D_ac,
#                              angle_pe_t=angle_pe_ac,
#                              angle_pe_r=angle_pe_sc,
#                              var_pj_t=var_pj_ac,
#                              var_pj_r=var_pj_sc,
#                              n=n_index)
#
# # INCLUDE BEAM SPREAD EFFECT
# link_budget.beam_spread(r0=r0)
#
# # SET UP LINK BUDGET (POWER and INTENSITY)
# P_r_0 = link_budget.P_r_func()
# I_r_0 = link_budget.I_r_0_func()
#
# # SET UP GAUSSIAN BEAM PROFILE FOR TRANSMITING BEAM
# radial_pos_t = np.linspace(-link_budget.w0, link_budget.w0, 50)
# I_t = gaussian_beam(link_budget.I_t_0, radial_pos_t, D_ac, D_ac)
#
# link_budget.print(elevation=elevation, index=index, t=time)
# # link_budget.plot(t=time, ranges=ranges)
# # plt.show()
#
# # Threshold SNR at receiver
# noise_sh_0 = terminal_sc.noise(noise_type="shot", P_r=P_r_0)
# noise_th_0 = terminal_sc.noise(noise_type="thermal")
# noise_bg_0 = terminal_sc.noise(noise_type="background", P_r=P_r_0, I_sun=I_sun)
# SNR_0_thres_sc, P_r_0_thres_sc, Np_0_quantum_limit_sc = terminal_sc.threshold()
#
#
# SNR = terminal_sc.SNR_func(P_r_0)
# BER = terminal_sc.BER_func()
# terminal_sc.print(type="terminal", index=index)
# terminal_sc.print(type="noise", index=index)
# print(SNR[index], BER[index])
# print(noise_sh_0[index], noise_th_0, noise_bg_0)
# print(noise_sh_0[index]/(noise_sh_0[index]+noise_th_0+noise_bg_0))
# ---------------------------------------------------------------------------------------------
# ------------------------------CONNECT-DIMENSION-1--&--DIMENSION-2----------------------------
# ----------------------------TURBULENCE----POINTING-ERROR----BER/SNR--------------------------
# ---------------------------------------------------------------------------------------------
# 1.    RETRIEVE STATISTICAL PARAMETERS FROM DIMENSION 1
# 2.    COMPUTE POWER AND INTENSITY FLUCTUATIONS
# 3.    COMPUTE NOISE FLUCTUATIONS
# 4.    DETERMINE RECEIVER SENSITIVITY (THRESHOLD)
# 5.    COMPUTE BER & FADE STATISTICS

# h_avg_array, noise_array, BER_avg_array, fade_time = connect_dim1_dim2(terminal_sc,
#                                                                        P_r_0, I_r_0,
#                                                                        zenith_angles_dim1,
#                                                                        zenith,
#                                                                        dim_1_results)
#
# P_r_0_current = P_r_0[index]
# I_r_0_current = I_r_0[index]
#
# # Retrieve statistical parameters for
# if current_zenith < zenith_angles_dim1[2]:
#     h_avg = h_avg_array[2]
#     noise = noise_array[2]
#     BER_avg = BER_avg_array[2]
# elif current_elevation < zenith_angles_dim1[1]:
#     h_avg = h_avg_array[1]
#     noise = noise_array[1]
#     BER_avg = BER_avg_array[1]
# else:
#     h_avg = h_avg_array[0]
#     noise = noise_array[0]
#     BER_avg = BER_avg_array[0]








# # print(h_avg, ' at ', np.rad2deg(current_elevation), np.rad2deg(current_zenith))
# network.save_data(h_avg, BER_avg, P_r_0_current, I_r_0_current, latency, index)

# # -----------------------------------------------------------------------
# #----------------------------------PLOT----------------------------------
# #------------------------------------------------------------------------
#
# link_geometry.plot()
# # link_budget.plot(D_ac, D_ac, radial_pos_t, radial_pos_r)
#
# turbulence_uplink.plot(t = t, plot="scint pdf")
# # turbulence_uplink.plot(t = t, plot="scint vs. zenith", zenith=zenith_angles_a)
# terminal_sc.plot(t = t, plot="pointing")
# terminal_sc.plot(t = t, plot="BER & SNR")
#
# plt.show()