import numpy as np
import csv
import sqlite3

from constants import *
from helper_functions import *
import Atmosphere as atm
import Terminal as LCT
import Link_budget as LB
from matplotlib import pyplot as plt
from PDF import distributions


#---------------------------------------------------------------------------------------------
#--------------------------------------DIMENSION-1-SIMULATION---------------------------------
#----------------------------TURBULENCE----POINTING-ERROR----BER/SNR--------------------------
#---------------------------------------------------------------------------------------------


# SET TIME INTERVAL OF THE SIMULATION OF DIMENSION 1
duration = 100.0
dt = 1.0e-4

t = np.arange(0.0, duration, dt)
steps = len(t)
# steps = 100
# N = 100
# t_dim0 = np.linspace(0.0, 100000, N)
# steps_dim0 = len(t_dim0)

zenith_angles_dim1 = np.flip(np.arange(0.0,
                                       zenith_max,
                                       np.deg2rad(1.0)))

elevation_angles_dim1 = np.pi/2 - zenith_angles_dim1

zenith_angles_a = np.arcsin( 1/ n_index * np.sin(zenith_angles_dim1) )
# print(zenith_angles_dim1, R_earth)
ranges_dim1 = np.sqrt((h_SC-h_AC)**2 + 2*h_SC*R_earth  + R_earth**2 * np.cos(zenith_angles_dim1)**2) \
                     - (R_earth+h_AC)*np.cos(zenith_angles_dim1)
# ranges_dim1 =
refractions     = np.array((1.0, 1.03, 1.3))
slew_rate = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)

#------------------------------------------------------------------------
#--------------------------TURBULENCE-ANALYSIS---------------------------
#------------------------------------------------------------------------

# SET UP TURBULENCE MODEL
turbulence_dim1 = atm.turbulence(
     range=ranges_dim1,
     D_r=D_sc,
     D_t=D_ac,
     link='up')

windspeed_rms = turbulence_dim1.windspeed(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
Cn_HVB    = turbulence_dim1.Cn(turbulence_model=turbulence_model)
r0 = turbulence_dim1.r0(zenith_angles_dim1)
var_rytov = turbulence_dim1.var_rytov(zenith_angles_a)
var_scint = turbulence_dim1.var_scint(PDF_type="lognormal", zenith_angles=zenith_angles_a)
var_bw    = turbulence_dim1.var_bw(D_t = D_ac, zenith_angles=zenith_angles_a)

# Simulate PDF for scintillation and beam wander
h_scint = turbulence_dim1.PDF(ranges=ranges_dim1, D_t=D_ac, PDF_type="lognormal", steps=steps, zenith_angles=zenith_angles_a)
r_bw, h_bw = turbulence_dim1.PDF(ranges=ranges_dim1, D_t=D_ac, PDF_type="rayleigh", steps=steps, zenith_angles=zenith_angles_a, effect="beam wander")

# ------------------------------------------------------------------------
# --------------------------ATTENUATION-ANALYSIS--------------------------
# ------------------------------------------------------------------------
# SET UP ATTENUATION MODEL
attenuation = atm.attenuation(range_link=ranges_dim1)
T_ext = attenuation.T_ext(zenith_angles=zenith_angles_dim1, type="standard_atmosphere", steps=steps)

# att.print()

#------------------------------------------------------------------------
#-------------------------POINTING-ERROR-ANALYSIS------------------------
#------------------------------------------------------------------------

terminal_dim1 = LCT.terminal_properties(D = D_ac,
                                       eff_quantum = eff_quantum_ac,
                                       BW = BW_ac,
                                       modulator = mod_ac,
                                       detection= detection_ac,
                                       FOV = FOV_ac,
                                       delta_wavelength=delta_wavelength_ac,
                                       M = M_ac,
                                       F = F_ac,
                                       )

# Pointing error distribution
r_pe, h_pe = terminal_dim1.pointing(zenith_angles_dim1,
                                  ranges_dim1,
                                  pointing_error=angle_pe_ac,
                                  jitter_variance=var_pj_ac,
                                  dist_type="gaussian",
                                  steps=steps,
                                  D_t=D_ac,
                                  angle_divergence=angle_div_ac,
                                  r0 = r0
                                  )

# SUM POINTING JITTER AND BEAM WANDER DISPLACEMENT FLUCTUATIONS
r_tot = r_pe #+ r_bw
# h_tot = gaussian_beam(h_scint, np.abs(r_tot), D_ac, D_ac)
# REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
h_tot = h_scint * h_pe

for i in range(len(h_tot)):
    for j in range(len(h_tot[0,:])):
        if h_tot[i, j] < 0.0:
            h_tot[i,j] = 0.0


# def gg_pdf(self, alpha, beta, steps):

# fig_h, axs = plt.subplots(1, 1)
# axs.scatter(terminal_dim1.angle_pe_x, terminal_dim1.angle_pe_y)
#
# axs[0].set_title(f'Pointing error displacement - Rayleigh distribution')
# axs[0].set_ylabel('Probability [-]')
# axs[0].set_xlabel('Radial angle error [microrad]')
# axs[1].plot(t, h_scint[0])
# axs[2].plot(t, r_tot[0])


# ------------------------------------------------------------------------
# -------------------------LINK-BUDGET-ANALYSIS---------------------------
# ------------------------------------------------------------------------
# SET UP LINK BUDGET (POWER) MODEL
link_budget_dim1 = LB.link_budget(ranges_dim1,
                                 P_t=P_ac,
                                 eff_t=eff_ac,
                                 eff_r=eff_sc,
                                 eff_coupling=eff_coupling_sc,
                                 D_t=D_sc,
                                 D_r=D_ac,
                                 angle_pe_t=angle_pe_ac,
                                 angle_pe_r=angle_pe_sc,
                                 var_pj_t=var_pj_ac,
                                 var_pj_r=var_pj_sc,
                                 n=n_index)


# INCLUDE BEAM SPREAD EFFECT
link_budget_dim1.beam_spread(r0=r0)

# SET UP LINK BUDGET (POWER and INTENSITY)
P_r_0 = link_budget_dim1.P_r_func()
I_r_0 = link_budget_dim1.I_r_0_func()

# SET UP GAUSSIAN BEAM PROFILE FOR TRANSMITING BEAM
radial_pos_t = np.linspace(-link_budget_dim1.w0, link_budget_dim1.w0, 50)
I_t = gaussian_beam(link_budget_dim1.I_t_0, radial_pos_t, D_ac, D_ac)

# link_budget_dim1.print(elevation=elevation_angles_dim1, index=1, t=np.array([1,2]))

# Threshold SNR at receiver
# noise_sh_0 = terminal_dim1.noise(noise_type="shot", P_r=P_r_0)
# noise_th_0 = terminal_dim1.noise(noise_type="thermal")
# noise_bg_0 = terminal_dim1.noise(noise_type="background", P_r=P_r_0, I_sun=I_sun)
# SNR_0_thres_sc, P_r_0_thres_sc, Np_0_quantum_limit_sc = terminal_dim1.threshold()

# turbulence_dim1.print()
# terminal_dim1.print(type="terminal", index=1)
# terminal_dim1.print(type="noise", index=1)

# SNR_0 = terminal_dim1.SNR_func(P_r_0)
# BER_0 = terminal_dim1.BER_func()

#------------------------------------------------------------------------
#-----------------------FLUCTUATING-SNR-&-BER-VALUES---------------------
#------------------------------------------------------------------------
P_r = np.zeros((len(zenith_angles_dim1), steps))

for i in range(len(zenith_angles_dim1)):
    P_r[i] = P_r_0[i] * h_tot[i]

# Threshold SNR at receiver
noise_sh = terminal_dim1.noise(noise_type="shot", P_r=P_r)
noise_th = terminal_dim1.noise(noise_type="thermal")
noise_bg = terminal_dim1.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
SNR_thres_sc, P_r_thres_sc, Np_quantum_limit_sc = terminal_dim1.threshold()


SNR = terminal_dim1.SNR_func(P_r)
BER = terminal_dim1.BER_func()
#------------------------------------------------------------------------
#-----------------------------FADE-STATISTICS----------------------------
#------------------------------------------------------------------------
data_metrics = ['elevation', 'P_r_0', 'h_scint', 'h_pe', 'h_bw', 'SNR', 'BER', 'number of fades', 'fade time', 'fractional fade time']
data_metrics_tuple = tuple(data_metrics)
number_of_metrics = len(data_metrics)
data = np.zeros((len(zenith_angles_dim1), number_of_metrics))
# data[0, :] = np.zeros(len(data_metrics))
data[:, 0] = np.rad2deg(elevation_angles_dim1)

for i in range(0, len(zenith_angles_dim1)):
    # data[i, 0] = np.rad2deg(elevation_angles_dim1[i])
    data[i, 1] = np.mean(P_r_0[i])
    data[i, 2] = np.mean(h_scint[i])
    data[i, 3] = np.mean(h_pe[i])
    data[i, 4] = np.mean(h_bw[i])
    data[i, 5] = np.mean(SNR[i])
    data[i, 6] = np.mean(BER[i])
    data[i, 7] = np.count_nonzero(P_r[i] < P_r_thres_sc[i])
    data[i, 8] = data[i, 7] * dt
    data[i, 9] = data[i, 8] / duration

# Add row to data array with zero values
data = np.vstack((np.zeros(len(data_metrics)), data))

#------------------------------------------------------------------------
#-----------------PLOT-RESULTS-(OPTIONAL)-AND-SAVE-TO-DATABASE-----------
#------------------------------------------------------------------------
terminal_dim1.plot(t, plot = "BER & SNR")
# turbulence_dim1.plot(t = t, plot='Cn')
terminal_dim1.plot(t = t, plot="pointing")
turbulence_dim1.plot(t = t, P_r=P_r, plot="scint pdf")
# turbulence_dim1.plot(t = t, plot='bw pdf')
# turbulence_dim1.print()
# plt.show()



con = sqlite3.connect("link_data.db")
cur = con.cursor()
cur.execute("CREATE TABLE performance_metrics(elevation, P_r_0, h_scint, h_pe, h_bw, SNR, BER, number_of_fades, fade_time, fractional_fade_time)")
cur.executemany("INSERT INTO performance_metrics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
con.commit()
cur.close()
con.close()
























# with open('dimension1_results.csv', 'w', newline='') as csvfile:
#     fieldnames = ['zenith', 'h_tot', 'r_tot']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     writer.writeheader()
#
#     for i in range(len(zenith_angles_dim1)):
#         writer.writerow({'zenith': zenith_angles_dim1[i], 'h_tot': h_tot[i], 'r_tot': r_tot[i]})






# print(dim_1_results.items())
    # h_tot[i] = list(h_tot[i])
    # r_tot[i] = list(r_tot[i])
    #
    # dim_1_results[str(np.rad2deg(zenith_angles_dim1[i]))] = [t, h_tot[i], r_tot[i]]



#     dim_1_results += [t, h_tot[i], r_tot[i]]
#
# np.savetxt("dimension1_results.csv", np.column_stack(dim_1_results), fmt='%.3e', delimiter="  ")

#     with open("dimension1_results.csv", "ab") as f:
#         f.write(b"\n")
#         np.savetxt(f, np.array([h_tot[i], r_tot[i]]), delimiter=",")
#     # with open("dimension1_results.csv", "ab") as f:
#     #     # f.write(b"\n")
#     #     np.savetxt(f, h_tot[i])
#
#
#     # print(np.rad2deg(zenith_angles_dim1[i]))
#     # print(h_tot[i])
#     # print(h_scint[i])
#     # print(abs(r_tot[i]))
#
# # print('1: ', dim_1_results[str(np.rad2deg(zenith_angles_dim1[0]))][0])
#
# # dim_1_results = np.array([r_norm, r_pe])
#
# # with open('dimension1_results.csv', 'w') as csv_file:
# #     writer = csv.writer(csv_file, delimiter=',')
#     # write timesteps
#     # writer.writerow(dim_1_results[str(np.rad2deg(zenith_angles_dim1[0]))][0])
#     # write other values
#     # for key, value in dim_1_results.items():
#     #     writer.writerow([key, value[0]])
#     #     writer.writerow([key, value[1]])
#     #     writer.writerow([key, value[2]])
#
# # np.savetxt('dimension1_results.csv', dim_1_results, delimiter=",")
#
# plt.show()