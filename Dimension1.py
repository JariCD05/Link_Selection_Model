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
#---------------------------------DIMENSION-1-SIMULATION-SETUP--------------------------------
#---------------------------------------------------------------------------------------------


# Here, the time interval is set for the simulation of dimension 1.
duration = 100.0
dt = 1.0e-4
t = np.arange(0.0, duration, dt)
# This is the number of samples (1.000.000)
steps = len(t)

# Here, the time interval is set for the simulation of dimension 0. This is not yet implemented at this point.
# steps = 100
# N = 100
# t_dim0 = np.linspace(0.0, 100000, N)
# steps_dim0 = len(t_dim0)

# A sequence of elevation and zenith angles are generated here.
# These angles will be looped, where for each angle a Monte Carlo simulation is done with 1.000.000 samples.
zenith_angles_dim1 = np.flip(np.arange(0.0,zenith_max, np.deg2rad(1.0)))
elevation_angles_dim1 = np.pi/2 - zenith_angles_dim1

# These angles are corrected with a refractive index.
zenith_angles_a = np.arcsin( 1/ n_index * np.sin(zenith_angles_dim1) )

# The range is computed, based on the aircraft height, spacecraft height and the zenith angles.
ranges_dim1 = np.sqrt((h_SC-h_AC)**2 + 2*h_SC*R_earth  + R_earth**2 * np.cos(zenith_angles_dim1)**2) \
                     - (R_earth+h_AC)*np.cos(zenith_angles_dim1)


# refractions     = np.array((1.0, 1.03, 1.3))

# The slew rate is computed for computation of the Cn^2 model, inside the turbulence class.
slew_rate = 1 / np.sqrt((R_earth+h_SC)**3 / mu_earth)

#------------------------------------------------------------------------
#--------------------------TURBULENCE-ANALYSIS---------------------------
#------------------------------------------------------------------------

# The turbulence class is initiated here. Inside the turbulence class, there are multiple functions that are run.
turbulence_dim1 = atm.turbulence(
     range=ranges_dim1,
     D_r=D_sc,
     link='up')

# Firstly, a windspeed profile is calculated, which is used for the Cn^2 model. This will then be used for the r0 profile.
windspeed_rms = turbulence_dim1.windspeed(slew=slew_rate, Vg=speed_AC, wind_model_type=wind_model_type)
Cn_HVB    = turbulence_dim1.Cn(turbulence_model=turbulence_model)
r0 = turbulence_dim1.r0(zenith_angles_dim1)
var_rytov = turbulence_dim1.var_rytov(zenith_angles_a)
var_scint = turbulence_dim1.var_scint(PDF_type="lognormal", zenith_angles=zenith_angles_a)
var_bw    = turbulence_dim1.var_bw(D_t = D_ac, zenith_angles=zenith_angles_a)

# Now, the turbulence variances can be computed and the Monte Carlo simulations for scintillation and beam wander are simulated.
h_scint = turbulence_dim1.PDF(ranges=ranges_dim1, w0=w0_ac, PDF_type="lognormal", steps=steps, zenith_angles=zenith_angles_a)
r_bw, h_bw = turbulence_dim1.PDF(ranges=ranges_dim1, w0=w0_ac, PDF_type="rayleigh", steps=steps, zenith_angles=zenith_angles_a, effect="beam wander")

# ------------------------------------------------------------------------
# --------------------------ATTENUATION-ANALYSIS--------------------------
# ------------------------------------------------------------------------
# The atmospheric attenutuation/extinction class is set up here.
attenuation = atm.attenuation(range_link=ranges_dim1)
h_ext = attenuation.T_ext(zenith_angles=zenith_angles_dim1, type="standard_atmosphere", steps=steps)


#------------------------------------------------------------------------
#-------------------------POINTING-ERROR-ANALYSIS------------------------
#------------------------------------------------------------------------

# #------------------------------------------------------------------------
# #-----------------------POINTING-ERROR-&-TERMINALS-----------------------
# #------------------------------------------------------------------------
#
terminal_ac = LCT.terminal_properties(D = D_ac,
                                       eff_quantum = eff_quantum_ac,
                                       BW = BW_ac,
                                       modulator = mod_ac,
                                       detection= detection_ac,
                                       FOV = FOV_ac,
                                       delta_wavelength=delta_wavelength_ac,
                                       M = M_ac,
                                       F = F_ac,
                                       )

terminal_sc = LCT.terminal_properties(D = D_ac,
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
r_pe, h_pe = terminal_sc.pointing(zenith_angles_dim1,
                                  ranges_dim1,
                                  pointing_error=angle_pe_ac,
                                  jitter_variance=var_pj_ac,
                                  dist_type="gaussian",
                                  steps=steps,
                                  w0=w0_ac,
                                  angle_divergence=angle_div_ac,
                                  r0 = r0
                                  )


# ------------------------------------------------------------------------
# -------------------------LINK-BUDGET-ANALYSIS---------------------------
# ------------------------------------------------------------------------
# The link budget class is initiated here.
link_budget_dim1 = LB.link_budget(ranges_dim1,
                                 P_t=P_ac,
                                 eff_t=eff_ac,
                                 eff_r=eff_sc,
                                 eff_coupling=eff_coupling_sc,
                                 D_t=D_sc,
                                 D_r=D_ac,
                                 w0 = w0_ac,
                                 angle_pe_t=angle_pe_ac,
                                 angle_pe_r=angle_pe_sc,
                                 var_pj_t=var_pj_ac,
                                 var_pj_r=var_pj_sc,
                                 n=n_index)


# The loss due to the turbulence beam spread effect is computed and taken as a static loss T_beamspread.
# This is then included in the link budget.
link_budget_dim1.beam_spread(r0=r0)


# The power and intensity at the receiver is computed from the link budget.
# All gains, losses and efficiencies are given in the link budget class.
P_r_0 = link_budget_dim1.P_r_0_func()
I_r_0 = link_budget_dim1.I_r_0_func()

#------------------------------------------------------------------------
#----------------------COMBINING-ALL-FLUCTUATION-LOSSES------------------
#------------------------------------------------------------------------

# The displacement fluctuations due to pointing error and beam wander are summed here.
r_tot = r_pe #+ r_bw
# h_tot = gaussian_beam(h_scint, np.abs(r_tot), D_ac, D_ac)

# Here the normalized power fluctuations are simply multiplied with each other.
# REF: REFERENCE POWER VECTORS FOR OPTICAL LEO DOWNLINK CHANNEL, D. GIGGENBACH ET AL. Fig.1.
h_tot = h_ext * h_scint * h_pe

# Making sure that all negative values (if present) are converted to ZERO values
for i in range(len(h_tot)):
    for j in range(len(h_tot[0,:])):
        if h_tot[i, j] < 0.0:
            h_tot[i,j] = 0.0

#------------------------------------------------------------------------
#-----------------------FLUCTUATING-SNR-&-BER-VALUES---------------------
#------------------------------------------------------------------------

# I_r = h_tot.transpose() * I_r_0
# w_0, w_r = beam_spread(D_sc, ranges_dim1)
# w_LT = beam_spread_turbulence(r0, w_0, w_r)
# r = np.linspace(0, D_sc/2, 1000)
# P_r = I_to_P(I_r, r, w_0, w_LT).transpose()


# The fluctuating signal power at the receiver is computed by multiplying the static power with the power fluctuations (h).
P_r = (h_tot.transpose() * P_r_0).transpose()
N_p = terminal_sc.Np_func(P_r)


# All relevant noise types are computed with analytical equations.
# These equations are approximations, based on the assumption of a gaussian distribution for each noise type.
noise_sh = terminal_sc.noise(noise_type="shot", P_r=P_r)
noise_th = terminal_sc.noise(noise_type="thermal")
noise_bg = terminal_sc.noise(noise_type="background", P_r=P_r, I_sun=I_sun)
noise_tot = terminal_sc.noise(noise_type="total")

# The threshold SNR is computed with the analytical relationship between SNR and BER, based on the modulation scheme.
# The threshold P_r is then computed from the threshold SNR, the noise and the detection technique.
# Finally, the photon count Np (expressed in # photons / bit) is then computed with Np = Pr / (Ep * Dr * eff_quantum)
SNR_thres_sc, P_r_thres_sc, Np_thres_sc = terminal_sc.threshold()

# The actually received SNR is computed for each sample, as it directly depends on the noise and the received power P_r.
# The BER is then computed with the analytical relationship between SNR and BER, based on the modulation scheme.
SNR = terminal_sc.SNR_func(P_r)
BER = terminal_sc.BER_func()
#------------------------------------------------------------------------
#-----------------------------FADE-STATISTICS----------------------------
#------------------------------------------------------------------------

# Here, the fading statistics are computed numerically.
# For each sample, the received power is evaluated and compared with the threshold power.
# When it is lower, it is counted as a fade. One fade is equal to the time step of 0.1 ms.
fades = np.zeros((len(zenith_angles_dim1), 3))
for i in range(0, len(zenith_angles_dim1)):
    # Number of fades
    fades[i, 0] = np.count_nonzero(P_r[i] < P_r_thres_sc[i])
    # Fade time
    fades[i, 1] = fades[i, 0] * dt
    # Fractional fade time
    fades[i, 2] = fades[i, 1] / duration

# ------------------------------------------------------------------------
# ---------------------------SAVE-TO-DATABASE-----------------------------
# ------------------------------------------------------------------------

# This is a list of the performance variables (metrics) that are computed in dimension 1 and saved to the database
data_metrics = ['elevation', 'P_r', 'h_scint', 'h_pe', 'h_bw', 'SNR', 'BER',
                'number of fades', 'fade time', 'fractional fade time',
                'P_r threshold', 'SNR threshold', 'Np threshold', 'Data rate', 'Np', 'noise']
data_metrics_tuple = tuple(data_metrics)
number_of_metrics = len(data_metrics)

# A data array is created here and filled with all the computed performance variables
data = np.zeros((len(zenith_angles_dim1), number_of_metrics))
data[:, 0] = np.rad2deg(elevation_angles_dim1)

for i in range(0, len(zenith_angles_dim1)):
    # data[i, 0] = np.rad2deg(elevation_angles_dim1[i])
    data[i, 1] = np.mean(P_r[i])
    data[i, 2] = np.mean(h_scint[i])
    data[i, 3] = np.mean(h_pe[i])
    data[i, 4] = np.mean(h_bw[i])
    data[i, 5] = np.mean(SNR[i])
    data[i, 6] = np.mean(BER[i])
    data[i, 7] = fades[i, 0]
    data[i, 8] = fades[i, 1]
    data[i, 9] = fades[i, 2]
    data[i, 10] = np.mean(P_r_thres_sc[i])
    data[i, 11] = SNR_thres_sc
    data[i, 12] = np.mean(Np_thres_sc[i])
    data[i, 13] = data_rate
    data[i, 14] = np.mean(N_p[i])
    data[i, 15] = np.mean(noise_tot[i])

# Add row to data array with zero values
data = np.vstack((np.zeros(len(data_metrics)), data))

# Save data to sqlite3 database, 14 values are saved for each elevation angle (14 columns and N rows)
con = sqlite3.connect("link_data_16_param.db")
cur = con.cursor()
cur.execute("CREATE TABLE performance_metrics(elevation, P_r_0, h_scint, h_pe, h_bw, SNR, BER, number_of_fades, fade_time, fractional_fade_time, P_r_threshold, SNR_threshold, Np_threshold, data_rate, N_p, noise)")
cur.executemany("INSERT INTO performance_metrics VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", data)
con.commit()
cur.close()
con.close()

#------------------------------------------------------------------------
#-------------------------PLOT-RESULTS-(OPTIONAL)------------------------
#------------------------------------------------------------------------

# print(data[:, 0])
# print(data[:, 1])
# print(data[:, 2])
# print(data[:, 3])
# print(data[:, 5])
# print(data[:, 6])

terminal_sc.plot(t = t, data=data, plot = "BER & SNR")
terminal_sc.plot(t = t, data=data, plot = "data vs elevation")
turbulence_dim1.plot(t = t, plot='Cn')
terminal_sc.plot(t = t, data=data, plot="pointing")
turbulence_dim1.plot(t = t, P_r=P_r, elevation=elevation_angles_dim1, plot="scint pdf")
turbulence_dim1.plot(t = t, plot='bw pdf')

link_budget_dim1.print(elevation=elevation_angles_dim1, index=1, t=np.array([1,2]))
turbulence_dim1.print()

# plt.show()
























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